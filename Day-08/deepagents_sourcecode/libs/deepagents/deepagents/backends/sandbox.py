"""기본 샌드박스 구현체 (`BaseSandbox`) — `SandboxBackendProtocol`을 구현합니다.

파일 목록 조회, grep, glob, read는 `execute()`를 통해 셸 명령으로 처리됩니다.
write는 `upload_files()`에 콘텐츠 전송을 위임합니다. edit은 페이로드가
`_EDIT_INLINE_MAX_BYTES` 미만일 경우 서버 측 `execute()`를 사용하고,
그보다 크면 old/new 문자열을 임시 파일로 업로드한 후 서버 측 replace 스크립트로
처리합니다.

구체적인 서브클래스는 `execute()`와 `upload_files()`를 구현하며,
나머지 모든 작업은 이 두 메서드로부터 파생됩니다.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
from abc import ABC, abstractmethod
from typing import Final

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import _get_file_type

logger = logging.getLogger(__name__)

_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json
import base64

# base64로 인코딩된 파라미터를 디코딩
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>&1"""
"""메타데이터와 함께 패턴에 일치하는 파일을 검색합니다.

셸 이스케이핑 문제를 방지하기 위해 base64로 인코딩된 파라미터를 사용합니다.
"""

_WRITE_CHECK_TEMPLATE = """python3 -c "
import os, sys, base64

path = base64.b64decode('{path_b64}').decode('utf-8')
if os.path.exists(path):
    print('Error: File already exists: ' + repr(path))
    sys.exit(1)
os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
" 2>&1"""
"""쓰기 작업의 사전 확인: 대상 파일이 존재하지 않는지 검증하고
상위 디렉터리를 생성합니다.

base64로 인코딩된 경로(소형)만 보간되며, 파일 내용은
`upload_files()`를 통해 별도로 전송됩니다.
"""

_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys, os, base64, json

payload = json.loads(base64.b64decode(sys.stdin.read().strip()).decode('utf-8'))
path, old, new = payload['path'], payload['old'], payload['new']
replace_all = payload.get('replace_all', False)

if not os.path.isfile(path):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)

with open(path, 'rb') as f:
    raw = f.read()

try:
    text = raw.decode('utf-8')
except UnicodeDecodeError:
    print(json.dumps({{'error': 'not_a_text_file'}}))
    sys.exit(0)

count = text.count(old)
if count == 0:
    print(json.dumps({{'error': 'string_not_found'}}))
    sys.exit(0)
if count > 1 and not replace_all:
    print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
    sys.exit(0)

result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
with open(path, 'wb') as f:
    f.write(result.encode('utf-8'))

print(json.dumps({{'count': count}}))
" 2>&1 <<'__DEEPAGENTS_EDIT_EOF__'
{payload_b64}
__DEEPAGENTS_EDIT_EOF__
"""
# DEEPAGENTS_EDIT_EOF 끝에 개행을 유지해야 입력 종료를 인식합니다.
# 일부 통합 환경에서는 필요하지 않을 수 있습니다.

"""`execute()`를 통한 서버 측 파일 편집.

파일을 읽고, 문자열 치환을 수행한 후 샌드박스 내에서 다시 씁니다.
페이로드(경로, old/new 문자열, replace_all 플래그)는 셸 이스케이핑 문제를 피하기
위해 heredoc stdin을 통해 base64 인코딩된 JSON으로 전달됩니다.

출력: 성공 시 `{{"count": N}}`, 실패 시 `{{"error": ...}}`를 포함하는 단일 줄 JSON.

`_EDIT_INLINE_MAX_BYTES` 미만 페이로드에 사용되며, 그보다 큰 페이로드는
old/new 문자열을 임시 파일로 전송하는 `_edit_via_upload()`로 폴백됩니다.

heredoc 피드를 개행으로 입력 종료를 감지하는 통합 환경에서 완료를 인식할 수
있도록 `__DEEPAGENTS_EDIT_EOF__` 뒤에 후행 개행을 유지합니다.
"""

_EDIT_INLINE_MAX_BYTES: Final = 50_000
"""인라인 서버 측 편집을 위한 old_string + new_string의 최대 결합 바이트 크기.

이 크기를 초과하는 페이로드는 일부 샌드박스 공급자가 execute() 요청 본문에 부과하는
크기 제한을 피하기 위해 _edit_via_upload (임시 파일 업로드 + 서버 측 치환)를 사용합니다.
"""

_EDIT_TMPFILE_TEMPLATE = """python3 -c "
import os, sys, json, base64

old_path = base64.b64decode('{old_path_b64}').decode('utf-8')
new_path = base64.b64decode('{new_path_b64}').decode('utf-8')
target = base64.b64decode('{target_b64}').decode('utf-8')
replace_all = {replace_all}

try:
    old = open(old_path, 'rb').read().decode('utf-8')
    new = open(new_path, 'rb').read().decode('utf-8')
except Exception as e:
    print(json.dumps({{'error': 'temp_read_failed', 'detail': str(e)}}))
    sys.exit(0)
finally:
    for p in (old_path, new_path):
        try: os.remove(p)
        except OSError: pass

if not os.path.isfile(target):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)

with open(target, 'rb') as f:
    raw = f.read()

try:
    text = raw.decode('utf-8')
except UnicodeDecodeError:
    print(json.dumps({{'error': 'not_a_text_file'}}))
    sys.exit(0)

count = text.count(old)
if count == 0:
    print(json.dumps({{'error': 'string_not_found'}}))
    sys.exit(0)
if count > 1 and not replace_all:
    print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
    sys.exit(0)

result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
with open(target, 'wb') as f:
    f.write(result.encode('utf-8'))

print(json.dumps({{'count': count}}))
" 2>&1"""
"""대형 페이로드를 위한 임시 파일 업로드 방식의 서버 측 파일 편집.

old/new 문자열을 `upload_files()`를 통해 임시 파일로 업로드한 후, 이 스크립트가
해당 파일을 읽어 소스 파일(샌드박스를 벗어나지 않음)에서 치환을 수행하고
임시 파일을 정리합니다.

출력: 성공 시 `{{"count": N}}`, 실패 시 `{{"error": ...}}`를 포함하는 단일 줄 JSON.
`_EDIT_COMMAND_TEMPLATE`과 동일한 성공 계약을 가지며, 추가로 업로드된 임시 파일을
읽을 수 없을 때 `{{"error": "temp_read_failed", "detail": ...}}`를 생성합니다.
"""

_READ_COMMAND_TEMPLATE = """python3 -c "
import os, sys, base64, json

MAX_OUTPUT_BYTES = 500 * 1024
MAX_BINARY_BYTES = 500 * 1024
TRUNCATION_MSG = '\\n\\n' + (
    '[Output was truncated due to size limits. '
    'This paginated read result exceeded the sandbox stdout limit. '
    'Continue reading with a larger offset or smaller limit to inspect the rest of the file.]'
)

path = base64.b64decode('{path_b64}').decode('utf-8')

if not os.path.isfile(path):
    print(json.dumps({{'error': 'file_not_found'}}))
    sys.exit(0)

if os.path.getsize(path) == 0:
    print(json.dumps({{'encoding': 'utf-8', 'content': 'System reminder: File exists but has empty contents'}}))
    sys.exit(0)

file_type = '{file_type}'
if file_type != 'text':
    file_size = os.path.getsize(path)
    if file_size > MAX_BINARY_BYTES:
        print(json.dumps({{'error': 'Binary file exceeds maximum preview size of ' + str(MAX_BINARY_BYTES) + ' bytes'}}))
        sys.exit(0)
    with open(path, 'rb') as f:
        raw = f.read()
    print(json.dumps({{'encoding': 'base64', 'content': base64.b64encode(raw).decode('ascii')}}))
    sys.exit(0)

with open(path, 'rb') as f:
    raw_prefix = f.read(8192)

try:
    raw_prefix.decode('utf-8')
except UnicodeDecodeError:
    with open(path, 'rb') as f:
        raw = f.read()
    print(json.dumps({{'encoding': 'base64', 'content': base64.b64encode(raw).decode('ascii')}}))
    sys.exit(0)

offset = {offset}
limit = {limit}
line_count = 0
returned_lines = 0
truncated = False
parts = []
current_bytes = 0
msg_bytes = len(TRUNCATION_MSG.encode('utf-8'))
effective_limit = MAX_OUTPUT_BYTES - msg_bytes

with open(path, 'r', encoding='utf-8', newline=None) as f:
    for raw_line in f:
        line_count += 1
        if line_count <= offset:
            continue
        if returned_lines >= limit:
            break

        line = raw_line.rstrip('\\n').rstrip('\\r')
        piece = line if returned_lines == 0 else '\\n' + line
        piece_bytes = len(piece.encode('utf-8'))
        if current_bytes + piece_bytes > effective_limit:
            truncated = True
            remaining_bytes = effective_limit - current_bytes
            if remaining_bytes > 0:
                prefix = piece.encode('utf-8')[:remaining_bytes].decode('utf-8', errors='ignore')
                if prefix:
                    parts.append(prefix)
                    current_bytes += len(prefix.encode('utf-8'))
            break

        parts.append(piece)
        current_bytes += piece_bytes
        returned_lines += 1

if returned_lines == 0 and not truncated:
    print(json.dumps({{'error': 'Line offset ' + str(offset) + ' exceeds file length (' + str(line_count) + ' lines)'}}))
    sys.exit(0)

text = ''.join(parts)
if truncated:
    text += TRUNCATION_MSG

print(json.dumps({{'encoding': 'utf-8', 'content': text}}))
" 2>&1"""
"""서버 측 페이지네이션으로 파일 내용을 읽습니다.

`execute()`를 통해 샌드박스에서 실행됩니다. 페이지네이션된 텍스트 읽기에서
전체 파일 전송을 피하기 위해 요청된 페이지만 반환합니다. 경로는 base64로
인코딩되며, `file_type`, `offset`, `limit`은 직접 보간됩니다
(내부 코드에서 오는 값이므로 안전).

출력: 성공 시 `{{"encoding": ..., "content": ...}}`, 실패 시 `{{"error": ...}}`를
포함하는 단일 줄 JSON.
"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """execute()를 핵심 추상 메서드로 하는 기본 샌드박스 구현체.

    이 클래스는 모든 프로토콜 메서드의 기본 구현을 제공합니다.
    파일 목록 조회, grep, glob은 `execute()`를 통해 셸 명령으로 처리됩니다.
    read는 페이지네이션 접근을 위해 `execute()`를 통해 서버 측 Python 스크립트를
    실행합니다. write는 `upload_files()`에 콘텐츠 전송을 위임합니다. edit은
    소형 페이로드에는 서버 측 스크립트를 사용하고, 대형 페이로드에는 old/new 문자열을
    임시 파일로 업로드한 후 서버 측 치환을 수행합니다.

    서브클래스는 반드시 `execute()`, `upload_files()`, `download_files()`,
    그리고 `id` 프로퍼티를 구현해야 합니다.
    """

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """샌드박스에서 명령을 실행하고 ExecuteResponse를 반환합니다.

        Args:
            command: 실행할 전체 셸 명령 문자열.
            timeout: 명령 완료까지 대기할 최대 시간(초).

                None이면 백엔드의 기본 타임아웃을 사용합니다.

        Returns:
            결합된 출력, 종료 코드, 절삭 플래그를 포함하는 ExecuteResponse.
        """

    def ls(self, path: str) -> LsResult:
        """os.scandir을 사용해 파일 메타데이터를 포함한 구조화된 목록을 반환합니다."""
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
        cmd = f"""python3 -c "
import os
import json
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null"""

        result = self.execute(cmd)

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
            except json.JSONDecodeError:
                continue

        return LsResult(entries=file_infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """서버 측 줄 기반 페이지네이션으로 파일 내용을 읽습니다.

        `execute()`를 통해 샌드박스에서 Python 스크립트를 실행하여
        파일을 읽고, 인코딩을 감지하며, 텍스트 파일에 대해 offset/limit 페이지네이션을
        적용합니다. 요청된 페이지만 전송되며, 텍스트 출력은 백엔드 stdout/로그 전송
        실패를 방지하기 위해 약 500 KiB로 제한됩니다. 이 제한을 초과하면 반환된
        내용이 절삭되며 다른 `offset`이나 더 작은 `limit`으로 페이지네이션을 계속하도록
        안내합니다.

        바이너리 파일(비 UTF-8)은 페이지네이션 없이 base64로 인코딩하여 반환됩니다.

        Args:
            file_path: 읽을 파일의 절대 경로.
            offset: 시작 줄 번호 (0 기반 인덱스).

                텍스트 파일에만 적용됩니다.
            limit: 반환할 최대 줄 수.

                텍스트 파일에만 적용됩니다.

        Returns:
            성공 시 `file_data`, 실패 시 `error`를 포함하는 `ReadResult`.
        """
        file_type = _get_file_type(file_path)
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")

        # 타입 검사를 우회하는 호출자를 방어하기 위해 int로 강제 변환합니다.
        cmd = _READ_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            file_type=file_type,
            offset=int(offset),
            limit=int(limit),
        )
        result = self.execute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            detail = output[:200] if output else "(empty)"
            return ReadResult(error=f"File '{file_path}': unexpected server response: {detail}")

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            return ReadResult(error=f"File '{file_path}': unexpected server response: {detail}")

        if "error" in data:
            return ReadResult(error=f"File '{file_path}': {data['error']}")

        return ReadResult(
            file_data=FileData(
                content=data["content"],
                encoding=data.get("encoding", "utf-8"),
            )
        )

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """이미 존재하는 경우 실패하며 새 파일을 생성합니다.

        Args:
            file_path: 새 파일의 절대 경로.
            content: 쓸 UTF-8 텍스트 내용.

        Returns:
            성공 시 `path`, 실패 시 `error`를 포함하는 `WriteResult`.
        """
        # 존재 여부 확인 + mkdir. 이 확인과 아래 업로드 사이에 TOCTOU 창이 있습니다 -
        # 그 사이에 다른 프로세스가 파일을 생성할 수 있습니다.
        # 작업을 분리하는 데 따른 고유한 한계입니다.
        path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
        check_cmd = _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)
        result = self.execute(check_cmd)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        responses = self.upload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            # 도달 불가능한 조건에 도달한 경우
            msg = f"Responses was expected to return 1 result, but it returned {len(responses)} with type {type(responses)}"
            raise AssertionError(msg)
        response = responses[0]
        if response.error:
            return WriteResult(error=f"Failed to write file '{file_path}': {response.error}")

        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """정확한 문자열 발생 횟수를 교체하여 파일을 편집합니다.

        소형 페이로드(`_EDIT_INLINE_MAX_BYTES` 미만의 결합 old/new)의 경우
        `execute()`를 통해 서버 측 Python 스크립트를 실행합니다 — 단일 왕복,
        파일 전송 없음. 대형 페이로드의 경우 old/new 문자열을 임시 파일로 업로드하고
        서버 측 치환 스크립트를 실행합니다 — 소스 파일은 샌드박스를 벗어나지 않습니다.

        Args:
            file_path: 편집할 파일의 절대 경로.
            old_string: 찾을 정확한 부분 문자열.
            new_string: 대체 문자열.
            replace_all: `True`이면 모든 발생을 교체합니다.

                `False`(기본값)이면 발생 횟수가 2개 이상일 때 오류를 반환합니다.

        Returns:
            성공 시 `path`와 `occurrences`, 실패 시 `error`를 포함하는 `EditResult`.
        """
        payload_size = len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))

        if payload_size <= _EDIT_INLINE_MAX_BYTES:
            return self._edit_inline(file_path, old_string, new_string, replace_all)

        return self._edit_via_upload(file_path, old_string, new_string, replace_all)

    def _edit_inline(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """`execute()`를 통한 서버 측 치환 — 단일 왕복."""
        payload = json.dumps(
            {
                "path": file_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
            }
        )
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
        result = self.execute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if "error" in data:
            return self._map_edit_error(data["error"], file_path, old_string)

        return EditResult(
            path=file_path,
            occurrences=data.get("count", 1),
        )

    def _edit_via_upload(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """old/new를 임시 파일로 업로드하고 서버 측에서 치환합니다.

        소스 파일은 샌드박스를 벗어나지 않습니다. old/new 문자열만
        `upload_files()`를 통해 전송되며, 서버 측 스크립트가 해당 파일을 읽고
        치환을 수행한 후 임시 파일을 정리합니다.
        """
        uid = base64.b32encode(os.urandom(10)).decode("ascii").lower()
        old_tmp = f"/tmp/.deepagents_edit_{uid}_old"  # noqa: S108  # 80비트 랜덤 uid를 가진 샌드박스 내부 임시 파일
        new_tmp = f"/tmp/.deepagents_edit_{uid}_new"  # noqa: S108

        resps = self.upload_files(
            [
                (old_tmp, old_string.encode("utf-8")),
                (new_tmp, new_string.encode("utf-8")),
            ]
        )
        if len(resps) < 2:  # noqa: PLR2004  # 정확히 2개의 응답을 기대함
            return EditResult(error=f"Error editing file '{file_path}': upload returned no response")
        for r in resps:
            if r.error:
                return EditResult(error=f"Error editing file '{file_path}': {r.error}")

        cmd = _EDIT_TMPFILE_TEMPLATE.format(
            old_path_b64=base64.b64encode(old_tmp.encode("utf-8")).decode("ascii"),
            new_path_b64=base64.b64encode(new_tmp.encode("utf-8")).decode("ascii"),
            target_b64=base64.b64encode(file_path.encode("utf-8")).decode("ascii"),
            replace_all=replace_all,
        )
        result = self.execute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            # 스크립트가 시작되지 않았거나 finally 블록이 실행되지 않았을 수 있으므로
            # 임시 파일을 최선의 노력으로 정리합니다.
            cleanup = self.execute(f"rm -f {shlex.quote(old_tmp)} {shlex.quote(new_tmp)}")
            if cleanup.exit_code != 0:
                logger.warning(
                    "Failed to clean up temp files for edit %s: %s",
                    file_path,
                    cleanup.output[:200],
                )
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if "error" in data:
            return self._map_edit_error(data["error"], file_path, old_string)

        return EditResult(
            path=file_path,
            occurrences=data.get("count", 1),
        )

    @staticmethod
    def _map_edit_error(error: str, file_path: str, old_string: str) -> EditResult:
        """서버 측 오류 코드를 `EditResult` 객체로 매핑합니다."""
        if error == "file_not_found":
            return EditResult(
                error=f"Error: File '{file_path}' not found",
            )
        if error == "not_a_text_file":
            return EditResult(
                error=f"Error: File '{file_path}' is not a text file",
            )
        if error == "string_not_found":
            return EditResult(
                error=f"Error: String not found in file: '{old_string}'",
            )
        if error == "multiple_occurrences":
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.",
            )
        return EditResult(error=f"Error editing file '{file_path}': {error}")

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """`grep -F`를 사용해 리터럴 문자열로 파일 내용을 검색합니다.

        Args:
            pattern: 검색할 리터럴 문자열 (정규식 아님).
            path: 검색할 디렉터리 또는 파일.

                기본값은 `"."`.
            glob: 검색을 제한하는 선택적 파일명 glob
                (예: `'*.py'`).

        Returns:
            `GrepMatch` 딕셔너리 목록을 포함하는 `GrepResult`, 실패 시 `error`.
        """
        search_path = shlex.quote(path or ".")

        # 구조화된 출력을 위한 grep 명령 구성
        grep_opts = "-rHnF"  # 재귀, 파일명 포함, 줄 번호 포함, 고정 문자열(리터럴)

        # glob 패턴이 지정된 경우 추가
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        # 셸을 위한 패턴 이스케이핑
        pattern_escaped = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = self.execute(cmd)

        output = result.output.rstrip()
        if not output:
            return GrepResult(matches=[])

        # grep 출력을 GrepMatch 객체로 파싱
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # 형식: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= 3:  # noqa: PLR2004  # grep 출력 필드 수
                matches.append(
                    {
                        "path": parts[0],
                        "line": int(parts[1]),
                        "text": parts[2],
                    }
                )

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """`GlobResult`를 반환하는 구조화된 glob 매칭."""
        # 이스케이핑 문제를 피하기 위해 패턴과 경로를 base64로 인코딩
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

        # JSON 출력을 FileInfo 딕셔너리로 파싱
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append(
                    {
                        "path": data["path"],
                        "is_dir": data["is_dir"],
                    }
                )
            except json.JSONDecodeError:
                continue

        return GlobResult(matches=file_infos)

    @property
    @abstractmethod
    def id(self) -> str:
        """샌드박스 백엔드의 고유 식별자."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """샌드박스에 여러 파일을 업로드합니다.

        구현체는 부분 성공을 지원해야 합니다 - 파일별 예외를 처리하고
        예외를 전파하는 대신 `FileUploadResponse` 객체에 오류를 반환합니다.

        upload_files는 상위 경로가 존재하는지 확인할 책임이 있습니다
        (사용자가 해당 디렉터리에 쓰기 권한이 있는 경우).
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """샌드박스에서 여러 파일을 다운로드합니다.

        구현체는 부분 성공을 지원해야 합니다 - 파일별 예외를 처리하고
        예외를 전파하는 대신 `FileDownloadResponse` 객체에 오류를 반환합니다.
        """
