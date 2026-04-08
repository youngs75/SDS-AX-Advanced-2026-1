"""백엔드 추상화 계층의 핵심 프로토콜 정의 모듈.

이 모듈은 Deep Agents 프레임워크에서 모든 백엔드 구현체가 따라야 하는
BackendProtocol 및 SandboxBackendProtocol을 정의합니다.

프로토콜 기반 설계의 이점:
- **다형성(Polymorphism)**: 동일한 인터페이스로 상태 저장소, 파일시스템, 데이터베이스 등
  다양한 백엔드를 투명하게 교체할 수 있습니다.
- **교체 가능성(Replaceability)**: 사용하는 쪽 코드 변경 없이 백엔드 구현체만
  바꿔 끼울 수 있어 유지보수성과 테스트 용이성이 크게 향상됩니다.

BackendProtocol은 파일 읽기/쓰기/수정/목록 조회/검색 등 기본 파일 연산을 정의하며,
SandboxBackendProtocol은 여기에 셸 명령 실행(execute) 기능을 추가하여
격리된 컨테이너·VM·원격 호스트 환경을 지원합니다.
"""

import abc
import asyncio
import inspect
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, NotRequired, TypeAlias

from langchain.tools import ToolRuntime
from typing_extensions import TypedDict

FileFormat = Literal["v1", "v2"]
r"""파일 저장 형식 버전.

- `'v1'`: 레거시 형식 — `content`가 `list[str]` (줄 단위로 `\\n` 분리),
    `encoding` 필드 없음.
- `'v2'`: 현행 형식 — `content`가 일반 `str` (UTF-8 텍스트 또는 base64 인코딩 바이너리),
    `encoding` 필드 포함 (`"utf-8"` 또는 `"base64"`).
"""

logger = logging.getLogger(__name__)

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]
"""파일 업로드/다운로드 연산의 표준화된 오류 코드.

LLM이 이해하고 복구를 시도할 수 있는 일반적이고 회복 가능한 오류를 나타냅니다:

- file_not_found: 요청한 파일이 존재하지 않음 (다운로드 시)
- permission_denied: 해당 연산에 대한 접근 권한 없음
- is_directory: 디렉터리를 단일 파일로 다운로드하려 시도함
- invalid_path: 경로 구문이 잘못되었거나 유효하지 않은 문자를 포함함
"""


@dataclass
class FileDownloadResponse:
    """단일 파일 다운로드 연산의 결과.

    배치 연산에서 부분 성공을 허용하도록 설계된 응답 구조입니다.

    파일 연산을 수행하는 LLM을 위해 회복 가능한 특정 조건에 대해
    `FileOperationError` 리터럴로 오류를 표준화합니다.

    Examples:
        >>> # 성공 시
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # 실패 시
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str
    """요청된 파일 경로. 배치 결과 처리 시 결과를 입력과 연결하거나
    오류 메시지 작성에 유용합니다."""

    content: bytes | None = None
    """성공 시 바이트 형태의 파일 내용, 실패 시 `None`."""

    error: FileOperationError | None = None
    """알려진 조건에 대한 `FileOperationError` 리터럴, 또는
    정규화할 수 없는 실패의 경우 백엔드 고유 오류 문자열.

    성공 시 `None`.
    """


@dataclass
class FileUploadResponse:
    """단일 파일 업로드 연산의 결과.

    배치 연산에서 부분 성공을 허용하도록 설계된 응답 구조입니다.

    파일 연산을 수행하는 LLM을 위해 회복 가능한 특정 조건에 대해
    `FileOperationError` 리터럴로 오류를 표준화합니다.

    Examples:
        >>> # 성공 시
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # 실패 시
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str
    """요청된 파일 경로.

    배치 결과 처리 시 결과를 입력과 연결하거나 오류 메시지 작성에 유용합니다.
    """

    error: FileOperationError | None = None
    """error: 알려진 조건에 대한 `FileOperationError` 리터럴, 또는
    정규화할 수 없는 실패의 경우 백엔드 고유 오류 문자열.

    성공 시 `None`.
    """


class FileInfo(TypedDict):
    """구조화된 파일 목록 정보.

    백엔드 간 최소 계약 구조입니다. `path`만 필수이며,
    나머지 필드는 백엔드에 따라 제공되지 않을 수 있습니다 (최선 제공).
    """

    path: str
    """절대 경로 또는 상대 경로."""

    is_dir: NotRequired[bool]
    """해당 항목이 디렉터리인지 여부."""

    size: NotRequired[int]
    """파일 크기 (바이트, 근사값)."""

    modified_at: NotRequired[str]
    """마지막 수정 시각의 ISO 8601 타임스탬프 (알려진 경우)."""


class GrepMatch(TypedDict):
    """grep 검색에서 발견된 단일 매칭 결과."""

    path: str
    """매칭이 발견된 파일의 경로."""

    line: int
    """매칭된 줄 번호 (1부터 시작)."""

    text: str
    """매칭된 줄의 내용."""


class FileData(TypedDict):
    """메타데이터를 포함한 파일 내용 저장 자료구조."""

    content: str
    """파일 내용 문자열 (UTF-8 텍스트 또는 base64 인코딩 바이너리)."""

    encoding: str
    """내용 인코딩: 텍스트는 `"utf-8"`, 바이너리는 `"base64"`."""

    created_at: NotRequired[str]
    """파일 생성 시각의 ISO 8601 타임스탬프."""

    modified_at: NotRequired[str]
    """마지막 수정 시각의 ISO 8601 타임스탬프."""


@dataclass
class ReadResult:
    """백엔드 read 연산의 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        file_data: 성공 시 FileData 딕셔너리, 실패 시 None.
    """

    error: str | None = None
    file_data: FileData | None = None


class _Unset:
    """명시적 파라미터 사용 여부를 감지하기 위한 센티넬 타입."""


Unset = _Unset()


def _normalize_files_update(
    files_update: dict[str, Any] | None | _Unset,
) -> dict[str, Any] | None:
    """파일 업데이트 값을 정규화합니다.

    _Unset 센티넬 값이면 None을 반환하고,
    실제 값이 전달된 경우 deprecated 경고를 발생시킨 후 해당 값을 반환합니다.
    """
    if isinstance(files_update, _Unset):
        return None

    warnings.warn(
        "`files_update` is deprecated and will be removed in v0.7. State updates are now handled internally by the backend.",
        DeprecationWarning,
        stacklevel=3,
    )
    return files_update


@dataclass(init=False)
class WriteResult:
    """백엔드 write 연산의 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        path: 성공 시 작성된 파일의 절대 경로, 실패 시 None.

    Examples:
        >>> WriteResult(path="/f.txt")
        >>> WriteResult(error="File exists")
    """

    error: str | None
    path: str | None
    files_update: dict[str, Any] | None

    def __init__(
        self,
        error: str | None = None,
        path: str | None = None,
        files_update: dict[str, Any] | None | _Unset = Unset,
    ) -> None:
        """WriteResult를 초기화합니다."""
        self.error = error
        self.path = path
        self.files_update = _normalize_files_update(files_update)


@dataclass(init=False)
class EditResult:
    """백엔드 edit 연산의 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        path: 성공 시 수정된 파일의 절대 경로, 실패 시 None.
        occurrences: 수행된 교체 횟수, 실패 시 None.

    Examples:
        >>> EditResult(path="/f.txt", occurrences=1)
        >>> EditResult(error="File not found")
    """

    error: str | None
    path: str | None
    files_update: dict[str, Any] | None
    occurrences: int | None

    def __init__(
        self,
        error: str | None = None,
        path: str | None = None,
        files_update: dict[str, Any] | None | _Unset = Unset,
        occurrences: int | None = None,
    ) -> None:
        """EditResult를 초기화합니다."""
        self.error = error
        self.path = path
        self.files_update = _normalize_files_update(files_update)
        self.occurrences = occurrences


@dataclass
class LsResult:
    """백엔드 ls 연산의 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        entries: 성공 시 파일 정보 딕셔너리 목록, 실패 시 None.
    """

    error: str | None = None
    entries: list["FileInfo"] | None = None


@dataclass
class GrepResult:
    """백엔드 grep 연산의 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        matches: 성공 시 grep 매칭 딕셔너리 목록, 실패 시 None.
    """

    error: str | None = None
    matches: list["GrepMatch"] | None = None


@dataclass
class GlobResult:
    """백엔드 glob 연산의 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        matches: 성공 시 매칭된 파일 정보 딕셔너리 목록, 실패 시 None.
    """

    error: str | None = None
    matches: list["FileInfo"] | None = None


# @abstractmethod를 사용하지 않은 이유: 일부 메서드만 구현한 하위 클래스가 깨지는 것을 방지
class BackendProtocol(abc.ABC):  # noqa: B024
    r"""플러거블 메모리 백엔드를 위한 통합 프로토콜.

    백엔드는 상태 저장소, 파일시스템, 데이터베이스 등 다양한 위치에 파일을 저장하면서
    파일 연산에 대한 균일한 인터페이스를 제공합니다.

    모든 파일 데이터는 다음 구조의 딕셔너리로 표현됩니다::

        {
            "content": str,  # 텍스트 내용(utf-8) 또는 base64 인코딩 바이너리
            "encoding": str,  # 텍스트는 "utf-8", 바이너리는 "base64"
            "created_at": str,  # ISO 형식 타임스탬프
            "modified_at": str,  # ISO 형식 타임스탬프
        }

    Note:
        레거시 데이터에는 `"content": list[str]` (줄 단위로 `\\n` 분리)이
        포함될 수 있습니다. 백엔드는 하위 호환성을 위해 이를 허용하되
        `DeprecationWarning`을 발생시킵니다.
    """

    def ls(self, path: str) -> "LsResult":
        """디렉터리 내 모든 파일을 메타데이터와 함께 목록 조회합니다.

        Args:
            path: 목록 조회할 디렉터리의 절대 경로. '/'로 시작해야 합니다.

        Returns:
            디렉터리 항목 또는 오류를 담은 LsResult.
        """
        # 하위 클래스가 deprecated ls_info를 재정의한 경우 경고 후 위임
        if type(self).ls_info is not BackendProtocol.ls_info:
            warnings.warn(
                "`ls_info` is deprecated and will be removed in v0.7; rename to `ls` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return LsResult(entries=self.ls_info(path))

        raise NotImplementedError

    async def als(self, path: str) -> "LsResult":
        """ls의 비동기 버전."""
        return await asyncio.to_thread(self.ls, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """파일 내용을 줄 번호와 함께 읽습니다.

        Args:
            file_path: 읽을 파일의 절대 경로. '/'로 시작해야 합니다.
            offset: 읽기 시작할 줄 번호 (0 기반 인덱스). 기본값: 0.
            limit: 읽을 최대 줄 수. 기본값: 2000.

        Returns:
            줄 번호가 포함된 파일 내용 문자열 (cat -n 형식, 1부터 시작).
            2000자를 초과하는 줄은 잘립니다.

            파일이 존재하지 않거나 읽을 수 없는 경우 오류 문자열을 반환합니다.
        """
        raise NotImplementedError

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """read의 비동기 버전."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """파일 내에서 리터럴 텍스트 패턴을 검색합니다.

        Args:
            pattern: 검색할 리터럴 문자열 (정규식 아님).

                파일 내용에서 정확한 부분 문자열 매칭을 수행합니다.

                예: "TODO"는 "TODO"가 포함된 모든 줄과 매칭됩니다.

            path: 검색할 디렉터리 경로 (선택).

                None이면 현재 작업 디렉터리에서 검색합니다.

                예: `'/workspace/src'`

            glob: 검색할 파일을 필터링하는 glob 패턴 (선택).

                내용이 아닌 파일명/경로로 필터링합니다.

                표준 glob 와일드카드를 지원합니다:

                - `*`: 파일명 내 임의의 문자열 매칭
                - `**`: 디렉터리를 재귀적으로 매칭
                - `?`: 단일 문자 매칭
                - `[abc]`: 집합에서 한 문자 매칭

        Examples:
            - `'*.py'` - Python 파일만 검색
            - `'**/*.txt'` - 모든 `.txt` 파일을 재귀 검색
            - `'src/**/*.js'` - src/ 하위의 JS 파일 검색
            - `'test[0-9].txt'` - `test0.txt`, `test1.txt` 등 검색

        Returns:
            매칭 결과 또는 오류를 담은 `GrepResult`.
        """
        # 하위 클래스가 deprecated grep_raw를 재정의한 경우 경고 후 위임
        if type(self).grep_raw is not BackendProtocol.grep_raw:
            warnings.warn(
                "`grep_raw` is deprecated and will be removed in v0.7; rename to `grep` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            result = self.grep_raw(pattern, path, glob)
            if isinstance(result, str):
                return GrepResult(error=result)
            return GrepResult(matches=result)

        raise NotImplementedError

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> "GrepResult":
        """grep의 비동기 버전."""
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> "GlobResult":
        """glob 패턴과 일치하는 파일을 찾습니다.

        Args:
            pattern: 파일 경로를 매칭할 와일드카드 glob 패턴.

                표준 glob 구문을 지원합니다:

                - `*`: 파일명/디렉터리 내 임의의 문자열 매칭
                - `**`: 디렉터리를 재귀적으로 매칭
                - `?`: 단일 문자 매칭
                - `[abc]`: 집합에서 한 문자 매칭

            path: 검색을 시작할 기준 디렉터리.

                기본값: `'/'` (루트).

                패턴은 이 경로를 기준으로 적용됩니다.

        Returns:
            매칭된 파일 목록 또는 오류를 담은 GlobResult.
        """
        # 하위 클래스가 deprecated glob_info를 재정의한 경우 경고 후 위임
        if type(self).glob_info is not BackendProtocol.glob_info:
            warnings.warn(
                "`glob_info` is deprecated and will be removed in v0.7; rename to `glob` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return GlobResult(matches=self.glob_info(pattern, path))

        raise NotImplementedError

    async def aglob(self, pattern: str, path: str = "/") -> "GlobResult":
        """glob의 비동기 버전."""
        return await asyncio.to_thread(self.glob, pattern, path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """파일시스템에 새 파일을 생성하여 내용을 씁니다. 파일이 이미 존재하면 오류를 반환합니다.

        Args:
            file_path: 파일을 생성할 절대 경로.

                '/'로 시작해야 합니다.
            content: 파일에 쓸 문자열 내용.

        Returns:
            WriteResult
        """
        raise NotImplementedError

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """write의 비동기 버전."""
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """기존 파일에서 정확한 문자열 교체를 수행합니다.

        Args:
            file_path: 수정할 파일의 절대 경로. `'/'`로 시작해야 합니다.
            old_string: 찾아서 교체할 정확한 문자열.

                공백과 들여쓰기를 포함하여 정확히 일치해야 합니다.
            new_string: old_string을 대체할 문자열.

                old_string과 달라야 합니다.
            replace_all: True이면 모든 발생을 교체합니다.

                False (기본값)이면 `old_string`이 파일 내에서 유일해야 하며,
                그렇지 않으면 수정이 실패합니다.

        Returns:
            EditResult
        """
        raise NotImplementedError

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """edit의 비동기 버전."""
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일을 샌드박스에 업로드합니다.

        이 API는 개발자가 직접 사용하거나 커스텀 도구를 통해 LLM에 노출하여
        사용할 수 있도록 설계되었습니다.

        Args:
            files: 업로드할 (경로, 내용) 튜플의 목록.

        Returns:
            입력 파일별 FileUploadResponse 객체 목록.

                응답 순서는 입력 순서와 일치합니다 (response[i]는 files[i]에 해당).

                파일별 성공/실패 여부는 error 필드로 확인합니다.

        Examples:
            ```python
            responses = sandbox.upload_files(
                [
                    ("/app/config.json", b"{...}"),
                    ("/app/data.txt", b"content"),
                ]
            )
            ```
        """
        raise NotImplementedError

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """upload_files의 비동기 버전."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """샌드박스에서 여러 파일을 다운로드합니다.

        이 API는 개발자가 직접 사용하거나 커스텀 도구를 통해 LLM에 노출하여
        사용할 수 있도록 설계되었습니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로별 `FileDownloadResponse` 객체 목록.

                응답 순서는 입력 순서와 일치합니다 (response[i]는 paths[i]에 해당).

                파일별 성공/실패 여부는 error 필드로 확인합니다.
        """
        raise NotImplementedError

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """download_files의 비동기 버전."""
        return await asyncio.to_thread(self.download_files, paths)

    # -- 더 이상 사용되지 않는 메서드 (deprecated) ----------------------------------

    def ls_info(self, path: str) -> list["FileInfo"]:
        """디렉터리 내 모든 파일을 메타데이터와 함께 목록 조회합니다.

        !!! warning "Deprecated"

            대신 `ls`를 사용하세요.
        """
        warnings.warn(
            "`ls_info` is deprecated and will be removed in v0.7; use `ls` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.ls(path)
        if result.error is not None:
            msg = "This behavior is only available via the new `ls` API."
            raise NotImplementedError(msg)
        return result.entries or []

    async def als_info(self, path: str) -> list["FileInfo"]:
        """ls_info의 비동기 버전.

        !!! warning "Deprecated"

            대신 `als`를 사용하세요.
        """
        warnings.warn(
            "`als_info` is deprecated and will be removed in v0.7; use `als` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = await self.als(path)
        if result.error is not None:
            msg = "This behavior is only available via the new `als` API."
            raise NotImplementedError(msg)
        return result.entries or []

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """glob 패턴과 일치하는 파일을 찾습니다.

        !!! warning "Deprecated"

            대신 `glob`을 사용하세요.
        """
        warnings.warn(
            "`glob_info` is deprecated and will be removed in v0.7; use `glob` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.glob(pattern, path)
        if result.error is not None:
            msg = "This behavior is only available via the new `glob` API."
            raise NotImplementedError(msg)
        return result.matches or []

    async def aglob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """glob_info의 비동기 버전.

        !!! warning "Deprecated"

            대신 `aglob`을 사용하세요.
        """
        warnings.warn(
            "`aglob_info` is deprecated and will be removed in v0.7; use `aglob` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = await self.aglob(pattern, path)
        if result.error is not None:
            msg = "This behavior is only available via the new `aglob` API."
            raise NotImplementedError(msg)
        return result.matches or []

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """파일 내에서 리터럴 텍스트 패턴을 검색합니다.

        !!! warning "Deprecated"

            대신 `grep`을 사용하세요.
        """
        warnings.warn(
            "`grep_raw` is deprecated and will be removed in v0.7; use `grep` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.grep(pattern, path, glob)
        if result.error is not None:
            return result.error
        return result.matches or []

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """grep_raw의 비동기 버전.

        !!! warning "Deprecated"

            대신 `agrep`을 사용하세요.
        """
        warnings.warn(
            "`agrep_raw` is deprecated and will be removed in v0.7; use `agrep` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = await self.agrep(pattern, path, glob)
        if result.error is not None:
            return result.error
        return result.matches or []


@dataclass
class ExecuteResponse:
    """코드 실행 결과.

    LLM 소비에 최적화된 단순화된 스키마입니다.
    """

    output: str
    """실행된 명령의 stdout과 stderr를 합친 출력."""

    exit_code: int | None = None
    """프로세스 종료 코드.

    0은 성공, 0이 아니면 실패를 나타냅니다.
    """

    truncated: bool = False
    """백엔드 제한으로 인해 출력이 잘렸는지 여부."""


class SandboxBackendProtocol(BackendProtocol):
    """셸 명령 실행 기능을 추가한 `BackendProtocol`의 확장.

    격리된 환경(컨테이너, VM, 원격 호스트)에서 실행되는 백엔드를 위해 설계되었습니다.

    셸 명령을 위한 `execute()`/`aexecute()`와 `id` 프로퍼티를 추가합니다.

    모든 상속된 파일 연산을 `execute()`에 위임하여 구현하는 기본 클래스는
    `BaseSandbox`를 참조하세요.
    """

    @property
    def id(self) -> str:
        """샌드박스 백엔드 인스턴스의 고유 식별자."""
        raise NotImplementedError

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """샌드박스 환경에서 셸 명령을 실행합니다.

        LLM 소비에 최적화된 단순화된 인터페이스입니다.

        Args:
            command: 실행할 전체 셸 명령 문자열.
            timeout: 명령 완료를 대기할 최대 시간 (초).

                None이면 백엔드의 기본 타임아웃을 사용합니다.

                백엔드 간 이식 가능한 동작을 위해 음수가 아닌 정수 값을 제공하세요.
                0은 타임아웃 없는 실행을 지원하는 백엔드에서 타임아웃을 비활성화할 수 있습니다.

        Returns:
            합산된 출력, 종료 코드, 잘림 플래그를 담은 `ExecuteResponse`.
        """
        raise NotImplementedError

    async def aexecute(
        self,
        command: str,
        *,
        # ASYNC109 - timeout은 asyncio.timeout() 계약이 아닌
        # 동기 구현체로 전달되는 시맨틱 파라미터입니다.
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """execute의 비동기 버전."""
        # 미들웨어 계층은 호출 전에 timeout 지원 여부를 검증합니다.
        # 이 가드는 미들웨어를 우회하는 직접 호출자를 보호합니다.
        if timeout is not None and execute_accepts_timeout(type(self)):
            return await asyncio.to_thread(self.execute, command, timeout=timeout)
        return await asyncio.to_thread(self.execute, command)


@lru_cache(maxsize=128)
def execute_accepts_timeout(cls: type[SandboxBackendProtocol]) -> bool:
    """백엔드 클래스의 `execute`가 `timeout` 키워드 인수를 받는지 확인합니다.

    일부 오래된 백엔드 패키지는 SDK 의존성의 하한을 설정하지 않아
    `SandboxBackendProtocol`에 추가된 `timeout` 키워드를 지원하지 않을 수 있습니다.

    반복적인 인트로스펙션 오버헤드를 방지하기 위해 클래스별로 결과를 캐시합니다.
    """
    try:
        sig = inspect.signature(cls.execute)
    except (ValueError, TypeError):
        # 시그니처 검사에 실패하면 timeout 미지원으로 간주하고 경고를 기록합니다.
        # 이는 백엔드 패키징 문제를 나타낼 수 있습니다.
        logger.warning(
            "Could not inspect signature of %s.execute; assuming timeout is not supported. This may indicate a backend packaging issue.",
            cls.__qualname__,
            exc_info=True,
        )
        return False
    else:
        return "timeout" in sig.parameters


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
