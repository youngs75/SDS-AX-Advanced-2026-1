"""`FilesystemBackend`: 파일시스템에 직접 접근하여 파일을 읽고 쓰는 백엔드."""

import base64
import json
import logging
import os
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import wcmatch.glob as wcglob

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileOperationError,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_file_type,
    check_empty_content,
    perform_string_replacement,
)

logger = logging.getLogger(__name__)


class FilesystemBackend(BackendProtocol):
    """파일시스템에 직접 접근하여 파일을 읽고 쓰는 백엔드.

    파일은 실제 파일시스템 경로로 접근한다. 상대 경로는 현재 작업 디렉토리
    기준으로 해석한다. 내용은 평문으로 읽고 쓰며, 메타데이터(타임스탬프)는
    파일시스템 stat에서 가져온다.

    !!! warning "보안 경고"

        이 백엔드는 에이전트에게 파일시스템 직접 읽기/쓰기 권한을 부여한다.
        적절한 환경에서만 주의하여 사용하라.

        **적절한 사용 사례:**

        - 로컬 개발 CLI (코딩 어시스턴트, 개발 도구)
        - CI/CD 파이프라인 (아래 보안 고려사항 참조)

        **부적절한 사용 사례:**

        - 웹 서버나 HTTP API — `StateBackend`, `StoreBackend`, 또는
            `SandboxBackend`를 사용하라

        **보안 위험:**

        - 에이전트가 시크릿(API 키, 인증 정보, `.env` 파일 등)을 포함한
            접근 가능한 모든 파일을 읽을 수 있다
        - 네트워크 도구와 결합하면 SSRF 공격을 통해 시크릿이 유출될 수 있다
        - 파일 수정은 영구적이며 되돌릴 수 없다

        **권장 안전 조치:**

        1. 민감한 작업을 검토하기 위해 Human-in-the-Loop (HITL) 미들웨어를 활성화하라
        2. 접근 가능한 파일시스템 경로에서 시크릿을 제외하라 (특히 CI/CD에서)
        3. 프로덕션 환경에서는 `StateBackend`, `StoreBackend` 또는 `SandboxBackend`를 선호하라

        일반적으로 이 백엔드는 Human-in-the-Loop (HITL) 미들웨어와 함께,
        또는 신뢰할 수 없는 워크로드를 실행해야 할 경우 적절히 샌드박스된
        환경 내에서 사용할 것을 권장한다.

        !!! note

            `virtual_mode=True`는 주로 가상 경로 의미론을 위한 것이다 (예:
            `CompositeBackend`와 함께 사용). `..`, `~` 같은 경로 탈출을 차단하고
            `root_dir` 외부의 절대 경로를 막는 경로 기반 가드레일을 제공할 수 있으나,
            샌드박싱이나 프로세스 격리는 제공하지 않는다. 기본값(`virtual_mode=False`)은
            `root_dir`이 설정되어 있어도 어떠한 보안도 제공하지 않는다.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool | None = None,  # noqa: FBT001
        max_file_size_mb: int = 10,
    ) -> None:
        """파일시스템 백엔드를 초기화한다.

        Args:
            root_dir: 파일 작업의 선택적 루트 디렉토리.

                기본값은 현재 작업 디렉토리이다.

                - `virtual_mode=False` (기본값): 상대 경로 해석에만 영향을 준다.
                - `virtual_mode=True`: 파일시스템 작업의 가상 루트 역할을 한다.

            virtual_mode: 가상 경로 모드를 활성화한다.

                **주요 사용 사례:** `CompositeBackend`와 함께 사용할 때 라우트 접두사를
                제거하고 정규화된 경로를 라우팅된 백엔드로 전달하는,
                백엔드에 독립적이고 안정적인 경로 의미론을 제공한다.

                `True`이면 모든 경로를 `root_dir`에 고정된 가상 절대 경로로 취급한다.
                경로 탈출 (`..`, `~`)을 차단하고 해석된 모든 경로가 `root_dir`
                내에 유지되는지 검증한다.

                `False` (기본값)이면 절대 경로를 그대로 사용하고 상대 경로는
                `root_dir` 하위에서 해석한다. 에이전트가 `root_dir` 외부 경로를
                선택하는 것에 대한 보안은 제공하지 않는다.

                - 절대 경로 (예: `/etc/passwd`)는 `root_dir`을 완전히 우회한다
                - `..`를 포함한 상대 경로는 `root_dir`을 벗어날 수 있다
                - 에이전트는 제한 없이 파일시스템에 접근할 수 있다

            max_file_size_mb: grep의 Python 폴백 검색 등에서 허용하는
                최대 파일 크기 (메가바이트).

                이 제한을 초과하는 파일은 검색 시 건너뛴다. 기본값은 10 MB.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        if virtual_mode is None:
            warnings.warn(
                "FilesystemBackend virtual_mode default will change in deepagents 0.5.0; "
                "please specify virtual_mode explicitly. "
                "Note: virtual_mode is for virtual path semantics (e.g., CompositeBackend routing) and optional path-based guardrails; "
                "it does not provide sandboxing or process isolation. "
                "Security note: leaving virtual_mode=False allows absolute paths and '..' to bypass root_dir. "
                "Consult the API reference for details.",
                DeprecationWarning,
                stacklevel=2,
            )
            virtual_mode = False
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """보안 검사를 포함하여 파일 경로를 해석한다.

        `virtual_mode=True`이면 들어오는 경로를 `self.cwd` 하위의 가상 절대
        경로로 취급하고, 경로 탈출 (`..`, `~`)을 차단하며 해석된 경로가
        루트 내에 유지되는지 확인한다.

        `virtual_mode=False`이면 레거시 동작을 유지한다: 절대 경로는 그대로
        허용하고, 상대 경로는 cwd 하위에서 해석한다.

        Args:
            key: 파일 경로 (절대, 상대, 또는 `virtual_mode=True`일 때 가상 경로).

        Returns:
            해석된 절대 `Path` 객체.

        Raises:
            ValueError: `virtual_mode`에서 경로 탈출이 시도되거나 해석된 경로가
                루트 디렉토리를 벗어나는 경우.
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                msg = "Path traversal not allowed"
                raise ValueError(msg)
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                msg = f"Path:{full} outside root directory: {self.cwd}"
                raise ValueError(msg) from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def _to_virtual_path(self, path: Path) -> str:
        """파일시스템 경로를 cwd 기준의 가상 경로로 변환한다.

        Args:
            path: 변환할 파일시스템 경로.

        Returns:
            `/`로 시작하는 포워드 슬래시 상대 경로 문자열.

        Raises:
            ValueError: 경로가 cwd 외부에 있는 경우.
            OSError: 심볼릭 링크 깨짐, 권한 거부 등으로 경로를 해석할 수 없는 경우.
        """
        return "/" + path.resolve().relative_to(self.cwd).as_posix()

    def ls(self, path: str) -> LsResult:  # noqa: C901, PLR0912, PLR0915  # virtual_mode 로직이 복잡함
        """지정된 디렉토리의 파일과 하위 디렉토리를 나열한다 (비재귀).

        Args:
            path: 파일을 나열할 절대 디렉토리 경로.

        Returns:
            디렉토리 바로 아래 파일과 디렉토리에 대한 `FileInfo` 유사 dict 리스트.
            디렉토리는 경로 끝에 `/`가 붙고 `is_dir=True`이다.
        """
        dir_path = self._resolve_path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return LsResult(entries=[])

        results: list[FileInfo] = []

        # 비교를 위해 cwd를 문자열로 변환
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # 직계 자식만 나열 (비재귀)
        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except OSError:
                    continue

                abs_path = str(child_path)

                if not self.virtual_mode:
                    # 비가상 모드: 절대 경로 사용
                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # 로컬 파일시스템 타임스탬프는 타임존 불필요
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # 로컬 파일시스템 타임스탬프는 타임존 불필요
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path + "/", "is_dir": True})
                else:
                    # 가상 모드: 크로스 플랫폼 지원을 위해 Path를 사용하여 cwd 접두사 제거
                    try:
                        virt_path = self._to_virtual_path(child_path)
                    except ValueError:
                        logger.debug("Skipping path outside root: %s", child_path)
                        continue
                    except OSError:
                        logger.warning("Could not resolve path: %s", child_path, exc_info=True)
                        continue

                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # 로컬 파일시스템 타임스탬프는 타임존 불필요
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # 로컬 파일시스템 타임스탬프는 타임존 불필요
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path + "/", "is_dir": True})
        except (OSError, PermissionError):
            pass

        # 경로 기준으로 결정적 순서를 유지한다
        results.sort(key=lambda x: x.get("path", ""))
        return LsResult(entries=results)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """요청된 줄 범위의 파일 내용을 읽는다.

        Args:
            file_path: 절대 또는 상대 파일 경로.
            offset: 읽기 시작 줄 오프셋 (0 인덱스).
            limit: 읽을 최대 줄 수.

        Returns:
            요청된 윈도우의 원시(미포맷) 내용이 담긴 ReadResult.
            줄 번호 포맷팅은 미들웨어에서 적용된다.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return ReadResult(error=f"File '{file_path}' not found")

        try:
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if _get_file_type(file_path) != "text":
                with os.fdopen(fd, "rb") as f:
                    raw = f.read()
                encoded = base64.standard_b64encode(raw).decode("ascii")
                return ReadResult(file_data=FileData(content=encoded, encoding="base64"))

            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            empty_msg = check_empty_content(content)
            if empty_msg:
                return ReadResult(file_data=FileData(content=empty_msg, encoding="utf-8"))

            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))

            if start_idx >= len(lines):
                return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")

            selected_lines = lines[start_idx:end_idx]
            return ReadResult(file_data=FileData(content="\n".join(selected_lines), encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """내용을 담아 새 파일을 생성한다.

        Args:
            file_path: 새 파일을 생성할 경로.
            content: 파일에 쓸 텍스트 내용.

        Returns:
            성공 시 경로가 담긴 `WriteResult`, 파일이 이미 존재하거나
                쓰기에 실패하면 오류 메시지가 담긴 `WriteResult`.
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists():
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        try:
            # 필요하면 부모 디렉토리를 생성한다
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # 심볼릭 링크를 통한 쓰기를 방지하기 위해 O_NOFOLLOW를 선호한다
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            return WriteResult(path=file_path)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """문자열 치환으로 파일을 편집한다.

        Args:
            file_path: 편집할 파일 경로.
            old_string: 검색하여 교체할 텍스트.
            new_string: 대체 텍스트.
            replace_all: `True`이면 모든 발생을 교체한다. `False` (기본값)이면
                정확히 하나의 발생이 있을 때만 교체한다.

        Returns:
            성공 시 경로와 발생 횟수가 담긴 `EditResult`, 파일을 찾지 못하거나
                교체에 실패하면 오류 메시지가 담긴 `EditResult`.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            # 안전하게 읽기
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            # old_string/new_string의 줄 끝 문자를 위의 텍스트 모드 읽기와
            # 맞게 정규화한다. Python 유니버설 개행 모드(newline=None 기본값)는
            # 읽기 시 \r\n과 단독 \r을 \n으로 변환한다.
            # 바이너리 모드 읽기(예: download_files)로 내용을 얻은 호출자는
            # \n만 있는 내용과 매칭되지 않는 \r\n이나 \r이 포함된
            # 문자열을 전달할 수 있다.
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")

            result = perform_string_replacement(content, old_string, new_string, replace_all)

            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            # 안전하게 쓰기
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(path=file_path, occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """파일에서 리터럴 텍스트 패턴을 검색한다.

        ripgrep을 우선 사용하고, 미설치 시 Python 검색으로 폴백한다.

        Args:
            pattern: 검색할 리터럴 문자열 (정규식 아님).
            path: 검색할 디렉토리 또는 파일 경로. 기본값은 현재 디렉토리.
            glob: 검색할 파일을 필터링하는 선택적 glob 패턴.

        Returns:
            매치 결과 또는 오류가 담긴 GrepResult.
        """
        # 기본 경로 해석
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return GrepResult(matches=[])

        if not base_full.exists():
            return GrepResult(matches=[])

        # 리터럴 검색을 위해 -F 플래그와 함께 ripgrep 우선 시도
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            # Python 폴백은 리터럴 검색을 위해 이스케이프된 패턴이 필요하다
            results = self._python_search(re.escape(pattern), base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return GrepResult(matches=matches)

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[tuple[int, str]]] | None:  # noqa: C901  # 로깅을 위해 except 절을 분리함
        """고정 문자열(리터럴) 모드로 ripgrep을 사용하여 검색한다.

        Args:
            pattern: 검색할 리터럴 문자열 (이스케이프 없음).
            base_full: 검색할 해석된 기본 경로.
            include_glob: 파일을 필터링할 선택적 glob 패턴.

        Returns:
            파일 경로를 `(줄 번호, 줄 텍스트)` 튜플 리스트에 매핑하는 dict.
                ripgrep을 사용할 수 없거나 타임아웃 시 `None`을 반환한다.
        """
        cmd = ["rg", "--json", "-F"]  # -F는 고정 문자열(리터럴) 모드를 활성화한다
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = self._to_virtual_path(p)
                except ValueError:
                    logger.debug("Skipping grep result outside root: %s", p)
                    continue
                except OSError:
                    logger.warning("Could not resolve grep result path: %s", p, exc_info=True)
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(self, pattern: str, base_full: Path, include_glob: str | None) -> dict[str, list[tuple[int, str]]]:  # noqa: C901, PLR0912
        """ripgrep을 사용할 수 없을 때 Python으로 폴백 검색한다.

        `max_file_size_bytes` 제한을 준수하며 파일을 재귀적으로 검색한다.

        Args:
            pattern: 리터럴 검색을 위한 이스케이프된 정규식 패턴 (re.escape 사용).
            base_full: 검색할 해석된 기본 경로.
            include_glob: 파일 이름을 필터링할 선택적 glob 패턴.

        Returns:
            파일 경로를 `(줄 번호, 줄 텍스트)` 튜플 리스트에 매핑하는 dict.
        """
        # 루프 내에서 사용하기 위해 이스케이프된 패턴을 한 번만 컴파일한다
        regex = re.compile(pattern)

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            try:
                if not fp.is_file():
                    continue
            except (PermissionError, OSError):
                continue
            if include_glob:
                rel_path = str(fp.relative_to(root))
                if not wcglob.globmatch(rel_path, include_glob, flags=wcglob.BRACE | wcglob.GLOBSTAR):
                    continue
            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue
            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = self._to_virtual_path(fp)
                        except ValueError:
                            logger.debug("Skipping grep result outside root: %s", fp)
                            continue
                        except OSError:
                            logger.warning("Could not resolve grep result path: %s", fp, exc_info=True)
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob(self, pattern: str, path: str = "/") -> GlobResult:  # noqa: C901, PLR0912  # virtual_mode 로직이 복잡함
        """glob 패턴에 맞는 파일을 찾는다.

        Args:
            pattern: 파일을 매칭할 glob 패턴 (예: `'*.py'`, `'**/*.txt'`).
            path: 검색할 기본 디렉토리. 기본값은 루트 (`/`).

        Returns:
            매칭된 파일 또는 오류가 담긴 GlobResult.
        """
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        if self.virtual_mode and ".." in Path(pattern).parts:
            msg = "Path traversal not allowed in glob pattern"
            raise ValueError(msg)

        search_path = self.cwd if path == "/" else self._resolve_path(path)
        if not search_path.exists() or not search_path.is_dir():
            return GlobResult(matches=[])

        results: list[FileInfo] = []
        try:
            # 테스트가 기대하는 대로 하위 디렉토리의 파일을 매칭하기 위해 재귀 glob 사용
            for matched_path in search_path.rglob(pattern):
                try:
                    is_file = matched_path.is_file()
                except (PermissionError, OSError):
                    continue
                if not is_file:
                    continue
                if self.virtual_mode:
                    try:
                        matched_path.resolve().relative_to(self.cwd)
                    except ValueError:
                        continue
                abs_path = str(matched_path)
                if not self.virtual_mode:
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # 로컬 파일시스템 타임스탬프는 타임존 불필요
                            }
                        )
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    # 가상 모드: 크로스 플랫폼 지원을 위해 Path 사용
                    try:
                        virt = self._to_virtual_path(matched_path)
                    except ValueError:
                        logger.debug("Skipping glob result outside root: %s", matched_path)
                        continue
                    except OSError:
                        logger.warning("Could not resolve glob result path: %s", matched_path, exc_info=True)
                        continue
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": virt,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),  # noqa: DTZ006  # 로컬 파일시스템 타임스탬프는 타임존 불필요
                            }
                        )
                    except OSError:
                        results.append({"path": virt, "is_dir": False})
        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일을 파일시스템에 업로드한다.

        Args:
            files: (경로, 내용) 튜플의 리스트. content는 bytes이다.

        Returns:
            입력 파일별 FileUploadResponse 객체 리스트.
            응답 순서는 입력 순서와 일치한다.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # 필요하면 부모 디렉토리를 생성한다
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileUploadResponse(path=path, error=error))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """파일시스템에서 여러 파일을 다운로드한다.

        Args:
            paths: 다운로드할 파일 경로 리스트.

        Returns:
            입력 경로별 FileDownloadResponse 객체 리스트.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                # OS가 지원하는 경우 심볼릭 링크 추적을 방지하기 위해
                # 플래그를 선택적으로 사용한다
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception as exc:
                error = _map_exception_to_standard_error(exc)
                if error is None:
                    raise
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses


def _map_exception_to_standard_error(exc: Exception) -> FileOperationError | None:
    """잡힌 예외를 표준화된 `FileOperationError` 코드로 매핑한다.

    분류는 예외 타입(표준 라이브러리 계층)만을 기준으로 한다.
    타입으로 분류할 수 없는 예외는 `None`을 반환하여 호출자가
    재발생 또는 `str(exc)` 폴백 여부를 결정하도록 한다.

    Args:
        exc: 분류할 예외.

    Returns:
        `FileOperationError` 리터럴, 또는 인식할 수 없는 경우 `None`.
    """
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, (NotADirectoryError, FileExistsError)):
        return "invalid_path"
    if isinstance(exc, ValueError):
        return "invalid_path"
    return None
