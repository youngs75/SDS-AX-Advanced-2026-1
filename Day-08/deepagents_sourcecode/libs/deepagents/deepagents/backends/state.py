"""StateBackend: LangGraph 에이전트 상태에 파일을 임시 저장하는 백엔드."""

import base64
import warnings
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_SEND
from langgraph.config import get_config

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileFormat,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_file_type,
    _glob_search_files,
    _to_legacy_file_data,
    create_file_data,
    file_data_to_string,
    grep_matches_from_files,
    perform_string_replacement,
    slice_read_response,
    update_file_data,
)


class StateBackend(BackendProtocol):
    """파일을 에이전트 상태에 임시 저장하는 백엔드.

    LangGraph의 상태 관리 및 체크포인팅을 활용합니다. 파일은 대화 스레드 내에서는
    유지되지만 스레드 간에는 공유되지 않습니다. 상태는 각 에이전트 스텝 이후
    자동으로 체크포인팅됩니다.

    읽기/쓰기는 LangGraph의 `CONFIG_KEY_READ` / `CONFIG_KEY_SEND`를 통해 처리되어
    상태 업데이트가 `files_update` dict로 반환되는 대신 적절한 채널 쓰기로 큐잉됩니다.
    """

    def __init__(
        self,
        runtime: object = None,
        *,
        file_format: FileFormat = "v2",
    ) -> None:
        r"""StateBackend를 초기화합니다.

        Args:
            runtime: 사용 중단됨 - 하위 호환성을 위해 허용되지만 무시됩니다.
                상태는 이제 `get_config()`를 통해 읽기/쓰기됩니다.
            file_format: 저장 포맷 버전. `"v1"`은 콘텐츠를 `list[str]`
                (`\\n`으로 분리된 라인) 형태로 저장하며 `encoding` 필드가 없습니다.
                `"v2"` (기본값)는 콘텐츠를 `encoding` 필드와 함께 일반 `str`로 저장합니다.
        """
        if runtime is not None:
            warnings.warn(
                "`runtime`을 StateBackend에 전달하는 방식은 사용 중단되었으며 "
                "v0.7에서 제거될 예정입니다. StateBackend는 이제 `get_config()`를 통해 "
                "상태를 읽고 씁니다. 단순히 `StateBackend()`를 사용하세요.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._file_format = file_format

    # ------------------------------------------------------------------
    # config 키를 통해 상태를 읽고 쓰는 내부 헬퍼 메서드
    # ------------------------------------------------------------------

    def _get_config(self) -> RunnableConfig:
        """현재 LangGraph config를 반환합니다. config가 없으면 명확한 오류를 발생시킵니다."""
        try:
            config = get_config()
        except RuntimeError:
            msg = (
                "StateBackend는 LangGraph 그래프 실행 내부에서 사용해야 합니다 "
                "(예: create_deep_agent를 통해). 그래프 컨텍스트 외부에서는 상태를 "
                "읽거나 쓸 수 없습니다. 파일을 미리 채우려면 invoke 시 전달하세요: "
                'agent.invoke({"messages": [...], "files": {...}})'
            )
            raise RuntimeError(msg) from None
        configurable = config.get("configurable", {})
        if CONFIG_KEY_READ not in configurable:
            msg = (
                "StateBackend는 LangGraph config에 CONFIG_KEY_READ / CONFIG_KEY_SEND가 "
                "필요합니다. 백엔드가 직접 호출되지 않고 그래프 노드나 툴 내부에서 "
                "사용되는지 확인하세요. 파일을 미리 채우려면 invoke 시 전달하세요: "
                'agent.invoke({"messages": [...], "files": {...}})'
            )
            raise RuntimeError(msg)
        return config

    def _read_files(self) -> dict[str, Any]:
        """Pregel 내부를 통해 현재 `files` 채널을 읽습니다.

        `CONFIG_KEY_READ`를 사용해 상태를 직접 읽습니다 — 이를 통해 StateBackend를
        한 번 초기화하고 어떤 그래프 컨텍스트(툴, 미들웨어 노드 등)에서든
        필요할 때 상태를 가져올 수 있습니다.

        `fresh=False`는 현재 superstep *시작* 시점의 값을 읽습니다
        (체크포인팅된 값 + 이전 스텝의 쓰기). 현재 스텝에서 큐잉된 쓰기는
        노드 경계까지 적용되지 않으므로, 같은 스텝 내 모든 호출은 일관된 스냅샷을 봅니다.
        """
        config = self._get_config()
        read = config["configurable"][CONFIG_KEY_READ]
        fresh = False
        return read("files", fresh) or {}

    def _send_files_update(self, update: dict[str, Any]) -> None:
        """Pregel 내부를 통해 `files` 채널에 쓰기를 큐잉합니다.

        이 헬퍼의 핵심 목적은 `backend.write` / `backend.edit` 호출자가
        상태 업데이트를 직접 알거나 관리하지 않아도 되도록 하는 것입니다 —
        백엔드가 내부적으로 처리합니다.

        `CONFIG_KEY_SEND`를 사용해 부분적인 `files` 업데이트를 직접 큐잉합니다 —
        `_read_files`와 같은 이유로 StateBackend를 한 번 초기화하고 어떤 그래프
        컨텍스트에서든 쓸 수 있습니다. `send`는 `(channel, value)` 튜플의 리스트를
        받으며, `files` 채널은 dict-merge reducer를 사용하므로 변경된 파일만 포함하면
        됩니다 — 변경되지 않은 파일은 reducer에 의해 보존됩니다.

        쓰기는 노드 경계까지 적용되지 않으므로 같은 스텝의 다른 호출에서는
        보이지 않습니다 (`_read_files`와 `fresh=False` 사용 참고).
        """
        config = self._get_config()
        send = config["configurable"][CONFIG_KEY_SEND]
        send([("files", update)])

    def _prepare_for_storage(self, file_data: FileData) -> dict[str, Any]:
        """FileData를 상태 저장에 사용되는 포맷으로 변환합니다.

        `file_format="v1"`이면 레거시 포맷을 반환합니다.
        """
        if self._file_format == "v1":
            return _to_legacy_file_data(file_data)
        return {**file_data}

    def ls(self, path: str) -> LsResult:
        """지정한 디렉토리의 파일과 하위 디렉토리를 나열합니다 (비재귀적).

        Args:
            path: 디렉토리의 절대 경로.

        Returns:
            디렉토리 바로 아래의 파일과 하위 디렉토리에 대한 FileInfo-like dict 목록.
            디렉토리는 경로 끝에 /가 붙고 is_dir=True입니다.
        """
        files = self._read_files()
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # 올바른 접두사 매칭을 위해 경로 끝에 슬래시 정규화
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # 파일이 지정한 디렉토리나 하위 디렉토리에 있는지 확인
            if not k.startswith(normalized_path):
                continue

            # 디렉토리 이후의 상대 경로 추출
            relative = k[len(normalized_path) :]

            # 상대 경로에 '/'가 있으면 하위 디렉토리에 있는 파일
            if "/" in relative:
                # 직계 하위 디렉토리 이름 추출
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # 현재 디렉토리에 직접 있는 파일
            # 하위 호환성: 크기 계산을 위해 레거시 list[str] 콘텐츠 처리
            raw = fd.get("content", "")
            size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            infos.append(
                {
                    "path": k,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # 결과에 디렉토리 추가
        infos.extend(FileInfo(path=subdir, is_dir=True, size=0, modified_at="") for subdir in sorted(subdirs))

        infos.sort(key=lambda x: x.get("path", ""))
        return LsResult(entries=infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """요청된 라인 범위의 파일 콘텐츠를 읽습니다.

        Args:
            file_path: 파일의 절대 경로.
            offset: 읽기 시작할 라인 오프셋 (0-indexed).
            limit: 읽을 최대 라인 수.

        Returns:
            요청된 윈도우에 대한 원시(비포맷) 콘텐츠가 담긴 ReadResult.
            라인 번호 포맷팅은 미들웨어에서 적용됩니다.
        """
        files = self._read_files()
        file_data = files.get(file_path)

        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")

        if _get_file_type(file_path) != "text":
            return ReadResult(file_data=file_data)

        sliced = slice_read_response(file_data, offset, limit)
        if isinstance(sliced, ReadResult):
            return sliced
        sliced_fd = FileData(
            content=sliced,
            encoding=file_data.get("encoding", "utf-8"),
        )
        if "created_at" in file_data:
            sliced_fd["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            sliced_fd["modified_at"] = file_data["modified_at"]
        return ReadResult(file_data=sliced_fd)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """콘텐츠로 새 파일을 생성합니다.

        업데이트는 `CONFIG_KEY_SEND`를 통해 직접 큐잉됩니다.
        """
        files = self._read_files()

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)
        self._send_files_update({file_path: self._prepare_for_storage(new_file_data)})
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """문자열 치환으로 파일을 편집합니다.

        업데이트는 `CONFIG_KEY_SEND`를 통해 직접 큐잉됩니다.
        """
        files = self._read_files()
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        self._send_files_update({file_path: self._prepare_for_storage(new_file_data)})
        return EditResult(path=file_path, occurrences=int(occurrences))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """상태 파일에서 리터럴 텍스트 패턴을 검색합니다."""
        files = self._read_files()
        return grep_matches_from_files(files, pattern, path if path is not None else "/", glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """glob 패턴과 일치하는 파일의 FileInfo를 반환합니다."""
        files = self._read_files()
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            if fd:
                # 하위 호환성: 크기 계산을 위해 레거시 list[str] 콘텐츠 처리
                raw = fd.get("content", "")
                size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            else:
                size = 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return GlobResult(matches=infos)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일을 상태에 업로드합니다.

        Args:
            files: 업로드할 (path, content) 튜플 목록

        Returns:
            입력 파일 각각에 대한 FileUploadResponse 객체 목록
        """
        msg = (
            "StateBackend는 아직 upload_files를 지원하지 않습니다. "
            "메모리에 파일을 저장하는 경우 invoke 시 직접 파일을 전달하여 업로드할 수 있습니다."
        )
        raise NotImplementedError(msg)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """상태에서 여러 파일을 다운로드합니다.

        Args:
            paths: 다운로드할 파일 경로 목록

        Returns:
            입력 경로 각각에 대한 FileDownloadResponse 객체 목록
        """
        state_files = self._read_files()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            file_data = state_files.get(path)

            if file_data is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            content_str = file_data_to_string(file_data)

            encoding = file_data.get("encoding", "utf-8")
            content_bytes = content_str.encode("utf-8") if encoding == "utf-8" else base64.standard_b64decode(content_str)
            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
