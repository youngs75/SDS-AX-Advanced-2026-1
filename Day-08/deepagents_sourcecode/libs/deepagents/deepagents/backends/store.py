"""StoreBackend: LangGraph의 BaseStore를 사용하는 영구 저장 백엔드."""

import base64
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from langgraph.config import get_config, get_store
from langgraph.runtime import get_runtime
from langgraph.store.base import BaseStore, Item
from langgraph.typing import ContextT, StateT

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

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


@dataclass
class BackendContext(Generic[StateT, ContextT]):
    """네임스페이스 팩토리 함수에 전달되는 컨텍스트."""

    state: StateT
    runtime: "Runtime[ContextT]"


# 네임스페이스 팩토리 함수의 타입 별칭
NamespaceFactory = Callable[[BackendContext[Any, Any]], tuple[str, ...]]

# 네임스페이스 구성 요소에 허용되는 문자: 영숫자, 사용자 ID에
# 흔히 쓰이는 문자 (하이픈, 언더스코어, 점, @, +, 콜론, 틸드).
_NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")


def _validate_namespace(namespace: tuple[str, ...]) -> tuple[str, ...]:
    """NamespaceFactory가 반환한 네임스페이스 튜플을 검증한다.

    각 구성 요소는 비어 있지 않은 문자열이어야 하며, 안전한 문자만
    포함해야 한다: 영숫자 (a-z, A-Z, 0-9), 하이픈 (-), 언더스코어 (_),
    점 (.), 골뱅이 (@), 플러스 (+), 콜론 (:), 틸드 (~).

    ``*``, ``?``, ``[``, ``]``, ``{``, ``}`` 등의 문자는 Store 조회에서
    와일드카드 또는 글로브 인젝션을 방지하기 위해 거부된다.

    Args:
        namespace: 검증할 네임스페이스 튜플.

    Returns:
        검증된 네임스페이스 튜플 (변경 없음).

    Raises:
        ValueError: 네임스페이스가 비어 있거나, 문자열이 아닌 요소를 포함하거나,
            빈 문자열이 있거나, 허용되지 않는 문자가 있을 경우.
    """
    if not namespace:
        msg = "Namespace tuple must not be empty."
        raise ValueError(msg)

    for i, component in enumerate(namespace):
        if not isinstance(component, str):
            msg = f"Namespace component at index {i} must be a string, got {type(component).__name__}."
            raise TypeError(msg)
        if not component:
            msg = f"Namespace component at index {i} must not be empty."
            raise ValueError(msg)
        if not _NAMESPACE_COMPONENT_RE.match(component):
            msg = (
                f"Namespace component at index {i} contains disallowed characters: {component!r}. "
                f"Only alphanumeric characters, hyphens, underscores, dots, @, +, colons, and tildes are allowed."
            )
            raise ValueError(msg)

    return namespace


class StoreBackend(BackendProtocol):
    """LangGraph의 BaseStore에 파일을 저장하는 영구 백엔드.

    LangGraph Store를 사용하여 대화를 초월한 영구·크로스-스레드 저장을 제공한다.
    파일은 네임스페이스로 구분되며 모든 스레드에 걸쳐 유지된다.

    네임스페이스에는 멀티 에이전트 격리를 위해 선택적으로 assistant_id를 포함할 수 있다.
    """

    def __init__(
        self,
        runtime: object = None,
        *,
        store: BaseStore | None = None,
        namespace: NamespaceFactory | None = None,
        file_format: FileFormat = "v2",
    ) -> None:
        r"""StoreBackend를 초기화한다.

        Args:
            runtime: 하위 호환성을 위해 수용하지만 무시되는 deprecated 인자.
                Store와 컨텍스트는 이제 ``get_store()`` / ``get_runtime()`` 으로
                획득한다.
            store: 선택적 ``BaseStore`` 인스턴스. 제공 시 해당 Store를 직접
                사용한다. ``None`` (기본값)이면 호출 시점에 ``get_store()``로
                Store를 획득하며, LangGraph 그래프 실행 컨텍스트가 필요하다.
            namespace: BackendContext를 받아 네임스페이스 튜플을 반환하는
                선택적 callable. 네임스페이스 해석에 완전한 유연성을 제공한다.
                현재 와일드카드인 * 는 금지된다.
                None이면 메타데이터의 legacy assistant_id 감지를 사용한다 (deprecated).

                !!! Note:
                    이 파라미터는 버전 0.5.0에서 **필수**가 될 예정이다.
                !!!! Warning:
                    이 API는 마이너 버전에서 변경될 수 있다.

            file_format: 저장 포맷 버전. `"v1"` (기본값)은 내용을 `list[str]`
                (줄 단위 분할, `\\n` 구분)로 저장하며 `encoding` 필드가 없다.
                `"v2"`는 내용을 평문 `str`과 `encoding` 필드로 저장한다.

        Example:
                    namespace=lambda ctx: ("filesystem", ctx.runtime.context.user_id)
        """
        if runtime is not None:
            warnings.warn(
                "Passing `runtime` to StoreBackend is deprecated and will be "
                "removed in v0.7. StoreBackend now obtains store "
                "and context via `get_store()` / `get_runtime()`. Simply use "
                "`StoreBackend()` or `StoreBackend(store=my_store)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._store = store
        self._namespace = namespace
        self._file_format = file_format

    def _get_store(self) -> BaseStore:
        """Store 인스턴스를 반환한다.

        초기화 시 전달된 Store가 있으면 그것을 사용하고,
        없으면 LangGraph 실행 컨텍스트에서 ``get_store()``로 획득한다.
        """
        if self._store is not None:
            return self._store
        try:
            return get_store()
        except (RuntimeError, KeyError):
            msg = (
                "StoreBackend must be used inside a LangGraph graph execution "
                "(e.g. via create_deep_agent), or initialized with an explicit "
                "store: StoreBackend(store=my_store)"
            )
            raise RuntimeError(msg) from None

    def _get_namespace(self) -> tuple[str, ...]:
        """Store 연산에 사용할 네임스페이스를 반환한다.

        초기화 시 namespace가 제공된 경우 BackendContext와 함께 호출한다.
        그렇지 않으면 메타데이터의 legacy assistant_id 감지를 사용한다 (deprecated).
        """
        if self._namespace is not None:
            try:
                runtime = get_runtime()
            except RuntimeError:
                runtime = None
            ctx = BackendContext(state=None, runtime=runtime)  # type: ignore[arg-type]
            return _validate_namespace(self._namespace(ctx))

        return self._get_namespace_legacy()

    def _get_namespace_legacy(self) -> tuple[str, ...]:
        """Legacy 네임스페이스 해석: 메타데이터에서 assistant_id를 확인한다.

        ``get_config()``로 메타데이터의 assistant_id를 탐색한다.
        기본값은 ``("filesystem",)``.

        .. deprecated::
            레거시 감지에 의존하는 대신 StoreBackend에 `namespace`를 전달하라.
        """
        warnings.warn(
            "StoreBackend without explicit `namespace` is deprecated and will be removed in v0.7. "
            "Pass `namespace=lambda ctx: (...)` to StoreBackend.",
            DeprecationWarning,
            stacklevel=3,
        )
        namespace = "filesystem"

        try:
            cfg = get_config()
        except Exception:  # noqa: BLE001  # 탄력적인 config 폴백을 위한 의도적 예외 처리
            return (namespace,)

        try:
            assistant_id = cfg.get("metadata", {}).get("assistant_id")
        except Exception:  # noqa: BLE001  # 탄력적인 config 폴백을 위한 의도적 예외 처리
            assistant_id = None

        if assistant_id:
            return (assistant_id, namespace)
        return (namespace,)

    def _convert_store_item_to_file_data(self, store_item: Item) -> FileData:
        """Store Item을 FileData 형식으로 변환한다.

        Args:
            store_item: 파일 데이터를 담고 있는 Store Item.

        Returns:
            content와 encoding을 포함하는 FileData dict.
            Store item에 created_at과 modified_at이 있으면 함께 포함한다.
        """
        raw_content = store_item.value.get("content")
        if raw_content is None:
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise ValueError(msg)

        # 하위 호환성: legacy list[str] 포맷 처리
        if isinstance(raw_content, list):
            warnings.warn(
                "Store item with list[str] content is deprecated and will be removed in v0.7. Content should be stored as a plain str.",
                DeprecationWarning,
                stacklevel=2,
            )
            content = "\n".join(raw_content)
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise TypeError(msg)

        result = FileData(
            content=content,
            encoding=store_item.value.get("encoding", "utf-8"),
        )
        if "created_at" in store_item.value and isinstance(store_item.value["created_at"], str):
            result["created_at"] = store_item.value["created_at"]
        if "modified_at" in store_item.value and isinstance(store_item.value["modified_at"], str):
            result["modified_at"] = store_item.value["modified_at"]
        return result

    def _convert_file_data_to_store_value(self, file_data: FileData) -> dict[str, Any]:
        """FileData를 store.put()에 적합한 dict로 변환한다.

        `file_format="v1"` 이면 `content`를 `list[str]`로,
        `encoding` 키 없이 legacy 포맷으로 반환한다.

        Args:
            file_data: 변환할 FileData.

        Returns:
            content와 encoding을 포함하는 dict.
            FileData에 created_at과 modified_at이 있으면 함께 포함한다.
        """
        if self._file_format == "v1":
            return _to_legacy_file_data(file_data)
        result: dict[str, Any] = {
            "content": file_data["content"],
            "encoding": file_data["encoding"],
        }
        if "created_at" in file_data:
            result["created_at"] = file_data["created_at"]
        if "modified_at" in file_data:
            result["modified_at"] = file_data["modified_at"]
        return result

    def _search_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002  # LangGraph BaseStore.search() API와 일치
        page_size: int = 100,
    ) -> list[Item]:
        """자동 페이지네이션으로 전체 결과를 조회하며 Store를 검색한다.

        Args:
            store: 검색할 Store.
            namespace: 검색할 계층적 경로 접두사.
            query: 자연어 검색을 위한 선택적 쿼리.
            filter: 결과를 필터링할 키-값 쌍.
            page_size: 페이지당 가져올 항목 수 (기본값: 100).

        Returns:
            검색 조건에 맞는 모든 항목의 리스트.

        Example:
            ```python
            store = _get_store(runtime)
            namespace = _get_namespace()
            all_items = _search_store_paginated(store, namespace)
            ```
        """
        all_items: list[Item] = []
        offset = 0
        while True:
            page_items = store.search(
                namespace,
                query=query,
                filter=filter,
                limit=page_size,
                offset=offset,
            )
            if not page_items:
                break
            all_items.extend(page_items)
            if len(page_items) < page_size:
                break
            offset += page_size

        return all_items

    def ls(self, path: str) -> LsResult:
        """지정된 디렉토리의 파일과 하위 디렉토리를 나열한다 (비재귀).

        Args:
            path: 디렉토리의 절대 경로.

        Returns:
            디렉토리 바로 아래 파일과 디렉토리에 대한 FileInfo 유사 dict 리스트.
            디렉토리는 경로 끝에 /가 붙고 is_dir=True이다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Store별 필터 의미론에 종속되지 않도록
        # 모든 항목을 가져온 후 경로 접두사로 로컬에서 필터링한다
        items = self._search_store_paginated(store, namespace)
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # 접두사 매칭을 위해 경로 끝에 슬래시를 붙여 정규화한다
        normalized_path = path if path.endswith("/") else path + "/"

        for item in items:
            # 지정된 디렉토리 또는 하위 디렉토리에 있는 파일인지 확인
            if not str(item.key).startswith(normalized_path):
                continue

            # 디렉토리 이후의 상대 경로를 추출한다
            relative = str(item.key)[len(normalized_path) :]

            # 상대 경로에 '/'가 있으면 하위 디렉토리에 있는 파일이다
            if "/" in relative:
                # 바로 아래 하위 디렉토리 이름을 추출한다
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # 현재 디렉토리에 직접 위치한 파일이다
            try:
                fd = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
            # 하위 호환성: 크기 계산을 위한 legacy list[str] content 처리
            raw = fd.get("content", "")
            size = len("\n".join(raw)) if isinstance(raw, list) else len(raw)
            infos.append(
                {
                    "path": item.key,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # 결과에 디렉토리를 추가한다
        infos.extend(FileInfo(path=subdir, is_dir=True, size=0, modified_at="") for subdir in sorted(subdirs))

        infos.sort(key=lambda x: x.get("path", ""))
        return LsResult(entries=infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """요청된 줄 범위의 파일 내용을 읽는다.

        Args:
            file_path: 절대 파일 경로.
            offset: 읽기 시작 줄 오프셋 (0 인덱스).
            limit: 읽을 최대 줄 수.

        Returns:
            요청된 윈도우의 원시(미포맷) 내용이 담긴 ReadResult.
            줄 번호 포맷팅은 미들웨어에서 적용된다.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Item | None = store.get(namespace, file_path)

        if item is None:
            return ReadResult(error=f"File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return ReadResult(error=str(e))

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

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Store 비동기 메서드를 사용하는 read의 비동기 버전.

        store.aget를 직접 사용하여 비동기 컨텍스트에서 동기 호출을 방지한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Item | None = await store.aget(namespace, file_path)

        if item is None:
            return ReadResult(error=f"File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return ReadResult(error=str(e))

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
        """내용을 담아 새 파일을 생성한다.

        성공 또는 오류 시 WriteResult를 반환한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # 파일 존재 여부 확인
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # 새 파일 생성
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)
        return WriteResult(path=file_path)

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Store 비동기 메서드를 사용하는 write의 비동기 버전.

        store.aget/aput를 직접 사용하여 비동기 컨텍스트에서 동기 호출을 방지한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # 비동기 메서드로 파일 존재 여부 확인
        existing = await store.aget(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # 비동기 메서드로 새 파일 생성
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        await store.aput(namespace, file_path, store_value)
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """문자열 치환으로 파일을 편집한다.

        성공 또는 오류 시 EditResult를 반환한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # 기존 파일 가져오기
        item = store.get(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return EditResult(error=f"Error: {e}")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Store에서 파일 업데이트
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        return EditResult(path=file_path, occurrences=int(occurrences))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Store 비동기 메서드를 사용하는 edit의 비동기 버전.

        store.aget/aput를 직접 사용하여 비동기 컨텍스트에서 동기 호출을 방지한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # 비동기 메서드로 기존 파일 가져오기
        item = await store.aget(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return EditResult(error=f"Error: {e}")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # 비동기 메서드로 Store에서 파일 업데이트
        store_value = self._convert_file_data_to_store_value(new_file_data)
        await store.aput(namespace, file_path, store_value)
        return EditResult(path=file_path, occurrences=int(occurrences))

    # legacy grep() 편의 메서드는 인터페이스를 간결하게 유지하기 위해 제거됨

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Store 파일에서 리터럴 텍스트 패턴을 검색한다."""
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        return grep_matches_from_files(files, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Store에서 glob 패턴에 맞는 파일을 찾는다."""
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            if fd:
                # 하위 호환성: 크기 계산을 위한 legacy list[str] content 처리
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
        """여러 파일을 Store에 업로드한다.

        바이너리 파일 (이미지, PDF 등)은 base64 인코딩 문자열로 저장한다.
        텍스트 파일은 utf-8 문자열로 저장한다.

        Args:
            files: (경로, 내용) 튜플의 리스트. content는 bytes이다.

        Returns:
            입력 파일별 FileUploadResponse 객체 리스트.
            응답 순서는 입력 순서와 일치한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                content_str = content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                content_str = base64.standard_b64encode(content).decode("ascii")
                encoding = "base64"

            file_data = create_file_data(content_str, encoding=encoding)
            store_value = self._convert_file_data_to_store_value(file_data)

            store.put(namespace, path, store_value)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Store에서 여러 파일을 다운로드한다.

        Args:
            paths: 다운로드할 파일 경로 리스트.

        Returns:
            입력 경로별 FileDownloadResponse 객체 리스트.
            응답 순서는 입력 순서와 일치한다.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            item = store.get(namespace, path)

            if item is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            file_data = self._convert_store_item_to_file_data(item)
            content_str = file_data_to_string(file_data)

            encoding = file_data["encoding"]
            content_bytes = base64.standard_b64decode(content_str) if encoding == "base64" else content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
