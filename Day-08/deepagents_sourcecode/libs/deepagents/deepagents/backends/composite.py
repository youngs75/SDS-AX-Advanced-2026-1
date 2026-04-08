"""경로 접두사별로 파일 작업을 라우팅하는 복합 백엔드.

경로 접두사에 따라 다른 백엔드로 작업을 라우팅합니다. 경로별로 다른 저장 전략이
필요할 때 사용하세요 (예: 임시 파일에는 state, 메모리에는 영구 저장소).

Examples:
    ```python
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend
    from deepagents.backends.store import StoreBackend

    composite = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})

    composite.write("/temp.txt", "ephemeral")
    composite.write("/memories/note.md", "persistent")
    ```
"""

from collections import defaultdict
from dataclasses import replace
from typing import cast

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
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
    execute_accepts_timeout,
)
from deepagents.backends.state import StateBackend


def _remap_grep_path(m: GrepMatch, route_prefix: str) -> GrepMatch:
    """경로 앞에 라우트 접두사를 붙인 새 GrepMatch를 생성합니다."""
    return cast(
        "GrepMatch",
        {
            **m,
            "path": f"{route_prefix[:-1]}{m['path']}",
        },
    )


def _strip_route_from_pattern(pattern: str, route_prefix: str) -> str:
    """패턴이 해당 라우트를 대상으로 할 때 glob 패턴에서 라우트 접두사를 제거합니다.

    패턴(앞의 `/` 무시)이 라우트 접두사(앞의 `/` 무시)로 시작하면
    중복되는 접두사를 제거하여 패턴이 백엔드의 내부 루트에 대해 상대적이 되도록 합니다.

    Args:
        pattern: glob 패턴. 절대 경로일 수 있습니다 (예: `/memories/**/*.md`).
        route_prefix: 라우트 접두사 (예: `/memories/`).

    Returns:
        라우트 접두사가 제거된 패턴, 또는 라우트와 일치하지 않으면 원본 패턴.
    """
    bare_pattern = pattern.lstrip("/")
    bare_prefix = route_prefix.strip("/") + "/"
    if bare_pattern.startswith(bare_prefix):
        return bare_pattern[len(bare_prefix) :]
    return pattern


def _remap_file_info_path(fi: FileInfo, route_prefix: str) -> FileInfo:
    """경로 앞에 라우트 접두사를 붙인 새 FileInfo를 생성합니다."""
    return cast(
        "FileInfo",
        {
            **fi,
            "path": f"{route_prefix[:-1]}{fi['path']}",
        },
    )


def _route_for_path(
    *,
    default: BackendProtocol,
    sorted_routes: list[tuple[str, BackendProtocol]],
    path: str,
) -> tuple[BackendProtocol, str, str | None]:
    """경로를 백엔드로 라우팅하고 해당 백엔드에 맞게 정규화합니다.

    선택된 백엔드, 해당 백엔드에 전달할 정규화된 경로,
    일치한 라우트 접두사(기본 백엔드 사용 시 None)를 반환합니다.

    정규화 규칙:
    - 경로가 정확히 라우트 루트에서 끝 슬래시 없이 일치하면 (예: "/memories"),
      해당 백엔드로 라우팅하고 backend_path "/"를 반환합니다.
    - 경로가 라우트 접두사로 시작하면 (예: "/memories/notes.txt"), 라우트 접두사를
      제거하고 결과가 "/"로 시작하도록 보장합니다.
    - 그 외에는 기본 백엔드와 원본 경로를 반환합니다.
    """
    for route_prefix, backend in sorted_routes:
        prefix_no_slash = route_prefix.rstrip("/")
        if path == prefix_no_slash:
            return backend, "/", route_prefix

        # startswith 검사에서 경계를 강제하기 위해 route_prefix가 /로 끝나도록 보장
        normalized_prefix = route_prefix if route_prefix.endswith("/") else f"{route_prefix}/"
        if path.startswith(normalized_prefix):
            suffix = path[len(normalized_prefix) :]
            backend_path = f"/{suffix}" if suffix else "/"
            return backend, backend_path, route_prefix
    return default, path, None


class CompositeBackend(BackendProtocol):
    """경로 접두사별로 파일 작업을 다른 백엔드로 라우팅합니다.

    경로를 라우트 접두사와 매칭하여 (길이 내림차순) 해당 백엔드로 위임합니다.
    일치하지 않는 경로는 기본 백엔드를 사용합니다.

    Attributes:
        default: 어떤 라우트와도 일치하지 않는 경로를 위한 백엔드.
        routes: 경로 접두사에서 백엔드로의 매핑 (예: {"/memories/": store_backend}).
        sorted_routes: 올바른 매칭을 위해 길이 내림차순으로 정렬된 라우트.

    Examples:
        ```python
        composite = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(), "/cache/": StoreBackend()})

        composite.write("/temp.txt", "data")
        composite.write("/memories/note.txt", "data")
        ```
    """

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """복합 백엔드를 초기화합니다.

        Args:
            default: 어떤 라우트와도 일치하지 않는 경로를 위한 백엔드.
            routes: 경로 접두사에서 백엔드로의 매핑. 접두사는 "/"로 시작해야 하며
                "/"로 끝나야 합니다 (예: "/memories/").
        """
        # 기본 백엔드
        self.default = default

        # 가상 라우트
        self.routes = routes

        # 올바른 접두사 매칭을 위해 길이 내림차순으로 라우트 정렬
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        backend, stripped_key, _route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=key,
        )
        return backend, stripped_key

    @staticmethod
    def _coerce_ls_result(raw: LsResult | list[FileInfo]) -> LsResult:
        """레거시 ``list[FileInfo]`` 반환값을 `LsResult`로 정규화합니다."""
        if isinstance(raw, LsResult):
            return raw
        return LsResult(entries=raw)

    def ls(self, path: str) -> LsResult:
        """디렉토리 내용을 나열합니다 (비재귀적).

        경로가 라우트와 일치하면 해당 백엔드만 조회합니다. 경로가 "/"이면
        기본 백엔드와 모든 가상 라우트 디렉토리를 집계합니다. 그 외에는 기본 백엔드를 조회합니다.

        Args:
            path: "/"로 시작하는 절대 디렉토리 경로.

        Returns:
            디렉토리 항목 또는 오류가 담긴 LsResult.

        Examples:
            ```python
            result = composite.ls("/")
            result = composite.ls("/memories/")
            ```
        """
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            ls_result = self._coerce_ls_result(backend.ls(backend_path))
            if ls_result.error:
                return ls_result
            return LsResult(entries=[_remap_file_info_path(fi, route_prefix) for fi in (ls_result.entries or [])])

        # 루트에서 기본 백엔드와 모든 라우팅된 백엔드 집계
        if path == "/":
            results: list[FileInfo] = []
            default_result = self._coerce_ls_result(self.default.ls(path))
            results.extend(default_result.entries or [])
            for route_prefix, _backend in self.sorted_routes:
                # 라우트 자체를 디렉토리로 추가 (예: /memories/)
                results.append(
                    FileInfo(
                        path=route_prefix,
                        is_dir=True,
                        size=0,
                        modified_at="",
                    )
                )

            results.sort(key=lambda x: x.get("path", ""))
            return LsResult(entries=results)

        # 경로가 라우트와 일치하지 않음: 기본 백엔드만 조회
        return self._coerce_ls_result(self.default.ls(path))

    async def als(self, path: str) -> LsResult:
        """ls의 비동기 버전."""
        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            ls_result = self._coerce_ls_result(await backend.als(backend_path))
            if ls_result.error:
                return ls_result
            return LsResult(entries=[_remap_file_info_path(fi, route_prefix) for fi in (ls_result.entries or [])])

        # 루트에서 기본 백엔드와 모든 라우팅된 백엔드 집계
        if path == "/":
            results: list[FileInfo] = []
            default_result = self._coerce_ls_result(await self.default.als(path))
            results.extend(default_result.entries or [])
            for route_prefix, _backend in self.sorted_routes:
                # 라우트 자체를 디렉토리로 추가 (예: /memories/)
                results.append(
                    {
                        "path": route_prefix,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    }
                )

            results.sort(key=lambda x: x.get("path", ""))
            return LsResult(entries=results)

        # 경로가 라우트와 일치하지 않음: 기본 백엔드만 조회
        return self._coerce_ls_result(await self.default.als(path))

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """적절한 백엔드로 라우팅하여 파일 콘텐츠를 읽습니다.

        Args:
            file_path: 파일의 절대 경로.
            offset: 읽기 시작할 라인 오프셋 (0-indexed).
            limit: 읽을 최대 라인 수.

        Returns:
            ReadResult
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """read의 비동기 버전."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    @staticmethod
    def _coerce_grep_result(raw: GrepResult | list[GrepMatch] | str) -> GrepResult:
        """레거시 ``list[GrepMatch] | str`` 반환값을 `GrepResult`로 정규화합니다."""
        if isinstance(raw, GrepResult):
            return raw
        if isinstance(raw, str):
            return GrepResult(error=raw)
        return GrepResult(matches=raw)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """파일에서 리터럴 텍스트 패턴을 검색합니다.

        경로에 따라 백엔드로 라우팅합니다: 특정 라우트는 해당 백엔드만 검색하고,
        "/" 또는 None은 모든 백엔드를 검색하며, 그 외에는 기본 백엔드를 검색합니다.

        Args:
            pattern: 검색할 리터럴 텍스트 (정규표현식 아님).
            path: 검색할 디렉토리. None이면 모든 백엔드를 검색합니다.
            glob: 파일 필터링을 위한 glob 패턴 (예: "*.py", "**/*.txt").
                콘텐츠가 아닌 파일명으로 필터링합니다.

        Returns:
            일치 결과 또는 오류가 담긴 GrepResult.

        Examples:
            ```python
            result = composite.grep("TODO", path="/memories/")
            result = composite.grep("error", path="/")
            result = composite.grep("import", path="/", glob="*.py")
            ```
        """
        if path is not None:
            backend, backend_path, route_prefix = _route_for_path(
                default=self.default,
                sorted_routes=self.sorted_routes,
                path=path,
            )
            if route_prefix is not None:
                grep_result = self._coerce_grep_result(backend.grep(pattern, backend_path, glob))
                if grep_result.error:
                    return grep_result
                return GrepResult(matches=[_remap_grep_path(m, route_prefix) for m in (grep_result.matches or [])])

        # path가 None 또는 "/"이면 기본 백엔드와 모든 라우팅된 백엔드를 검색하고 병합
        # 그 외에는 기본 백엔드만 검색
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            default_result = self._coerce_grep_result(self.default.grep(pattern, path, glob))
            if default_result.error:
                return default_result
            all_matches.extend(default_result.matches or [])

            for route_prefix, backend in self.routes.items():
                grep_result = self._coerce_grep_result(backend.grep(pattern, "/", glob))
                if grep_result.error:
                    return grep_result
                all_matches.extend(_remap_grep_path(m, route_prefix) for m in (grep_result.matches or []))

            return GrepResult(matches=all_matches)
        # 경로가 지정되었지만 라우트와 일치하지 않음 - 기본 백엔드만 검색
        return self._coerce_grep_result(self.default.grep(pattern, path, glob))

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """grep의 비동기 버전.

        라우팅 동작 및 파라미터에 대한 자세한 설명은 grep()을 참고하세요.
        """
        if path is not None:
            backend, backend_path, route_prefix = _route_for_path(
                default=self.default,
                sorted_routes=self.sorted_routes,
                path=path,
            )
            if route_prefix is not None:
                grep_result = self._coerce_grep_result(await backend.agrep(pattern, backend_path, glob))
                if grep_result.error:
                    return grep_result
                return GrepResult(matches=[_remap_grep_path(m, route_prefix) for m in (grep_result.matches or [])])

        # path가 None 또는 "/"이면 기본 백엔드와 모든 라우팅된 백엔드를 검색하고 병합
        # 그 외에는 기본 백엔드만 검색
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            default_result = self._coerce_grep_result(await self.default.agrep(pattern, path, glob))
            if default_result.error:
                return default_result
            all_matches.extend(default_result.matches or [])

            for route_prefix, backend in self.routes.items():
                grep_result = self._coerce_grep_result(await backend.agrep(pattern, "/", glob))
                if grep_result.error:
                    return grep_result
                all_matches.extend(_remap_grep_path(m, route_prefix) for m in (grep_result.matches or []))

            return GrepResult(matches=all_matches)
        # 경로가 지정되었지만 라우트와 일치하지 않음 - 기본 백엔드만 검색
        return self._coerce_grep_result(await self.default.agrep(pattern, path, glob))

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """경로 접두사별로 라우팅하여 glob 패턴과 일치하는 파일을 찾습니다."""
        results: list[FileInfo] = []

        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            glob_result = backend.glob(pattern, backend_path)
            matches = glob_result.matches if isinstance(glob_result, GlobResult) else glob_result
            if isinstance(glob_result, GlobResult) and glob_result.error:
                return glob_result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (matches or [])])

        # 경로가 특정 라우트와 일치하지 않음 - 기본 백엔드와 모든 라우팅된 백엔드 검색
        default_result = self.default.glob(pattern, path)
        default_matches = default_result.matches if isinstance(default_result, GlobResult) else default_result
        results.extend(default_matches or [])

        for route_prefix, backend in self.routes.items():
            route_pattern = _strip_route_from_pattern(pattern, route_prefix)
            sub_result = backend.glob(route_pattern, "/")
            sub_matches = sub_result.matches if isinstance(sub_result, GlobResult) else sub_result
            results.extend(_remap_file_info_path(fi, route_prefix) for fi in (sub_matches or []))

        # 결정론적 순서 정렬
        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        """glob의 비동기 버전."""
        results: list[FileInfo] = []

        backend, backend_path, route_prefix = _route_for_path(
            default=self.default,
            sorted_routes=self.sorted_routes,
            path=path,
        )
        if route_prefix is not None:
            glob_result = await backend.aglob(pattern, backend_path)
            matches = glob_result.matches if isinstance(glob_result, GlobResult) else glob_result
            if isinstance(glob_result, GlobResult) and glob_result.error:
                return glob_result
            return GlobResult(matches=[_remap_file_info_path(fi, route_prefix) for fi in (matches or [])])

        # 경로가 특정 라우트와 일치하지 않음 - 기본 백엔드와 모든 라우팅된 백엔드 검색
        default_result = await self.default.aglob(pattern, path)
        default_matches = default_result.matches if isinstance(default_result, GlobResult) else default_result
        results.extend(default_matches or [])

        for route_prefix, backend in self.routes.items():
            route_pattern = _strip_route_from_pattern(pattern, route_prefix)
            sub_result = await backend.aglob(route_pattern, "/")
            sub_matches = sub_result.matches if isinstance(sub_result, GlobResult) else sub_result
            results.extend(_remap_file_info_path(fi, route_prefix) for fi in (sub_matches or []))

        # 결정론적 순서 정렬
        results.sort(key=lambda x: x.get("path", ""))
        return GlobResult(matches=results)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """적절한 백엔드로 라우팅하여 새 파일을 생성합니다.

        Args:
            file_path: 파일의 절대 경로.
            content: 문자열 형태의 파일 콘텐츠.

        Returns:
            성공 메시지 또는 Command 객체, 파일이 이미 존재하면 오류.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.write(stripped_key, content)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """write의 비동기 버전."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.awrite(stripped_key, content)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """적절한 백엔드로 라우팅하여 파일을 편집합니다.

        Args:
            file_path: 파일의 절대 경로.
            old_string: 찾아서 교체할 문자열.
            new_string: 교체 문자열.
            replace_all: True이면 모든 일치 항목을 교체합니다.

        Returns:
            성공 메시지 또는 Command 객체, 실패 시 오류 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """edit의 비동기 버전."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.aedit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.path is not None:
            res = replace(res, path=file_path)
        return res

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """기본 백엔드를 통해 셸 명령을 실행합니다.

        파일 작업과 달리 실행은 경로 라우팅이 불가능하며 — 항상
        기본 백엔드로 위임합니다.

        Args:
            command: 실행할 셸 명령.
            timeout: 명령 완료를 기다릴 최대 시간(초).

                None이면 백엔드의 기본 타임아웃을 사용합니다.

        Returns:
            출력, 종료 코드, 잘림 플래그가 담긴 ExecuteResponse.

        Raises:
            NotImplementedError: 기본 백엔드가 `SandboxBackendProtocol`이 아닌 경우
                (즉, 실행을 지원하지 않는 경우).
        """
        if isinstance(self.default, SandboxBackendProtocol):
            if timeout is not None and execute_accepts_timeout(type(self.default)):
                return self.default.execute(command, timeout=timeout)
            return self.default.execute(command)

        # execute 툴의 런타임 검사가 올바르게 작동하면 이 코드에 도달하지 않지만,
        # 안전 폴백으로 포함합니다.
        msg = (
            "기본 백엔드가 명령 실행을 지원하지 않습니다 (SandboxBackendProtocol). "
            "실행을 활성화하려면 SandboxBackendProtocol을 구현하는 기본 백엔드를 제공하세요."
        )
        raise NotImplementedError(msg)

    async def aexecute(
        self,
        command: str,
        *,
        # ASYNC109 - timeout은 asyncio.timeout() 계약이 아닌 하위
        # 백엔드 구현으로 전달되는 시맨틱 파라미터입니다.
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """execute의 비동기 버전.

        파라미터 및 동작에 대한 자세한 설명은 `execute()`를 참고하세요.
        """
        if isinstance(self.default, SandboxBackendProtocol):
            if timeout is not None and execute_accepts_timeout(type(self.default)):
                return await self.default.aexecute(command, timeout=timeout)
            return await self.default.aexecute(command)

        # execute 툴의 런타임 검사가 올바르게 작동하면 이 코드에 도달하지 않지만,
        # 안전 폴백으로 포함합니다.
        msg = (
            "기본 백엔드가 명령 실행을 지원하지 않습니다 (SandboxBackendProtocol). "
            "실행을 활성화하려면 SandboxBackendProtocol을 구현하는 기본 백엔드를 제공하세요."
        )
        raise NotImplementedError(msg)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """효율성을 위해 백엔드별 배치 처리로 여러 파일을 업로드합니다.

        파일을 대상 백엔드별로 그룹화하고, 각 백엔드의 upload_files를
        해당 백엔드의 모든 파일로 한 번 호출한 뒤, 원래 순서대로 결과를 병합합니다.

        Args:
            files: 업로드할 (path, content) 튜플 목록.

        Returns:
            입력 파일 각각에 대한 FileUploadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.
        """
        # 결과 리스트 사전 할당
        results: list[FileUploadResponse | None] = [None] * len(files)

        # 원래 인덱스를 추적하면서 백엔드별로 파일 그룹화
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # 각 백엔드의 배치 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출을 위한 데이터 추출
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # 모든 파일로 백엔드를 한 번 호출
            batch_responses = backend.upload_files(batch_files)

            # 원래 인덱스에 원래 경로로 응답 배치
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # 원래 경로
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileUploadResponse]", results)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """upload_files의 비동기 버전."""
        # 결과 리스트 사전 할당
        results: list[FileUploadResponse | None] = [None] * len(files)

        # 원래 인덱스를 추적하면서 백엔드별로 파일 그룹화
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # 각 백엔드의 배치 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출을 위한 데이터 추출
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # 모든 파일로 백엔드를 한 번 호출
            batch_responses = await backend.aupload_files(batch_files)

            # 원래 인덱스에 원래 경로로 응답 배치
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # 원래 경로
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileUploadResponse]", results)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """효율성을 위해 백엔드별 배치 처리로 여러 파일을 다운로드합니다.

        경로를 대상 백엔드별로 그룹화하고, 각 백엔드의 download_files를
        해당 백엔드의 모든 경로로 한 번 호출한 뒤, 원래 순서대로 결과를 병합합니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로 각각에 대한 FileDownloadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.
        """
        # 결과 리스트 사전 할당
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # 각 백엔드의 배치 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출을 위한 데이터 추출
            indices, stripped_paths = zip(*batch, strict=False)

            # 모든 경로로 백엔드를 한 번 호출
            batch_responses = backend.download_files(list(stripped_paths))

            # 원래 인덱스에 원래 경로로 응답 배치
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # 원래 경로
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileDownloadResponse]", results)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """download_files의 비동기 버전."""
        # 결과 리스트 사전 할당
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # 각 백엔드의 배치 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출을 위한 데이터 추출
            indices, stripped_paths = zip(*batch, strict=False)

            # 모든 경로로 백엔드를 한 번 호출
            batch_responses = await backend.adownload_files(list(stripped_paths))

            # 원래 인덱스에 원래 경로로 응답 배치
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # 원래 경로
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return cast("list[FileDownloadResponse]", results)
