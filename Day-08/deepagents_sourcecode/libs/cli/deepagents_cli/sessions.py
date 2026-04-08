"""LangGraph 체크포인트 저장소에서 스레드 데이터를 읽고 형식을 지정합니다.

세션 도우미는 SQLite 지원 체크포인터를 쿼리하고, 캐시된 스레드 메타데이터를 파생하고, `/threads` 및 텍스트 스레드 선택기에서 사용하는
프레젠테이션 계층 데이터를 제공합니다.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import aiosqlite
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from deepagents_cli.output import OutputFormat

logger = logging.getLogger(__name__)

_aiosqlite_patched = False
_jsonplus_serializer: JsonPlusSerializer | None = None
_message_count_cache: dict[str, tuple[str | None, int]] = {}
_MAX_MESSAGE_COUNT_CACHE = 4096
_initial_prompt_cache: dict[str, tuple[str | None, str | None]] = {}
_MAX_INITIAL_PROMPT_CACHE = 4096
_recent_threads_cache: dict[tuple[str | None, int], list[ThreadInfo]] = {}
_MAX_RECENT_THREADS_CACHE_KEYS = 16


# ---------------------------------------------------------------------------
# SQLite compatibility patches and shared caches
# ---------------------------------------------------------------------------

def _patch_aiosqlite() -> None:
    """누락된 경우 aiosqlite.Connection을 `is_alive()`로 패치하세요.

    langgraph-checkpoint>=2.1.0에서 필요합니다. 참조:
    https://github.com/langchain-ai/langgraph/issues/6583

    """
    global _aiosqlite_patched  # noqa: PLW0603  # Module-level flag requires global statement
    if _aiosqlite_patched:
        return

    import aiosqlite as _aiosqlite

    if not hasattr(_aiosqlite.Connection, "is_alive"):

        def _is_alive(self: _aiosqlite.Connection) -> bool:
            """연결이 아직 살아 있는지 확인하십시오.

            Returns:
                연결이 살아 있으면 True이고, 그렇지 않으면 False입니다.

            """
            return bool(self._running and self._connection is not None)

        # Dynamically adding a method to aiosqlite.Connection at runtime.
        # Type checkers can't understand this monkey-patch, so we suppress the
        # "attr-defined" error that would otherwise be raised.
        _aiosqlite.Connection.is_alive = _is_alive  # type: ignore[attr-defined]

    _aiosqlite_patched = True


@asynccontextmanager
async def _connect() -> AsyncIterator[aiosqlite.Connection]:
    """aiosqlite를 가져와서 호환성 패치를 적용하고 연결합니다.

    이 모듈의 모든 데이터베이스 기능에서 사용되는 지연된 가져오기 + 패치 + 연결 순서를 중앙 집중화합니다.

    Yields:
        세션 데이터베이스에 대한 열린 aiosqlite 연결입니다.

    """
    import aiosqlite as _aiosqlite

    _patch_aiosqlite()

    async with _aiosqlite.connect(str(get_db_path()), timeout=30.0) as conn:
        yield conn


class ThreadInfo(TypedDict):
    """`list_threads`에서 반환된 스레드 메타데이터입니다."""

    thread_id: str
    """스레드의 고유 식별자입니다."""

    agent_name: str | None
    """스레드를 소유한 에이전트의 이름입니다."""

    updated_at: str | None
    """마지막 업데이트의 ISO 타임스탬프입니다."""

    created_at: NotRequired[str | None]
    """스레드 생성의 ISO 타임스탬프(가장 빠른 체크포인트)"""

    git_branch: NotRequired[str | None]
    """스레드가 생성될 때 Git 분기가 활성화됩니다."""

    initial_prompt: NotRequired[str | None]
    """스레드의 첫 번째 인간 메시지입니다."""

    message_count: NotRequired[int]
    """스레드의 메시지 수입니다."""

    latest_checkpoint_id: NotRequired[str | None]
    """캐시 무효화에 대한 최신 체크포인트 ID입니다."""

    cwd: NotRequired[str | None]
    """스레드가 마지막으로 사용된 작업 디렉터리입니다."""


class _CheckpointSummary(NamedTuple):
    """스레드의 최신 체크포인트에서 추출된 구조화된 데이터입니다."""

    message_count: int
    """최신 체크포인트의 메시지 수입니다."""

    initial_prompt: str | None
    """최신 체크포인트에서 최초의 인간 메시지가 복구되었습니다."""


# ---------------------------------------------------------------------------
# Formatting helpers shared by the CLI and Textual thread picker
# ---------------------------------------------------------------------------

def format_timestamp(iso_timestamp: str | None) -> str:
    """표시할 ISO 타임스탬프 형식을 지정합니다(예: '12월 30일, 오후 6시 10분').

    Args:
        iso_timestamp: ISO 8601 타임스탬프 문자열 또는 `None`.

    Returns:
        형식이 지정된 타임스탬프 문자열이거나 유효하지 않은 경우 빈 문자열입니다.

    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
        return (
            dt.strftime("%b %d, %-I:%M%p")
            .lower()
            .replace("am", "am")
            .replace("pm", "pm")
        )
    except (ValueError, TypeError):
        logger.debug(
            "Failed to parse timestamp %r; displaying as blank",
            iso_timestamp,
            exc_info=True,
        )
        return ""


def format_relative_timestamp(iso_timestamp: str | None) -> str:
    """ISO 타임스탬프를 상대 시간(예: '5분 전', '2시간 전')으로 형식화합니다.

    Args:
        iso_timestamp: ISO 8601 타임스탬프 문자열 또는 `None`.

    Returns:
        유효하지 않은 경우 상대 시간 문자열이거나 빈 문자열입니다.

    """
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp).astimezone()
    except (ValueError, TypeError):
        logger.debug(
            "Failed to parse timestamp %r; displaying as blank",
            iso_timestamp,
            exc_info=True,
        )
        return ""

    delta = datetime.now(tz=dt.tzinfo) - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:  # noqa: PLR2004
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:  # noqa: PLR2004
        return f"{days}d ago"
    months = days // 30
    if months < 12:  # noqa: PLR2004
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def format_path(path: str | None) -> str:
    """표시할 파일 시스템 경로 형식을 지정합니다.

    사용자 홈 디렉토리 아래의 경로는 `~`을 기준으로 표시됩니다. 다른 모든 경로는 있는 그대로 반환됩니다.

    Args:
        path: 절대 파일 시스템 경로 또는 `None`.

    Returns:
        형식화된 경로 문자열 또는 경로가 거짓인 경우 빈 문자열입니다.

    """
    if not path:
        return ""
    try:
        home = str(Path.home())
        if path == home:
            return "~"
        prefix = home + "/"
        if path.startswith(prefix):
            return "~/" + path[len(prefix) :]
    except (RuntimeError, KeyError, OSError):
        logger.debug(
            "Could not resolve home directory for path formatting", exc_info=True
        )
        return path
    else:
        return path


_db_path: Path | None = None


def get_db_path() -> Path:
    """글로벌 데이터베이스에 대한 경로를 가져옵니다.

    반복되는 파일 시스템 작업을 피하기 위해 첫 번째 성공적인 호출 후에 결과가 캐시됩니다.

    Returns:
        SQLite 데이터베이스 파일의 경로입니다.

    """
    global _db_path  # noqa: PLW0603  # Module-level cache requires global statement
    if _db_path is not None:
        return _db_path
    db_dir = Path.home() / ".deepagents"
    db_dir.mkdir(parents=True, exist_ok=True)
    _db_path = db_dir / "sessions.db"
    return _db_path


def generate_thread_id() -> str:
    """전체 UUID7 문자열로 새 스레드 ID를 생성합니다.

    Returns:
        UUID7 문자열(생성 시간에 따른 자연 정렬을 위해 시간 순서).

    """
    from uuid_utils import uuid7

    return str(uuid7())


# ---------------------------------------------------------------------------
# Thread listing, cache population, and checkpoint inspection
# ---------------------------------------------------------------------------

async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    """데이터베이스에 테이블이 있는지 확인하십시오.

    Returns:
        테이블이 존재하면 True, 그렇지 않으면 False입니다.

    """
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    async with conn.execute(query, (table,)) as cursor:
        return await cursor.fetchone() is not None


async def list_threads(
    agent_name: str | None = None,
    limit: int = 20,
    include_message_count: bool = False,
    sort_by: str = "updated",
    branch: str | None = None,
) -> list[ThreadInfo]:
    """체크포인트 테이블의 스레드를 나열합니다.

    Args:
        agent_name: 에이전트 이름으로 선택적으로 필터링합니다.
        limit: 반환할 최대 스레드 수입니다.
        include_message_count: 메시지 수를 포함할지 여부입니다.
        sort_by: 정렬 필드 — `"updated"` 또는 `"created"`.
        branch: Git 브랜치 이름으로 선택적 필터링.

    Returns:
        `thread_id`, `agent_name`이 포함된 `ThreadInfo` 사전 목록
            `updated_at`, `created_at`, `latest_checkpoint_id`, `git_branch`, `cwd` 및
            선택적으로 `message_count`.

    Raises:
        ValueError: `sort_by`이(가) `"updated"` 또는 `"created"`이 아닌 경우.

    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        if sort_by not in {"updated", "created"}:
            msg = f"Invalid sort_by {sort_by!r}; expected 'updated' or 'created'"
            raise ValueError(msg)
        order_col = "created_at" if sort_by == "created" else "updated_at"

        where_clauses: list[str] = []
        params_list: list[str | int] = []

        if agent_name:
            where_clauses.append("json_extract(metadata, '$.agent_name') = ?")
            params_list.append(agent_name)
        if branch:
            where_clauses.append("json_extract(metadata, '$.git_branch') = ?")
            params_list.append(branch)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT thread_id,
                   json_extract(metadata, '$.agent_name') as agent_name,
                   MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                   MAX(checkpoint_id) as latest_checkpoint_id,
                   MIN(json_extract(metadata, '$.updated_at')) as created_at,
                   MAX(json_extract(metadata, '$.git_branch')) as git_branch,
                   MAX(json_extract(metadata, '$.cwd')) as cwd
            FROM checkpoints
            {where_sql}
            GROUP BY thread_id
            ORDER BY {order_col} DESC
            LIMIT ?
        """  # noqa: S608  # where_sql/order_col derived from controlled internal values; user values use ? placeholders
        params: tuple = (*params_list, limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            threads: list[ThreadInfo] = [
                ThreadInfo(
                    thread_id=r[0],
                    agent_name=r[1],
                    updated_at=r[2],
                    latest_checkpoint_id=r[3],
                    created_at=r[4],
                    git_branch=r[5],
                    cwd=r[6],
                )
                for r in rows
            ]

        # Fetch message counts if requested
        if include_message_count and threads:
            await _populate_message_counts(conn, threads)

        # Only cache unfiltered results so the thread selector modal
        # doesn't receive branch-filtered or differently-sorted data.
        if sort_by == "updated" and branch is None:
            _cache_recent_threads(agent_name, limit, threads)
        return threads


async def populate_thread_message_counts(threads: list[ThreadInfo]) -> list[ThreadInfo]:
    """기존 스레드 목록에 대해 `message_count`을(를) 채웁니다.

    이는 `/threads` 모달에서 행을 빠르게 렌더링한 다음 두 번째 스레드 목록 쿼리를 실행하지 않고 백그라운드에서 백필 계산을 수행하는 데
    사용됩니다.

    Args:
        threads: 제자리에 보강할 스레드 행입니다.

    Returns:
        `message_count` 값이 채워진 동일한 목록 개체입니다.

    """
    if not threads:
        return threads

    async with _connect() as conn:
        await _populate_message_counts(conn, threads)
    return threads


async def populate_thread_checkpoint_details(
    threads: list[ThreadInfo],
    *,
    include_message_count: bool = True,
    include_initial_prompt: bool = True,
) -> list[ThreadInfo]:
    """기존 스레드 목록에 대한 체크포인트 파생 필드를 채웁니다.

    이는 `/threads` 모달에서 하나의 백그라운드 패스에서 행을 강화하는 데 사용되므로 최신 체크포인트는 행당 최대 한 번 가져오고 역직렬화됩니다.

    Args:
        threads: 제자리에 보강할 스레드 행입니다.
        include_message_count: `message_count`을 채울지 여부입니다.
        include_initial_prompt: `initial_prompt`을 채울지 여부입니다.

    Returns:
        누락된 체크포인트 파생 필드가 채워진 동일한 목록 개체입니다.

    """
    if not threads or (not include_message_count and not include_initial_prompt):
        return threads

    async with _connect() as conn:
        await _populate_checkpoint_fields(
            conn,
            threads,
            include_message_count=include_message_count,
            include_initial_prompt=include_initial_prompt,
        )
    return threads


async def prewarm_thread_message_counts(limit: int | None = None) -> None:
    """더 빠른 `/threads` 열기를 위한 사전 준비 스레드 선택기 캐시입니다.

    최근 스레드의 제한된 목록을 가져오고 현재 표시되는 열에 대한 검사점 파생 필드를 메모리 내 캐시에 채웁니다. 앱 시작 중에 백그라운드 작업자에서
    실행되도록 되어 있습니다.

    Args:
        limit: 예열할 최대 스레드입니다. `None`인 경우 `get_thread_limit()`을 사용합니다.

    """
    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return

    try:
        from deepagents_cli.model_config import load_thread_config

        cfg = load_thread_config()
        threads = await list_threads(limit=thread_limit, include_message_count=False)
        if threads:
            await populate_thread_checkpoint_details(
                threads,
                include_message_count=cfg.columns.get("messages", False),
                include_initial_prompt=cfg.columns.get("initial_prompt", False),
            )
        _cache_recent_threads(None, thread_limit, threads)
    except (OSError, sqlite3.Error):
        logger.debug("Could not prewarm thread selector cache", exc_info=True)
    except Exception:
        logger.warning(
            "Unexpected error while prewarming thread selector cache",
            exc_info=True,
        )


def get_cached_threads(
    agent_name: str | None = None,
    limit: int | None = None,
) -> list[ThreadInfo] | None:
    """가능한 경우 캐시된 최근 스레드를 가져옵니다.

    Args:
        agent_name: 선택적 에이전트 이름 필터 키입니다.
        limit: 요청된 최대 행입니다. `None`인 경우 `get_thread_limit()`을 사용합니다.

    Returns:
        사용 가능한 경우 캐시된 행의 복사본, 그렇지 않은 경우 `None`.

    """

    def _copy_with_cached_counts(rows: list[ThreadInfo]) -> list[ThreadInfo]:
        copied_rows = _copy_threads(rows)
        apply_cached_thread_message_counts(copied_rows)
        apply_cached_thread_initial_prompts(copied_rows)
        return copied_rows

    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit < 1:
        return None

    exact = _recent_threads_cache.get((agent_name, thread_limit))
    if exact is not None:
        return _copy_with_cached_counts(exact)

    best_key: tuple[str | None, int] | None = None
    for key in _recent_threads_cache:
        cache_agent, cache_limit = key
        if cache_agent != agent_name or cache_limit < thread_limit:
            continue
        if best_key is None or cache_limit < best_key[1]:
            best_key = key

    if best_key is None:
        return None

    return _copy_with_cached_counts(_recent_threads_cache[best_key][:thread_limit])


def apply_cached_thread_message_counts(threads: list[ThreadInfo]) -> int:
    """최신 상태가 일치하면 캐시된 메시지 수를 스레드 행에 적용합니다.

    Args:
        threads: 제자리에서 변경할 행을 스레드합니다.

    Returns:
        캐시에서 채워진 행 수입니다.

    """
    populated = 0
    for thread in threads:
        if "message_count" in thread:
            continue
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached = _message_count_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["message_count"] = cached[1]
        populated += 1
    return populated


def apply_cached_thread_initial_prompts(threads: list[ThreadInfo]) -> int:
    """최신 상태가 일치하면 캐시된 초기 프롬프트를 스레드 행에 적용합니다.

    Args:
        threads: 제자리에서 변경할 행을 스레드합니다.

    Returns:
        캐시에서 채워진 행 수입니다.

    """
    populated = 0
    for thread in threads:
        if "initial_prompt" in thread:
            continue
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        cached = _initial_prompt_cache.get(thread_id)
        if cached is None or cached[0] != freshness:
            continue
        thread["initial_prompt"] = cached[1]
        populated += 1
    return populated


async def _populate_message_counts(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
) -> None:
    """캐시 인식 조회를 사용하여 스레드 행에 `message_count`을 채웁니다."""
    await _populate_checkpoint_fields(
        conn,
        threads,
        include_message_count=True,
        include_initial_prompt=False,
    )


async def _get_jsonplus_serializer() -> JsonPlusSerializer:
    """캐시된 JsonPlus 직렬 변환기를 반환하여 UI 루프에서 로드합니다."""
    global _jsonplus_serializer  # noqa: PLW0603  # Module-level cache requires global statement
    if _jsonplus_serializer is not None:
        return _jsonplus_serializer

    loop = asyncio.get_running_loop()
    _jsonplus_serializer = await loop.run_in_executor(None, _create_jsonplus_serializer)
    return _jsonplus_serializer


def _create_jsonplus_serializer() -> JsonPlusSerializer:
    """JsonPlus 직렬 변환기를 가져오고 생성합니다.

    Returns:
        준비된 `JsonPlusSerializer` 인스턴스.

    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer()


def _cache_message_count(thread_id: str, freshness: str | None, count: int) -> None:
    """신선도 토큰을 사용하여 스레드의 메시지 수를 캐시합니다."""
    if len(_message_count_cache) >= _MAX_MESSAGE_COUNT_CACHE and (
        thread_id not in _message_count_cache
    ):
        oldest = next(iter(_message_count_cache))
        _message_count_cache.pop(oldest, None)
    _message_count_cache[thread_id] = (freshness, count)


def _cache_initial_prompt(
    thread_id: str,
    freshness: str | None,
    initial_prompt: str | None,
) -> None:
    """신선도 토큰을 사용하여 스레드의 초기 프롬프트를 캐시합니다."""
    if len(_initial_prompt_cache) >= _MAX_INITIAL_PROMPT_CACHE and (
        thread_id not in _initial_prompt_cache
    ):
        oldest = next(iter(_initial_prompt_cache))
        _initial_prompt_cache.pop(oldest, None)
    _initial_prompt_cache[thread_id] = (freshness, initial_prompt)


def _thread_freshness(thread: ThreadInfo) -> str | None:
    """스레드 행에 대한 캐시 신선도 토큰을 반환합니다."""
    return thread.get("latest_checkpoint_id") or thread.get("updated_at")


def _cache_recent_threads(
    agent_name: str | None,
    limit: int,
    threads: list[ThreadInfo],
) -> None:
    """빠른 선택기 시작을 위해 최근 스레드 행의 복사본을 저장합니다."""
    key = (agent_name, max(1, limit))
    if len(_recent_threads_cache) >= _MAX_RECENT_THREADS_CACHE_KEYS and (
        key not in _recent_threads_cache
    ):
        _recent_threads_cache.clear()
    _recent_threads_cache[key] = _copy_threads(threads)


def _copy_threads(threads: list[ThreadInfo]) -> list[ThreadInfo]:
    """얕은 복사된 스레드 행을 반환합니다."""
    return [ThreadInfo(**thread) for thread in threads]


async def _count_messages_from_checkpoint(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> int:
    """가장 최근의 체크포인트 Blob에서 메시지 수를 셉니다.

    `durability='exit'`을 사용하면 메시지가 쓰기 테이블이 아닌 체크포인트 Blob에 저장됩니다. 이 함수는 체크포인트를 역직렬화하고
    채널_값의 메시지 수를 셉니다.

    Args:
        conn: 데이터베이스 연결.
        thread_id: 메시지를 계산할 스레드 ID입니다.
        serde: 체크포인트 데이터 디코딩을 위한 직렬 변환기입니다.

    Returns:
        체크포인트의 메시지 수, 또는 찾을 수 없는 경우 0입니다.

    """
    return (await _load_latest_checkpoint_summary(conn, thread_id, serde)).message_count


async def _extract_initial_prompt(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> str | None:
    """최신 체크포인트에서 첫 번째 사람 메시지를 추출합니다.

    Args:
        conn: 데이터베이스 연결.
        thread_id: 추출할 스레드 ID입니다.
        serde: 체크포인트 데이터 디코딩을 위한 직렬 변환기입니다.

    Returns:
        첫 번째 사람의 메시지 내용입니다. 찾을 수 없으면 없음입니다.

    """
    summary = await _load_latest_checkpoint_summary(conn, thread_id, serde)
    return summary.initial_prompt


async def populate_thread_initial_prompts(threads: list[ThreadInfo]) -> None:
    """백그라운드에서 스레드 행에 대해 `initial_prompt`을 채웁니다.

    Args:
        threads: 제자리에 보강할 스레드 행입니다.

    """
    if not threads:
        return

    async with _connect() as conn:
        await _populate_checkpoint_fields(
            conn,
            threads,
            include_message_count=False,
            include_initial_prompt=True,
        )


async def _populate_checkpoint_fields(
    conn: aiosqlite.Connection,
    threads: list[ThreadInfo],
    *,
    include_message_count: bool,
    include_initial_prompt: bool,
) -> None:
    """일괄 처리된 최신 행 패스로 체크포인트 파생 스레드 필드를 채웁니다."""
    serde = await _get_jsonplus_serializer()

    # Phase 1: apply cache hits, collect threads that need DB fetch.
    uncached: list[ThreadInfo] = []
    for thread in threads:
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        needs_count = False
        needs_prompt = False

        if include_message_count:
            cached = _message_count_cache.get(thread_id)
            if cached is not None and cached[0] == freshness:
                thread["message_count"] = cached[1]
            else:
                needs_count = True

        if include_initial_prompt and "initial_prompt" not in thread:
            cached_prompt = _initial_prompt_cache.get(thread_id)
            if cached_prompt is not None and cached_prompt[0] == freshness:
                thread["initial_prompt"] = cached_prompt[1]
            else:
                needs_prompt = True

        if needs_count or needs_prompt:
            uncached.append(thread)

    if not uncached:
        return

    # Phase 2: batch-fetch all uncached threads.
    uncached_ids = [t["thread_id"] for t in uncached]
    batch_results = await _load_latest_checkpoint_summaries_batch(
        conn, uncached_ids, serde
    )

    # Phase 3: apply results and update caches.
    for thread in uncached:
        thread_id = thread["thread_id"]
        freshness = _thread_freshness(thread)
        summary = batch_results.get(thread_id, _CheckpointSummary(0, None))

        if include_message_count and "message_count" not in thread:
            thread["message_count"] = summary.message_count
            _cache_message_count(thread_id, freshness, summary.message_count)
        if include_initial_prompt and "initial_prompt" not in thread:
            thread["initial_prompt"] = summary.initial_prompt
            _cache_initial_prompt(thread_id, freshness, summary.initial_prompt)


_SQLITE_MAX_VARIABLE_NUMBER = 500
"""SQL 쿼리당 최대 `?` 자리 표시자.

SQLite는 단일 쿼리가 가질 수 있는 `?` 매개변수 수를 제한합니다(기본값은 999, 일부 빌드에서는 더 낮음). 사용자가 수백 개의 스레드를 축적하고
`/threads` 모달이 이를 모두 한 번에 가져오는 경우 `IN (?, ?, ...)` 절이 해당 제한을 초과할 수 있습니다. 안전을 유지하기 위해 이
크기로 청크합니다.
"""


async def _load_latest_checkpoint_summaries_batch(
    conn: aiosqlite.Connection,
    thread_ids: list[str],
    serde: JsonPlusSerializer,
) -> dict[str, _CheckpointSummary]:
    """여러 스레드에 대한 최신 체크포인트 요약을 일괄 로드합니다.

    창 함수를 사용하여 스레드당 최신 체크포인트를 가져오고 SQLite 변수 제한 안전을 위해 청크당 하나의 쿼리를 실행합니다.

    Args:
        conn: 데이터베이스 연결.
        thread_ids: 조회할 스레드 ID입니다.
        serde: 체크포인트 Blob을 디코딩하기 위한 직렬 변환기입니다.

    Returns:
        스레드 ID를 체크포인트 요약에 매핑하는 Dict입니다.

    """
    if not thread_ids:
        return {}

    results: dict[str, _CheckpointSummary] = {}

    for start in range(0, len(thread_ids), _SQLITE_MAX_VARIABLE_NUMBER):
        chunk = thread_ids[start : start + _SQLITE_MAX_VARIABLE_NUMBER]
        placeholders = ",".join("?" * len(chunk))
        query = f"""
            SELECT thread_id, type, checkpoint FROM (
                SELECT thread_id, type, checkpoint,
                       ROW_NUMBER() OVER (
                           PARTITION BY thread_id ORDER BY checkpoint_id DESC
                       ) AS rn
                FROM checkpoints
                WHERE thread_id IN ({placeholders})
            ) WHERE rn = 1
        """  # noqa: S608  # placeholders built from len(chunk); user values use ? params
        async with conn.execute(query, chunk) as cursor:
            rows = await cursor.fetchall()

        loop = asyncio.get_running_loop()
        for row in rows:
            tid, type_str, checkpoint_blob = row
            if not type_str or not checkpoint_blob:
                results[tid] = _CheckpointSummary(message_count=0, initial_prompt=None)
                continue
            try:
                data = await loop.run_in_executor(
                    None, serde.loads_typed, (type_str, checkpoint_blob)
                )
                results[tid] = _summarize_checkpoint(data)
            except Exception:
                logger.warning(
                    "Failed to deserialize checkpoint for thread %s; "
                    "message count and initial prompt may be incomplete",
                    tid,
                    exc_info=True,
                )
                results[tid] = _CheckpointSummary(message_count=0, initial_prompt=None)

    return results


async def _load_latest_checkpoint_summary(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> _CheckpointSummary:
    """최신 체크포인트 행에서 체크포인트 파생 요약 데이터를 로드합니다.

    Returns:
        최신 체크포인트 행에서 추출된 메시지 수 및 프롬프트 데이터입니다.

    """
    query = """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id DESC
        LIMIT 1
    """
    async with conn.execute(query, (thread_id,)) as cursor:
        row = await cursor.fetchone()
        if not row or not row[0] or not row[1]:
            return _CheckpointSummary(message_count=0, initial_prompt=None)

        type_str, checkpoint_blob = row
        try:
            data = serde.loads_typed((type_str, checkpoint_blob))
        except (ValueError, TypeError, KeyError, AttributeError):
            logger.warning(
                "Failed to deserialize checkpoint for thread %s; "
                "message count and initial prompt may be incomplete",
                thread_id,
                exc_info=True,
            )
            return _CheckpointSummary(message_count=0, initial_prompt=None)

    return _summarize_checkpoint(data)


def _summarize_checkpoint(data: object) -> _CheckpointSummary:
    """체크포인트 데이터에서 메시지 수와 초기 인간 프롬프트를 추출합니다.

    Returns:
        디코딩된 체크포인트 페이로드에 대한 구조화된 요약입니다.

    """
    messages = _checkpoint_messages(data)
    return _CheckpointSummary(
        message_count=len(messages),
        initial_prompt=_initial_prompt_from_messages(messages),
    )


def _checkpoint_messages(data: object) -> list[object]:
    """디코딩된 페이로드가 예상한 모양이면 체크포인트 메시지를 반환합니다."""
    if not isinstance(data, dict):
        return []

    payload = cast("dict[str, object]", data)
    channel_values = payload.get("channel_values")
    if not isinstance(channel_values, dict):
        return []

    channel_values_dict = cast("dict[str, object]", channel_values)
    messages = channel_values_dict.get("messages")
    if not isinstance(messages, list):
        return []

    return cast("list[object]", messages)


def _initial_prompt_from_messages(messages: list[object]) -> str | None:
    """체크포인트 메시지 목록에서 첫 번째 사람 메시지 내용을 반환합니다."""
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            return _coerce_prompt_text(getattr(msg, "content", None))
    return None


def _coerce_prompt_text(content: object) -> str | None:
    """체크포인트 메시지 내용을 표시 가능한 텍스트로 정규화합니다.

    Returns:
        표시 가능한 프롬프트 텍스트 또는 내용이 비어 있는 경우 `None`.

    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_dict = cast("dict[str, object]", part)
                text = part_dict.get("text")
                parts.append(text if isinstance(text, str) else "")
            else:
                parts.append(str(part))
        joined = " ".join(parts).strip()
        return joined or None
    if content is None:
        return None
    return str(content)


# ---------------------------------------------------------------------------
# Public thread helpers and CLI-facing commands
# ---------------------------------------------------------------------------

async def get_most_recent(agent_name: str | None = None) -> str | None:
    """선택적으로 에이전트별로 필터링된 최신 thread_id를 가져옵니다.

    Returns:
        가장 최근의 thread_id 또는 스레드가 없는 경우 None입니다.

    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        if agent_name:
            query = """
                SELECT thread_id FROM checkpoints
                WHERE json_extract(metadata, '$.agent_name') = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """
            params: tuple = (agent_name,)
        else:
            query = (
                "SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            )
            params = ()

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_thread_agent(thread_id: str) -> str | None:
    """스레드에 대한 Agent_name을 가져옵니다.

    Returns:
        스레드와 연관된 에이전트 이름이거나, 찾을 수 없는 경우 없음입니다.

    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None

        query = """
            SELECT json_extract(metadata, '$.agent_name')
            FROM checkpoints
            WHERE thread_id = ?
            LIMIT 1
        """
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def thread_exists(thread_id: str) -> bool:
    """체크포인트에 스레드가 있는지 확인합니다.

    Returns:
        스레드가 존재하면 True이고, 그렇지 않으면 False입니다.

    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        query = "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1"
        async with conn.execute(query, (thread_id,)) as cursor:
            row = await cursor.fetchone()
            return row is not None


async def find_similar_threads(thread_id: str, limit: int = 3) -> list[str]:
    """주어진 접두어로 시작하는 ID를 가진 스레드를 찾습니다.

    Args:
        thread_id: 스레드 ID와 일치하는 접두사입니다.
        limit: 반환할 일치 스레드의 최대 수입니다.

    Returns:
        지정된 접두사로 시작하는 스레드 ID 목록입니다.

    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        query = """
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id LIKE ?
            ORDER BY thread_id
            LIMIT ?
        """
        prefix = thread_id + "%"
        async with conn.execute(query, (prefix, limit)) as cursor:
            rows = await cursor.fetchall()
            return [r[0] for r in rows]


async def delete_thread(thread_id: str) -> bool:
    """스레드 체크포인트를 삭제합니다.

    Returns:
        스레드가 삭제된 경우 True이고, 스레드가 없으면 False입니다.

    """
    async with _connect() as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False

        cursor = await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        deleted = cursor.rowcount > 0
        if await _table_exists(conn, "writes"):
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await conn.commit()
        if deleted:
            _message_count_cache.pop(thread_id, None)
            for key, rows in list(_recent_threads_cache.items()):
                filtered = [row for row in rows if row["thread_id"] != thread_id]
                _recent_threads_cache[key] = filtered
        return deleted


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """글로벌 데이터베이스용 AsyncSqliteSaver를 가져옵니다.

    Yields:
        체크포인트 지속성을 위한 AsyncSqliteSaver 인스턴스.

    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    _patch_aiosqlite()

    async with AsyncSqliteSaver.from_conn_string(str(get_db_path())) as checkpointer:
        yield checkpointer


_DEFAULT_THREAD_LIMIT = 20


def get_thread_limit() -> int:
    """`DA_CLI_RECENT_THREADS`에서 스레드 목록 제한을 읽습니다.

    변수가 설정되지 않거나 정수가 아닌 값을 포함하는 경우 `_DEFAULT_THREAD_LIMIT`로 대체됩니다. 결과는 최소 1로 고정됩니다.

    Returns:
        표시할 스레드 수입니다.

    """
    import os

    raw = os.environ.get("DA_CLI_RECENT_THREADS")
    if raw is None:
        return _DEFAULT_THREAD_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid DA_CLI_RECENT_THREADS value %r, using default %d",
            raw,
            _DEFAULT_THREAD_LIMIT,
        )
        return _DEFAULT_THREAD_LIMIT


async def list_threads_command(
    agent_name: str | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
    branch: str | None = None,
    verbose: bool = False,
    relative: bool | None = None,
    *,
    output_format: OutputFormat = "text",
) -> None:
    """`deepagents threads list`에 대한 CLI 처리기.

    최근 대화 스레드 테이블을 가져오고 표시하며 선택적으로 에이전트 이름 또는 git 분기로 필터링됩니다.

    Args:
        agent_name: 이 에이전트에 속한 스레드만 표시합니다.

            `None`인 경우 모든 에이전트에 대한 스레드가 표시됩니다.
        limit: 표시할 최대 스레드 수입니다.

            `None`인 경우 `DA_CLI_RECENT_THREADS`에서 읽거나 기본값으로 돌아갑니다.
        sort_by: 정렬 필드 — `"updated"` 또는 `"created"`.

            `None`인 경우 구성(`~/.deepagents/config.toml`)에서 읽습니다.
        branch: 이 git 브랜치의 스레드만 표시합니다.
        verbose: `True`인 경우 모든 열(분기, 생성됨, 프롬프트)을 표시합니다.
        relative: 타임스탬프를 상대 시간으로 표시합니다(예: '5분 전').

            `None`인 경우 구성(`~/.deepagents/config.toml`)에서 읽습니다.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    """
    from deepagents_cli.model_config import (
        load_thread_relative_time,
        load_thread_sort_order,
    )

    if sort_by is None:
        raw = load_thread_sort_order()
        sort_by = "created" if raw == "created_at" else "updated"
    if relative is None:
        relative = load_thread_relative_time()

    fmt_ts = format_relative_timestamp if relative else format_timestamp

    limit = get_thread_limit() if limit is None else max(1, limit)

    threads = await list_threads(
        agent_name,
        limit=limit,
        include_message_count=True,
        sort_by=sort_by,
        branch=branch,
    )

    if verbose and threads:
        await populate_thread_checkpoint_details(
            threads, include_message_count=False, include_initial_prompt=True
        )

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json("threads list", list(threads))
        return

    from rich.markup import escape as escape_markup
    from rich.table import Table

    from deepagents_cli import theme
    from deepagents_cli.config import console

    if not threads:
        filters = []
        if agent_name:
            filters.append(f"agent '{escape_markup(agent_name)}'")
        if branch:
            filters.append(f"branch '{escape_markup(branch)}'")
        if filters:
            console.print(
                f"[yellow]No threads found for {' and '.join(filters)}.[/yellow]"
            )
        else:
            console.print("[yellow]No threads found.[/yellow]")
        console.print("[dim]Start a conversation with: deepagents[/dim]")
        return

    title_parts = []
    if agent_name:
        title_parts.append(f"agent '{escape_markup(agent_name)}'")
    if branch:
        title_parts.append(f"branch '{escape_markup(branch)}'")

    title_filter = f" for {' and '.join(title_parts)}" if title_parts else ""
    sort_label = "created" if sort_by == "created" else "updated"
    title = f"Recent Threads{title_filter} (last {limit}, by {sort_label})"

    table = Table(title=title, show_header=True, header_style=f"bold {theme.PRIMARY}")
    table.add_column("Thread ID", style="bold")
    table.add_column("Agent")
    table.add_column("Messages", justify="right")
    if verbose:
        table.add_column("Created")
    table.add_column("Updated" if sort_by == "updated" else "Last Used")
    if verbose:
        table.add_column("Branch")
        table.add_column("Location")
        table.add_column("Prompt", max_width=40, no_wrap=True)

    prompt_max = 40

    for t in threads:
        row: list[str] = [
            t["thread_id"],
            t["agent_name"] or "unknown",
            str(t.get("message_count", 0)),
        ]
        if verbose:
            row.append(fmt_ts(t.get("created_at")))
        row.append(fmt_ts(t.get("updated_at")))
        if verbose:
            prompt = " ".join((t.get("initial_prompt") or "").split())
            if len(prompt) > prompt_max:
                prompt = prompt[: prompt_max - 3] + "..."
            row.extend(
                [
                    t.get("git_branch") or "",
                    format_path(t.get("cwd")),
                    prompt,
                ]
            )
        table.add_row(*row)

    console.print()
    console.print(table)
    if len(threads) >= limit:
        console.print(
            f"[dim]Showing last {limit} threads. "
            "Override with -n/--limit or DA_CLI_RECENT_THREADS.[/dim]"
        )
    console.print()


async def delete_thread_command(
    thread_id: str,
    *,
    dry_run: bool = False,
    output_format: OutputFormat = "text",
) -> None:
    """CLI 처리기: deepagents 스레드 삭제.

    Args:
        thread_id: 삭제할 스레드의 ID입니다.
        dry_run: `True`인 경우 변경하지 않고 어떤 일이 발생하는지 인쇄하세요.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    """
    if dry_run:
        exists = await thread_exists(thread_id)
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json(
                "threads delete",
                {"thread_id": thread_id, "exists": exists, "dry_run": True},
            )
            return

        from rich.markup import escape as escape_markup

        from deepagents_cli.config import console

        escaped_id = escape_markup(thread_id)
        if exists:
            console.print(f"Would delete thread '{escaped_id}'.")
        else:
            console.print(f"Thread '{escaped_id}' not found. Nothing to delete.")
        console.print("No changes made.", style="dim")
        return

    deleted = await delete_thread(thread_id)

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json("threads delete", {"thread_id": thread_id, "deleted": deleted})
        return

    from rich.markup import escape as escape_markup

    from deepagents_cli import theme
    from deepagents_cli.config import console

    escaped_id = escape_markup(thread_id)
    if deleted:
        console.print(f"[green]Thread '{escaped_id}' deleted.[/green]")
    else:
        console.print(
            f"Thread '{escaped_id}' not found or already deleted.",
            style=theme.MUTED,
        )
