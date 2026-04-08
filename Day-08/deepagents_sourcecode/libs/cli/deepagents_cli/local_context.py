"""시스템 프롬프트에 로컬 컨텍스트를 삽입하기 위한 미들웨어입니다.

백엔드를 통해 bash 스크립트를 실행하여 git 상태, 프로젝트 구조, 패키지 관리자, 런타임 및 디렉터리 레이아웃을 감지합니다. 스크립트는 백엔드(로컬 셸
또는 원격 샌드박스) 내부에서 실행되므로 에이전트가 실행되는 위치에 관계없이 동일한 감지 논리가 작동합니다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NotRequired,
    Protocol,
    cast,
    runtime_checkable,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import ExecuteResponse
    from deepagents.middleware.summarization import SummarizationEvent
    from langgraph.runtime import Runtime

    from deepagents_cli.mcp_tools import MCPServerInfo


_TOOL_NAME_DISPLAY_LIMIT = 10
"""시스템 프롬프트에서 MCP 서버당 표시되는 최대 도구 이름 수입니다."""

_DETECT_SCRIPT_TIMEOUT = 30
"""환경 감지 스크립트의 시간 초과(초)입니다."""


def _build_mcp_context(servers: list[MCPServerInfo]) -> str:
    """시스템 프롬프트에 대한 MCP 서버/도구 인벤토리 형식을 지정합니다.

Args:
        servers: 연결된 MCP 서버 메타데이터 목록입니다.

Returns:
        형식화된 마크다운 문자열 또는 서버가 없는 경우 `""`입니다.

    """
    if not servers:
        return ""

    total_tools = sum(len(s.tools) for s in servers)
    lines = [f"**MCP Servers** ({len(servers)} servers, {total_tools} tools):"]

    for server in servers:
        if not server.tools:
            lines.append(f"- **{server.name}** ({server.transport}): (no tools)")
            continue

        names = [t.name for t in server.tools]
        if len(names) > _TOOL_NAME_DISPLAY_LIMIT:
            shown = ", ".join(names[:_TOOL_NAME_DISPLAY_LIMIT])
            remaining = len(names) - _TOOL_NAME_DISPLAY_LIMIT
            lines.append(
                f"- **{server.name}** ({server.transport}): "
                f"{shown}, and {remaining} more"
            )
        else:
            lines.append(
                f"- **{server.name}** ({server.transport}): {', '.join(names)}"
            )

    return "\n".join(lines)


@runtime_checkable
class _ExecutableBackend(Protocol):
    """`execute(command) -> ExecuteResponse`을 지원하는 모든 백엔드."""

    def execute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse: ...


@runtime_checkable
class _AsyncExecutableBackend(Protocol):
    """비동기 `aexecute` 메서드를 제공하는 모든 백엔드."""

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # Timeout is forwarded to backend, not used as asyncio timeout
    ) -> ExecuteResponse: ...


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context detection script
#
# Outputs markdown describing the current working environment. Each section
# is guarded so that missing tools or unsupported environments are silently
# skipped -- external tools like git, tree, python3, and node are checked
# with `command -v` before use.
#
# The script is built from section functions so each piece can be tested
# independently. Independent sections run as parallel background subshells;
# see build_detect_script() for the orchestration logic.
# ---------------------------------------------------------------------------


def _section_header() -> str:
    """CWD 라인 및 IN_GIT 플래그(다른 섹션에서 사용됨)

Returns:
        헤더를 인쇄하고 `CWD` / `IN_GIT`을 설정하는 Bash 스니펫입니다.

    """
    return r"""CWD="$(pwd)"
echo "## Local Context"
echo ""
echo "**Current Directory**: \`${CWD}\`"
echo ""

# --- Check git once ---
IN_GIT=false
if command -v git >/dev/null 2>&1 \
    && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=true
fi"""


def _section_project() -> str:
    """언어, 모노레포, git 루트, 가상 환경 감지.

Returns:
        Bash 스니펫(헤더에서 `CWD` / `IN_GIT` 필요)

    """
    return r"""# --- Project ---
PROJ_LANG=""
[ -f pyproject.toml ] || [ -f setup.py ] && PROJ_LANG="python"
[ -z "$PROJ_LANG" ] && [ -f package.json ] && PROJ_LANG="javascript/typescript"
[ -z "$PROJ_LANG" ] && [ -f Cargo.toml ] && PROJ_LANG="rust"
[ -z "$PROJ_LANG" ] && [ -f go.mod ] && PROJ_LANG="go"
[ -z "$PROJ_LANG" ] && { [ -f pom.xml ] || [ -f build.gradle ]; } && PROJ_LANG="java"

MONOREPO=false
{ [ -f lerna.json ] || [ -f pnpm-workspace.yaml ] \
  || [ -d packages ] || { [ -d libs ] && [ -d apps ]; } \
  || [ -d workspaces ]; } && MONOREPO=true

ROOT=""
$IN_GIT && ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

ENVS=""
{ [ -d .venv ] || [ -d venv ]; } && ENVS=".venv"
[ -d node_modules ] && ENVS="${ENVS:+${ENVS}, }node_modules"

HAS_PROJECT=false
{ [ -n "$PROJ_LANG" ] || { [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ]; } \
  || $MONOREPO || [ -n "$ENVS" ]; } && HAS_PROJECT=true

if $HAS_PROJECT; then
  echo "**Project**:"
  [ -n "$PROJ_LANG" ] && echo "- Language: ${PROJ_LANG}"
  [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && echo "- Project root: \`${ROOT}\`"
  $MONOREPO && echo "- Monorepo: yes"
  [ -n "$ENVS" ] && echo "- Environments: ${ENVS}"
  echo ""
fi"""


def _section_package_managers() -> str:
    """Python 및 Node 패키지 관리자 감지.

Returns:
        Bash 스니펫(독립형).

    """
    return r"""# --- Package managers ---
PKG=""
if [ -f uv.lock ]; then PKG="Python: uv"
elif [ -f poetry.lock ]; then PKG="Python: poetry"
elif [ -f Pipfile.lock ] || [ -f Pipfile ]; then PKG="Python: pipenv"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.uv\]' pyproject.toml 2>/dev/null; then PKG="Python: uv"
  elif grep -q '\[tool\.poetry\]' pyproject.toml 2>/dev/null; then PKG="Python: poetry"
  else PKG="Python: pip"
  fi
elif [ -f requirements.txt ]; then PKG="Python: pip"
fi

NODE_PKG=""
if [ -f bun.lockb ] || [ -f bun.lock ]; then NODE_PKG="Node: bun"
elif [ -f pnpm-lock.yaml ]; then NODE_PKG="Node: pnpm"
elif [ -f yarn.lock ]; then NODE_PKG="Node: yarn"
elif [ -f package-lock.json ] || [ -f package.json ]; then NODE_PKG="Node: npm"
fi
[ -n "$NODE_PKG" ] && PKG="${PKG:+${PKG}, }${NODE_PKG}"
[ -n "$PKG" ] && echo "**Package Manager**: ${PKG}" && echo ""
"""


def _section_runtimes() -> str:
    """Python 및 Node 런타임 버전 감지.

Returns:
        Bash 스니펫(독립형).

    """
    return r"""# --- Runtimes ---
RT=""
if command -v python3 >/dev/null 2>&1; then
  PV="$(python3 --version 2>/dev/null | awk '{print $2}')"
  [ -n "$PV" ] && RT="Python ${PV}"
fi
if command -v node >/dev/null 2>&1; then
  NV="$(node --version 2>/dev/null | sed 's/^v//')"
  [ -n "$NV" ] && RT="${RT:+${RT}, }Node ${NV}"
fi
[ -n "$RT" ] && echo "**Runtimes**: ${RT}" && echo ""
"""


def _section_git() -> str:
    """Git 브랜치, 메인 브랜치, 커밋되지 않은 변경 사항.

Returns:
        Bash 스니펫(헤더에서 `IN_GIT` 필요)

    """
    return r"""# --- Git ---
if $IN_GIT; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  GT="**Git**: Current branch \`${BRANCH}\`"

  MAINS=""
  for b in $(git branch 2>/dev/null | sed 's/^[* ]*//'); do
    case "$b" in
      main) MAINS="${MAINS:+${MAINS}, }\`main\`" ;;
      master) MAINS="${MAINS:+${MAINS}, }\`master\`" ;;
    esac
  done
  [ -n "$MAINS" ] && GT="${GT}, main branch available: ${MAINS}"

  DC=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  if [ "$DC" -gt 0 ]; then
    if [ "$DC" -eq 1 ]; then GT="${GT}, 1 uncommitted change"
    else GT="${GT}, ${DC} uncommitted changes"
    fi
  fi

  echo "$GT"
  echo ""
fi"""


def _section_test_command() -> str:
    """테스트 명령 감지(make test / pytest / npm test).

Returns:
        Bash 스니펫(독립형).

    """
    return r"""# --- Test command ---
TC=""
if [ -f Makefile ] && grep -qE '^tests?:' Makefile 2>/dev/null; then TC="make test"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.pytest' pyproject.toml 2>/dev/null \
      || [ -f pytest.ini ] || [ -d tests ] || [ -d test ]; then
    TC="pytest"
  fi
elif [ -f package.json ] \
    && grep -q '"test"' package.json 2>/dev/null; then
  TC="npm test"
fi
[ -n "$TC" ] && echo "**Run Tests**: \`${TC}\`" && echo ""
"""


def _section_files() -> str:
    """디렉토리 목록(필터링됨, 최대 20개)

Returns:
        Bash 스니펫(독립형).

    """
    return r"""# --- Files ---
EXCL='node_modules|__pycache__|\.pytest_cache'
EXCL="${EXCL}|\.mypy_cache|\.ruff_cache|\.tox"
EXCL="${EXCL}|\.coverage|\.eggs|dist|build"
FILES=$(
  { ls -1 2>/dev/null; [ -e .deepagents ] && echo .deepagents; } |
  grep -vE "^(${EXCL})$" |
  sort -u
)
if [ -n "$FILES" ]; then
  TOTAL=$(echo "$FILES" | wc -l | tr -d ' ')
  SHOWN_FILES=$(echo "$FILES" | head -20)
  SHOWN=$(echo "$SHOWN_FILES" | wc -l | tr -d ' ')
  echo "**Files** (${SHOWN} shown):"
  echo "$SHOWN_FILES" | while IFS= read -r f; do
    if [ -d "$f" ]; then echo "- ${f}/"
    else echo "- ${f}"
    fi
  done
  [ "$SHOWN" -lt "$TOTAL" ] && echo "... ($((TOTAL - SHOWN)) more files)"
  echo ""
fi"""


def _section_tree() -> str:
    """`tree -L 3` 출력.

Returns:
        Bash 스니펫(독립형).

    """
    return r"""# --- Tree ---
if command -v tree >/dev/null 2>&1; then
  TREE_EXCL='node_modules|.venv|__pycache__|.pytest_cache'
  TREE_EXCL="${TREE_EXCL}|.git|.mypy_cache|.ruff_cache"
  TREE_EXCL="${TREE_EXCL}|.tox|.coverage|.eggs|dist|build"
  T=$(tree -L 3 --noreport --dirsfirst \
    -I "$TREE_EXCL" 2>/dev/null | head -22)
  if [ -n "$T" ]; then
    echo "**Tree** (3 levels):"
    echo '```text'
    echo "$T"
    echo '```'
    echo ""
  fi
fi"""


def _section_makefile() -> str:
    """Makefile의 처음 20줄(모노레포의 git 루트로 대체)

Returns:
        Bash 스니펫(`_section_project`에서 `ROOT` 필요, 헤더에서 `CWD` 필요)

    """
    return r"""# --- Makefile ---
MK=""
if [ -f Makefile ]; then
  MK="Makefile"
elif [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && [ -f "${ROOT}/Makefile" ]; then
  MK="${ROOT}/Makefile"
fi
if [ -n "$MK" ]; then
  echo "**Makefile** (\`${MK}\`, first 20 lines):"
  echo '```makefile'
  head -20 "$MK"
  TL=$(wc -l < "$MK" | tr -d ' ')
  [ "$TL" -gt 20 ] && echo "... (truncated)"
  echo '```'
fi"""


def build_detect_script() -> str:
    """모든 섹션 기능을 전체 감지 스크립트에 연결합니다.

    독립 섹션은 임시 파일에 기록하는 병렬 백그라운드 작업으로 실행되며 결과는 원래 표시 순서로 연결됩니다. 헤더(CWD/IN_GIT) 및 프로젝트
    섹션(ROOT 설정)이 먼저 실행됩니다. 이후 섹션은 해당 변수에 따라 달라지기 때문입니다.

Returns:
        `backend.execute()`에 대한 bash heredoc를 완료하세요.

    """
    # Header + project run synchronously (set CWD, IN_GIT, ROOT for others)
    serial_prefix = f"{_section_header()}\n{_section_project()}"

    # These sections are independent — run them in parallel.
    # Subshells inherit parent variables (IN_GIT, ROOT, CWD) via fork.
    # Individual exit codes are not tracked because sections legitimately
    # exit non-zero when they have nothing to report (e.g. no runtimes).
    parallel_sections = [
        ("02_pkgmgr", _section_package_managers()),
        ("03_runtimes", _section_runtimes()),
        ("04_git", _section_git()),
        ("05_testcmd", _section_test_command()),
        ("06_files", _section_files()),
        ("07_tree", _section_tree()),
        ("08_makefile", _section_makefile()),
    ]

    # Build parallel wrapper: each section runs in a subshell writing to a
    # temp file. Stderr is captured per-section to prevent noise leakage.
    parallel_setup = "_DCT=$(mktemp -d) || exit 1\ntrap 'rm -rf \"$_DCT\"' EXIT"
    parallel_block = "\n".join(
        f'(\n{body}\n) > "$_DCT/{name}" 2>"$_DCT/{name}.err" &'
        for name, body in parallel_sections
    )
    cat_line = "cat " + " ".join(f'"$_DCT/{name}"' for name, _ in parallel_sections)

    body = f"{serial_prefix}\n{parallel_setup}\n{parallel_block}\nwait\n{cat_line}"
    return f"bash <<'__DETECT_CONTEXT_EOF__'\n{body}\n__DETECT_CONTEXT_EOF__\n"


DETECT_CONTEXT_SCRIPT = build_detect_script()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class LocalContextState(AgentState):
    """로컬 컨텍스트 미들웨어의 상태입니다."""

    local_context: NotRequired[str]
    """형식화된 로컬 컨텍스트: cwd, 프로젝트, 패키지 관리자,
    런타임, git, 테스트 명령, 파일, 트리, Makefile.

    """

    _local_context_refreshed_at_cutoff: NotRequired[Annotated[int, PrivateStateAttr]]
    """마지막으로 새로 고친 요약 이벤트의 컷오프 인덱스입니다.

    LangGraph 체크포인트 상태(스레드별로 격리됨) 및 비공개(`PrivateStateAttr`를 통해 하위 에이전트에 노출되지 않음)로 저장됩니다.
    동일한 요약 이벤트에 대한 탐지 스크립트의 중복 재실행을 방지하는 데 사용됩니다.

    """


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class LocalContextMiddleware(AgentMiddleware):
    """로컬 컨텍스트(git 상태, 프로젝트 구조 등)를 시스템 프롬프트에 삽입합니다.

    첫 번째 상호작용 시와 각 요약 이벤트 후에 다시 `backend.execute()`을 통해 bash 감지 스크립트를 실행하고 결과를 상태로 저장하며
    모든 모델 호출 시 시스템 프롬프트에 추가합니다.

    스크립트는 백엔드 내부에서 실행되므로 로컬 셸과 원격 샌드박스 모두에서 작동합니다.

    """

    state_schema = LocalContextState

    def __init__(
        self,
        backend: _ExecutableBackend | _AsyncExecutableBackend,
        *,
        mcp_server_info: list[MCPServerInfo] | None = None,
    ) -> None:
        """셸 실행을 지원하는 백엔드로 초기화합니다.

Args:
            backend: 셸 명령 실행을 제공하는 백엔드 인스턴스입니다.
            mcp_server_info: 시스템 프롬프트에 포함할 MCP 서버 메타데이터입니다.

        """
        self.backend = backend
        self._mcp_context = _build_mcp_context(mcp_server_info or [])

    @staticmethod
    def _handle_detect_result(result: ExecuteResponse) -> str | None:
        """탐지 스크립트 출력을 검증하고 상태 저장을 위해 정규화합니다.

Args:
            result: 백엔드의 실행 결과입니다.

Returns:
            스트립된 스크립트 출력 또는 실패/빈 출력의 경우 `None`.

        """
        output = result.output.strip() if result.output else ""
        if result.exit_code is None or result.exit_code != 0:
            logger.warning(
                "Local context detection script %s; "
                "context will be omitted. Output: %.200s",
                f"exited with code {result.exit_code}"
                if result.exit_code is not None
                else "did not report an exit code",
                output or "(empty)",
            )
            return None
        if not output:
            logger.debug(
                "Local context detection script succeeded but produced no output"
            )
        return output or None

    def _run_detect_script(self) -> str | None:
        """환경 감지 스크립트를 실행합니다.

Returns:
            스트립된 스크립트 출력 또는 실패/빈 출력의 경우 `None`.

        """
        backend = self.backend
        if not isinstance(backend, _ExecutableBackend):
            logger.debug(
                "Skipping sync local context detection; backend %s only "
                "supports async execution",
                type(backend).__name__,
            )
            return None
        try:
            result = backend.execute(
                DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT
            )
        except NotImplementedError:
            # Expected for async-only backends (e.g. HarborSandbox) that
            # define a stub execute() raising NotImplementedError.
            logger.debug(
                "Backend %s does not support sync execute; "
                "context detection deferred to async path",
                type(backend).__name__,
            )
            return None
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(backend).__name__,
                exc_info=True,
            )
            return None

        return LocalContextMiddleware._handle_detect_result(result)

    # override - state parameter is intentionally narrowed from
    # AgentState to LocalContextState for type safety within this middleware.
    def before_agent(  # type: ignore[override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """첫 번째 상호 작용에서 컨텍스트 감지를 실행하고 요약 후 새로 고칩니다.

        첫 번째 호출에서 탐지 스크립트를 실행하고 결과를 저장합니다. 요약 이벤트(상태에서 새 `_summarization_event`로 표시됨) 후에
        스크립트를 다시 실행하여 세션 중에 발생한 환경 변경 사항을 캡처합니다.

Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.

Returns:
            성공 시 `local_context`이 채워진 상태 업데이트입니다. 에
                요약 후 새로 고침이 실패하면 재시도 루프를 방지하기 위해 컷오프(`local_context` 없이)를 기록하는 상태 업데이트를
                반환합니다.

                컨텍스트가 이미 설정되어 있고 새로 고침이 필요하지 않거나 초기 감지에 실패한 경우 `None`을 반환합니다.

        """
        # --- Post-summarization refresh ---
        # _summarization_event is a private field from SummarizationState.
        # At runtime the merged state dict contains all middleware fields;
        # accessed as untyped dict value because LocalContextState does not
        # (and should not) redeclare it.
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = self._run_detect_script()
                if output:
                    return {
                        "local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                # Script failed — record cutoff to avoid retry loop,
                # keep existing local_context.
                return {"_local_context_refreshed_at_cutoff": cutoff}

        # --- Initial detection (first invocation) ---
        if state.get("local_context"):
            return None

        output = self._run_detect_script()
        if output:
            return {"local_context": output}
        return None

    async def _arun_detect_script(self) -> str | None:
        """환경 감지 스크립트를 비동기적으로 실행합니다.

        백엔드가 `_AsyncExecutableBackend`을 구현할 때 `aexecute`을 선호합니다. 동기화 전용 백엔드에 대한 스레드 풀에서
        동기화 감지 스크립트 실행으로 대체됩니다.

Returns:
            스트립된 스크립트 출력 또는 실패/빈 출력의 경우 `None`.

        """
        backend = self.backend
        if not (
            isinstance(backend, _AsyncExecutableBackend)
            and asyncio.iscoroutinefunction(backend.aexecute)
        ):
            try:
                return await asyncio.to_thread(self._run_detect_script)
            except Exception:
                logger.warning(
                    "Local context detection via sync fallback failed "
                    "(backend: %s); context will be omitted from system prompt",
                    type(backend).__name__,
                    exc_info=True,
                )
                return None
        try:
            result = await backend.aexecute(
                DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT
            )
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(backend).__name__,
                exc_info=True,
            )
            return None

        return LocalContextMiddleware._handle_detect_result(result)

    async def abefore_agent(  # type: ignore[override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """비동기 실행 컨텍스트에서 사용하기 위한 `before_agent`의 비동기 변형입니다.

Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.

Returns:
            성공 시 `local_context`이 채워진 상태 업데이트입니다. 에
                요약 후 새로 고침이 실패하면 재시도 루프를 방지하기 위해 컷오프(`local_context` 없이)를 기록하는 상태 업데이트를
                반환합니다.

                컨텍스트가 이미 설정되어 있고 새로 고침이 필요하지 않거나 초기 감지에 실패한 경우 `None`을 반환합니다.

        """
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = await self._arun_detect_script()
                if output:
                    return {
                        "local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                return {"_local_context_refreshed_at_cutoff": cutoff}

        if state.get("local_context"):
            return None

        output = await self._arun_detect_script()
        if output:
            return {"local_context": output}
        return None

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        """가능한 경우 시스템 프롬프트에 로컬 컨텍스트 및 MCP 정보를 추가합니다.

Args:
            request: 잠재적으로 수정하려는 모델 요청입니다.

Returns:
            컨텍스트가 추가된 수정된 요청 또는 `None`.

        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("local_context", "")

        parts = [p for p in (local_context, self._mcp_context) if p]
        if not parts:
            return None

        system_prompt = request.system_prompt or ""
        new_prompt = system_prompt + "\n\n" + "\n\n".join(parts)
        return request.override(system_prompt=new_prompt)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """시스템 프롬프트에 로컬 컨텍스트를 삽입합니다.

Args:
            request: 처리 중인 모델 요청입니다.
            handler: 수정된 요청으로 호출할 핸들러 함수입니다.

Returns:
            핸들러의 모델 응답입니다.

        """
        modified_request = self._get_modified_request(request)
        return handler(modified_request or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """시스템 프롬프트(비동기)에 로컬 컨텍스트를 삽입합니다.

Args:
            request: 처리 중인 모델 요청입니다.
            handler: 수정된 요청으로 호출할 비동기 처리기 함수입니다.

Returns:
            핸들러의 모델 응답입니다.

        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request or request)


__all__ = ["LocalContextMiddleware"]
