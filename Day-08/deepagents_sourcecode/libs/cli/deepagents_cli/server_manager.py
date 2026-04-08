"""CLI에 대한 서버 수명주기 조정.

다음의 전체 흐름을 처리하는 `start_server_and_get_agent`을 제공합니다.

1. CLI 인수에서 `ServerConfig` 빌드 2. `ServerConfig.to_env()`을 통해 환경 변수에 구성 작성 3. 작업 공간
스캐폴딩(langgraph.json, checkpointer, pyproject) 4. `langgraph dev` 서버 시작 5. `RemoteAgent`
클라이언트 반환

또한 호출자가 try/finally 해제를 중복할 필요가 없도록 서버 시작 및 정리 보장을 래핑하는 비동기 컨텍스트 관리자인 `server_session`도
제공합니다.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from deepagents_cli.mcp_tools import MCPSessionManager
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess

from deepagents_cli._env_vars import SERVER_ENV_PREFIX
from deepagents_cli._server_config import ServerConfig
from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)


def _set_or_clear_server_env(name: str, value: str | None) -> None:
    """`DEEPAGENTS_CLI_SERVER_*` 환경 변수를 설정하거나 지웁니다.

    Args:
        name: `DEEPAGENTS_CLI_SERVER_` 뒤의 접미사입니다.
        value: 설정할 문자열 값 또는 변수를 지우려면 `None`입니다.

    """
    key = f"{SERVER_ENV_PREFIX}{name}"
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _apply_server_config(config: ServerConfig) -> None:
    """`ServerConfig`을 `DEEPAGENTS_CLI_SERVER_*` 환경 변수에 씁니다.

    변수 세트와 해당 직렬화 형식이 여기와 판독기(`ServerConfig.from_env()`)에서 독립적으로 유지 관리되는 대신 한
    위치(`ServerConfig` 데이터 클래스)에 정의되도록 `ServerConfig.to_env()`을 사용합니다.

    Args:
        config: 서버 구성이 완전히 해결되었습니다.

    """
    for suffix, value in config.to_env().items():
        _set_or_clear_server_env(suffix, value)


def _capture_project_context() -> ProjectContext | None:
    """서버 하위 프로세스에 대한 사용자의 프로젝트 컨텍스트를 캡처합니다.

    Returns:
        명시적인 프로젝트 컨텍스트 또는 cwd를 확인할 수 없는 경우 `None`입니다.

    """
    try:
        return ProjectContext.from_user_cwd(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory for server")
        return None


# ------------------------------------------------------------------
# Workspace scaffolding
# ------------------------------------------------------------------


def _scaffold_workspace(work_dir: Path) -> None:
    """모든 필수 파일이 포함된 서버 작업 디렉터리를 준비합니다.

    서버 그래프 진입점을 *work_dir*에 복사하고 `langgraph dev`이 부팅하는 데 필요한 보조 파일(검사포인터 모듈,
    `pyproject.toml`, `langgraph.json`)을 생성합니다.

    Args:
        work_dir: 서버의 cwd가 될 임시 디렉터리입니다.

    """
    from deepagents_cli.server import generate_langgraph_json

    server_graph_src = Path(__file__).parent / "server_graph.py"
    server_graph_dst = work_dir / "server_graph.py"
    shutil.copy2(server_graph_src, server_graph_dst)

    _write_checkpointer(work_dir)
    _write_pyproject(work_dir)

    # Relative paths resolve against the subprocess cwd, which
    # ServerProcess.start() sets to work_dir (server.py). Using absolute paths
    # here breaks Windows because importlib treats backslash paths as module names.
    generate_langgraph_json(
        work_dir,
        graph_ref="./server_graph.py:graph",
        checkpointer_path="./checkpointer.py:create_checkpointer",
    )


def _write_checkpointer(work_dir: Path) -> None:
    """환경에서 DB 경로를 읽는 체크포인터 모듈을 작성합니다.

    생성된 모듈은 런타임 시 DB 경로 env var를 읽으므로 해당 경로는 생성된 소스에 적용되지 않습니다. 이는 다른 곳에서 사용되는
    `DEEPAGENTS_CLI_SERVER_*` env-var 통신 패턴과 일치합니다.

    Args:
        work_dir: 서버 작업 디렉토리.

    """
    from deepagents_cli.sessions import get_db_path

    # Set the env var that the generated module will read at import time.
    os.environ[f"{SERVER_ENV_PREFIX}DB_PATH"] = str(get_db_path())

    db_path_var = f"{SERVER_ENV_PREFIX}DB_PATH"
    content = f'''\
"""Persistent SQLite checkpointer for the LangGraph dev server."""

import os
from contextlib import asynccontextmanager


@asynccontextmanager
async def create_checkpointer():
    """Yield an AsyncSqliteSaver connected to the CLI sessions DB.

    The database path is read from the `{db_path_var}` env var
    (set by the CLI before server startup) rather than hard-coded, so
    the checkpointer module works without code generation.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = os.environ.get("{db_path_var}")
    if not db_path:
        raise RuntimeError(
            "{db_path_var} not set. The CLI must set this "
            "env var before server startup."
        )
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        yield saver
'''
    (work_dir / "checkpointer.py").write_text(content)


def _write_pyproject(work_dir: Path) -> None:
    """서버 작업 디렉터리에 대한 최소한의 pyproject.toml을 작성합니다.

    `langgraph dev` 서버는 프로젝트 종속성을 설치해야 합니다. SDK를 전이적으로 가져오는 CLI 패키지를 가리킵니다.

    Args:
        work_dir: 서버 작업 디렉토리.

    """
    cli_dir = Path(__file__).parent.parent
    content = f"""[project]
name = "deepagents-server-runtime"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
    "deepagents-cli @ file://{cli_dir}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (work_dir / "pyproject.toml").write_text(content)


# ------------------------------------------------------------------
# Server startup
# ------------------------------------------------------------------


async def start_server_and_get_agent(
    *,
    assistant_id: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = 2024,
) -> tuple[RemoteAgent, ServerProcess, MCPSessionManager | None]:
    """LangGraph 서버를 시작하고 연결된 원격 에이전트 클라이언트를 반환합니다.

    Args:
        assistant_id: 에이전트 식별자.
        model_name: 모델 사양 문자열.
        model_params: 추가 모델 kwargs.
        auto_approve: 모든 도구를 자동 승인합니다.
        interrupt_shell_only: HITL 대신 미들웨어를 통해 셸 명령의 유효성을 검사합니다.
        shell_allow_list: `ShellAllowListMiddleware`에 대한 제한적인 셸 허용 목록입니다.
        sandbox_type: 샌드박스 유형.
        sandbox_id: 재사용할 기존 샌드박스 ID입니다.
        sandbox_setup: 샌드박스의 설정 스크립트 경로입니다.
        enable_shell: 셸 실행 도구를 활성화합니다.
        enable_ask_user: Ask_user 도구를 활성화합니다.
        mcp_config_path: MCP 구성 경로입니다.
        no_mcp: MCP를 비활성화합니다.
        trust_project_mcp: 프로젝트 MCP 서버를 신뢰하십시오.
        interactive: 에이전트가 대화형인지 여부입니다.
        host: 서버 호스트.
        port: 서버 포트.

    Returns:
        `(remote_agent, server_process, mcp_session_manager)`의 튜플입니다.
            `mcp_session_manager`은 현재 항상 `None`입니다(MCP 수명 주기는 서버 측에서 처리됩니다).

    """
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess

    project_context = _capture_project_context()

    config = ServerConfig.from_cli_args(
        project_context=project_context,
        model_name=model_name,
        model_params=model_params,
        assistant_id=assistant_id,
        auto_approve=auto_approve,
        interrupt_shell_only=interrupt_shell_only,
        shell_allow_list=shell_allow_list,
        sandbox_type=sandbox_type,
        sandbox_id=sandbox_id,
        sandbox_setup=sandbox_setup,
        enable_shell=enable_shell,
        enable_ask_user=enable_ask_user,
        mcp_config_path=mcp_config_path,
        no_mcp=no_mcp,
        trust_project_mcp=trust_project_mcp,
        interactive=interactive,
    )
    _apply_server_config(config)

    work_dir = Path(tempfile.mkdtemp(prefix="deepagents_server_"))
    _scaffold_workspace(work_dir)

    server = ServerProcess(
        host=host, port=port, config_dir=work_dir, owns_config_dir=True
    )
    try:
        await server.start()
    except Exception:
        server.stop()
        raise

    agent = RemoteAgent(
        url=server.url,
        graph_name="agent",
    )

    return agent, server, None


# ------------------------------------------------------------------
# Session context manager
# ------------------------------------------------------------------


@asynccontextmanager
async def server_session(
    *,
    assistant_id: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
    enable_shell: bool = True,
    enable_ask_user: bool = False,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    interactive: bool = True,
    host: str = "127.0.0.1",
    port: int = 2024,
) -> AsyncIterator[tuple[RemoteAgent, ServerProcess]]:
    """서버를 시작하고 정리를 보장하는 비동기 컨텍스트 관리자입니다.

    호출자가 서버를 중지하기 위해 try/finally 패턴을 복제할 필요가 없도록 `start_server_and_get_agent`을 래핑합니다.

    Args:
        assistant_id: 에이전트 식별자.
        model_name: 모델 사양 문자열.
        model_params: 추가 모델 kwargs.
        auto_approve: 모든 도구를 자동 승인합니다.
        interrupt_shell_only: HITL 대신 미들웨어를 통해 셸 명령의 유효성을 검사합니다.
        shell_allow_list: `ShellAllowListMiddleware`에 대한 제한적인 셸 허용 목록입니다.
        sandbox_type: 샌드박스 유형.
        sandbox_id: 재사용할 기존 샌드박스 ID입니다.
        sandbox_setup: 샌드박스의 설정 스크립트 경로입니다.
        enable_shell: 셸 실행 도구를 활성화합니다.
        enable_ask_user: Ask_user 도구를 활성화합니다.
        mcp_config_path: MCP 구성 경로입니다.
        no_mcp: MCP를 비활성화합니다.
        trust_project_mcp: 프로젝트 MCP 서버를 신뢰하십시오.
        interactive: 에이전트가 대화형인지 여부입니다.
        host: 서버 호스트.
        port: 서버 포트.

    Yields:
        `(remote_agent, server_process)`의 튜플입니다.

    """
    server_proc: ServerProcess | None = None
    mcp_session_manager: MCPSessionManager | None = None
    try:
        agent, server_proc, mcp_session_manager = await start_server_and_get_agent(
            assistant_id=assistant_id,
            model_name=model_name,
            model_params=model_params,
            auto_approve=auto_approve,
            interrupt_shell_only=interrupt_shell_only,
            shell_allow_list=shell_allow_list,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_setup=sandbox_setup,
            enable_shell=enable_shell,
            enable_ask_user=enable_ask_user,
            mcp_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            interactive=interactive,
            host=host,
            port=port,
        )
        yield agent, server_proc
    finally:
        if mcp_session_manager is not None:
            try:
                await mcp_session_manager.cleanup()
            except Exception:
                logger.warning("MCP session cleanup failed", exc_info=True)
        if server_proc is not None:
            server_proc.stop()
