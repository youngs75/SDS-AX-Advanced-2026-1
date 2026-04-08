"""`langgraph dev`에 대한 서버측 그래프 진입점입니다.

이 모듈은 생성된 `langgraph.json`에 의해 참조되며 LangGraph 서버가 로드하고 제공할 수 있는 모듈 수준 변수로 CLI 에이전트 그래프를
노출합니다.

그래프는 `make_graph()`을 통해 모듈 가져오기 시 생성됩니다. 이는 `ServerConfig.from_env()`에서 구성을 읽습니다. 이는
CLI가 `ServerConfig.to_env()`을 통해 구성을 *쓰기*하는 데 사용하는 것과 동일한 데이터 클래스입니다. 이 공유 스키마는 양측이 동기화
상태를 유지하도록 보장합니다.
"""

from __future__ import annotations

import atexit
import logging
import sys
import traceback
from typing import Any

from deepagents_cli._server_config import ServerConfig
from deepagents_cli.project_utils import ProjectContext, get_server_project_context

logger = logging.getLogger(__name__)

# Module-level sandbox state kept alive for the server process lifetime.
_sandbox_cm: Any = None
_sandbox_backend: Any = None


def _build_tools(
    config: ServerConfig,
    project_context: ProjectContext | None,
) -> tuple[list[Any], list[Any] | None]:
    """서버 구성을 기반으로 도구 목록을 수집합니다.

    내장 도구(Tavily를 사용할 수 있는 경우 조건부로 웹 검색 포함)와 MCP 도구(활성화된 경우)를 로드합니다.

    MCP 검색은 `asyncio.run`을 통해 동기적으로 실행됩니다. 이 함수는 모듈 수준 그래프 구성 중에(서버의 비동기 이벤트 루프를 사용할 수
    있기 전) 호출되기 때문입니다.

    Args:
        config: 역직렬화된 서버 구성.
        project_context: MCP 검색을 위한 프로젝트 컨텍스트가 해결되었습니다.

    Returns:
        `(tools, mcp_server_info)`의 튜플입니다.

    Raises:
        FileNotFoundError: MCP 구성 파일을 찾을 수 없는 경우.
        RuntimeError: MCP 도구 로딩이 실패한 경우.

    """
    from deepagents_cli.config import settings
    from deepagents_cli.tools import fetch_url, web_search

    tools: list[Any] = [fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    mcp_server_info: list[Any] | None = None
    if not config.no_mcp:
        import asyncio

        from deepagents_cli.mcp_tools import resolve_and_load_mcp_tools

        try:
            mcp_tools, _, mcp_server_info = asyncio.run(
                resolve_and_load_mcp_tools(
                    explicit_config_path=config.mcp_config_path,
                    no_mcp=config.no_mcp,
                    trust_project_mcp=config.trust_project_mcp,
                    project_context=project_context,
                )
            )
        except FileNotFoundError:
            logger.exception("MCP config file not found: %s", config.mcp_config_path)
            raise
        except RuntimeError:
            logger.exception(
                "Failed to load MCP tools (config: %s)", config.mcp_config_path
            )
            raise

        tools.extend(mcp_tools)
        if mcp_tools:
            logger.info("Loaded %d MCP tool(s)", len(mcp_tools))

    return tools, mcp_server_info


def make_graph() -> Any:  # noqa: ANN401
    """환경 기반 구성에서 CLI 에이전트 그래프를 생성합니다.

    `ServerConfig.from_env()`(CLI 프로세스에서 사용하는 `ServerConfig.to_env()`의 반대)을 통해
    `DEEPAGENTS_CLI_SERVER_*` 환경 변수를 읽고, 모델을 확인하고, 도구를 조합하고, 에이전트 그래프를 컴파일합니다.

    Returns:
        LangGraph 에이전트 그래프를 컴파일했습니다.

    """
    config = ServerConfig.from_env()
    project_context = get_server_project_context()

    from deepagents_cli.agent import create_cli_agent, load_async_subagents
    from deepagents_cli.config import create_model, settings

    if project_context is not None:
        settings.reload_from_environment(start_path=project_context.user_cwd)

    result = create_model(config.model, extra_kwargs=config.model_params)
    result.apply_to_settings()

    tools, mcp_server_info = _build_tools(config, project_context)

    # Create sandbox backend if a sandbox provider is configured.
    # The context manager is held open at module level and cleaned up via
    # atexit so the sandbox lives for the entire server process lifetime.
    global _sandbox_cm, _sandbox_backend  # noqa: PLW0603
    sandbox_backend = None
    if config.sandbox_type:
        from deepagents_cli.integrations.sandbox_factory import create_sandbox

        try:
            _sandbox_cm = create_sandbox(
                config.sandbox_type,
                sandbox_id=config.sandbox_id,
                setup_script_path=config.sandbox_setup,
            )
            _sandbox_backend = _sandbox_cm.__enter__()  # noqa: PLC2801  # Context manager kept open for server process lifetime
            sandbox_backend = _sandbox_backend

            def _cleanup_sandbox() -> None:
                if _sandbox_cm is not None:
                    _sandbox_cm.__exit__(None, None, None)

            atexit.register(_cleanup_sandbox)
        except ImportError:
            logger.exception(
                "Sandbox provider '%s' is not installed", config.sandbox_type
            )
            print(  # noqa: T201  # stderr fallback — logger may not reach parent process
                f"Sandbox provider '{config.sandbox_type}' is not installed",
                file=sys.stderr,
            )
            sys.exit(1)
        except NotImplementedError:
            logger.exception("Sandbox type '%s' is not supported", config.sandbox_type)
            print(  # noqa: T201  # stderr fallback — logger may not reach parent process
                f"Sandbox type '{config.sandbox_type}' is not supported",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as exc:
            logger.exception("Sandbox creation failed for '%s'", config.sandbox_type)
            print(  # noqa: T201  # stderr fallback — logger may not reach parent process
                f"Sandbox creation failed for '{config.sandbox_type}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    async_subagents = load_async_subagents() or None

    agent, _ = create_cli_agent(
        model=result.model,
        assistant_id=config.assistant_id,
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type=config.sandbox_type,
        system_prompt=config.system_prompt,
        interactive=config.interactive,
        auto_approve=config.auto_approve,
        interrupt_shell_only=config.interrupt_shell_only,
        shell_allow_list=config.shell_allow_list,
        enable_ask_user=config.enable_ask_user,
        enable_memory=config.enable_memory,
        enable_skills=config.enable_skills,
        enable_shell=config.enable_shell,
        mcp_server_info=mcp_server_info,
        cwd=project_context.user_cwd if project_context is not None else config.cwd,
        project_context=project_context,
        async_subagents=async_subagents,
    )
    return agent


try:
    graph = make_graph()
except Exception as exc:
    logger.critical("Failed to initialize server graph", exc_info=True)
    print(  # noqa: T201  # stderr fallback — logger may not reach parent process
        f"Failed to initialize server graph: {exc}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    sys.exit(1)
