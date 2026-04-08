"""deepagents CLI용 MCP(모델 컨텍스트 프로토콜) 도구 로더.

이 모듈은 Claude Desktop 스타일 JSON 구성을 지원하는 `langchain-mcp-adapters`을 사용하여 MCP 서버를 로드하고 관리하는
비동기 기능을 제공합니다. 또한 사용자 수준 및 프로젝트 수준 위치에서 `.mcp.json` 파일의 자동 검색을 지원합니다.
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

    from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """단일 MCP 도구에 대한 메타데이터입니다."""

    name: str
    """도구 이름(서버 이름 접두사를 포함할 수 있음)"""

    description: str
    """도구의 기능에 대한 사람이 읽을 수 있는 설명입니다."""


@dataclass
class MCPServerInfo:
    """연결된 MCP 서버 및 해당 도구에 대한 메타데이터입니다."""

    name: str
    """MCP 구성의 서버 이름입니다."""

    transport: str
    """전송 유형(`stdio`, `sse` 또는 `http`)."""

    tools: list[MCPToolInfo] = field(default_factory=list)
    """이 서버에 의해 노출되는 도구입니다."""


_SUPPORTED_REMOTE_TYPES = {"sse", "http"}
"""원격 MCP 서버(SSE 및 HTTP)에 지원되는 전송 유형입니다."""


def _resolve_server_type(server_config: dict[str, Any]) -> str:
    """서버 구성의 전송 유형을 결정합니다.

    `type` 및 `transport` 필드 이름을 모두 지원하며 기본값은 `stdio`입니다.

Args:
        server_config: 서버 구성 사전.

Returns:
        전송 유형 문자열(`stdio`, `sse` 또는 `http`).

    """
    t = server_config.get("type")
    if t is not None:
        return t
    return server_config.get("transport", "stdio")


def _validate_server_config(server_name: str, server_config: dict[str, Any]) -> None:
    """단일 서버 구성의 유효성을 검사합니다.

Args:
        server_name: 서버의 이름입니다.
        server_config: 서버 구성 사전.

Raises:
        TypeError: 구성 필드에 잘못된 유형이 있는 경우.
        ValueError: 필수 필드가 누락되었거나 서버 유형이 지원되지 않는 경우.

    """
    if not isinstance(server_config, dict):
        error_msg = f"Server '{server_name}' config must be a dictionary"
        raise TypeError(error_msg)

    server_type = _resolve_server_type(server_config)

    if server_type in _SUPPORTED_REMOTE_TYPES:
        # SSE/HTTP server validation - requires url field
        if "url" not in server_config:
            error_msg = (
                f"Server '{server_name}' with type '{server_type}'"
                " missing required 'url' field"
            )
            raise ValueError(error_msg)

        # headers is optional but must be correct type if present
        headers = server_config.get("headers")
        if headers is not None and not isinstance(headers, dict):
            error_msg = f"Server '{server_name}' 'headers' must be a dictionary"
            raise TypeError(error_msg)
    elif server_type == "stdio":
        # stdio server validation
        if "command" not in server_config:
            error_msg = f"Server '{server_name}' missing required 'command' field"
            raise ValueError(error_msg)

        # args and env are optional but must be correct type if present
        if "args" in server_config and not isinstance(server_config["args"], list):
            error_msg = f"Server '{server_name}' 'args' must be a list"
            raise TypeError(error_msg)

        if "env" in server_config and not isinstance(server_config["env"], dict):
            error_msg = f"Server '{server_name}' 'env' must be a dictionary"
            raise TypeError(error_msg)
    else:
        error_msg = (
            f"Server '{server_name}' has unsupported transport type '{server_type}'. "
            "Supported types: stdio, sse, http"
        )
        raise ValueError(error_msg)


def load_mcp_config(config_path: str) -> dict[str, Any]:
    """JSON 파일에서 MCP 구성을 로드하고 검증합니다.

    다양한 서버 유형을 지원합니다:

    - stdio: `command`, `args`, `env` 필드가 있는 프로세스 기반 서버(기본값) - sse: `type: "sse"`, `url`
    및 선택적 `headers`이 있는 서버 전송 이벤트 서버 - http: `type: "http"`, `url` 및 선택적 `headers`이 있는
    HTTP 기반 서버

Args:
        config_path: MCP JSON 구성 파일의 경로(Claude Desktop 형식)

Returns:
        구문 분석된 구성 사전.

Raises:
        FileNotFoundError: 구성 파일이 존재하지 않는 경우.
        json.JSONDecodeError: 구성 파일에 잘못된 JSON이 포함된 경우.
        TypeError: 구성 필드에 잘못된 유형이 있는 경우.
        ValueError: 구성에 필수 필드가 누락된 경우.

    """
    path = Path(config_path)

    if not path.exists():
        error_msg = f"MCP config file not found: {config_path}"
        raise FileNotFoundError(error_msg)

    try:
        with path.open(encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in MCP config file: {e.msg}"
        raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e

    # Validate required fields
    if "mcpServers" not in config:
        error_msg = (
            "MCP config must contain 'mcpServers' field. "
            'Expected format: {"mcpServers": {"server-name": {...}}}'
        )
        raise ValueError(error_msg)

    if not isinstance(config["mcpServers"], dict):
        error_msg = "'mcpServers' field must be a dictionary"
        raise TypeError(error_msg)

    if not config["mcpServers"]:
        error_msg = "'mcpServers' field is empty - no servers configured"
        raise ValueError(error_msg)

    # Validate each server config
    for server_name, server_config in config["mcpServers"].items():
        _validate_server_config(server_name, server_config)

    return config


def _resolve_project_config_base(project_context: ProjectContext | None) -> Path:
    """프로젝트 수준 MCP 구성 조회를 위한 기본 디렉터리를 확인합니다.

Args:
        project_context: 명시적인 프로젝트 경로 컨텍스트(사용 가능한 경우)

Returns:
        존재하는 경우 프로젝트 루트이고, 그렇지 않은 경우 사용자 작업 디렉터리입니다.

    """
    if project_context is not None:
        return project_context.project_root or project_context.user_cwd

    from deepagents_cli.project_utils import find_project_root

    return find_project_root() or Path.cwd()


def discover_mcp_configs(
    *, project_context: ProjectContext | None = None
) -> list[Path]:
    """표준 위치에서 MCP 구성 파일을 찾으십시오.

    우선 순위에 따라 세 개의 경로를 확인합니다(가장 낮은 것부터 가장 높은 것까지).

    1. `~/.deepagents/.mcp.json`(사용자 수준 전역) 2.
    `<project-root>/.deepagents/.mcp.json`(프로젝트 하위 디렉터리) 3.
    `<project-root>/.mcp.json`(프로젝트 루트, Claude Code compat)

    프로젝트 루트는 제공되면 `project_context`에서 결정되고, 그렇지 않으면 `find_project_root()`에 의해 CWD로
    대체됩니다.

Returns:
        기존 구성 파일 경로 목록(우선 순위가 가장 낮은 것부터 높은 것까지).

    """
    user_dir = Path.home() / ".deepagents"
    project_root = _resolve_project_config_base(project_context)

    candidates = [
        user_dir / ".mcp.json",
        project_root / ".deepagents" / ".mcp.json",
        project_root / ".mcp.json",
    ]

    found: list[Path] = []
    for path in candidates:
        try:
            if path.is_file():
                found.append(path)
        except OSError:
            logger.warning("Could not check MCP config %s", path, exc_info=True)
    return found


def classify_discovered_configs(
    config_paths: list[Path],
) -> tuple[list[Path], list[Path]]:
    """검색된 구성 경로를 사용자 수준과 프로젝트 수준으로 분할합니다.

    사용자 수준 구성은 `~/.deepagents/`에 있습니다. 다른 모든 것은 프로젝트 수준으로 간주됩니다.

Args:
        config_paths: `discover_mcp_configs`에서 반환된 경로입니다.

Returns:
        `(user_configs, project_configs)`의 튜플입니다.

    """
    user_dir = Path.home() / ".deepagents"
    user: list[Path] = []
    project: list[Path] = []
    for path in config_paths:
        try:
            if path.resolve().is_relative_to(user_dir.resolve()):
                user.append(path)
            else:
                project.append(path)
        except (OSError, ValueError):
            project.append(path)
    return user, project


def extract_stdio_server_commands(
    config: dict[str, Any],
) -> list[tuple[str, str, list[str]]]:
    """구문 분석된 MCP 구성에서 stdio 서버 항목을 추출합니다.

Args:
        config: `mcpServers` 키를 사용하여 MCP 구성 사전을 구문 분석했습니다.

Returns:
        각 stdio 서버에 대한 `(server_name, command, args)` 목록입니다.

    """
    results: list[tuple[str, str, list[str]]] = []
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return results
    for name, srv in servers.items():
        if not isinstance(srv, dict):
            continue
        if _resolve_server_type(srv) == "stdio":
            results.append((name, srv.get("command", ""), srv.get("args", [])))
    return results


def _filter_project_stdio_servers(config: dict[str, Any]) -> dict[str, Any]:
    """stdio 서버가 제거된 *config*의 복사본을 반환합니다.

    원격(SSE/HTTP) 서버는 로컬 코드를 실행하지 않기 때문에 유지됩니다.

Args:
        config: 구문 분석된 MCP 구성 dict.

Returns:
        필터링된 구성 사전

    """
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        return config
    filtered = {
        name: srv
        for name, srv in servers.items()
        if isinstance(srv, dict) and _resolve_server_type(srv) != "stdio"
    }
    return {"mcpServers": filtered}


def merge_mcp_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """서버 이름별로 여러 MCP 구성 사전을 병합합니다.

    이후 항목은 동일한 서버 이름(`mcpServers`의 간단한 `dict.update`)에 대해 이전 항목을 재정의합니다.

Args:
        configs: 구문 분석된 구성 사전의 정렬된 목록입니다(각각 `mcpServers` 키 포함).

Returns:
        `mcpServers`이(가) 결합된 구성이 병합되었습니다.

    """
    merged: dict[str, Any] = {}
    for cfg in configs:
        servers = cfg.get("mcpServers")
        if isinstance(servers, dict):
            merged.update(servers)
    return {"mcpServers": merged}


def load_mcp_config_lenient(config_path: Path) -> dict[str, Any] | None:
    """MCP 구성 파일을 로드하고 오류가 발생하면 None을 반환합니다.

    자동 검색에 적합한 관대한 오류 처리로 `load_mcp_config`을 래핑합니다. 누락된 파일은 자동으로 건너뜁니다. 구문 분석 및 유효성 검사
    오류는 경고로 기록됩니다.

Args:
        config_path: MCP 구성 파일의 경로입니다.

Returns:
        구문 분석된 config dict 또는 파일이 없거나 유효하지 않은 경우 None입니다.

    """
    try:
        return load_mcp_config(str(config_path))
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning("Skipping unreadable MCP config %s: %s", config_path, e)
        return None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Skipping invalid MCP config %s: %s", config_path, e)
        return None


class MCPSessionManager:
    """상태 저장 stdio 서버에 대한 영구 MCP 세션을 관리합니다.

    이 관리자는 stdio MCP 서버에 대한 영구 세션을 생성하고 유지하여 모든 도구 호출에서 서버가 다시 시작되는 것을 방지합니다. 세션은 명시적으로
    정리될 때까지 활성 상태로 유지됩니다.

    """

    def __init__(self) -> None:
        """세션 관리자를 초기화합니다."""
        self.client: MultiServerMCPClient | None = None
        self.exit_stack = AsyncExitStack()

    async def cleanup(self) -> None:
        """모든 관리 세션을 정리하고 연결을 닫습니다."""
        await self.exit_stack.aclose()


def _check_stdio_server(server_name: str, server_config: dict[str, Any]) -> None:
    """stdio 서버의 명령이 PATH에 있는지 확인하세요.

Args:
        server_name: 서버 이름(오류 메시지용)
        server_config: `command` 키가 있는 서버 구성 사전입니다.

Raises:
        RuntimeError: 명령이 구성에서 누락되었거나 PATH에서 찾을 수 없는 경우.

    """
    command = server_config.get("command")
    if command is None:
        msg = f"MCP server '{server_name}': missing 'command' in config."
        raise RuntimeError(msg)
    if shutil.which(command) is None:
        msg = (
            f"MCP server '{server_name}': command '{command}' not found on PATH. "
            "Install it or check your MCP config."
        )
        raise RuntimeError(msg)


async def _check_remote_server(server_name: str, server_config: dict[str, Any]) -> None:
    """원격 MCP 서버 URL에 대한 네트워크 연결을 확인하십시오.

    MCP 세션 핸드셰이크 전에 DNS 오류, 연결 거부 및 네트워크 시간 초과를 조기에 감지하기 위해 2초 시간 초과로 경량 HEAD 요청을 보냅니다.
    HTTP 오류 응답(4xx, 5xx)은 실패로 처리되지 않습니다. 전송 오류, 잘못된 URL 및 OS 수준 소켓 오류만 발생합니다.

Args:
        server_name: 서버 이름(오류 메시지용)
        server_config: `url` 키가 있는 서버 구성 사전입니다.

Raises:
        RuntimeError: 서버 URL에 연결할 수 없거나 유효하지 않은 경우.

    """
    import httpx

    url = server_config.get("url")
    if url is None:
        msg = f"MCP server '{server_name}': missing 'url' in config."
        raise RuntimeError(msg)
    try:
        async with httpx.AsyncClient() as client:
            await client.head(url, timeout=2)
    except (httpx.TransportError, httpx.InvalidURL, OSError) as exc:
        msg = (
            f"MCP server '{server_name}': URL '{url}' is unreachable: {exc}. "
            "Check that the URL is correct and the server is running."
        )
        raise RuntimeError(msg) from exc


async def _load_tools_from_config(
    config: dict[str, Any],
) -> tuple[list[BaseTool], MCPSessionManager, list[MCPServerInfo]]:
    """검증된 구성 및 로드 도구에서 MCP 연결을 구축합니다.

    이는 `get_mcp_tools`(명시적 경로) 및 `resolve_and_load_mcp_tools`(자동 검색)에서 사용되는 공유 구현입니다.

Args:
        config: `mcpServers` 키를 사용하여 검증된 MCP 구성 dict입니다.

Returns:
        `(tools_list, session_manager, server_infos)`의 튜플입니다.

Raises:
        RuntimeError: MCP 서버가 생성되거나 연결되지 않는 경우.

    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
    )
    from langchain_mcp_adapters.tools import load_mcp_tools

    # Pre-flight health checks (best-effort early detection; the session setup
    # below has its own error handling for TOCTOU races).
    errors: list[str] = []
    for server_name, server_config in config["mcpServers"].items():
        server_type = _resolve_server_type(server_config)
        try:
            if server_type in _SUPPORTED_REMOTE_TYPES:
                await _check_remote_server(server_name, server_config)
            elif server_type == "stdio":
                _check_stdio_server(server_name, server_config)
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        msg = "Pre-flight health check(s) failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise RuntimeError(msg)

    # Create connections dict for MultiServerMCPClient
    # Convert Claude Desktop format to langchain-mcp-adapters format
    connections: dict[str, Connection] = {}
    for server_name, server_config in config["mcpServers"].items():
        server_type = _resolve_server_type(server_config)

        if server_type in _SUPPORTED_REMOTE_TYPES:
            # langchain-mcp-adapters uses "streamable_http" for HTTP transport
            if server_type == "http":
                conn: Connection = StreamableHttpConnection(
                    transport="streamable_http",
                    url=server_config["url"],
                )
            else:
                conn = SSEConnection(
                    transport="sse",
                    url=server_config["url"],
                )
            if "headers" in server_config:
                conn["headers"] = server_config["headers"]
            connections[server_name] = conn
        else:
            # stdio server connection (default)
            connections[server_name] = StdioConnection(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env") or None,
                transport="stdio",
            )

    # Create session manager to track persistent sessions
    manager = MCPSessionManager()

    try:
        client = MultiServerMCPClient(connections=connections)
        manager.client = client
    except Exception as e:
        await manager.cleanup()
        error_msg = f"Failed to initialize MCP client: {e}"
        raise RuntimeError(error_msg) from e

    try:
        all_tools: list[BaseTool] = []
        server_infos: list[MCPServerInfo] = []
        for server_name, server_config in config["mcpServers"].items():
            session = await manager.exit_stack.enter_async_context(
                client.session(server_name)
            )
            tools = await load_mcp_tools(
                session, server_name=server_name, tool_name_prefix=True
            )
            all_tools.extend(tools)
            server_infos.append(
                MCPServerInfo(
                    name=server_name,
                    transport=_resolve_server_type(server_config),
                    tools=[
                        MCPToolInfo(name=t.name, description=t.description or "")
                        for t in tools
                    ],
                )
            )
    except Exception as e:
        await manager.cleanup()
        error_msg = (
            f"Failed to load tools from MCP server '{server_name}': {e}\n"
            "For stdio servers: Check that the command and args are correct,"
            " and that the MCP server is installed"
            " (e.g., run 'npx -y <package>' manually to test).\n"
            "For sse/http servers: Check that the URL is correct"
            " and the server is running."
        )
        raise RuntimeError(error_msg) from e

    return all_tools, manager, server_infos


async def get_mcp_tools(
    config_path: str,
) -> tuple[list[BaseTool], MCPSessionManager, list[MCPServerInfo]]:
    """상태 저장 세션을 사용하여 구성 파일에서 MCP 도구를 로드합니다.

    여러 서버 유형 지원: - stdio: 영구 세션이 있는 하위 프로세스로 MCP 서버를 생성합니다. - sse/http: URL을 통해 원격 MCP
    서버에 연결합니다.

    stdio 서버의 경우 이는 도구 호출 전반에 걸쳐 활성 상태로 유지되는 영구 세션을 생성하여 서버가 다시 시작되지 않도록 합니다. 세션은
    `MCPSessionManager`에 의해 관리되며 완료되면 `session_manager.cleanup()`로 정리되어야 합니다.

Args:
        config_path: MCP JSON 구성 파일의 경로입니다.

Returns:
        Tuple of `(tools_list, session_manager, server_infos)` where: - tools_list:
                                                                      LangChain
                                                                      `BaseTool` 개체 목록 -
                                                                      session_manager:
                                                                      `MCPSessionManager`
                                                                      인스턴스(완료되면
                                                                      `cleanup()` 호출) -
                                                                      server_infos: 서버별
                                                                      메타데이터가 포함된
                                                                      `MCPServerInfo` 목록

    """
    config = load_mcp_config(config_path)
    return await _load_tools_from_config(config)


async def resolve_and_load_mcp_tools(
    *,
    explicit_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool | None = None,
    project_context: ProjectContext | None = None,
) -> tuple[list[BaseTool], MCPSessionManager | None, list[MCPServerInfo]]:
    """MCP 구성 및 로드 도구를 해결합니다.

    표준 위치에서 구성을 자동으로 검색하고 병합합니다. `explicit_config_path`이 제공되면 가장 높은 우선 순위의 소스로 추가됩니다(해당
    파일의 오류는 치명적임).

Args:
        explicit_config_path: 자동 검색된 구성 위에 추가할 추가 구성 파일입니다(가장 높은 우선순위). 오류는 치명적입니다.
        no_mcp: True인 경우 모든 MCP 로딩을 비활성화합니다.
        trust_project_mcp: 프로젝트 수준 stdio 서버 신뢰를 제어합니다.

            - `True`: 모든 프로젝트 stdio 서버를 허용합니다(플래그/프롬프트 승인).
            - `False`: 프로젝트 stdio 서버를 필터링하고 경고를 기록합니다.
            - `None` (default): 영구 신뢰 저장소를 확인하십시오. 지문이 일치하면 허용합니다. 그렇지 않으면 필터링 + 경고합니다.
        project_context: 구성 검색 및 신뢰 확인을 위한 명시적 프로젝트 경로 컨텍스트입니다.

Returns:
        `(tools_list, session_manager, server_infos)`의 튜플입니다.

            도구가 로드되지 않으면 `([], None, [])`을 반환합니다.

Raises:
        RuntimeError: MCP 서버 구성이 유효하지 않거나 생성/연결에 실패한 경우.

    """
    if no_mcp:
        return [], None, []

    # Auto-discovery
    try:
        config_paths = discover_mcp_configs(project_context=project_context)
    except (OSError, RuntimeError):
        logger.warning("MCP config auto-discovery failed", exc_info=True)
        config_paths = []

    # Classify discovered configs and apply trust filtering
    user_configs, project_configs = classify_discovered_configs(config_paths)

    configs: list[dict[str, Any]] = []

    # User-level configs are always trusted
    for path in user_configs:
        cfg = load_mcp_config_lenient(path)
        if cfg is not None:
            configs.append(cfg)

    # Project-level configs need trust gating for stdio servers
    for path in project_configs:
        cfg = load_mcp_config_lenient(path)
        if cfg is None:
            continue

        stdio_servers = extract_stdio_server_commands(cfg)
        if not stdio_servers:
            # No stdio servers — safe to load (remote only)
            configs.append(cfg)
            continue

        if trust_project_mcp is True:
            configs.append(cfg)
        elif trust_project_mcp is False:
            filtered = _filter_project_stdio_servers(cfg)
            if filtered.get("mcpServers"):
                configs.append(filtered)
            skipped = [
                f"{name}: {cmd} {' '.join(args)}" for name, cmd, args in stdio_servers
            ]
            logger.warning(
                "Skipped untrusted project stdio MCP servers: %s",
                "; ".join(skipped),
            )
        else:
            # None — check trust store
            from deepagents_cli.mcp_trust import (
                compute_config_fingerprint,
                is_project_mcp_trusted,
            )

            project_root = str(_resolve_project_config_base(project_context).resolve())
            fingerprint = compute_config_fingerprint(project_configs)
            if is_project_mcp_trusted(project_root, fingerprint):
                configs.append(cfg)
            else:
                filtered = _filter_project_stdio_servers(cfg)
                if filtered.get("mcpServers"):
                    configs.append(filtered)
                skipped = [
                    f"{name}: {cmd} {' '.join(args)}"
                    for name, cmd, args in stdio_servers
                ]
                logger.warning(
                    "Skipped untrusted project stdio MCP servers "
                    "(config changed or not yet approved): %s",
                    "; ".join(skipped),
                )

    # Explicit path is highest precedence — errors are fatal
    if explicit_config_path:
        config_path = (
            str(project_context.resolve_user_path(explicit_config_path))
            if project_context is not None
            else explicit_config_path
        )
        configs.append(load_mcp_config(config_path))

    if not configs:
        return [], None, []

    merged = merge_mcp_configs(configs)
    if not merged.get("mcpServers"):
        return [], None, []

    # Validate each server in the merged config
    try:
        for server_name, server_config in merged["mcpServers"].items():
            _validate_server_config(server_name, server_config)
    except (TypeError, ValueError) as e:
        msg = f"Invalid MCP server configuration: {e}"
        raise RuntimeError(msg) from e

    return await _load_tools_from_config(merged)
