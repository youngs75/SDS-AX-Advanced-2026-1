"""CLI에서 사용되는 에이전트 그래프를 빌드하고 구성합니다.

이 모듈은 모델, 미들웨어 스택, 도구, 파일 시스템 백엔드 및 선택적 샌드박스 통합을 TUI 및 비대화형 진입점 모두에서 사용되는 실행 가능한 그래프로
조합합니다.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from deepagents.backends.sandbox import SandboxBackendProtocol
    from deepagents.middleware.async_subagents import AsyncSubAgent
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentState
    from langchain.messages import ToolCall
    from langchain.tools import BaseTool
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.pregel import Pregel
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from deepagents_cli.mcp_tools import MCPServerInfo
    from deepagents_cli.output import OutputFormat

from langchain.agents.middleware.types import AgentMiddleware

from deepagents_cli import theme
from deepagents_cli.config import (
    _ShellAllowAll,
    config,
    console,
    get_default_coding_instructions,
    get_glyphs,
    settings,
)
from deepagents_cli.configurable_model import ConfigurableModelMiddleware
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.local_context import (
    LocalContextMiddleware,
    _AsyncExecutableBackend,
    _ExecutableBackend,
)
from deepagents_cli.project_utils import ProjectContext, get_server_project_context
from deepagents_cli.subagents import list_subagents
from deepagents_cli.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    render_with_unicode_markers,
    strip_dangerous_unicode,
    summarize_issues,
)

logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "agent"
"""`-a` 플래그가 제공되지 않을 때 사용되는 기본 에이전트 이름입니다."""

REQUIRE_COMPACT_TOOL_APPROVAL: bool = True
"""`True`, `compact_conversation`에는 다른 제한 도구와 마찬가지로 HITL 승인이 필요합니다."""


class ShellAllowListMiddleware(AgentMiddleware):
    """HITL 인터럽트 없이 허용 목록에 대해 셸 명령의 유효성을 검사합니다.

    에이전트가 셸 도구(`SHELL_TOOL_NAMES`의 모든 도구)를 호출하면 이 미들웨어는 **실행 전에** 구성된 허용 목록과 비교하여 명령을
    확인합니다. 거부된 명령은 오류 `ToolMessage` 개체로 반환됩니다. 그래프는 절대 일시 중지되지 않으므로 LangSmith 추적은 단일 연속
    실행으로 유지됩니다.

    추적을 조각화하는 인터럽트/재개 주기를 방지하려면 비대화형 모드에서 이 미들웨어를 사용하십시오.

    """

    def __init__(self, allow_list: list[str]) -> None:
        """명령의 유효성을 검사하기 위해 셸 허용 목록을 사용하여 초기화합니다.

Args:
            allow_list: 허용되는 명령 이름(예: `["ls", "cat", "grep"]`). `SHELL_ALLOW_ALL`이 아닌 비어
                        있지 않은 제한 목록이어야 합니다.

Raises:
            ValueError: `allow_list`이 비어 있는 경우.
            TypeError: `allow_list`이 `SHELL_ALLOW_ALL` 파수꾼인 경우.

        """
        from deepagents_cli.config import SHELL_ALLOW_ALL

        super().__init__()
        if not allow_list:
            msg = "allow_list must not be empty; disable shell access instead"
            raise ValueError(msg)
        if isinstance(allow_list, type(SHELL_ALLOW_ALL)):
            msg = (
                "SHELL_ALLOW_ALL should not be used with "
                "ShellAllowListMiddleware; use auto_approve=True instead"
            )
            raise TypeError(msg)
        self._allow_list = list(allow_list)

    def _validate_tool_call(self, request: ToolCallRequest) -> ToolMessage | None:
        """쉘 명령이 허용되지 않으면 오류 도구 메시지를 반환합니다.

Args:
            request: 도구 호출 요청이 처리되고 있습니다.

Returns:
            쉘 명령을 거부해야 할 경우 오류 `ToolMessage`, 그렇지 않으면 `None`.

        """
        from langchain_core.messages import ToolMessage as LCToolMessage

        from deepagents_cli.config import SHELL_TOOL_NAMES, is_shell_command_allowed

        tool_name = request.tool_call["name"]
        if tool_name not in SHELL_TOOL_NAMES:
            return None

        args = request.tool_call.get("args") or {}
        command = args.get("command", "")
        if is_shell_command_allowed(command, self._allow_list):
            logger.debug("Shell command allowed: %r", command)
            return None

        logger.warning("Shell command rejected by allow-list: %r", command)
        allowed_str = ", ".join(self._allow_list)
        return LCToolMessage(
            content=(
                f"Shell command rejected: `{command}` is not in the allow-list. "
                f"Allowed commands: {allowed_str}. "
                f"Please use an allowed command or try another approach."
            ),
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """허용되지 않는 쉘 명령을 거부합니다. 다른 모든 것을 통과하십시오.

Args:
            request: 도구 호출 요청이 처리되고 있습니다.
            handler: 미들웨어 체인의 다음 핸들러입니다.

Returns:
            도구 실행 결과 또는 거부된 셸 명령에 대한 오류 `ToolMessage`입니다.

        """
        if (rejection := self._validate_tool_call(request)) is not None:
            return rejection
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """허용되지 않는 쉘 명령을 거부합니다. 다른 모든 것을 통과하십시오.

Args:
            request: 도구 호출 요청이 처리되고 있습니다.
            handler: 미들웨어 체인의 다음 핸들러입니다.

Returns:
            도구 실행 결과 또는 거부된 셸 명령에 대한 오류 `ToolMessage`입니다.

        """
        if (rejection := self._validate_tool_call(request)) is not None:
            return rejection
        return await handler(request)


def load_async_subagents(config_path: Path | None = None) -> list[AsyncSubAgent]:
    """`config.toml`에서 비동기 하위 에이전트 정의를 로드합니다.

    각 하위 테이블이 원격 LangGraph 배포를 정의하는 `[async_subagents]` 섹션을 읽습니다.

    ```toml
    [async_subagents.researcher]
    description = "Research agent"
    url = "https://my-deployment.langsmith.dev"
    graph_id = "agent"
    ```

Args:
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

Returns:
        `AsyncSubAgent` 사양 목록(섹션이 없거나 잘못된 경우 비어 있음)

    """
    if config_path is None:
        config_path = Path.home() / ".deepagents" / "config.toml"

    if not config_path.exists():
        return []

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as e:
        logger.warning("Could not read async subagents from %s: %s", config_path, e)
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Could not read async subagents "
            f"from {config_path}: {e}",
        )
        return []

    section = data.get("async_subagents")
    if not isinstance(section, dict):
        return []

    required = {"description", "graph_id"}
    agents: list[AsyncSubAgent] = []
    for name, spec in section.items():
        if not isinstance(spec, dict):
            logger.warning("Skipping async subagent '%s': expected a table", name)
            continue
        missing = required - spec.keys()
        if missing:
            logger.warning(
                "Skipping async subagent '%s': missing fields %s", name, missing
            )
            continue
        agent: AsyncSubAgent = {
            "name": name,
            "description": spec["description"],
            "graph_id": spec["graph_id"],
        }
        if "url" in spec and isinstance(spec["url"], str):
            agent["url"] = spec["url"]
        if "headers" in spec and isinstance(spec["headers"], dict):
            agent["headers"] = spec["headers"]
        agents.append(agent)

    return agents


def list_agents(*, output_format: OutputFormat = "text") -> None:
    """사용 가능한 모든 에이전트를 나열합니다.

Args:
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    """
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json("list", [])
            return
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ "
            "when you first use them.[/dim]",
            style=theme.MUTED,
        )
        return

    if output_format == "json":
        from deepagents_cli.output import write_json

        agents = []
        for agent_path in sorted(agents_dir.iterdir()):
            if agent_path.is_dir():
                agent_name = agent_path.name
                agents.append(
                    {
                        "name": agent_name,
                        "path": str(agent_path),
                        "has_agents_md": (agent_path / "AGENTS.md").exists(),
                        "is_default": agent_name == DEFAULT_AGENT_NAME,
                    }
                )
        write_json("list", agents)
        return

    from rich.markup import escape as escape_markup

    console.print("\n[bold]Available Agents:[/bold]\n", style=theme.PRIMARY)

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = escape_markup(agent_path.name)
            agent_md = agent_path / "AGENTS.md"
            is_default = agent_path.name == DEFAULT_AGENT_NAME
            default_label = " [dim](default)[/dim]" if is_default else ""

            bullet = get_glyphs().bullet
            if agent_md.exists():
                console.print(
                    f"  {bullet} [bold]{agent_name}[/bold]{default_label}",
                    style=theme.PRIMARY,
                )
                console.print(
                    f"    {escape_markup(str(agent_path))}",
                    style=theme.MUTED,
                )
            else:
                console.print(
                    f"  {bullet} [bold]{agent_name}[/bold]{default_label}"
                    " [dim](incomplete)[/dim]",
                    style=theme.WARNING,
                )
                console.print(
                    f"    {escape_markup(str(agent_path))}",
                    style=theme.MUTED,
                )

    console.print()


def reset_agent(
    agent_name: str,
    source_agent: str | None = None,
    *,
    dry_run: bool = False,
    output_format: OutputFormat = "text",
) -> None:
    """에이전트를 기본값으로 재설정하거나 다른 에이전트에서 복사합니다.

Args:
        agent_name: 재설정할 에이전트의 이름입니다.
        source_agent: 기본값 대신 이 에이전트에서 AGENTS.md를 복사합니다.
        dry_run: `True`인 경우 변경하지 않고 어떤 일이 발생하는지 인쇄하세요.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

Raises:
        SystemExit: 소스 에이전트를 찾을 수 없는 경우.

    """
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "AGENTS.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no AGENTS.md\n"
                "  Available agents: deepagents agents list"
            )
            raise SystemExit(1)

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if dry_run:
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json(
                "reset",
                {
                    "agent": agent_name,
                    "reset_to": source_agent or "default",
                    "path": str(agent_dir),
                    "dry_run": True,
                },
            )
            return
        exists = "remove and recreate" if agent_dir.exists() else "create"
        console.print(f"Would {exists} {agent_dir} with {action_desc} prompt.")
        console.print("No changes made.", style=theme.MUTED)
        return

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        if output_format != "json":
            console.print(
                f"Removed existing agent directory: {agent_dir}", style=theme.WARNING
            )

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "AGENTS.md"
    agent_md.write_text(source_content)

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json(
            "reset",
            {
                "agent": agent_name,
                "reset_to": source_agent or "default",
                "path": str(agent_dir),
            },
        )
        return

    console.print(
        f"{get_glyphs().checkmark} Agent '{agent_name}' reset to {action_desc}",
        style=theme.PRIMARY,
    )
    console.print(f"Location: {agent_dir}\n", style=theme.MUTED)


MODEL_IDENTITY_RE = re.compile(r"### Model Identity\n\n.*?(?=###|\Z)", re.DOTALL)
"""시스템 프롬프트의 `### Model Identity` 섹션을 다음 제목 또는 문자열 끝까지 일치시킵니다."""


def build_model_identity_section(
    name: str | None,
    provider: str | None = None,
    context_limit: int | None = None,
    unsupported_modalities: frozenset[str] = frozenset(),
) -> str:
    """시스템 프롬프트에 대한 `### Model Identity` 섹션을 빌드합니다.

Args:
        name: 모델 식별자(예: `claude-opus-4-6`).
        provider: 공급자 식별자(예: `anthropic`).
        context_limit: 모델 프로필의 최대 입력 토큰입니다.
        unsupported_modalities: 모델 프로필에서 지원되는 것으로 표시되지 않은 입력 양식(예: `{"audio", "video"}`)

Returns:
        제목과 후행 줄 바꿈을 포함하는 섹션 텍스트 또는 `name`이 거짓인 경우 빈 문자열입니다.

    """
    if not name:
        return ""
    section = f"### Model Identity\n\nYou are running as model `{name}`"
    if provider:
        section += f" (provider: {provider})"
    section += ".\n"
    if context_limit:
        section += f"Your context window is {context_limit:,} tokens.\n"
    if unsupported_modalities:
        items = sorted(unsupported_modalities)
        if len(items) == 1:
            joined = items[0]
        elif len(items) == 2:  # noqa: PLR2004
            joined = f"{items[0]} and {items[1]}"
        else:
            joined = ", ".join(items[:-1]) + f", and {items[-1]}"
        section += (
            f"{joined.capitalize()} input may not be available for this model. "
            "Do not attempt to read or process these content types.\n"
        )
    section += "\n"
    return section


def get_system_prompt(
    assistant_id: str,
    sandbox_type: str | None = None,
    *,
    interactive: bool = True,
    cwd: str | Path | None = None,
) -> str:
    """에이전트에 대한 기본 시스템 프롬프트를 가져옵니다.

    `system_prompt.md`에서 기본 시스템 프롬프트 템플릿을 로드하고 동적 섹션(모델 ID, 작업 디렉터리, 기술 경로, 실행 모드)을
    삽입합니다.

Args:
        assistant_id: 경로 참조에 대한 에이전트 식별자
        sandbox_type: 샌드박스 공급자 유형(`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`,
                      `'runloop'`).

            `None`인 경우 에이전트가 로컬 모드에서 작동 중입니다.
        interactive: `False`인 경우 프롬프트는 헤드리스 비대화형 실행(루프에 사람이 없음)에 맞춰 조정됩니다.
        cwd: 프롬프트에 표시된 작업 디렉터리를 재정의합니다.

Returns:
        시스템 프롬프트 문자열

Example:
        ```txt
        You are running as model {MODEL} (provider: {PROVIDER}).

        Your context window is {CONTEXT_WINDOW} tokens.

        ... {CONDITIONAL SECTIONS} ...
        ```

    """
    template = (Path(__file__).parent / "system_prompt.md").read_text()

    skills_path = f"~/.deepagents/{assistant_id}/skills"

    if interactive:
        mode_description = "an interactive CLI on the user's computer"
        interactive_preamble = (
            "The user sends you messages and you respond with text and tool "
            "calls. Your tools run on the user's machine. The user can see "
            "your responses and tool outputs in real time, so keep them "
            "informed — but don't over-explain."
        )
        ambiguity_guidance = (
            "- If the request is ambiguous, ask questions before acting.\n"
            "- If asked how to approach something, explain first, then act."
        )
    else:
        mode_description = (
            "non-interactive (headless) mode — there is no human operator "
            "monitoring your output in real time"
        )
        interactive_preamble = (
            "You received a single task and must complete it fully and "
            "autonomously. There is no human available to answer follow-up "
            "questions, so do NOT ask for clarification — make reasonable "
            "assumptions and proceed."
        )
        ambiguity_guidance = (
            "- Do NOT ask clarifying questions — there is no human to answer "
            "them. Make reasonable assumptions and proceed.\n"
            "- If you encounter ambiguity, choose the most reasonable "
            "interpretation and note your assumption briefly.\n"
            "- Always use non-interactive command variants — no human is "
            "available to respond to prompts. Examples: `npm init -y` not "
            "`npm init`, `apt-get install -y` not `apt-get install`, "
            "`yes |` or `--no-input`/`--non-interactive` flags where "
            "available. Never run commands that block waiting for stdin."
        )

    model_identity_section = build_model_identity_section(
        settings.model_name,
        provider=settings.model_provider,
        context_limit=settings.model_context_limit,
        unsupported_modalities=settings.model_unsupported_modalities,
    )

    # 작업 디렉터리 섹션 빌드(로컬 및 샌드박스)
    if sandbox_type:
        working_dir = get_default_working_dir(sandbox_type)
        working_dir_section = (
            f"### Current Working Directory\n\n"
            f"You are operating in a **remote Linux sandbox** at `{working_dir}`.\n\n"
            f"All code execution and file operations happen in this sandbox "
            f"environment.\n\n"
            f"**Important:**\n"
            f"- The CLI is running locally on the user's machine, but you execute "
            f"code remotely\n"
            f"- Use `{working_dir}` as your working directory for all operations\n"
            f"- **You do NOT have access to the user's local filesystem.** Paths "
            f"like `/Users/...`, `/home/<local-user>/...`, `C:\\...`, etc. do not "
            f"exist in this sandbox. Never reference or attempt to read/write local "
            f"paths — all files must be within the sandbox at `{working_dir}`\n"
            f"- When delegating to subagents, ensure they also use sandbox paths "
            f"(`{working_dir}/...`), not local paths\n\n"
        )
    else:
        if cwd is not None:
            resolved_cwd = Path(cwd)
        else:
            try:
                resolved_cwd = Path.cwd()
            except OSError:
                logger.warning(
                    "Could not determine working directory for system prompt",
                    exc_info=True,
                )
                resolved_cwd = Path()
        cwd = resolved_cwd
        working_dir_section = (
            f"### Current Working Directory\n\n"
            f"The filesystem backend is currently operating in: `{cwd}`\n\n"
            f"### File System and Paths\n\n"
            f"**IMPORTANT - Path Handling:**\n"
            f"- All file paths must be absolute paths (e.g., `{cwd}/file.txt`)\n"
            f"- Use the working directory to construct absolute paths\n"
            f"- Example: To create a file in your working directory, "
            f"use `{cwd}/research_project/file.md`\n"
            f"- Never use relative paths - always construct full absolute paths\n\n"
        )

    result = (
        template.replace("{mode_description}", mode_description)
        .replace("{interactive_preamble}", interactive_preamble)
        .replace("{ambiguity_guidance}", ambiguity_guidance)
        .replace("{model_identity_section}", model_identity_section)
        .replace("{working_dir_section}", working_dir_section)
        .replace("{skills_path}", skills_path)
    )

    # 대체되지 않은 자리 표시자 감지(템플릿 오타에 대한 심층 방어)
    unreplaced = re.findall(r"\{[a-z_]+\}", result)
    if unreplaced:
        logger.warning("System prompt contains unreplaced placeholders: %s", unreplaced)

    return result


def _format_write_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """승인 메시지를 위한 write_file 도구 호출 형식을 지정합니다.

Returns:
        write_file 도구 호출에 대한 형식화된 설명 문자열입니다.

    """
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")

    action = "Overwrite" if Path(file_path).exists() else "Create"

    return f"Action: {action} file"


def _format_edit_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """승인 메시지에 대한 edit_file 도구 호출 형식을 지정합니다.

Returns:
        edit_file 도구 호출에 대한 형식화된 설명 문자열입니다.

    """
    args = tool_call["args"]
    replace_all = bool(args.get("replace_all", False))

    scope = "all occurrences" if replace_all else "single occurrence"
    return f"Action: Replace text ({scope})"


def _format_web_search_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """승인 메시지에 대한 web_search 도구 호출 형식을 지정합니다.

Returns:
        web_search 도구 호출에 대한 형식화된 설명 문자열입니다.

    """
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return (
        f"Query: {query}\nMax results: {max_results}\n\n"
        f"{get_glyphs().warning}  This will use Tavily API credits"
    )


def _format_fetch_url_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """승인 메시지를 위한 fetch_url 도구 호출 형식을 지정합니다.

Returns:
        fetch_url 도구 호출에 대한 형식화된 설명 문자열입니다.

    """
    args = tool_call["args"]
    url = str(args.get("url", "unknown"))
    display_url = strip_dangerous_unicode(url)
    timeout = args.get("timeout", 30)
    safety = check_url_safety(url)

    warning_lines: list[str] = []
    if not safety.safe:
        detail = format_warning_detail(safety.warnings)
        warning_lines.append(f"{get_glyphs().warning}  URL warning: {detail}")
    if safety.decoded_domain:
        warning_lines.append(
            f"{get_glyphs().warning}  Decoded domain: {safety.decoded_domain}"
        )

    warning_block = "\n".join(warning_lines)
    if warning_block:
        warning_block = f"\n{warning_block}"

    return (
        f"URL: {display_url}\nTimeout: {timeout}s\n\n"
        f"{get_glyphs().warning}  Will fetch and convert web content to markdown"
        f"{warning_block}"
    )


def _format_task_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """승인 프롬프트에 대한 작업(하위 에이전트) 도구 호출 형식을 지정합니다.

    작업 도구 서명은 다음과 같습니다. task(description: str, subagent_type: str) 설명에는 하위 에이전트로 전송될 모든
    지침이 포함되어 있습니다.

Returns:
        작업 도구 호출에 대한 형식화된 설명 문자열입니다.

    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # 표시하기에 너무 길면 설명을 자릅니다.
    description_preview = description
    if len(description) > 500:  # noqa: PLR2004  # Subagent description length threshold
        description_preview = description[:500] + "..."

    glyphs = get_glyphs()
    separator = glyphs.box_horizontal * 40
    warning_msg = "Subagent will have access to file operations and shell commands"
    return (
        f"Subagent Type: {subagent_type}\n\n"
        f"{glyphs.warning} {warning_msg} {glyphs.warning}\n\n"
        f"Task Instructions:\n"
        f"{separator}\n"
        f"{description_preview}"
    )


def _format_execute_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """승인 프롬프트를 위한 실행 도구 호출 형식을 지정합니다.

Returns:
        실행 도구 호출에 대한 형식화된 설명 문자열입니다.

    """
    args = tool_call["args"]
    command_raw = str(args.get("command", "N/A"))
    command = strip_dangerous_unicode(command_raw)
    project_context = get_server_project_context()
    effective_cwd = (
        str(project_context.user_cwd)
        if project_context is not None
        else str(Path.cwd())
    )
    lines = [f"Execute Command: {command}", f"Working Directory: {effective_cwd}"]

    issues = detect_dangerous_unicode(command_raw)
    if issues:
        summary = summarize_issues(issues)
        lines.append(f"{get_glyphs().warning}  Hidden Unicode detected: {summary}")
        raw_marked = render_with_unicode_markers(command_raw)
        if len(raw_marked) > 220:  # noqa: PLR2004  # UI display truncation threshold
            raw_marked = raw_marked[:220] + "..."
        lines.append(f"Raw: {raw_marked}")

    return "\n".join(lines)


def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    """모든 게이트 도구에 대해 인간 개입(Human-In-The-Loop) 인터럽트 설정을 구성합니다.

    부작용이 있거나 외부 리소스(셸 실행, 파일 쓰기/편집, 웹 검색, URL 가져오기, 작업 위임)에 액세스할 수 있는 모든 도구는 자동 승인이
    활성화되지 않는 한 승인 프롬프트 뒤에 표시됩니다.

Returns:
        인터럽트 구성에 대한 사전 매핑 도구 이름입니다.

    """
    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,  # type: ignore[typeddict-item]  # Callable description narrower than TypedDict expects
    }

    async_subagent_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": "Launch, update, or cancel a remote async subagent.",
    }

    interrupt_map: dict[str, InterruptOnConfig] = {
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
        "launch_async_subagent": async_subagent_interrupt_config,
        "update_async_subagent": async_subagent_interrupt_config,
        "cancel_async_subagent": async_subagent_interrupt_config,
    }

    if REQUIRE_COMPACT_TOOL_APPROVAL:
        interrupt_map["compact_conversation"] = {
            "allowed_decisions": ["approve", "reject"],
            "description": (
                "Offloads older messages to backend storage and "
                "replaces them with a summary, freeing context "
                "window space. Recent messages are kept as-is. "
                "Full history remains available for retrieval."
            ),
        }

    return interrupt_map


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    interactive: bool = True,
    auto_approve: bool = False,
    interrupt_shell_only: bool = False,
    shell_allow_list: list[str] | None = None,
    enable_ask_user: bool = True,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
    mcp_server_info: list[MCPServerInfo] | None = None,
    cwd: str | Path | None = None,
    project_context: ProjectContext | None = None,
    async_subagents: list[AsyncSubAgent] | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """유연한 옵션으로 CLI 구성 에이전트를 생성하세요.

    이는 내부 및 외부 코드(예: 벤치마킹 프레임워크)에서 모두 사용할 수 있는 deepagents CLI 에이전트를 생성하기 위한 주요 진입점입니다.

Args:
        model: 사용할 LLM 모델(예: `'anthropic:claude-sonnet-4-6'`)
        assistant_id: 메모리/상태 저장을 위한 에이전트 식별자
        tools: 에이전트에게 제공할 추가 도구
        sandbox: 원격 실행을 위한 선택적 샌드박스 백엔드(예: `ModalSandbox`).

            `None`인 경우 로컬 파일 시스템 + 셸을 사용합니다.
        sandbox_type: 샌드박스 공급자 유형(`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`,
                      `'runloop'`). 시스템 프롬프트 생성에 사용됩니다.
        system_prompt: 기본 시스템 프롬프트를 재정의합니다.

            `None`인 경우 `sandbox_type`, `assistant_id` 및 `interactive`을 기반으로 생성합니다.
        interactive: `False`인 경우 자동 생성된 시스템 프롬프트는 헤드리스 비대화형 실행에 맞게 조정됩니다.
                     `system_prompt`이 명시적으로 제공되면 무시됩니다.
        auto_approve: `True`인 경우 인간 개입(Human-In-The-Loop) 인터럽트를 트리거하는 도구가 없습니다. 모든 호출(셸
                      실행, 파일 쓰기/편집, 웹 검색, URL 가져오기)이 자동으로 실행됩니다.

            `False`인 경우 승인 메뉴를 통한 사용자 확인을 위해 도구가 일시 중지됩니다. 게이트 도구의 전체 목록은
            `_add_interrupt_on`을 참조하세요.
        interrupt_shell_only: `True`인 경우 모든 HITL 인터럽트가 비활성화됩니다. 대신 쉘 명령은 구성된 허용 목록에 대해
                              `ShellAllowListMiddleware`에 의해 인라인으로 검증됩니다.

            추적이 여러 LangSmith 실행으로 분할되는 것을 방지하기 위해 제한적인 셸 허용 목록과 함께 비대화형 모드에서 사용됩니다.

            `auto_approve`이 `True`인 경우(인터럽트는 이미 비활성화되어 있음) 또는 `shell_allow_list`이
            `SHELL_ALLOW_ALL`인 경우에는 효과가 없습니다.
        shell_allow_list: CLI 프로세스에서 전달된 명시적 제한적 셸 허용 목록입니다. 제공되면(그리고
                          `interrupt_shell_only`은 `True`임)
                          `settings.shell_allow_list`(서버 하위 프로세스 환경에서 설정되지 않을 수 있음)을 읽는
                          대신 직접 사용됩니다.
        enable_ask_user: 에이전트이 명확한 질문을 할 수 있도록 `AskUserMiddleware`을(를) 활성화하세요.

            비대화형 모드에서는 비활성화됩니다.
        enable_memory: 영구 메모리에 대해 `MemoryMiddleware` 활성화
        enable_skills: 사용자 정의 에이전트 기술에 대해 `SkillsMiddleware` 활성화
        enable_shell: `LocalShellBackend`을(를) 통해 쉘 실행을 활성화합니다(로컬 모드에서만). 활성화되면 `execute`
                      도구를 사용할 수 있습니다.
        checkpointer: 세션 지속성을 위한 선택적 체크포인터입니다. `None`인 경우 그래프는 체크포인터 없이 컴파일됩니다.
        mcp_server_info: 시스템 프롬프트에 표시되는 MCP 서버 메타데이터입니다.
        cwd: 에이전트의 파일 시스템 백엔드 및 시스템 프롬프트에 대한 작업 디렉터리를 재정의합니다.
        project_context: 프로젝트 `AGENTS.md` 파일, 기술, 하위 에이전트 및 MCP 신뢰와 같은 프로젝트에 민감한 동작에 대한
                         명시적 프로젝트 경로 컨텍스트입니다.
        async_subagents: 비동기 하위 에이전트 도구로 노출되는 원격 LangGraph 배포입니다.

            `config.toml`의 `[async_subagents]`에서 로드되거나 직접 전달됩니다.

Returns:
        `(agent_graph, backend)`의 2튜플

            - `agent_graph`: 실행 준비가 완료된 LangGraph Pregel 인스턴스 구성
            - `composite_backend`: 파일 작업용 `CompositeBackend`

    """
    tools = tools or []
    effective_cwd = (
        Path(cwd)
        if cwd is not None
        else (project_context.user_cwd if project_context is not None else None)
    )

    # 영구 메모리용 에이전트 디렉터리 설정(활성화된 경우)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "AGENTS.md"
        if not agent_md.exists():
            # 사용자 정의를 위한 빈 파일 생성
            # 기본 명령어는 get_system_prompt()에서 새로 로드됩니다.
            agent_md.touch()

    # 기술 디렉토리(활성화된 경우)
    skills_dir = None
    user_agent_skills_dir = None
    project_skills_dir = None
    project_agent_skills_dir = None
    if enable_skills:
        skills_dir = settings.ensure_user_skills_dir(assistant_id)
        user_agent_skills_dir = settings.get_user_agent_skills_dir()
        project_skills_dir = (
            project_context.project_skills_dir()
            if project_context is not None
            else settings.get_project_skills_dir()
        )
        project_agent_skills_dir = (
            project_context.project_agent_skills_dir()
            if project_context is not None
            else settings.get_project_agent_skills_dir()
        )

    # 파일 시스템에서 사용자 정의 하위 에이전트 로드
    custom_subagents: list[SubAgent | CompiledSubAgent] = []
    restrictive_shell_allow_list: list[str] | None = None
    if interrupt_shell_only and not auto_approve:
        # 명시적으로 전달된 허용 목록을 선호합니다(CLI 프로세스에 의해 설정됨).
        # ServerConfig를 통해 전달됨)  다음에 대해서만 설정으로 돌아갑니다.
        # 통과하지 않는 직접 호출자(예: 벤치마킹 프레임워크)
        # 서버 하위 프로세스 경로.
        if shell_allow_list:
            restrictive_shell_allow_list = list(shell_allow_list)
        elif settings.shell_allow_list and not isinstance(
            settings.shell_allow_list, _ShellAllowAll
        ):
            restrictive_shell_allow_list = list(settings.shell_allow_list)
        else:
            logger.warning(
                "interrupt_shell_only=True but no restrictive shell allow-list "
                "available; falling back to standard HITL interrupts"
            )

    user_agents_dir = settings.get_user_agents_dir(assistant_id)
    project_agents_dir = (
        project_context.project_agents_dir()
        if project_context is not None
        else settings.get_project_agents_dir()
    )

    for subagent_meta in list_subagents(
        user_agents_dir=user_agents_dir,
        project_agents_dir=project_agents_dir,
    ):
        subagent: SubAgent = {
            "name": subagent_meta["name"],
            "description": subagent_meta["description"],
            "system_prompt": subagent_meta["system_prompt"],
        }
        if subagent_meta["model"]:
            subagent["model"] = subagent_meta["model"]
        if restrictive_shell_allow_list is not None:
            subagent["middleware"] = [
                ShellAllowListMiddleware(restrictive_shell_allow_list)
            ]
        custom_subagents.append(subagent)

    if restrictive_shell_allow_list is not None:
        from deepagents.middleware.subagents import (
            GENERAL_PURPOSE_SUBAGENT,
            SubAgent as RuntimeSubAgent,
        )

        if not any(
            subagent["name"] == GENERAL_PURPOSE_SUBAGENT["name"]
            for subagent in custom_subagents
        ):
            general_purpose_subagent: RuntimeSubAgent = {
                "name": GENERAL_PURPOSE_SUBAGENT["name"],
                "description": GENERAL_PURPOSE_SUBAGENT["description"],
                "system_prompt": GENERAL_PURPOSE_SUBAGENT["system_prompt"],
                "middleware": [ShellAllowListMiddleware(restrictive_shell_allow_list)],
            }
            custom_subagents.append(general_purpose_subagent)

    # 활성화된 기능을 기반으로 미들웨어 스택 구축
    agent_middleware = []
    agent_middleware.append(ConfigurableModelMiddleware())

    # 토큰 상태: 그래프 상태에 _context_tokens를 추가합니다(체크포인트됨, 아님
    # 모델에게 전달됨)  미들웨어보다 먼저 등록되어야 합니다.
    # 채널을 읽어보세요.
    from deepagents_cli.token_state import TokenStateMiddleware

    agent_middleware.append(TokenStateMiddleware())

    # Ask_user 미들웨어 추가(도구를 사용할 수 있도록 일찍 출시되어야 함)
    if enable_ask_user:
        from deepagents_cli.ask_user import AskUserMiddleware

        agent_middleware.append(AskUserMiddleware())

    # 메모리 미들웨어 추가
    if enable_memory:
        memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
        project_agent_md_paths = (
            project_context.project_agent_md_paths()
            if project_context is not None
            else settings.get_project_agent_md_path()
        )
        memory_sources.extend(str(p) for p in project_agent_md_paths)

        agent_middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(),
                sources=memory_sources,
            )
        )

    # 기술 미들웨어 추가
    if enable_skills:
        # 가장 낮은 우선순위부터 가장 높은 우선순위까지:
        # 내장 -> 사용자 .deepagents -> 사용자 .agents
        # -> 프로젝트 .deepagents -> 프로젝트 .agents
        # -> 사용자 .claude(실험적) -> 프로젝트 .claude(실험적)
        sources = [str(settings.get_built_in_skills_dir())]
        sources.extend([str(skills_dir), str(user_agent_skills_dir)])
        if project_skills_dir:
            sources.append(str(project_skills_dir))
        if project_agent_skills_dir:
            sources.append(str(project_agent_skills_dir))

        # 실험적: Claude Code 스킬 디렉토리
        user_claude_skills_dir = settings.get_user_claude_skills_dir()
        if user_claude_skills_dir.exists():
            sources.append(str(user_claude_skills_dir))
        project_claude_skills_dir = settings.get_project_claude_skills_dir()
        if project_claude_skills_dir:
            sources.append(str(project_claude_skills_dir))

        agent_middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(),
                sources=sources,
            )
        )

    # 조건부 설정: 로컬 및 원격 샌드박스
    if sandbox is None:
        # ========== 로컬 모드 ==========
        root_dir = effective_cwd if effective_cwd is not None else Path.cwd()
        if enable_shell:
            # 셸 명령을 위한 환경 만들기
            # 사용자의 원래 LANGSMITH_PROJECT를 복원하여 코드가 별도로 추적되도록 합니다.
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            # 파일 시스템 + 셸 실행에는 LocalShellBackend를 사용하세요.
            # SDK의 FilesystemMiddleware는 명령별 시간 초과를 노출합니다.
            # 실행 도구에서 기본적으로.
            backend = LocalShellBackend(
                root_dir=root_dir,
                inherit_env=True,
                env=shell_env,
            )
        else:
            # 셸 액세스 없음 - 일반 FilesystemBackend 사용
            backend = FilesystemBackend(root_dir=root_dir)
    else:
        # ========== 원격 샌드박스 모드 ==========
        backend = sandbox  # 원격 샌드박스(ModalSandbox 등)
        # 참고: 샌드박스 모드에서는 쉘 미들웨어가 사용되지 않습니다.
        # 파일 작업 및 실행 도구는 샌드박스 백엔드에서 제공됩니다.

    # 로컬 컨텍스트 미들웨어(git 정보, 디렉토리 트리 등)
    if isinstance(backend, (_ExecutableBackend, _AsyncExecutableBackend)):
        agent_middleware.append(
            LocalContextMiddleware(backend=backend, mcp_server_info=mcp_server_info)
        )

    # Interrupt_shell_only가 활성화된 경우 셸 허용 목록 미들웨어를 추가합니다.
    shell_middleware_added = False
    if restrictive_shell_allow_list is not None:
        agent_middleware.append(ShellAllowListMiddleware(restrictive_shell_allow_list))
        shell_middleware_added = True

    # 사용자 정의 시스템 프롬프트 가져오기 또는 사용
    if system_prompt is None:
        system_prompt = get_system_prompt(
            assistant_id=assistant_id,
            sandbox_type=sandbox_type,
            interactive=interactive,
            cwd=effective_cwd,
        )

    # auto_approve / shell_middleware_add를 기반으로 Interrupt_on을 구성합니다.
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if auto_approve or shell_middleware_added:  # noqa: SIM108  # if-else clearer than ternary for dual-path config
        # HITL 중단 없음 - 도구가 자동으로 실행됩니다.
        # shell_middleware_add가 True이면 쉘 유효성 검사는 다음에 의해 처리됩니다.
        # 허용되지 않는 것을 거부하는 ShellAllowListMiddleware(위에 추가됨)
        # 명령은 오류 ToolMessage로 인라인되어 전체 실행을 유지합니다.
        # 단일 LangSmith 추적.
        interrupt_on = {}
    else:
        # 파괴적인 작업을 위한 전체 HITL
        interrupt_on = _add_interrupt_on()  # type: ignore[assignment]  # InterruptOnConfig is compatible at runtime

    # 라우팅을 사용하여 복합 백엔드 설정
    # 로컬 FilesystemBackend의 경우 오염을 방지하기 위해 대규모 도구 결과를 /tmp로 라우팅합니다.
    # 작업 디렉토리. 샌드박스 백엔드의 경우 특별한 라우팅이 필요하지 않습니다.
    if sandbox is None:
        # 로컬 모드: 대규모 결과를 고유한 임시 디렉터리로 라우팅
        large_results_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_large_results_"),
            virtual_mode=True,
        )
        conversation_history_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="deepagents_conversation_history_"),
            virtual_mode=True,
        )
        composite_backend = CompositeBackend(
            default=backend,
            routes={
                "/large_tool_results/": large_results_backend,
                "/conversation_history/": conversation_history_backend,
            },
        )
    else:
        # 샌드박스 모드: 특별한 라우팅이 필요하지 않습니다.
        composite_backend = CompositeBackend(
            default=backend,
            routes={},
        )

    from deepagents.middleware.summarization import create_summarization_tool_middleware

    agent_middleware.append(
        create_summarization_tool_middleware(model, composite_backend)
    )

    # 에이전트 만들기
    all_subagents: list[SubAgent | CompiledSubAgent | AsyncSubAgent] = [
        *custom_subagents,
        *(async_subagents or []),
    ]
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
        subagents=all_subagents or None,
    ).with_config(config)
    return agent, composite_backend
