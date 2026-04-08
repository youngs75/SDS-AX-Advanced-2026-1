"""CLI-서버 하위 프로세스 통신 채널에 대해 입력된 구성입니다.

CLI는 `langgraph dev` 하위 프로세스를 생성하고 `DEEPAGENTS_CLI_SERVER_` 접두사가 붙은 환경 변수를 통해 구성을 전달합니다.
이 모듈은 변수 세트, 직렬화 형식 및 기본값이 한 곳에서 정의되도록 양측이 공유하는 단일 `ServerConfig` 데이터 클래스를 제공합니다. CLI는
`to_env()`을 사용하여 구성을 작성하고 서버 그래프는 `from_env()`을 사용하여 이를 다시 읽습니다.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_cli._env_vars import SERVER_ENV_PREFIX

if TYPE_CHECKING:
    from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)

_DEFAULT_ASSISTANT_ID = "agent"


def _read_env_bool(suffix: str, *, default: bool = False) -> bool:
    """환경에서 `DEEPAGENTS_CLI_SERVER_*` 부울을 읽습니다.

    부울 환경 변수는 `'true'` / `'false'` 규칙을 사용합니다(대소문자를 구분하지 않음). 누락된 변수는 *기본값*으로 돌아갑니다.

Args:
        suffix: `DEEPAGENTS_CLI_SERVER_` 접두사 뒤의 변수 이름 접미사.
        default: 변수가 없을 때의 값입니다.

Returns:
        구문 분석된 부울입니다.

    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return default
    return raw.lower() == "true"


def _read_env_json(suffix: str) -> Any:  # noqa: ANN401
    """JSON으로 인코딩된 `DEEPAGENTS_CLI_SERVER_*` 변수를 읽습니다.

Args:
        suffix: `DEEPAGENTS_CLI_SERVER_` 접두사 뒤의 변수 이름 접미사.

Returns:
        구문 분석된 JSON 값 또는 변수가 없는 경우 `None`입니다.

Raises:
        ValueError: 변수가 있지만 유효한 JSON이 아닌 경우.

    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = (
            f"Failed to parse {SERVER_ENV_PREFIX}{suffix} as JSON: {exc}. "
            f"Value was: {raw[:200]!r}"
        )
        raise ValueError(msg) from exc


def _read_env_str(suffix: str) -> str | None:
    """선택적 `DEEPAGENTS_CLI_SERVER_*` 문자열 변수를 읽습니다.

Args:
        suffix: `DEEPAGENTS_CLI_SERVER_` 접두사 뒤의 변수 이름 접미사.

Returns:
        문자열 값 또는 없는 경우 `None`입니다.

    """
    return os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")


def _read_env_optional_bool(suffix: str) -> bool | None:
    """세 가지 상태 `DEEPAGENTS_CLI_SERVER_*` 부울(`True` / `False` / `None`)을 읽습니다.

    `None`이 고유한 의미를 전달하는 설정에 사용됩니다(예: "지정되지 않음, 기본 논리 사용").

Args:
        suffix: `DEEPAGENTS_CLI_SERVER_` 접두사 뒤의 변수 이름 접미사.

Returns:
        변수가 없는 경우 `True`, `False` 또는 `None`입니다.

    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    return raw.lower() == "true"


@dataclass(frozen=True)
class ServerConfig:
    """CLI에서 서버 하위 프로세스로 전달된 전체 구성 페이로드입니다.

    서버 그래프(별도의 Python 인터프리터에서 실행됨)가 메모리를 공유하지 않고도 CLI의 의도를 재구성할 수 있도록
    `DEEPAGENTS_CLI_SERVER_*` 환경 변수에서 직렬화됩니다.

    """

    model: str | None = None
    model_params: dict[str, Any] | None = None
    assistant_id: str = _DEFAULT_ASSISTANT_ID
    system_prompt: str | None = None
    auto_approve: bool = False
    interrupt_shell_only: bool = False
    shell_allow_list: list[str] | None = None
    interactive: bool = True
    enable_shell: bool = True
    enable_ask_user: bool = False
    enable_memory: bool = True
    enable_skills: bool = True
    sandbox_type: str | None = None
    sandbox_id: str | None = None
    sandbox_setup: str | None = None
    cwd: str | None = None
    project_root: str | None = None
    mcp_config_path: str | None = None
    no_mcp: bool = False
    trust_project_mcp: bool | None = None

    def __post_init__(self) -> None:
        """필드를 정규화하고 불변성을 검증합니다.

Raises:
            ValueError: `shell_allow_list`이 빈 목록인 경우.

        """
        if self.sandbox_type == "none":
            object.__setattr__(self, "sandbox_type", None)
        if self.shell_allow_list is not None and len(self.shell_allow_list) == 0:
            msg = "shell_allow_list must be None or non-empty"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------

    def to_env(self) -> dict[str, str | None]:
        """이 구성을 `DEEPAGENTS_CLI_SERVER_*` env-var 매핑으로 직렬화합니다.

        `None` 값은 호출자가 `os.environ`의 각 변수를 반복하고 설정하거나 지울 수 있도록 환경에서 변수를 *지워야* 함을 나타냅니다(빈
        문자열로 설정하지 않음).

Returns:
            env-var 접미사(접두사 제외)를 해당 항목에 매핑하는 Dict
                문자열 값 또는 `None`.

        """
        return {
            "MODEL": self.model,
            "MODEL_PARAMS": (
                json.dumps(self.model_params) if self.model_params is not None else None
            ),
            "ASSISTANT_ID": self.assistant_id,
            "SYSTEM_PROMPT": self.system_prompt,
            "AUTO_APPROVE": str(self.auto_approve).lower(),
            "INTERRUPT_SHELL_ONLY": str(self.interrupt_shell_only).lower(),
            "SHELL_ALLOW_LIST": (
                ",".join(self.shell_allow_list)
                if self.shell_allow_list is not None
                else None
            ),
            "INTERACTIVE": str(self.interactive).lower(),
            "ENABLE_SHELL": str(self.enable_shell).lower(),
            "ENABLE_ASK_USER": str(self.enable_ask_user).lower(),
            "ENABLE_MEMORY": str(self.enable_memory).lower(),
            "ENABLE_SKILLS": str(self.enable_skills).lower(),
            "SANDBOX_TYPE": self.sandbox_type,
            "SANDBOX_ID": self.sandbox_id,
            "SANDBOX_SETUP": self.sandbox_setup,
            "CWD": self.cwd,
            "PROJECT_ROOT": self.project_root,
            "MCP_CONFIG_PATH": self.mcp_config_path,
            "NO_MCP": str(self.no_mcp).lower(),
            "TRUST_PROJECT_MCP": (
                str(self.trust_project_mcp).lower()
                if self.trust_project_mcp is not None
                else None
            ),
        }

    @classmethod
    def from_env(cls) -> ServerConfig:
        """`DEEPAGENTS_CLI_SERVER_*` 환경 변수에서 `ServerConfig`을 재구성합니다.

        이는 `to_env()`의 반대이며 CLI 구성을 복구하기 위해 서버 하위 프로세스 내부에서 호출됩니다.

Returns:
            환경에서 채워진 `ServerConfig`.

        """
        return cls(
            model=_read_env_str("MODEL"),
            model_params=_read_env_json("MODEL_PARAMS"),
            assistant_id=_read_env_str("ASSISTANT_ID") or _DEFAULT_ASSISTANT_ID,
            system_prompt=_read_env_str("SYSTEM_PROMPT"),
            auto_approve=_read_env_bool("AUTO_APPROVE"),
            interrupt_shell_only=_read_env_bool("INTERRUPT_SHELL_ONLY"),
            shell_allow_list=(
                [cmd.strip() for cmd in raw.split(",") if cmd.strip()]
                if (raw := _read_env_str("SHELL_ALLOW_LIST"))
                else None
            )
            or None,
            interactive=_read_env_bool("INTERACTIVE", default=True),
            enable_shell=_read_env_bool("ENABLE_SHELL", default=True),
            enable_ask_user=_read_env_bool("ENABLE_ASK_USER"),
            enable_memory=_read_env_bool("ENABLE_MEMORY", default=True),
            enable_skills=_read_env_bool("ENABLE_SKILLS", default=True),
            sandbox_type=_read_env_str("SANDBOX_TYPE"),
            sandbox_id=_read_env_str("SANDBOX_ID"),
            sandbox_setup=_read_env_str("SANDBOX_SETUP"),
            cwd=_read_env_str("CWD"),
            project_root=_read_env_str("PROJECT_ROOT"),
            mcp_config_path=_read_env_str("MCP_CONFIG_PATH"),
            no_mcp=_read_env_bool("NO_MCP"),
            trust_project_mcp=_read_env_optional_bool("TRUST_PROJECT_MCP"),
        )

    # ------------------------------------------------------------------
    # 공장
    # ------------------------------------------------------------------

    @classmethod
    def from_cli_args(
        cls,
        *,
        project_context: ProjectContext | None,
        model_name: str | None,
        model_params: dict[str, Any] | None,
        assistant_id: str,
        auto_approve: bool,
        interrupt_shell_only: bool = False,
        shell_allow_list: list[str] | None = None,
        sandbox_type: str = "none",
        sandbox_id: str | None,
        sandbox_setup: str | None,
        enable_shell: bool,
        enable_ask_user: bool,
        mcp_config_path: str | None,
        no_mcp: bool,
        trust_project_mcp: bool | None,
        interactive: bool,
    ) -> ServerConfig:
        """구문 분석된 CLI 인수에서 `ServerConfig`을 빌드합니다.

        원시 직렬화된 값이 항상 절대적이고 모호하지 않도록 경로 정규화(예: 사용자의 작업 디렉터리에 대한 상대 MCP 구성 경로 확인)를 처리합니다.

Args:
            project_context: 명시적인 사용자/프로젝트 경로 컨텍스트.
            model_name: 모델 사양 문자열.
            model_params: 추가 모델 kwargs.
            assistant_id: 에이전트 식별자.
            auto_approve: 모든 도구를 자동 승인합니다.
            interrupt_shell_only: HITL 대신 미들웨어를 통해 셸 명령의 유효성을 검사합니다.
            shell_allow_list: `ShellAllowListMiddleware`에 대한 서버 하위 프로세스로 전달하기 위한 제한적인 셸
                              허용 목록입니다.
            sandbox_type: 샌드박스 유형.
            sandbox_id: 재사용할 기존 샌드박스 ID입니다.
            sandbox_setup: 샌드박스의 설정 스크립트 경로입니다.
            enable_shell: 셸 실행 도구를 활성화합니다.
            enable_ask_user: Ask_user 도구를 활성화합니다.
            mcp_config_path: MCP 구성 경로입니다.
            no_mcp: MCP를 비활성화합니다.
            trust_project_mcp: 프로젝트 MCP 서버를 신뢰하십시오.
            interactive: 에이전트가 대화형인지 여부입니다.

Returns:
            완전히 해결된 `ServerConfig`.

        """
        normalized_mcp = _normalize_path(mcp_config_path, project_context, "MCP config")

        return cls(
            model=model_name,
            model_params=model_params,
            assistant_id=assistant_id,
            auto_approve=auto_approve,
            interrupt_shell_only=interrupt_shell_only,
            shell_allow_list=shell_allow_list,
            interactive=interactive,
            enable_shell=enable_shell,
            enable_ask_user=enable_ask_user,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_setup=_normalize_path(
                sandbox_setup, project_context, "sandbox setup"
            ),
            cwd=(
                str(project_context.user_cwd) if project_context is not None else None
            ),
            project_root=(
                str(project_context.project_root)
                if project_context is not None
                and project_context.project_root is not None
                else None
            ),
            mcp_config_path=normalized_mcp,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
        )


def _normalize_path(
    raw_path: str | None,
    project_context: ProjectContext | None,
    label: str,
) -> str | None:
    """절대 상대 경로를 해결합니다.

    서버 하위 프로세스는 다른 작업 디렉터리에서 실행되므로 직렬화하기 전에 사용자의 원래 cwd에 대해 상대 경로를 확인해야 합니다.

Args:
        raw_path: CLI 인수의 경로(상대적일 수 있음)
        project_context: 경로 확인을 위한 사용자/프로젝트 컨텍스트입니다.
        label: 사람이 읽을 수 있는 오류 메시지 레이블(예: "MCP config")

Returns:
        절대 경로 문자열 또는 *raw_path*가 `None`이거나 비어 있는 경우 `None`입니다.

Raises:
        ValueError: 경로를 확인할 수 없는 경우.

    """
    if not raw_path:
        return None
    try:
        if project_context is not None:
            return str(project_context.resolve_user_path(raw_path))
        return str(Path(raw_path).expanduser().resolve())
    except OSError as exc:
        msg = (
            f"Could not resolve {label} path {raw_path!r}: {exc}. "
            "Ensure the path exists and is accessible."
        )
        raise ValueError(msg) from exc
