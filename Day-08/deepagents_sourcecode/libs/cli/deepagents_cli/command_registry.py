"""통합 슬래시 명령 레지스트리.

모든 슬래시 명령은 `COMMANDS`의 `SlashCommand` 항목으로 한 번 선언됩니다. 우회 계층 고정 세트 및 자동 완성 튜플은 자동으로
파생됩니다. 다른 파일은 명령 메타데이터를 하드 코딩해서는 안 됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_cli.skills.load import ExtendedSkillMetadata


class BypassTier(StrEnum):
    """명령이 메시지 대기열을 건너뛸 수 있는지 여부를 제어하는 ​​분류입니다."""

    ALWAYS = "always"
    """스레드 중간 전환을 포함하여 사용 중인 상태에 관계없이 실행됩니다."""

    CONNECTING = "connecting"
    """에이전트/셸 중에는 우회하지 않고 초기 서버 연결 중에만 우회합니다."""

    IMMEDIATE_UI = "immediate_ui"
    """즉시 모달 UI를 엽니다. `_defer_action` 콜백을 통해 실제 작업이 연기되었습니다."""

    SIDE_EFFECT_FREE = "side_effect_free"
    """즉시 부작용을 실행하십시오. 유휴 상태가 될 때까지 채팅 출력을 연기합니다."""

    QUEUED = "queued"
    """앱이 사용 중일 때는 대기열에서 기다려야 합니다."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SlashCommand:
    """단일 슬래시 명령 정의."""

    name: str
    """정식 명령 이름(예: `/quit`)."""

    description: str
    """사용자에게 표시되는 간단한 설명입니다."""

    bypass_tier: BypassTier
    """대기열 우회 분류."""

    hidden_keywords: str = ""
    """유사 일치를 위한 공백으로 구분된 용어입니다(표시되지 않음)."""

    aliases: tuple[str, ...] = ()
    """대체 이름(예: `/quit`의 경우 `("/q",)`)"""


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        name="/clear",
        description="Clear chat and start new thread",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="reset",
    ),
    SlashCommand(
        name="/editor",
        description="Open prompt in external editor ($EDITOR)",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/mcp",
        description="Show active MCP servers and tools",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
        hidden_keywords="servers",
    ),
    SlashCommand(
        name="/model",
        description="Switch or configure model (--model-params, --default)",
        bypass_tier=BypassTier.IMMEDIATE_UI,
    ),
    SlashCommand(
        name="/offload",
        description="Free up context window space by offloading older messages",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="compact",
        aliases=("/compact",),
    ),
    SlashCommand(  # Static alias; not auto-generated from skill discovery
        name="/remember",
        description="Update memory and skills from conversation",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(  # Static alias; not auto-generated from skill discovery
        name="/skill-creator",
        description="Guide for creating effective agent skills",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/threads",
        description="Browse and resume previous threads",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="continue history sessions",
    ),
    SlashCommand(
        name="/trace",
        description="Open current thread in LangSmith",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/tokens",
        description="Token usage",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="cost",
    ),
    SlashCommand(
        name="/reload",
        description="Reload config from environment variables and .env",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="refresh",
    ),
    SlashCommand(
        name="/theme",
        description="Switch color theme",
        bypass_tier=BypassTier.IMMEDIATE_UI,
        hidden_keywords="dark light color appearance",
    ),
    SlashCommand(
        name="/update",
        description="Check for and install updates",
        bypass_tier=BypassTier.QUEUED,
        hidden_keywords="upgrade",
    ),
    SlashCommand(
        name="/auto-update",
        description="Toggle automatic updates on or off",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/changelog",
        description="Open changelog in browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/version",
        description="Show version",
        bypass_tier=BypassTier.CONNECTING,
    ),
    SlashCommand(
        name="/feedback",
        description="Submit a bug report or feature request",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/docs",
        description="Open documentation in browser",
        bypass_tier=BypassTier.SIDE_EFFECT_FREE,
    ),
    SlashCommand(
        name="/help",
        description="Show help",
        bypass_tier=BypassTier.QUEUED,
    ),
    SlashCommand(
        name="/quit",
        description="Exit app",
        bypass_tier=BypassTier.ALWAYS,
        hidden_keywords="close leave",
        aliases=("/q",),
    ),
)
"""모든 슬래시 명령."""


# ---------------------------------------------------------------------------
# Derived bypass-tier frozensets
# ---------------------------------------------------------------------------


def _build_bypass_set(tier: BypassTier) -> frozenset[str]:
    """계층에 대한 명령 이름(별칭 포함)의 고정 집합을 구축합니다.

Args:
        tier: 수집할 우회 계층입니다.

Returns:
        `tier`에 속하는 모든 이름과 별칭의 고정 집합입니다.

    """
    names: set[str] = set()
    for cmd in COMMANDS:
        if cmd.bypass_tier == tier:
            names.add(cmd.name)
            names.update(cmd.aliases)
    return frozenset(names)


ALWAYS_IMMEDIATE: frozenset[str] = _build_bypass_set(BypassTier.ALWAYS)
"""사용 중 상태와 관계없이 실행되는 명령입니다."""

BYPASS_WHEN_CONNECTING: frozenset[str] = _build_bypass_set(BypassTier.CONNECTING)
"""초기 서버 연결 중에만 우회하는 명령입니다."""

IMMEDIATE_UI: frozenset[str] = _build_bypass_set(BypassTier.IMMEDIATE_UI)
"""실제 작업을 연기하고 모달 UI를 즉시 여는 명령입니다."""

SIDE_EFFECT_FREE: frozenset[str] = _build_bypass_set(BypassTier.SIDE_EFFECT_FREE)
"""부작용이 즉시 발생하는 명령. 유휴 상태까지 채팅 출력이 연기됩니다."""

QUEUE_BOUND: frozenset[str] = _build_bypass_set(BypassTier.QUEUED)
"""앱이 사용 중일 때 대기열에서 대기해야 하는 명령입니다."""

ALL_CLASSIFIED: frozenset[str] = (
    ALWAYS_IMMEDIATE
    | BYPASS_WHEN_CONNECTING
    | IMMEDIATE_UI
    | SIDE_EFFECT_FREE
    | QUEUE_BOUND
)
"""5개 계층 모두의 통합 — 드리프트 테스트에 사용됩니다."""
# ---------------------------------------------------------------------------
# Autocomplete tuples
# ---------------------------------------------------------------------------

SLASH_COMMANDS: list[tuple[str, str, str]] = [
    (cmd.name, cmd.description, cmd.hidden_keywords) for cmd in COMMANDS
]
"""`SlashCommandController`에 대한 `(name, description, hidden_keywords)` 튜플."""


def parse_skill_command(command: str) -> tuple[str, str]:
    """`/skill:<name>` 명령에서 스킬 이름과 인수를 추출합니다.

Args:
        command: 전체 명령 문자열(예: `/skill:web-research find X`).

Returns:
        `(skill_name, args)`의 튜플입니다.

            스킬 이름은 소문자로 정규화됩니다. 명령의 접두사 뒤에 스킬 이름이 없으면 둘 다 빈 문자열입니다.

    """
    after_prefix = command[len("/skill:") :].strip()
    parts = after_prefix.split(maxsplit=1)
    if not parts or not parts[0]:
        return "", ""
    skill_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return skill_name, args


_STATIC_SKILL_ALIASES: frozenset[str] = frozenset({"remember", "skill-creator"})
"""전용 최상위 슬래시 명령이 있는 내장 스킬 이름입니다.

`COMMANDS`에 `/<name>` 편의 별칭이 존재하므로 `/skill:<name>` 형식이 중복되는 스킬만 나열합니다.  여기에 모든 명령 이름을
추가하지 **않습니다**. 이렇게 하면 슬래시 명령과 이름을 공유하는 관련 없는 사용자 기술이 자동으로 억제됩니다(예: `model`이라는 사용자 기술은
여전히 ​​`/skill:model`로 표시되어야 함).
"""


def build_skill_commands(
    skills: list[ExtendedSkillMetadata],
) -> list[tuple[str, str, str]]:
    """발견된 기술에 대한 자동 완성 튜플을 구축합니다.

    각 스킬은 설명과 퍼지 일치를 위한 숨겨진 키워드로 스킬 이름이 포함된 `/skill:<name>` 항목이 됩니다.

    `COMMANDS`에 이미 전용 슬래시 명령이 있는 스킬(예: `remember` → `/remember`)은 자동 완성 항목이 중복되는 것을 방지하기
    위해 제외됩니다.

Args:
        skills: 검색된 스킬 메타데이터 목록입니다.

Returns:
        `(name, description, hidden_keywords)` 튜플 목록입니다.

    """
    return [
        (f"/skill:{skill['name']}", skill["description"], skill["name"])
        for skill in skills
        if skill["name"] not in _STATIC_SKILL_ALIASES
    ]
