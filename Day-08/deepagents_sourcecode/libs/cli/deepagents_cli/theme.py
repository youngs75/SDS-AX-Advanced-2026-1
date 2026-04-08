"""CLI의 LangChain 브랜드 색상 및 의미 상수입니다.

Python 코드에 사용되는 색상 값에 대한 단일 정보 소스(Rich 마크업, `Content.styled`, `Content.from_markup`).
CSS 측 스타일은 텍스트 CSS 변수를 참조해야 합니다. 내장 변수(`$primary`, `$background`, `$text-muted`,
`$error-muted` 등)는 `DeepAgentsApp.__init__`의 `register_theme()`을 통해 설정되는 반면, 몇 가지 앱 관련
변수(`$mode-bash`, `$mode-command`, `$skill`, `$skill-hover`, `$tool`, `$tool-hover`)는
`App.get_theme_variable_defaults()`을 통해 이러한 상수로 뒷받침됩니다.

사용자 정의 CSS 변수 값이 필요한 코드는 `get_css_variable_defaults(dark=...)`을 호출해야 합니다. 전체 의미 색상 팔레트를
보려면 `ThemeEntry.REGISTRY`를 통해 `ThemeColors` 인스턴스를 검색하세요.

사용자는 `[themes.<name>]` 섹션 아래의 `~/.deepagents/config.toml`에서 사용자 정의 테마를 정의할 수 있습니다. 각각의 새
테마 섹션에는 `label`(str)이 포함되어야 합니다. `dark`(부울) 생략된 경우 기본값은 `False`입니다(어두운 테마의 경우 `True`로
설정). 색상 필드는 선택 사항이며 `dark` 플래그를 기반으로 내장된 어두운/밝은 팔레트로 대체됩니다. 이름이 내장 테마와 일치하는 섹션은 색상을 바꾸지
않고 해당 색상을 재정의합니다. 자세한 내용은 `_load_user_themes()`을 참조하세요.
"""


from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from textual.app import App

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand palette — dark  (originally tokyonight-inspired, LangChain blue primary)
# ---------------------------------------------------------------------------
LC_DARK = "#11121D"
"""배경 - 눈에 보이는 파란색 색조로 순수한 검정색과 구별됩니다."""
LC_CARD = "#1A1B2E"
"""표면/카드 — 배경보다 확실히 높아졌습니다."""
LC_BORDER_DK = "#25283B"
"""어두운 배경의 테두리입니다."""


LC_BORDER_LT = "#3A3E57"
"""더 밝거나 호버링된 배경의 테두리."""


LC_BODY = "#C0CAF5"
"""본문 텍스트 - 어두운 배경에 고대비가 표시됩니다."""
LC_BLUE = "#7AA2F7"
"""기본 액센트 파란색."""


LC_PURPLE = "#BB9AF7"
"""보조 악센트/배지/레이블."""


LC_GREEN = "#9ECE6A"
"""성공/긍정적 지표."""


LC_AMBER = "#EB8B46"
"""경고/주의 표시."""


LC_PINK = "#F7768E"
"""오류/파괴적인 행동."""


LC_MUTED = "#545C7E"
"""음소거/보조 텍스트."""


LC_GREEN_BG = "#1C2A38"
"""차이점 추가를 위한 미묘한 녹색 색조의 배경."""


LC_PINK_BG = "#2A1F32"
"""차이점 제거/오류를 위한 미묘한 분홍색 배경입니다."""


LC_PANEL = "#25283B"
"""패널 — 차별화된 섹션 배경(표면 위)"""
LC_SKILL = "#A78BFA"
"""스킬 호출 액센트 — 테두리 및 헤더 텍스트."""
LC_SKILL_HOVER = "#C4B5FD"
"""스킬 호출 호버 — 대화형 피드백을 위한 더 가벼운 변형입니다."""
LC_TOOL = LC_AMBER
"""도구 호출 강조 — 테두리 및 헤더 텍스트."""
LC_TOOL_HOVER = "#FFCB91"
"""도구 호출 호버 — 대화형 피드백을 위한 더 가벼운 변형입니다."""

# ---------------------------------------------------------------------------
# Brand palette — light
# ---------------------------------------------------------------------------
LC_LIGHT_BG = "#F5F5F7"
"""배경 - 따뜻한 중간색 흰색."""
LC_LIGHT_SURFACE = "#EAEAEE"
"""표면/카드 — 배경보다 약간 어둡습니다."""
LC_LIGHT_BORDER = "#C8CAD0"
"""밝은 배경의 테두리."""


LC_LIGHT_BORDER_HVR = "#A0A4B0"
"""호버/포커스 표면의 테두리."""


LC_LIGHT_BODY = "#24283B"
"""본문 텍스트 — 밝은 배경에 고대비가 표시됩니다."""
LC_LIGHT_BLUE = "#2E5EAA"
"""기본 강조 파란색(밝은 배경 대비를 위해 어두워짐)"""


LC_LIGHT_PURPLE = "#7C3AED"
"""보조 악센트(밝은 배경 대비를 위해 어둡게)."""


LC_LIGHT_GREEN = "#3A7D0A"
"""성공/양성(밝은 배경 대비를 위해 어두워짐)."""


LC_LIGHT_AMBER = "#B45309"
"""경고/주의(밝은 배경 대비를 위해 어둡게 표시됨)"""


LC_LIGHT_PINK = "#BE185D"
"""오류/파괴적입니다(밝은 배경 대비를 위해 어두워짐)."""


LC_LIGHT_MUTED = "#6B7280"
"""밝은 배경의 음소거/보조 텍스트입니다."""


LC_LIGHT_GREEN_BG = "#DCFCE7"
"""차이점 추가를 위한 미묘한 녹색 색조의 배경."""


LC_LIGHT_PINK_BG = "#FEE2E2"
"""차이점 제거/오류를 위한 미묘한 분홍색 배경입니다."""


LC_LIGHT_PANEL = "#E0E1E6"
"""밝은 테마를 위한 패널 - 차별화된 섹션 배경."""
LC_LIGHT_SKILL = "#7C3AED"
"""스킬 호출 액센트(밝은 배경 대비를 위해 어둡게)."""


LC_LIGHT_SKILL_HOVER = "#6D28D9"
"""스킬 호출 호버(밝은 배경 대비를 위해 어두워짐)"""


LC_LIGHT_TOOL = LC_LIGHT_AMBER
"""도구 호출 강조(밝은 배경 대비를 위해 어둡게)."""


LC_LIGHT_TOOL_HOVER = "#78350F"
"""도구 호출 호버(밝은 배경 대비를 위해 어두워짐)"""



# ---------------------------------------------------------------------------
# Semantic constants  (ANSI color names for Rich console output)
#
# These are ANSI color names resolved by the user's terminal palette, so they
# adapt to both dark and light terminal backgrounds automatically. They are
# used in Rich's `Console.print()` (non-interactive output, help screens,
# `non_interactive.py`, `main.py`).
#
# Textual widget code should NOT use these. Instead, call
# `get_theme_colors(self.app)` to obtain the active theme's `ThemeColors`
# (hex values), or reference CSS variables (`$primary`, `$muted`, etc.).
# ---------------------------------------------------------------------------
PRIMARY = "blue"
"""제목, 테두리, 링크 및 활성 요소에 대한 기본 악센트입니다."""


PRIMARY_DEV = "bright_red"
"""편집 가능한(개발) 설치에서 실행할 때 사용되는 악센트입니다."""


SUCCESS = "green"
"""긍정적인 결과 - 도구 성공, 승인된 조치."""
WARNING = "yellow"
"""주의 및 알림 상태 — 자동 승인 꺼짐, 도구 호출 보류 중, 알림."""
MUTED = "bright_black"
"""강조되지 않은 텍스트 — 타임스탬프, 보조 라벨."""
MODE_BASH = "red"
"""셸 모드 표시기 - 테두리, 프롬프트 및 메시지 접두사."""
MODE_COMMAND = "magenta"
"""명령 모드 표시기 - 테두리, 프롬프트 및 메시지 접두사."""
# Diff colors
DIFF_ADD_FG = "green"
"""인라인 diff에 추가된 라인 전경."""


DIFF_ADD_BG = "green"
"""인라인 diff에 라인 배경을 추가했습니다."""


DIFF_REMOVE_FG = "red"
"""인라인 diff에서 제거된 라인 전경."""


DIFF_REMOVE_BG = "red"
"""인라인 diff에서 제거된 라인 배경."""


DIFF_CONTEXT = "bright_black"
"""인라인 diff의 변경되지 않은 컨텍스트 줄."""


# Tool call widget
TOOL_BORDER = "bright_black"
"""도구 호출 카드 테두리."""


TOOL_HEADER = "yellow"
"""도구 호출 헤더, 슬래시 명령 토큰 및 승인 메뉴 명령."""


# File listing colors
FILE_PYTHON = "blue"
"""도구 호출 파일 목록의 Python 파일."""


FILE_CONFIG = "yellow"
"""도구 호출 파일 목록의 구성/데이터 파일."""


FILE_DIR = "green"
"""도구 호출 파일 목록의 디렉터리입니다."""


SPINNER = "blue"
"""스피너 색상을 로드 중입니다."""



# ---------------------------------------------------------------------------
# Theme variant dataclass
# ---------------------------------------------------------------------------


_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
"""`#7AA2F7`과 같은 7자리 16진수 색상 문자열과 일치합니다.

Textual의 `Color.parse`도 검증할 수 있지만 여기로 가져오면 Textual을 `theme.py`로 끌어오는데, 그렇지 않으면 프레임워크 깊이가
0인 순수 Python입니다.
"""



@dataclass(frozen=True, slots=True)
class ThemeColors:
    """하나의 테마 변형에 대한 의미 색상의 전체 세트입니다.

    모든 필드는 7자의 16진수 색상 문자열(예: `'#7AA2F7'`)이어야 합니다.

    """


    primary: str
    """제목, 테두리, 링크 및 활성 요소에 악센트를 줍니다."""


    secondary: str
    """배지, 라벨 및 장식 하이라이트에 대한 보조 악센트입니다."""


    accent: str
    """주목을 끄는 대비 액센트는 기본/보조와 구별됩니다."""


    panel: str
    """차별화된 단면 배경(표면 위)"""


    success: str
    """긍정적인 결과 - 도구 성공, 승인된 조치."""
    warning: str
    """주의 및 알림 상태 - 보류 중인 도구 호출, 알림입니다."""
    error: str
    """오류 및 파괴적인 조치 표시기."""


    muted: str
    """강조되지 않은 텍스트 — 타임스탬프, 보조 라벨."""
    mode_bash: str
    """셸 모드 표시기 - 테두리, 프롬프트 및 메시지 접두사."""
    mode_command: str
    """명령 모드 표시기 - 테두리, 프롬프트 및 메시지 접두사."""
    skill: str
    """스킬 호출 액센트 — 테두리 및 헤더 텍스트."""
    skill_hover: str
    """스킬 호출 호버 - 대화형 피드백을 위한 대조 변형입니다."""
    tool: str
    """도구 호출 강조 — 테두리 및 헤더 텍스트."""
    tool_hover: str
    """도구 호출 호버 — 대화형 피드백을 위한 대조 변형입니다."""
    foreground: str
    """기본 본문 텍스트입니다."""


    background: str
    """기본 애플리케이션 배경."""


    surface: str
    """높은 카드/패널 배경."""


    def __post_init__(self) -> None:
        """모든 필드가 유효한 16진수 색상인지 확인하세요.

        Raises:
            ValueError: 필드가 7자의 16진수 색상 문자열이 아닌 경우.

        """

        for f in fields(self):
            val = getattr(self, f.name)
            if not _HEX_RE.match(val):
                msg = (
                    f"ThemeColors.{f.name} must be a 7-char hex color"
                    f" (#RRGGBB), got {val!r}"
                )
                raise ValueError(msg)

    @classmethod
    def merged(cls, base: ThemeColors, overrides: dict[str, str]) -> ThemeColors:
        """베이스에 재정의를 오버레이하여 새 `ThemeColors`을(를) 만듭니다.

        `overrides`에 있는 필드는 해당 기본 값을 대체합니다. 누락된 필드는 `base`에서 상속됩니다. 이를 통해 사용자는 사용자 정의하려는
        색상만 지정할 수 있습니다.

        Args:
            base: `overrides`에 없는 모든 필드에 대해 대체 색상이 설정되었습니다.
            overrides: 필드 이름을 16진수 색상으로 매핑합니다. 알 수 없는 키는 자동으로 무시됩니다.

        Returns:
            값이 병합된 새로운 `ThemeColors`입니다.

        """

        valid_names = {f.name for f in fields(cls)}
        kwargs = {f.name: getattr(base, f.name) for f in fields(cls)}
        kwargs.update({k: v for k, v in overrides.items() if k in valid_names})
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Built-in theme color sets
# ---------------------------------------------------------------------------

DARK_COLORS = ThemeColors(
    primary=LC_BLUE,
    secondary=LC_PURPLE,
    accent=LC_GREEN,
    panel=LC_PANEL,
    success=LC_GREEN,
    warning=LC_AMBER,
    error=LC_PINK,
    muted=LC_MUTED,
    mode_bash=LC_PINK,
    mode_command=LC_PURPLE,
    skill=LC_SKILL,
    skill_hover=LC_SKILL_HOVER,
    tool=LC_TOOL,
    tool_hover=LC_TOOL_HOVER,
    foreground=LC_BODY,
    background=LC_DARK,
    surface=LC_CARD,
)
"""어두운 LangChain 테마에 대한 색상 세트입니다."""


LIGHT_COLORS = ThemeColors(
    primary=LC_LIGHT_BLUE,
    secondary=LC_LIGHT_PURPLE,
    accent=LC_LIGHT_GREEN,
    panel=LC_LIGHT_PANEL,
    success=LC_LIGHT_GREEN,
    warning=LC_LIGHT_AMBER,
    error=LC_LIGHT_PINK,
    muted=LC_LIGHT_MUTED,
    mode_bash=LC_LIGHT_PINK,
    mode_command=LC_LIGHT_PURPLE,
    skill=LC_LIGHT_SKILL,
    skill_hover=LC_LIGHT_SKILL_HOVER,
    tool=LC_LIGHT_TOOL,
    tool_hover=LC_LIGHT_TOOL_HOVER,
    foreground=LC_LIGHT_BODY,
    background=LC_LIGHT_BG,
    surface=LC_LIGHT_SURFACE,
)
"""밝은 LangChain 테마에 대한 색상 세트입니다."""



# ---------------------------------------------------------------------------
# Available themes  (name → display label, dark flag, colors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThemeEntry:
    """등록된 테마의 메타데이터입니다."""


    label: str
    """테마 선택기에 사람이 읽을 수 있는 라벨이 표시됩니다."""


    dark: bool
    """이것이 어두운 테마 변형인지 여부입니다."""


    colors: ThemeColors
    """색상 세트가 해결되었습니다."""


    custom: bool = True
    """이 테마를 `register_theme()`을 통해 Textual에 등록해야 하는지 여부입니다.

    LangChain 브랜드 테마 및 사용자 정의 테마의 경우 `True`입니다. `False` Textual이 이미 알고 있는 Textual 내장
    테마입니다.

    """


    REGISTRY: ClassVar[Mapping[str, ThemeEntry]]
    """등록된 모든 테마 항목은 텍스트 테마 이름으로 입력됩니다.

    모듈 로드 후 읽기 전용입니다(`MappingProxyType`).

    """


    def __post_init__(self) -> None:
        """레이블이 비어 있지 않은 문자열인지 확인하십시오.

        Raises:
            ValueError: `label`이 비어 있거나 공백만 있는 경우.

        """

        if not self.label.strip():
            msg = "ThemeEntry.label must be a non-empty string"
            raise ValueError(msg)


def _builtin_themes() -> dict[str, ThemeEntry]:
    """내장 테마 항목을 변경 가능한 사전으로 반환합니다.

    Returns:
        `ThemeEntry` 인스턴스에 내장된 테마 이름의 사전입니다.

    """

    r: dict[str, ThemeEntry] = {}
    r["langchain"] = ThemeEntry(
        label="LangChain Dark",
        dark=True,
        colors=DARK_COLORS,
    )
    r["langchain-light"] = ThemeEntry(
        label="LangChain Light",
        dark=False,
        colors=LIGHT_COLORS,
    )
    # Textual built-in themes — not registered via register_theme() (Textual's
    # own $primary, $background, etc. apply). The `colors` field provides
    # fallback values for app-specific CSS vars ($mode-bash, $mode-command) and
    # Python-side styling.  For standard properties (primary, secondary, etc.),
    # get_theme_colors() dynamically resolves from the actual Textual theme at
    # runtime so the Python and CSS color systems stay in sync.

    def _bi(label: str, *, is_dark: bool) -> ThemeEntry:
        return ThemeEntry(
            label=label,
            dark=is_dark,
            colors=DARK_COLORS if is_dark else LIGHT_COLORS,
            custom=False,
        )

    r["textual-dark"] = _bi("Textual Dark", is_dark=True)
    r["textual-light"] = _bi("Textual Light", is_dark=False)
    r["textual-ansi"] = _bi("Terminal (ANSI)", is_dark=False)
    # Popular community themes (all ship with Textual >= 8.0)
    r["atom-one-dark"] = _bi("Atom One Dark", is_dark=True)
    r["atom-one-light"] = _bi("Atom One Light", is_dark=False)
    r["catppuccin-frappe"] = _bi("Catppuccin Frappé", is_dark=True)
    r["catppuccin-latte"] = _bi("Catppuccin Latte", is_dark=False)
    r["catppuccin-macchiato"] = _bi("Catppuccin Macchiato", is_dark=True)
    r["catppuccin-mocha"] = _bi("Catppuccin Mocha", is_dark=True)
    r["dracula"] = _bi("Dracula", is_dark=True)
    r["flexoki"] = _bi("Flexoki", is_dark=True)
    r["gruvbox"] = _bi("Gruvbox", is_dark=True)
    r["monokai"] = _bi("Monokai", is_dark=True)
    r["nord"] = _bi("Nord", is_dark=True)
    r["rose-pine"] = _bi("Rosé Pine", is_dark=True)
    r["rose-pine-dawn"] = _bi("Rosé Pine Dawn", is_dark=False)
    r["rose-pine-moon"] = _bi("Rosé Pine Moon", is_dark=True)
    r["solarized-dark"] = _bi("Solarized Dark", is_dark=True)
    r["solarized-light"] = _bi("Solarized Light", is_dark=False)
    r["tokyo-night"] = _bi("Tokyo Night", is_dark=True)
    return r


_BUILTIN_NAMES: frozenset[str] = frozenset(_builtin_themes())
"""내장 테마의 이름입니다.

기본 제공 이름과 일치하는 사용자 `[themes.<name>]` 섹션은 새 테마를 생성하는 대신 색상을 재정의합니다. 자동으로 동기화를 유지하기 위해
`_builtin_themes()`에서 파생되었습니다.
"""



def _load_user_themes(
    builtins: dict[str, ThemeEntry],
    *,
    config_path: Path | None = None,
) -> None:
    """`config.toml`에서 `builtins`(변형됨)로 사용자 정의 테마를 로드합니다.

    **새 테마** — 각 `[themes.<name>]` 섹션(`<name>`은 기본 제공되지 않음)에는 다음이 있어야 합니다.

    - `label` (str) — 테마 선택기에 표시되는 사람이 읽을 수 있는 이름입니다. - `dark` (부울, 선택 사항) — 다크 모드 변형인지
    여부.

        기본값은 `False`(밝음)입니다.

    **내장 재정의** — `<name>`이 내장 테마와 일치하면 색상 필드만 읽혀집니다. `label` 및 `dark`은 내장에서 상속됩니다.

    모든 `ThemeColors` 필드는 선택 사항입니다. 새 테마의 경우 생략된 필드는 `dark` 플래그를 기반으로 내장된 어둡거나 밝은 팔레트로
    대체됩니다.

    기본 제공 재정의의 경우 생략된 필드는 기존 기본 제공 색상을 유지합니다.

    잘못된 테마(잘못된 16진수, 필수 키 누락)는 경고로 기록되고 건너뛰어지며 시작 시 충돌이 발생하지 않습니다.

    예 `config.toml` 조각:

    ```toml
    # New custom theme
    [themes.my-solarized]
    label = "My Solarized"
    dark = true
    primary = "#268BD2"
    warning = "#B58900"

    # Override built-in theme colors
    [themes.langchain]
    primary = "#FF5500"
    ```

    Args:
        builtins: 업데이트할 변경 가능한 사전(새 테마가 추가되고 기본 제공 재정의가 기존 항목을 대체함)
        config_path: 구성 파일 경로를 재정의합니다(테스트).

    """

    if config_path is None:
        try:
            config_path = Path.home() / ".deepagents" / "config.toml"
        except RuntimeError:
            logger.debug("Cannot determine home directory; skipping user theme loading")
            return

    import tomllib

    try:
        if not config_path.exists():
            return

        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as exc:
        logger.warning(
            "Could not read %s for user themes: %s",
            config_path,
            exc,
        )
        return

    themes_section: Any = data.get("themes")
    if not isinstance(themes_section, dict) or not themes_section:
        return

    valid_color_names = {f.name for f in fields(ThemeColors)}
    reserved = {"label", "dark"}

    for name, section in themes_section.items():
        if not isinstance(section, dict):
            logger.warning("Ignoring non-table [themes.%s]", name)
            continue

        # --- Parse color overrides (shared by built-in overrides & new themes)
        color_overrides: dict[str, str] = {}
        for k, v in section.items():
            if k in reserved:
                continue
            if not isinstance(v, str):
                logger.warning(
                    "User theme '%s' field '%s' must be a string, got %s; ignoring",
                    name,
                    k,
                    type(v).__name__,
                )
                continue
            if k in valid_color_names:
                color_overrides[k] = v
            else:
                logger.warning(
                    "User theme '%s' has unknown color field '%s'; ignoring",
                    name,
                    k,
                )

        # --- Built-in override: merge color tweaks into the existing entry
        if name in _BUILTIN_NAMES:
            existing = builtins.get(name)
            if existing is None:
                logger.warning(
                    "Built-in theme '%s' not in builtins dict; skipping override",
                    name,
                )
                continue
            if not color_overrides:
                continue
            try:
                colors = ThemeColors.merged(existing.colors, color_overrides)
            except ValueError as exc:
                logger.warning(
                    "Built-in theme '%s' color override invalid: %s; skipping",
                    name,
                    exc,
                )
                continue
            builtins[name] = ThemeEntry(
                label=existing.label,
                dark=existing.dark,
                colors=colors,
                custom=existing.custom,
            )
            continue

        # --- New custom theme: label required, dark defaults to False (light)
        label = section.get("label")
        if not isinstance(label, str) or not label.strip():
            logger.warning(
                "User theme '%s' missing required 'label' (str); skipping",
                name,
            )
            continue

        dark = section.get("dark", False)
        if not isinstance(dark, bool):
            logger.warning(
                "User theme '%s': 'dark' must be true or false, got %s (%r);"
                " defaulting to light",
                name,
                type(dark).__name__,
                dark,
            )
            dark = False

        base = DARK_COLORS if dark else LIGHT_COLORS
        try:
            colors = ThemeColors.merged(base, color_overrides)
        except ValueError as exc:
            logger.warning(
                "User theme '%s' has invalid colors: %s; skipping",
                name,
                exc,
            )
            continue

        builtins[name] = ThemeEntry(
            label=label,
            dark=dark,
            colors=colors,
            custom=True,
        )


def _build_registry(
    *, config_path: Path | None = None
) -> MappingProxyType[str, ThemeEntry]:
    """테마 레지스트리(내장 + 사용자 테마)를 빌드하고 고정합니다.

    Args:
        config_path: 구성 파일 경로를 재정의합니다(테스트).

    Returns:
        `ThemeEntry` 인스턴스에 대한 테마 이름의 읽기 전용 매핑입니다.

    """

    r = _builtin_themes()
    _load_user_themes(r, config_path=config_path)
    return MappingProxyType(r)


ThemeEntry.REGISTRY = _build_registry()
"""`ThemeEntry` 인스턴스에 대한 텍스트 테마 이름의 읽기 전용 매핑입니다.

`_build_registry()`을 통해 구축되었으므로 변경 가능한 스테이징 dict의 범위는 함수 호출로 지정되며 동결 후에는 변경할 수 없습니다.
`ThemeEntry`의 `ClassVar` 선언은 유형을 제공합니다. 이 할당은 값을 제공합니다.
"""


DEFAULT_THEME = "langchain"
"""기본 설정이 저장되지 않은 경우 사용되는 테마 이름입니다."""



def reload_registry() -> MappingProxyType[str, ThemeEntry]:
    """디스크에서 테마 레지스트리를 다시 빌드하고 `ThemeEntry.REGISTRY`을 업데이트하세요.

    `/reload`이 앱을 다시 시작하지 않고도 구성 변경 사항을 선택할 수 있도록 사용자 정의 테마에 대해
    `~/.deepagents/config.toml`을 다시 읽습니다.

    Returns:
        새로운 고정된 레지스트리.

    """

    ThemeEntry.REGISTRY = _build_registry()
    return ThemeEntry.REGISTRY


def get_css_variable_defaults(
    *, dark: bool = True, colors: ThemeColors | None = None
) -> dict[str, str]:
    """해당 모드에 대한 사용자 정의 CSS 변수 기본값을 반환합니다.

    대부분의 스타일은 Textual의 내장 CSS 변수(`$primary`, `$text-muted`, `$error-muted` 등)에 의해 처리됩니다.
    이 함수는 상응하는 텍스트가 없는 앱별 의미 변수만 반환합니다.

    Args:
        dark: `colors`이 없음인 경우 `DARK_COLORS` 또는 `LIGHT_COLORS`을 선택합니다.
        colors: 명시적 색상을 사용하도록 설정되었습니다. `dark`보다 우선합니다.

    Returns:
        16진수 색상 값에 대한 CSS 변수 이름의 사전입니다.

    """

    c = colors if colors is not None else (DARK_COLORS if dark else LIGHT_COLORS)
    return {
        "mode-bash": c.mode_bash,
        "mode-command": c.mode_command,
        "skill": c.skill,
        "skill-hover": c.skill_hover,
        "tool": c.tool,
        "tool-hover": c.tool_hover,
    }


def _resolve_app(widget_or_app: object) -> object:
    """위젯이나 앱을 앱 인스턴스로 해결합니다.

    Args:
        widget_or_app: 텍스트 `App` 또는 탑재된 위젯.

    Returns:
        해결된 앱 인스턴스입니다.

    """

    return (
        widget_or_app.app  # type: ignore[attr-defined]
        if hasattr(type(widget_or_app), "app")
        else widget_or_app
    )


def _colors_from_textual_theme(app: object) -> ThemeColors:
    """앱의 활성 텍스트 테마에서 `ThemeColors`을(를) 구성합니다.

    Python 측 스타일이 CSS와 일치하도록 해결된 테마에서 표준 속성(기본, 보조 등)을 읽습니다.  `muted`은 무조건 어둡거나 밝은 기반으로
    돌아갑니다(텍스트와 동일하지 않음). `mode_bash`은 테마의 `error` 색상에서 파생되고 `mode_command`은
    `secondary`에서 파생되며, 16진수가 아닌 경우 기본 팔레트로 돌아갑니다.

    16진수가 아닌 값(예: ANSI 테마의 `ansi_blue`)이 감지되어 자동으로 기본 팔레트로 대체됩니다.

    Args:
        app: 텍스트 앱 인스턴스.

    Returns:
        `ThemeColors`은 활성 테마에서 파생되었습니다.

    """

    ct = app.current_theme  # type: ignore[attr-defined]
    dark: bool = ct.dark
    base = DARK_COLORS if dark else LIGHT_COLORS

    def _hex_or(val: str | None, fallback: str) -> str:
        """유효한 `#RRGGBB` 16진수 색상이면 `val`을 반환하고, 그렇지 않으면 `fallback`을 반환합니다.

        Args:
            val: 활성 텍스트 테마의 색상 문자열(`None` 또는 `ansi_blue`과 같은 16진수가 아닌 이름일 수 있음)
            fallback: 기본 팔레트의 16진수 값이 보장됩니다.

        Returns:
            `#RRGGBB`과 일치하면 `val`, 그렇지 않으면 `fallback`입니다.

        """

        if val is not None and _HEX_RE.match(val):
            return val
        return fallback

    return ThemeColors(
        primary=_hex_or(ct.primary, base.primary),
        secondary=_hex_or(ct.secondary, base.secondary),
        accent=_hex_or(ct.accent, base.accent),
        panel=_hex_or(ct.panel, base.panel),
        success=_hex_or(ct.success, base.success),
        warning=_hex_or(ct.warning, base.warning),
        error=_hex_or(ct.error, base.error),
        muted=base.muted,
        mode_bash=_hex_or(ct.error, base.mode_bash),
        mode_command=_hex_or(ct.secondary, base.mode_command),
        # No Textual equivalent — always use base palette.
        skill=base.skill,
        skill_hover=base.skill_hover,
        # Derived from Textual's warning color (shared amber hue).
        tool=_hex_or(ct.warning, base.tool),
        # No Textual equivalent — always base palette (may diverge from
        # tool in custom themes that override warning).
        tool_hover=base.tool_hover,
        foreground=_hex_or(ct.foreground, base.foreground),
        background=_hex_or(ct.background, base.background),
        surface=_hex_or(ct.surface, base.surface),
    )


def get_theme_colors(widget_or_app: App | object | None = None) -> ThemeColors:
    """활성 텍스트 테마에 대해 `ThemeColors`을 반환합니다.

    사용자 정의 테마(LangChain 브랜드 및 사용자 정의)의 경우 레지스트리에서 미리 빌드된 `ThemeColors`이 직접 반환됩니다.  텍스트
    내장 테마의 경우 색상이 실제 테마 속성에서 동적으로 확인되므로 Python 측 스타일이 CSS 변수와 동기화됩니다.

    텍스트 위젯 코드는 리치 콘솔 출력 전용인 모듈 수준 ANSI 상수를 읽는 대신 이를 호출해야 합니다.

    Args:
        widget_or_app: 텍스트 `App`, 마운트된 위젯 또는 `None`.

    Returns:
        활성 테마의 경우 `ThemeColors`입니다.

    """

    if widget_or_app is None:
        # Fall back to the active Textual app context var when no explicit
        # widget/app is passed (e.g. from @staticmethod helpers).
        try:
            from textual._context import active_app  # noqa: PLC2701

            widget_or_app = active_app.get()
        except (ImportError, LookupError):
            return DARK_COLORS
    app = _resolve_app(widget_or_app)
    entry = ThemeEntry.REGISTRY.get(app.theme)  # type: ignore[attr-defined]
    # Custom themes (LC-branded / user-defined) use pre-built colors.
    if entry is not None and entry.custom:
        return entry.colors
    # Built-in or unrecognized themes — derive from the resolved Textual
    # theme so Python styling matches CSS.
    try:
        return _colors_from_textual_theme(app)
    except Exception:
        logger.warning("Could not resolve theme colors dynamically", exc_info=True)
        if entry is not None:
            return entry.colors
        return DARK_COLORS
