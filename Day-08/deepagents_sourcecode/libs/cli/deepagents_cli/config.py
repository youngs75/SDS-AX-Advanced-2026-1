"""CLI에 대한 중앙 구성 및 모델 부트스트랩 논리입니다.

이 모듈은 지연 환경/부트스트랩 로딩, 문자 모양 및 콘솔 설정, 공급자/모델 확인, 셸 안전 확인, 패키지의 나머지 부분에서 사용되는 `settings`
싱글톤을 소유합니다.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shlex
import sys
import threading
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from deepagents_cli._version import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy bootstrap: dotenv loading, LANGSMITH_PROJECT override, and start-path
# detection are deferred until first access of `settings` (via module
# `__getattr__`).  This avoids disk I/O and path traversal during import for
# callers that never touch `settings` (e.g. `deepagents --help`).
# ---------------------------------------------------------------------------

_bootstrap_done = False
"""`_ensure_bootstrap()`이 실행되었는지 여부."""

_bootstrap_lock = threading.Lock()
"""기본 스레드와 사전 준비 작업자 스레드의 동시 액세스로부터 `_ensure_bootstrap()`을 보호합니다."""

_singleton_lock = threading.Lock()
"""`_get_console` / `_get_settings`에서 게으른 싱글톤 구성을 보호합니다."""

_bootstrap_start_path: Path | None = None
"""dotenv 및 프로젝트 검색을 위해 부트스트랩 시간에 캡처된 작업 디렉터리입니다."""

_original_langsmith_project: str | None = None
"""CLI가 에이전트 추적을 위해 이를 재정의하기 전 호출자의 `LANGSMITH_PROJECT` 값입니다.

dotenv 로드 후 `LANGSMITH_PROJECT` 재정의 전에 `_ensure_bootstrap()` 내부에서 캡처되므로 `.env` 전용 값이
표시됩니다.
"""


def _find_dotenv_from_start_path(start_path: Path) -> Path | None:
    """명시적인 시작 경로에서 위쪽으로 가장 가까운 `.env` 파일을 찾습니다.

Args:
        start_path:

Returns:
        가장 가까운 `.env` 파일의 경로입니다. 파일을 찾을 수 없으면 `None`입니다.

    """
    current = start_path.expanduser().resolve()
    for parent in [current, *list(current.parents)]:
        candidate = parent / ".env"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            logger.warning("Could not inspect .env candidate %s", candidate)
            continue
    return None


# Global user-level .env (~/.deepagents/.env); sentinel when Path.home() fails.
try:
    _GLOBAL_DOTENV_PATH = Path.home() / ".deepagents" / ".env"
except RuntimeError:
    _GLOBAL_DOTENV_PATH = Path("/nonexistent/.deepagents/.env")


def _load_dotenv(*, start_path: Path | None = None) -> bool:
    """프로젝트 및 전역 `.env` 파일에서 환경 변수를 로드합니다.

    순서대로 로드됩니다(첫 번째 쓰기 성공, `override=False`):

    1. 프로젝트/CWD `.env` — 프로젝트별 값 2. `~/.deepagents/.env` — 전역 사용자 기본값

    두 레이어 모두 `override=False`(python-dotenv 기본값)을 사용하므로 쉘에서 내보낸 변수가 항상 dotenv 파일보다
    우선합니다. 프로젝트가 먼저 로드되므로 유효한 우선순위는 다음과 같습니다.

    ```text
    shell env (incl. inline `VAR=x`)  >  project `.env`  >  global `.env`
    ```

    !!! 메모

        동일한 이름의 셸 내보내기와 충돌하지 않고 자격 증명의 범위를 CLI로 지정하려면 `DEEPAGENTS_CLI_` env-var 접두사를
        사용하세요(`deepagents_cli.model_config`의 `resolve_env_var` 참조).

Args:
        start_path: 프로젝트 `.env` 검색에 사용할 디렉터리입니다.

Returns:
        적어도 하나의 dotenv 파일이 로드된 경우 `True`, 그렇지 않은 경우 `False`.

    """
    import dotenv

    loaded = False

    # 1. Project/CWD .env — loads first so project values are set before the
    # global file, which can only fill in vars not already present.
    dotenv_path: Path | str | None = None
    try:
        if start_path is None:
            loaded = dotenv.load_dotenv(override=False) or loaded
        else:
            dotenv_path = _find_dotenv_from_start_path(start_path)
            if dotenv_path is not None:
                loaded = (
                    dotenv.load_dotenv(dotenv_path=dotenv_path, override=False)
                    or loaded
                )
    except (OSError, ValueError):
        logger.warning(
            "Could not read project dotenv at %s; project env vars will not be loaded",
            dotenv_path or start_path or "cwd",
            exc_info=True,
        )

    # 2. Global (~/.deepagents/.env) — fills in any vars not already set by
    # the shell or the project dotenv.
    # try/except wraps both is_file() and load_dotenv() to cover the TOCTOU
    # window where the file can vanish between stat and open.
    try:
        if _GLOBAL_DOTENV_PATH.is_file() and dotenv.load_dotenv(
            dotenv_path=_GLOBAL_DOTENV_PATH, override=False
        ):
            loaded = True
            logger.debug("Loaded global dotenv: %s", _GLOBAL_DOTENV_PATH)
    except (OSError, ValueError):
        logger.warning(
            "Could not read global dotenv at %s; global defaults will not be applied",
            _GLOBAL_DOTENV_PATH,
            exc_info=True,
        )

    return loaded


def _ensure_bootstrap() -> None:
    """일회성 부트스트랩 실행: dotenv 로딩 및 `LANGSMITH_PROJECT` 재정의.

    멱등성 및 스레드 안전성 - 후속 호출은 작동하지 않습니다. `settings`에 처음 액세스할 때 `_get_settings()`에 의해 자동으로
    호출됩니다.

    부분적인 실패(예: 잘못된 형식의 `.env`)가 여전히 부트스트랩을 완료로 표시하도록 플래그가 `finally`에 설정되어 무한 재시도 루프를
    방지합니다. 예외는 ERROR 수준에서 포착되어 기록됩니다. CLI는 환경을 있는 그대로 진행합니다.

    """
    global _bootstrap_done, _bootstrap_start_path, _original_langsmith_project  # noqa: PLW0603

    if _bootstrap_done:
        return

    with _bootstrap_lock:
        if _bootstrap_done:  # double-check after acquiring lock
            return

        try:
            from deepagents_cli.project_utils import (
                get_server_project_context as _get_server_project_context,
            )

            ctx = _get_server_project_context()
            _bootstrap_start_path = ctx.user_cwd if ctx else None
            _load_dotenv(start_path=_bootstrap_start_path)

            # Capture AFTER dotenv loading so .env-only values are visible,
            # but BEFORE the override below replaces it.
            _original_langsmith_project = os.environ.get("LANGSMITH_PROJECT")

            # CRITICAL: Override LANGSMITH_PROJECT to route agent traces to a
            # separate project. LangSmith reads LANGSMITH_PROJECT at invocation
            # time, so we override it here and preserve the user's original
            # value for shell commands.
            from deepagents_cli._env_vars import LANGSMITH_PROJECT

            deepagents_project = os.environ.get(LANGSMITH_PROJECT)
            if deepagents_project:
                os.environ["LANGSMITH_PROJECT"] = deepagents_project

            # Propagate prefixed LangSmith env vars to canonical names.
            # The CLI resolves prefixed vars via resolve_env_var(), but the
            # LangSmith SDK reads os.environ directly and has no knowledge
            # of the DEEPAGENTS_CLI_ prefix. Setting canonical vars here
            # bridges that gap.
            from deepagents_cli.model_config import _ENV_PREFIX

            for canonical in (
                "LANGSMITH_API_KEY",
                "LANGCHAIN_API_KEY",
                "LANGSMITH_TRACING",
                "LANGCHAIN_TRACING_V2",
            ):
                prefixed = f"{_ENV_PREFIX}{canonical}"
                if prefixed not in os.environ:
                    continue
                prefixed_val = os.environ[prefixed]
                if canonical not in os.environ:
                    # Propagate (including empty string for explicit disable).
                    os.environ[canonical] = prefixed_val
                elif os.environ[canonical] != prefixed_val:
                    logger.warning(
                        "Both %s and %s are set with different values; "
                        "the LangSmith SDK will use %s while the CLI "
                        "prefers %s. Unset one to avoid confusion.",
                        canonical,
                        prefixed,
                        canonical,
                        prefixed,
                    )
        except Exception:
            logger.exception(
                "Bootstrap failed; .env values and LANGSMITH_PROJECT override "
                "may be missing. The CLI will proceed with environment as-is.",
            )
        finally:
            _bootstrap_done = True


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from rich.console import Console

    # Static type stubs for lazy module attributes resolved by __getattr__.
    # At runtime these are created on first access by _get_settings() /
    # _get_console() and cached in globals().
    settings: Settings
    console: Console

MODE_PREFIXES: dict[str, str] = {
    "shell": "!",
    "command": "/",
}
"""각 비정규 모드를 해당 트리거 문자에 매핑합니다."""

MODE_DISPLAY_GLYPHS: dict[str, str] = {
    "shell": "$",
    "command": "/",
}
"""각 비정규 모드를 프롬프트/UI에 표시된 표시 문자 모양으로 매핑합니다."""

if MODE_PREFIXES.keys() != MODE_DISPLAY_GLYPHS.keys():
    _only_prefixes = MODE_PREFIXES.keys() - MODE_DISPLAY_GLYPHS.keys()
    _only_glyphs = MODE_DISPLAY_GLYPHS.keys() - MODE_PREFIXES.keys()
    msg = (
        "MODE_PREFIXES and MODE_DISPLAY_GLYPHS have mismatched keys: "
        f"only in PREFIXES={_only_prefixes}, only in GLYPHS={_only_glyphs}"
    )
    raise ValueError(msg)

PREFIX_TO_MODE: dict[str, str] = {v: k for k, v in MODE_PREFIXES.items()}
"""역방향 조회: 트리거 문자 -> 모드 이름."""


class CharsetMode(StrEnum):
    """TUI 디스플레이를 위한 문자 세트 모드입니다."""

    UNICODE = "unicode"
    """항상 유니코드 문자 모양을 사용하십시오(예: `⏺`, `✓`, `…`)."""

    ASCII = "ascii"
    """항상 ASCII 안전 대체(예: `(*)`, `[OK]`, `...`)를 사용하세요."""

    AUTO = "auto"
    """런타임 시 문자 세트 지원을 감지하고 유니코드 또는 ASCII를 선택합니다."""


@dataclass(frozen=True)
class Glyphs:
    """TUI 표시용 문자 모양입니다."""

    tool_prefix: str  # ⏺ vs (*)
    ellipsis: str  # … vs ...
    checkmark: str  # ✓ vs [OK]
    error: str  # ✗ vs [X]
    circle_empty: str  # ○ vs [ ]
    circle_filled: str  # ● vs [*]
    output_prefix: str  # ⎿ vs L
    spinner_frames: tuple[str, ...]  # Braille vs ASCII spinner
    pause: str  # ⏸ vs ||
    newline: str  # ⏎ vs \\n
    warning: str  # ⚠ vs [!]
    question: str  # ? vs [?]
    arrow_up: str  # up arrow vs ^
    arrow_down: str  # down arrow vs v
    bullet: str  # bullet vs -
    cursor: str  # cursor vs >

    # Box-drawing characters
    box_vertical: str  # │ vs |
    box_horizontal: str  # ─ vs -
    box_double_horizontal: str  # ═ vs =

    # Diff-specific
    gutter_bar: str  # ▌ vs |

    # Status bar
    git_branch: str  # "↗" vs "git:"


UNICODE_GLYPHS = Glyphs(
    tool_prefix="⏺",
    ellipsis="…",
    checkmark="✓",
    error="✗",
    circle_empty="○",
    circle_filled="●",
    output_prefix="⎿",
    spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    pause="⏸",
    newline="⏎",
    warning="⚠",
    question="?",
    arrow_up="↑",
    arrow_down="↓",
    bullet="•",
    cursor="›",  # noqa: RUF001  # Intentional Unicode glyph
    # Box-drawing characters
    box_vertical="│",
    box_horizontal="─",
    box_double_horizontal="═",
    gutter_bar="▌",
    git_branch="↗",
)
"""전체 유니코드를 지원하는 터미널에 대한 문자 집합입니다."""

ASCII_GLYPHS = Glyphs(
    tool_prefix="(*)",
    ellipsis="...",
    checkmark="[OK]",
    error="[X]",
    circle_empty="[ ]",
    circle_filled="[*]",
    output_prefix="L",
    spinner_frames=("(-)", "(\\)", "(|)", "(/)"),
    pause="||",
    newline="\\n",
    warning="[!]",
    question="[?]",
    arrow_up="^",
    arrow_down="v",
    bullet="-",
    cursor=">",
    # Box-drawing characters
    box_vertical="|",
    box_horizontal="-",
    box_double_horizontal="=",
    gutter_bar="|",
    git_branch="git:",
)
"""7비트 ASCII로 제한된 터미널에 대한 문자 집합입니다."""

_glyphs_cache: Glyphs | None = None
"""감지된 문자 모양에 대한 모듈 수준 캐시입니다."""

_editable_cache: tuple[bool, str | None] | None = None
"""편집 가능한 설치 정보를 위한 모듈 수준 캐시: (is_editable, source_path)"""

_langsmith_url_cache: tuple[str, str] | None = None
"""성공적인 LangSmith 프로젝트 URL 조회를 위한 모듈 수준 캐시입니다."""

_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS = 2.0
"""LangSmith 프로젝트 URL 조회를 기다리는 최대 시간(초)입니다.

추적 메타데이터가 CLI 흐름을 중단하지 않도록 짧게 유지합니다.
"""


def _resolve_editable_info() -> tuple[bool, str | None]:
    """PEP 610 `direct_url.json`을 한 번 구문 분석하고 두 결과를 모두 캐시합니다.

Returns:
        (is_editable, contracted_source_path)의 튜플입니다. 경로가 사용자의 홈 디렉토리에 속할 경우 경로는 `~`로
        축소되고, 설치를 편집할 수 없거나 경로를 사용할 수 없는 경우에는 `None`입니다.

    """
    global _editable_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _editable_cache is not None:
        return _editable_cache

    editable = False
    path: str | None = None

    try:
        dist = distribution("deepagents-cli")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            editable = data.get("dir_info", {}).get("editable", False)
            if editable:
                url = data.get("url", "")
                if url.startswith("file://"):
                    path = unquote(urlparse(url).path)
                    home = str(Path.home())
                    if path.startswith(home):
                        path = "~" + path[len(home) :]
    except (PackageNotFoundError, FileNotFoundError, json.JSONDecodeError, TypeError):
        logger.debug(
            "Failed to read editable install info from PEP 610 metadata",
            exc_info=True,
        )

    _editable_cache = (editable, path)
    return _editable_cache


def _is_editable_install() -> bool:
    """deepagents-cli가 편집 가능 모드로 설치되어 있는지 확인하세요.

    PEP 610 `direct_url.json` 메타데이터를 사용하여 편집 가능한 설치를 감지합니다.

Returns:
        편집 가능 모드로 설치된 경우 `True`, 그렇지 않은 경우 `False`.

    """
    return _resolve_editable_info()[0]


def _get_editable_install_path() -> str | None:
    """편집 가능한 설치를 위해 `~` 계약 소스 디렉터리를 반환합니다.

    편집할 수 없는 설치의 경우 또는 경로를 결정할 수 없는 경우 `None`을 반환합니다.

    """
    return _resolve_editable_info()[1]


def _detect_charset_mode() -> CharsetMode:
    """터미널 문자 세트 기능을 자동 감지합니다.

Returns:
        환경 및 터미널 인코딩을 기반으로 감지된 CharsetMode입니다.

    """
    env_mode = os.environ.get("UI_CHARSET_MODE", "auto").lower()
    if env_mode == "unicode":
        return CharsetMode.UNICODE
    if env_mode == "ascii":
        return CharsetMode.ASCII

    # Auto: check stdout encoding and LANG
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return CharsetMode.UNICODE
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    if "utf" in lang.lower():
        return CharsetMode.UNICODE
    return CharsetMode.ASCII


def get_glyphs() -> Glyphs:
    """현재 문자 세트 모드에 대한 글리프 세트를 가져옵니다.

Returns:
        문자 세트 모드 감지를 기반으로 하는 적절한 Glyphs 인스턴스입니다.

    """
    global _glyphs_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _glyphs_cache is not None:
        return _glyphs_cache

    mode = _detect_charset_mode()
    _glyphs_cache = ASCII_GLYPHS if mode == CharsetMode.ASCII else UNICODE_GLYPHS
    return _glyphs_cache


def reset_glyphs_cache() -> None:
    """글리프 캐시를 재설정합니다(테스트용)."""
    global _glyphs_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _glyphs_cache = None


def is_ascii_mode() -> bool:
    """터미널이 ASCII 문자셋 모드인지 확인하세요.

    `_detect_charset_mode` 및 `CharsetMode`을 모두 가져오지 않고도 위젯이 문자 세트에서 분기할 수 있도록 하는 편리한
    래퍼입니다.

Returns:
        `True` 감지된 문자 세트 모드가 ASCII인 경우.

    """
    return _detect_charset_mode() == CharsetMode.ASCII


def newline_shortcut() -> str:
    """개행 키보드 단축키에 대한 플랫폼 고유 라벨을 반환합니다.

    macOS에서는 수정자 레이블을 "옵션"으로 지정하는 반면, 다른 플랫폼에서는 가장 안정적인 터미널 간 단축키로 Ctrl+J를 사용합니다.

Returns:
        사람이 읽을 수 있는 바로가기 문자열입니다. 예: `'Option+Enter'` 또는 `'Ctrl+J'`.

    """
    return "Option+Enter" if sys.platform == "darwin" else "Ctrl+J"


_UNICODE_BANNER = f"""
██████╗  ███████╗ ███████╗ ██████╗    ▄▓▓▄
██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗  ▓•███▙
██║  ██║ █████╗   █████╗   ██████╔╝  ░▀▀████▙▖
██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝      █▓████▙▖
██████╔╝ ███████╗ ███████╗ ██║          ▝█▓█████▙
╚═════╝  ╚══════╝ ╚══════╝ ╚═╝           ░▜█▓████▙
                                          ░█▀█▛▀▀▜▙▄
                                        ░▀░▀▒▛░░  ▝▀▘

 █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
                                                  v{__version__}
"""
_ASCII_BANNER = f"""
 ____  ____  ____  ____
|  _ \\| ___|| ___||  _ \\
| | | | |_  | |_  | |_) |
| |_| |  _| |  _| |  __/
|____/|____||____||_|

    _    ____  ____  _   _  _____  ____
   / \\  / ___|| ___|| \\ | ||_   _|/ ___|
  / _ \\| |  _ | |_  |  \\| |  | |  \\___ \\
 / ___ \\ |_| ||  _| | |\\  |  | |   ___) |
/_/   \\_\\____||____||_| \\_|  |_|  |____/
                                  v{__version__}
"""


def get_banner() -> str:
    """현재 문자 세트 모드에 적합한 배너를 가져옵니다.

Returns:
        텍스트 아트 배너 문자열(문자 세트 모드 기반 유니코드 또는 ASCII)입니다.

            편집 가능 모드로 설치된 경우 "(로컬)" 접미사가 포함됩니다.

    """
    if _detect_charset_mode() == CharsetMode.ASCII:
        banner = _ASCII_BANNER
    else:
        banner = _UNICODE_BANNER

    if _is_editable_install():
        banner = banner.replace(f"v{__version__}", f"v{__version__} (local)")

    return banner


MAX_ARG_LENGTH = 150
"""UI의 도구 인수 값에 대한 문자 제한입니다.

더 긴 값은 `tool_display`에서 `truncate_value`만큼 줄임표로 잘립니다.
"""

config: RunnableConfig = {
    "recursion_limit": 1000,
}
"""기본 LangGraph 실행 가능 구성입니다.

기본 LangGraph 최대 한도에 도달하지 않고 깊게 중첩된 에이전트 그래프를 수용하려면 `recursion_limit`을 1000으로 설정합니다.
"""

_git_branch_cache: dict[str, str | None] = {}
"""해결된 git 브랜치 이름의 cwd별 캐시입니다.

동일한 세션 내에서 반복되는 `git rev-parse` 하위 프로세스 호출을 방지합니다. `str(Path.cwd())`로 입력됨; `None` 값은
디렉터리가 git 저장소 내부에 없음을 나타냅니다.
"""


def _get_git_branch() -> str | None:
    """현재 git 브랜치 이름을 반환하거나 저장소에 없으면 `None`을 반환합니다."""
    import subprocess  # noqa: S404

    try:
        cwd = str(Path.cwd())
    except OSError:
        logger.debug("Could not determine cwd for git branch lookup", exc_info=True)
        return None
    if cwd in _git_branch_cache:
        return _git_branch_cache[cwd]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            branch = result.stdout.strip() or None
            _git_branch_cache[cwd] = branch
            return branch
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.debug("Could not determine git branch", exc_info=True)
    _git_branch_cache[cwd] = None
    return None


def build_stream_config(
    thread_id: str,
    assistant_id: str | None,
    *,
    sandbox_type: str | None = None,
) -> RunnableConfig:
    """LangGraph 스트림 구성 사전을 빌드합니다.

    CLI 및 SDK 버전을 `metadata["versions"]`에 삽입하여 LangSmith 추적을 특정 릴리스와 연관시킬 수 있습니다.

    CLI가 *두 버전*을 모두 설정하는 이유:

    * `create_deep_agent`은 `versions: {"deepagents": "X.Y.Z"}`을(를)
        `with_config`을 통해 컴파일된 그래프. 스트림 시간에 LangGraph는 그래프 구성을 여기에 전달된 런타임 구성과 병합합니다.
        메타데이터 병합이 얕기 때문에(최상위 키의 경우 사실상 `{**graph_meta, **runtime_meta}`) `versions` 키를
        포함하는 두 구성 모두 런타임 dict가 그래프 dict를 완전히 **대체**하므로 SDK 버전이 손실됩니다.
    * 여기에 SDK 버전을 포함하면 병합 후에도 유지됩니다.

    CLI에서 시작되는 LangSmith 추적이 기본 SDK 사용과 구별될 수 있도록 `ls_integration` 메타데이터를 포함합니다.

Args:
        thread_id: CLI 세션 스레드 식별자입니다.
        assistant_id: 에이전트/보조자 식별자입니다(있는 경우).
        sandbox_type: 추적 메타데이터에 대한 샌드박스 제공자 이름 또는 활성화된 샌드박스가 없는 경우 `None`입니다.

Returns:
        `configurable` 및 `metadata` 키를 사용하여 사전을 구성합니다.

    """
    import contextlib
    import importlib.metadata as importlib_metadata
    from datetime import UTC, datetime

    try:
        cwd = str(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory", exc_info=True)
        cwd = ""

    # Include SDK version alongside CLI version — see docstring for why.
    versions: dict[str, str] = {"deepagents-cli": __version__}
    with contextlib.suppress(importlib_metadata.PackageNotFoundError):
        versions["deepagents"] = importlib_metadata.version("deepagents")

    metadata: dict[str, Any] = {
        "versions": versions,
        "ls_integration": "deepagents-cli",
    }
    from deepagents_cli._env_vars import USER_ID

    user_id = os.environ.get(USER_ID)
    if user_id:
        metadata["user_id"] = user_id
    if cwd:
        metadata["cwd"] = cwd
    if assistant_id:
        metadata.update(
            {
                "assistant_id": assistant_id,
                "agent_name": assistant_id,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
    branch = _get_git_branch()
    if branch:
        metadata["git_branch"] = branch
    if sandbox_type and sandbox_type != "none":
        metadata["sandbox_type"] = sandbox_type
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
    }


class _ShellAllowAll(list):  # noqa: FURB189  # sentinel type, not a general-purpose list subclass
    """무제한 셸 액세스를 위한 Sentinel 하위 클래스입니다.

    일반 목록 대신 전용 유형을 사용하면 소비자는 ID 확인(`is`)과 달리 직렬화/복사 후에도 유지되는 `isinstance` 확인을 사용할 수
    있습니다.

    """


SHELL_ALLOW_ALL: list[str] = _ShellAllowAll(["__ALL__"])
"""`--shell-allow-list=all`에 대해 `parse_shell_allow_list`에서 반환된 센티널 값입니다."""


def parse_shell_allow_list(allow_list_str: str | None) -> list[str] | None:
    """문자열에서 셸 허용 목록을 구문 분석합니다.

Args:
        allow_list_str: 쉼표로 구분된 명령 목록입니다. 안전한 기본값의 경우 `'recommended'`, 모든 명령을 허용하려면
                        `'all'`입니다.

            `'all'`은 유일한 값이어야 합니다. `'recommended'`과 달리 쉼표로 구분된 목록 내에서는 인식되지 않습니다.

            사용자 정의 명령과 병합하기 위해 목록에 `'recommended'`을 포함할 수도 있습니다.

Returns:
        허용되는 명령 목록, `'all'`이 지정된 경우 `SHELL_ALLOW_ALL`,
            또는 허용 목록이 구성되지 않은 경우 `None`입니다.

Raises:
        ValueError: `'all'`이 다른 명령과 결합된 경우.

    """
    if not allow_list_str:
        return None

    # Special value 'all' allows any shell command
    if allow_list_str.strip().lower() == "all":
        return SHELL_ALLOW_ALL

    # Special value 'recommended' uses our curated safe list
    if allow_list_str.strip().lower() == "recommended":
        return list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    # Split by comma and strip whitespace
    commands = [cmd.strip() for cmd in allow_list_str.split(",") if cmd.strip()]

    # Reject ambiguous input: 'all' mixed with other commands
    if any(cmd.lower() == "all" for cmd in commands):
        msg = (
            "Cannot combine 'all' with other commands in --shell-allow-list. "
            "Use '--shell-allow-list all' alone to allow any command."
        )
        raise ValueError(msg)

    # If "recommended" is in the list, merge with recommended commands
    result = []
    for cmd in commands:
        if cmd.lower() == "recommended":
            result.extend(RECOMMENDED_SAFE_SHELL_COMMANDS)
        else:
            result.append(cmd)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cmd in result:
        if cmd not in seen:
            seen.add(cmd)
            unique.append(cmd)
    return unique


def _read_config_toml_skills_dirs() -> list[str] | None:
    """`~/.deepagents/config.toml`에서 `[skills].extra_allowed_dirs`을(를) 읽습니다.

Returns:
        경로 문자열 목록 또는 키가 없거나 파일이 있는 경우 `None`
            읽을 수 없습니다.

    """
    import tomllib

    from deepagents_cli.model_config import DEFAULT_CONFIG_PATH

    try:
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return None
    except (PermissionError, OSError, tomllib.TOMLDecodeError):
        logger.warning(
            "Could not read skills config from %s",
            DEFAULT_CONFIG_PATH,
            exc_info=True,
        )
        return None

    skills_section = data.get("skills", {})
    dirs = skills_section.get("extra_allowed_dirs")
    if isinstance(dirs, list):
        return dirs
    return None


def _parse_extra_skills_dirs(
    env_raw: str | None,
    config_toml_dirs: list[str] | None = None,
) -> list[Path] | None:
    """env var 및 config.toml의 추가 기술 디렉터리를 병합합니다.

    추가 기술 디렉터리는 해결된 기술 경로가 신뢰할 수 있는 루트 내에 있는지 확인하기 위해 `load_skill_content`에서 사용하는 격리 허용
    목록을 확장합니다. 새로운 기술 검색 위치를 추가하지 **않습니다** — 기술은 여전히 ​​표준 디렉터리에서만 검색됩니다. 이는 표준 기술 디렉터리
    내부의 심볼릭 링크가 경로 포함 검사에서 거부되지 않고 사용자가 지정한 위치의 대상을 합법적으로 가리킬 수 있도록 하기 위해 존재합니다.

    env var(`DEEPAGENTS_CLI_EXTRA_SKILLS_DIRS`, 콜론으로 구분)가 우선합니다. 설정되면 `config.toml` 값이
    무시됩니다.

Args:
        env_raw: `DEEPAGENTS_CLI_EXTRA_SKILLS_DIRS`(콜론으로 구분) 값 또는 설정되지 않은 경우 `None`입니다.
        config_toml_dirs: `~/.deepagents/config.toml`에 있는 `[skills].extra_allowed_dirs`의
                          경로 문자열 목록입니다.

Returns:
        해결된 `Path` 객체 목록 또는 구성되지 않은 경우 `None`.

    """
    # Env var takes precedence when set
    if env_raw:
        dirs = [
            Path(p.strip()).expanduser().resolve()
            for p in env_raw.split(":")
            if p.strip()
        ]
        return dirs or None

    if config_toml_dirs:
        dirs = [
            Path(p).expanduser().resolve()
            for p in config_toml_dirs
            if isinstance(p, str) and p.strip()
        ]
        return dirs or None

    return None


@dataclass
class Settings:
    """deepagents-cli에 대한 전역 설정 및 환경 감지.

    이 클래스는 시작 시 한 번 초기화되며 다음에 대한 액세스를 제공합니다. - 사용 가능한 모델 및 API 키 - 현재 프로젝트 정보 - 도구
    가용성(예: Tavily) - 파일 시스템 경로

    """

    openai_api_key: str | None
    """OpenAI API 키(사용 가능한 경우)"""

    anthropic_api_key: str | None
    """가능한 경우 Anthropic API 키입니다."""

    google_api_key: str | None
    """사용 가능한 경우 Google API 키입니다."""

    nvidia_api_key: str | None
    """NVIDIA API 키(사용 가능한 경우)"""

    tavily_api_key: str | None
    """사용 가능한 경우 Tavily API 키입니다."""

    google_cloud_project: str | None
    """VertexAI 인증을 위한 Google Cloud 프로젝트 ID입니다."""

    deepagents_langchain_project: str | None
    """deepagents 에이전트 추적을 위한 LangSmith 프로젝트 이름입니다."""

    user_langchain_project: str | None
    """환경의 원본 `LANGSMITH_PROJECT`(사용자 코드용)"""

    model_name: str | None = None
    """현재 활성 모델 이름으로, 모델 생성 후 설정됩니다."""

    model_provider: str | None = None
    """제공자 식별자(예: `openai`, `anthropic`, `google_genai`)."""

    model_context_limit: int | None = None
    """모델 프로필의 최대 입력 토큰 수입니다."""

    model_unsupported_modalities: frozenset[str] = frozenset()
    """모델 프로필에서 지원되는 것으로 표시되지 않은 입력 양식입니다."""

    project_root: Path | None = None
    """현재 프로젝트 루트 디렉터리 또는 git 프로젝트에 없는 경우 `None`입니다."""

    shell_allow_list: list[str] | None = None
    """사용자 승인이 필요하지 않은 셸 명령입니다."""

    extra_skills_dirs: list[Path] | None = None
    """기술 경로 격리 허용 목록에 추가 디렉터리가 추가되었습니다.

    이는 새로운 기술 발견 위치를 추가하지 않습니다. 기술은 여전히 ​​표준 디렉터리에서만 발견됩니다. 표준 기술 디렉터리 내부의 심볼릭 링크가
    `load_skill_content`의 격리 검사에서 거부되지 않고 이러한 추가 위치의 대상을 가리킬 수 있도록 존재합니다.

    `DEEPAGENTS_CLI_EXTRA_SKILLS_DIRS` env var(콜론으로 구분) 또는 `~/.deepagents/config.toml`의
    `[skills].extra_allowed_dirs`을 통해 설정합니다.

    """

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> Settings:
        """현재 환경을 감지하여 설정을 만듭니다.

Args:
            start_path: 프로젝트 감지를 시작할 디렉터리(기본값은 cwd)

Returns:
            감지된 구성이 있는 설정 인스턴스

        """
        # Detect API keys (normalize empty strings to None).
        from deepagents_cli.model_config import resolve_env_var

        openai_key = resolve_env_var("OPENAI_API_KEY")
        anthropic_key = resolve_env_var("ANTHROPIC_API_KEY")
        google_key = resolve_env_var("GOOGLE_API_KEY")
        nvidia_key = resolve_env_var("NVIDIA_API_KEY")
        tavily_key = resolve_env_var("TAVILY_API_KEY")
        google_cloud_project = resolve_env_var("GOOGLE_CLOUD_PROJECT")

        # Detect LangSmith configuration
        # DEEPAGENTS_CLI_LANGSMITH_PROJECT: Project for deepagents agent tracing
        # user_langchain_project: User's ORIGINAL LANGSMITH_PROJECT (before override)
        # When accessed via the module-level `settings` singleton,
        # _ensure_bootstrap() has already run and may have overridden
        # LANGSMITH_PROJECT. We use the saved original value, not the
        # current os.environ value. Direct callers should ensure
        # bootstrap has run if they depend on the override.
        from deepagents_cli._env_vars import (
            EXTRA_SKILLS_DIRS,
            LANGSMITH_PROJECT,
            SHELL_ALLOW_LIST,
        )

        deepagents_langchain_project = resolve_env_var(LANGSMITH_PROJECT)
        user_langchain_project = _original_langsmith_project  # Use saved original!

        # Detect project
        from deepagents_cli.project_utils import find_project_root

        project_root = find_project_root(start_path)

        # Parse shell command allow-list from environment
        # Format: comma-separated list of commands (e.g., "ls,cat,grep,pwd")

        shell_allow_list_str = os.environ.get(SHELL_ALLOW_LIST)
        shell_allow_list = parse_shell_allow_list(shell_allow_list_str)

        # Parse extra skill containment roots from env var or config.toml.
        # These extend the path allowlist for load_skill_content but do not
        # add new skill discovery locations.
        extra_skills_dirs = _parse_extra_skills_dirs(
            os.environ.get(EXTRA_SKILLS_DIRS),
            _read_config_toml_skills_dirs(),
        )

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            nvidia_api_key=nvidia_key,
            tavily_api_key=tavily_key,
            google_cloud_project=google_cloud_project,
            deepagents_langchain_project=deepagents_langchain_project,
            user_langchain_project=user_langchain_project,
            project_root=project_root,
            shell_allow_list=shell_allow_list,
            extra_skills_dirs=extra_skills_dirs,
        )

    def reload_from_environment(self, *, start_path: Path | None = None) -> list[str]:
        """환경 변수 및 프로젝트 파일에서 선택한 설정을 다시 로드합니다.

        런타임 시 변경될 것으로 예상되는 필드(API 키, Google Cloud 프로젝트, 프로젝트 루트, 셸 허용 목록, LangSmith 추적
        프로젝트)만 새로고침됩니다.

        런타임 모델 상태(`model_name`, `model_provider`, `model_context_limit`)와 원래 사용자
        LangSmith 프로젝트(`user_langchain_project`)는 의도적으로 보존됩니다. 즉, `reloadable_fields`에
        없으며 이 방법으로 건드리지 않습니다.

        !!! 메모

            `.env` 파일은 `override=False`을 사용하여 로드되므로 쉘에서 내보낸 변수가 항상 우선합니다.  `.env`에서 쉘에서
            내보낸 키를 재정의하려면 `DEEPAGENTS_CLI_` 접두사(예: `DEEPAGENTS_CLI_OPENAI_API_KEY`)를
            사용하세요.

Args:
            start_path: 프로젝트 검색을 시작할 디렉터리입니다(기본값은 cwd).

Returns:
            사람이 읽을 수 있는 변경 설명 목록입니다.

        """
        _load_dotenv(start_path=start_path)

        api_key_fields = {
            "openai_api_key",
            "anthropic_api_key",
            "google_api_key",
            "nvidia_api_key",
            "tavily_api_key",
        }
        """API 키를 보유하는 필드 — 변경 보고서의 값을 마스킹하는 데 사용됩니다.
        따라서 비밀은 일반 텍스트로 기록되지 않습니다.
        """

        reloadable_fields = (
            "openai_api_key",
            "anthropic_api_key",
            "google_api_key",
            "nvidia_api_key",
            "tavily_api_key",
            "google_cloud_project",
            "deepagents_langchain_project",
            "project_root",
            "shell_allow_list",
            "extra_skills_dirs",
        )
        """`/reload`에 필드가 새로 고쳐졌습니다.

        런타임 모델 상태(`model_name`, `model_provider`, `model_context_limit`) 및 원래 사용자
        LangSmith 프로젝트는 의도적으로 제외됩니다. 이는 한 번 설정되며 다시 로드할 때 변경되어서는 안 됩니다.

        """

        previous = {field: getattr(self, field) for field in reloadable_fields}

        from deepagents_cli._env_vars import (
            EXTRA_SKILLS_DIRS,
            LANGSMITH_PROJECT,
            SHELL_ALLOW_LIST,
        )

        try:
            shell_allow_list = parse_shell_allow_list(os.environ.get(SHELL_ALLOW_LIST))
        except ValueError:
            logger.warning(
                "Invalid %s during reload; keeping previous value",
                SHELL_ALLOW_LIST,
            )
            shell_allow_list = previous["shell_allow_list"]

        try:
            from deepagents_cli.project_utils import find_project_root

            project_root = find_project_root(start_path)
        except OSError:
            logger.warning(
                "Could not detect project root during reload; keeping previous value"
            )
            project_root = previous["project_root"]

        from deepagents_cli.model_config import resolve_env_var

        refreshed = {
            "openai_api_key": resolve_env_var("OPENAI_API_KEY"),
            "anthropic_api_key": resolve_env_var("ANTHROPIC_API_KEY"),
            "google_api_key": resolve_env_var("GOOGLE_API_KEY"),
            "nvidia_api_key": resolve_env_var("NVIDIA_API_KEY"),
            "tavily_api_key": resolve_env_var("TAVILY_API_KEY"),
            "google_cloud_project": resolve_env_var("GOOGLE_CLOUD_PROJECT"),
            "deepagents_langchain_project": resolve_env_var(LANGSMITH_PROJECT),
            "project_root": project_root,
            "shell_allow_list": shell_allow_list,
            "extra_skills_dirs": _parse_extra_skills_dirs(
                os.environ.get(EXTRA_SKILLS_DIRS),
                _read_config_toml_skills_dirs(),
            ),
        }

        for field, value in refreshed.items():
            setattr(self, field, value)

        # Sync the LANGSMITH_PROJECT env var so LangSmith tracing picks up
        # the change
        new_project = refreshed["deepagents_langchain_project"]
        if new_project:
            os.environ["LANGSMITH_PROJECT"] = new_project
        elif previous["deepagents_langchain_project"]:
            # Override was previously active but new value is unset; restore.
            if _original_langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = _original_langsmith_project
            else:
                os.environ.pop("LANGSMITH_PROJECT", None)

        def _display(field: str, value: object) -> str:
            if field in api_key_fields:
                return "set" if value else "unset"
            return str(value)

        changes: list[str] = []
        for field in reloadable_fields:
            old_value = previous[field]
            new_value = refreshed[field]
            if old_value != new_value:
                changes.append(
                    f"{field}: {_display(field, old_value)} -> "
                    f"{_display(field, new_value)}"
                )
        return changes

    @property
    def has_openai(self) -> bool:
        """OpenAI API 키가 구성되어 있는지 확인하세요."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Anthropic API 키가 구성되어 있는지 확인하세요."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Google API 키가 구성되어 있는지 확인하세요."""
        return self.google_api_key is not None

    @property
    def has_nvidia(self) -> bool:
        """NVIDIA API 키가 구성되어 있는지 확인하세요."""
        return self.nvidia_api_key is not None

    @property
    def has_vertex_ai(self) -> bool:
        """VertexAI를 사용할 수 있는지 확인하세요(Google Cloud 프로젝트 세트, API 키 없음).

        VertexAI는 인증을 위해 ADC(애플리케이션 기본 자격 증명)를 사용하므로 GOOGLE_CLOUD_PROJECT가 설정되고
        GOOGLE_API_KEY가 설정되지 않은 경우 VertexAI로 가정합니다.

        """
        return self.google_cloud_project is not None and self.google_api_key is None

    @property
    def has_tavily(self) -> bool:
        """Tavily API 키가 구성되어 있는지 확인하세요."""
        return self.tavily_api_key is not None

    @property
    def user_deepagents_dir(self) -> Path:
        """기본 사용자 수준 .deepagents 디렉터리를 가져옵니다.

Returns:
            ~/.deepagents 경로

        """
        return Path.home() / ".deepagents"

    @staticmethod
    def get_user_agent_md_path(agent_name: str) -> Path:
        """특정 에이전트에 대한 사용자 수준 AGENTS.md 경로를 가져옵니다.

        파일 존재 여부에 관계없이 경로를 반환합니다.

Args:
            agent_name: 대리인의 이름

Returns:
            ~/.deepagents/{agent_name}/AGENTS.md 경로

        """
        return Path.home() / ".deepagents" / agent_name / "AGENTS.md"

    def get_project_agent_md_path(self) -> list[Path]:
        """프로젝트 수준 AGENTS.md 경로를 가져옵니다.

        `{project_root}/.deepagents/AGENTS.md` 및 `{project_root}/AGENTS.md`을 모두 확인하여
        존재하는 모든 항목을 반환합니다. 둘 다 존재하는 경우 둘 다 로드되고 해당 명령이 먼저 `.deepagents/AGENTS.md`과
        결합됩니다.

Returns:
            기존 AGENTS.md 경로.

                파일이 둘 다 없거나 프로젝트에 없는 경우 비어 있고, 하나만 있는 경우 항목이 하나이고, 두 위치 모두에 파일이 있는 경우
                항목이 두 개입니다.

        """
        if not self.project_root:
            return []
        from deepagents_cli.project_utils import find_project_agent_md

        return find_project_agent_md(self.project_root)

    @staticmethod
    def _is_valid_agent_name(agent_name: str) -> bool:
        """잘못된 파일 시스템 경로 및 보안 문제를 방지하려면 유효성을 검사하십시오.

Returns:
            에이전트 이름이 유효하면 True이고, 그렇지 않으면 False입니다.

        """
        if not agent_name or not agent_name.strip():
            return False
        # Allow only alphanumeric, hyphens, underscores, and whitespace
        return bool(re.match(r"^[a-zA-Z0-9_\-\s]+$", agent_name))

    def get_agent_dir(self, agent_name: str) -> Path:
        """글로벌 에이전트 디렉터리 경로를 가져옵니다.

Args:
            agent_name: 대리인의 이름

Returns:
            ~/.deepagents/{agent_name} 경로

Raises:
            ValueError: 에이전트 이름에 잘못된 문자가 포함된 경우.

        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        return Path.home() / ".deepagents" / agent_name

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """글로벌 에이전트 디렉터리가 있는지 확인하고 해당 경로를 반환합니다.

Args:
            agent_name: 대리인의 이름

Returns:
            ~/.deepagents/{agent_name} 경로

Raises:
            ValueError: 에이전트 이름에 잘못된 문자가 포함된 경우.

        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        agent_dir = self.get_agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def get_user_skills_dir(self, agent_name: str) -> Path:
        """특정 에이전트에 대한 사용자 수준 기술 디렉터리 경로를 가져옵니다.

Args:
            agent_name: 대리인의 이름

Returns:
            ~/.deepagents/{agent_name}/skills/ 경로

        """
        return self.get_agent_dir(agent_name) / "skills"

    def ensure_user_skills_dir(self, agent_name: str) -> Path:
        """사용자 수준 기술 디렉터리가 있는지 확인하고 해당 경로를 반환합니다.

Args:
            agent_name: 대리인의 이름

Returns:
            ~/.deepagents/{agent_name}/skills/ 경로

        """
        skills_dir = self.get_user_skills_dir(agent_name)
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_project_skills_dir(self) -> Path | None:
        """프로젝트 수준 기술 디렉터리 경로를 가져옵니다.

Returns:
            {project_root}/.deepagents/skills/ 경로 또는 프로젝트에 없는 경우 없음

        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "skills"

    def ensure_project_skills_dir(self) -> Path | None:
        """프로젝트 수준 기술 디렉터리가 있는지 확인하고 해당 경로를 반환합니다.

Returns:
            {project_root}/.deepagents/skills/ 경로 또는 프로젝트에 없는 경우 없음

        """
        if not self.project_root:
            return None
        skills_dir = self.get_project_skills_dir()
        if skills_dir is None:
            return None
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_user_agents_dir(self, agent_name: str) -> Path:
        """사용자 정의 하위 에이전트 정의에 대한 사용자 수준 에이전트 디렉터리 경로를 가져옵니다.

Args:
            agent_name: CLI 에이전트의 이름(예: "deepagents")

Returns:
            ~/.deepagents/{agent_name}/agents/ 경로

        """
        return self.get_agent_dir(agent_name) / "agents"

    def get_project_agents_dir(self) -> Path | None:
        """사용자 정의 하위 에이전트 정의에 대한 프로젝트 수준 에이전트 디렉터리 경로를 가져옵니다.

Returns:
            {project_root}/.deepagents/agents/ 경로 또는 프로젝트에 없는 경우 없음

        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "agents"

    @property
    def user_agents_dir(self) -> Path:
        """기본 사용자 수준 `.agents` 디렉터리(`~/.agents`)를 가져옵니다.

Returns:
            `~/.agents` 경로

        """
        return Path.home() / ".agents"

    def get_user_agent_skills_dir(self) -> Path:
        """사용자 수준 `~/.agents/skills/` 디렉터리를 가져옵니다.

        이는 도구에 구애받지 않는 기술에 대한 일반적인 별칭 경로입니다.

Returns:
            `~/.agents/skills/` 경로

        """
        return self.user_agents_dir / "skills"

    def get_project_agent_skills_dir(self) -> Path | None:
        """프로젝트 수준 `.agents/skills/` 디렉터리를 가져옵니다.

        이는 도구에 구애받지 않는 기술에 대한 일반적인 별칭 경로입니다.

Returns:
            `{project_root}/.agents/skills/` 경로 또는 프로젝트에 없는 경우 `None` 경로

        """
        if not self.project_root:
            return None
        return self.project_root / ".agents" / "skills"

    @staticmethod
    def get_user_claude_skills_dir() -> Path:
        """사용자 수준 `~/.claude/skills/` 디렉터리를 가져옵니다(실험적).

        Claude Code와 교차 도구 기술 공유를 위한 편의 다리입니다. 이는 실험적이므로 삭제될 수 있습니다.

Returns:
            `~/.claude/skills/` 경로

        """
        return Path.home() / ".claude" / "skills"

    def get_project_claude_skills_dir(self) -> Path | None:
        """프로젝트 수준 `.claude/skills/` 디렉터리를 가져옵니다(실험적).

        Claude Code와 교차 도구 기술 공유를 위한 편의 다리입니다. 이는 실험적이므로 삭제될 수 있습니다.

Returns:
            `{project_root}/.claude/skills/` 경로 또는 프로젝트에 없는 경우 `None` 경로입니다.

        """
        if not self.project_root:
            return None
        return self.project_root / ".claude" / "skills"

    @staticmethod
    def get_built_in_skills_dir() -> Path:
        """CLI와 함께 제공되는 기본 제공 기술이 포함된 디렉터리를 가져옵니다.

Returns:
            패키지 내의 `built_in_skills/` 디렉터리 경로입니다.

        """
        return Path(__file__).parent / "built_in_skills"

    def get_extra_skills_dirs(self) -> list[Path]:
        """사용자가 구성한 추가 기술 디렉터리를 가져옵니다.

        `DEEPAGENTS_CLI_EXTRA_SKILLS_DIRS`(콜론으로 구분된 경로) 또는 `~/.deepagents/config.toml`의
        `[skills].extra_allowed_dirs`을 통해 설정합니다.

Returns:
            추가 기술 디렉터리 경로 목록 또는 구성되지 않은 경우 빈 목록입니다.

        """
        return self.extra_skills_dirs or []


class SessionState:
    """앱, 어댑터 및 에이전트 전체에서 공유되는 변경 가능한 세션 상태입니다.

    세션 중에 키 바인딩이나 HITL 승인 메뉴의 '모두 자동 승인' 옵션을 통해 전환할 수 있는 자동 승인과 같은 런타임 플래그를 추적합니다.

    `auto_approve` 플래그는 도구 호출(셸 실행, 파일 쓰기/편집, 웹 검색, URL 가져오기)을 실행하기 전에 사용자 확인이 필요한지 여부를
    제어합니다.

    """

    def __init__(self, auto_approve: bool = False, no_splash: bool = False) -> None:
        """선택적 플래그를 사용하여 세션 상태를 초기화합니다.

Args:
            auto_approve: 메시지를 표시하지 않고 도구 호출을 자동 승인할지 여부입니다.

                Shift+Tab 또는 HITL 승인 메뉴를 통해 런타임에 전환할 수 있습니다.
            no_splash: 시작 시 스플래시 화면 표시를 건너뛸지 여부입니다.

        """
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.exit_hint_until: float | None = None
        self.exit_hint_handle = None
        from deepagents_cli.sessions import generate_thread_id

        self.thread_id = generate_thread_id()

    def toggle_auto_approve(self) -> bool:
        """자동 승인을 전환하고 새 상태를 반환합니다.

        텍스트 앱에서 Shift+Tab 키 바인딩으로 호출됩니다.

        자동 승인이 켜져 있으면 모든 도구 호출이 메시지 없이 실행됩니다.

Returns:
            전환 후 새로운 `auto_approve` 상태.

        """
        self.auto_approve = not self.auto_approve
        return self.auto_approve


SHELL_TOOL_NAMES: frozenset[str] = frozenset({"bash", "shell", "execute"})
"""쉘/명령 실행 도구로 인식되는 도구 이름입니다.

실제로 SDK 및 CLI 백엔드에는 `'execute'`만 등록됩니다. `'bash'` 및 `'shell'`은 이전 버전과 호환되는 별칭으로 이전되고 유지되는
레거시 이름입니다.
"""

DANGEROUS_SHELL_PATTERNS = (
    "$(",  # Command substitution
    "`",  # Backtick command substitution
    "$'",  # ANSI-C quoting (can encode dangerous chars via escape sequences)
    "\n",  # Newline (command injection)
    "\r",  # Carriage return (command injection)
    "\t",  # Tab (can be used for injection in some shells)
    "<(",  # Process substitution (input)
    ">(",  # Process substitution (output)
    "<<<",  # Here-string
    "<<",  # Here-doc (can embed commands)
    ">>",  # Append redirect
    ">",  # Output redirect
    "<",  # Input redirect
    "${",  # Variable expansion with braces (can run commands via ${var:-$(cmd)})
)
"""쉘 주입 위험을 나타내는 리터럴 하위 문자열입니다.

기본 명령이 허용 목록에 있는 경우에도 리디렉션, 대체 연산자 또는 제어 문자를 통해 임의 실행을 포함하는 명령을 거부하기 위해
`contains_dangerous_patterns`에서 사용됩니다.
"""

RECOMMENDED_SAFE_SHELL_COMMANDS = (
    # Directory listing
    "ls",
    "dir",
    # File content viewing (read-only)
    "cat",
    "head",
    "tail",
    # Text searching (read-only)
    "grep",
    "wc",
    "strings",
    # Text processing (read-only, no shell execution)
    "cut",
    "tr",
    "diff",
    "md5sum",
    "sha256sum",
    # Path utilities
    "pwd",
    "which",
    # System info (read-only)
    "uname",
    "hostname",
    "whoami",
    "id",
    "groups",
    "uptime",
    "nproc",
    "lscpu",
    "lsmem",
    # Process viewing (read-only)
    "ps",
)
"""읽기 전용 명령은 비대화형 모드에서 자동 승인됩니다.

리더와 포맷터만 포함됩니다. 셸, 편집기, 인터프리터, 패키지 관리자, 네트워크 도구, 아카이버 및 GTFOBins/LOOBins의 모든 항목은 의도적으로
제외됩니다. 파일 쓰기 및 주입 벡터는 `DANGEROUS_SHELL_PATTERNS`에 의해 별도로 차단됩니다.
"""


def contains_dangerous_patterns(command: str) -> bool:
    """명령에 위험한 쉘 패턴이 포함되어 있는지 확인하십시오.

    이러한 패턴은 안전해 보이는 명령 내에 임의의 명령을 삽입하여 허용 목록 유효성 검사를 우회하는 데 사용될 수 있습니다. 검사에는 리터럴 하위 문자열
    패턴(리디렉션, 대체 연산자 등)과 단순 변수 확장(`$VAR`) 및 백그라운드 연산자(`&`)에 대한 정규식 패턴이 모두 포함됩니다.

Args:
        command: 확인할 쉘 명령입니다.

Returns:
        위험한 패턴이 발견되면 True이고, 그렇지 않으면 False입니다.

    """
    if any(pattern in command for pattern in DANGEROUS_SHELL_PATTERNS):
        return True

    # Bare variable expansion ($VAR without braces) can leak sensitive paths.
    # We already block ${ and $( above; this catches plain $HOME, $IFS, etc.
    if re.search(r"\$[A-Za-z_]", command):
        return True

    # Standalone & (background execution) changes the execution model and
    # should not be allowed.  We check for & that is NOT part of &&.
    return bool(re.search(r"(?<![&])&(?![&])", command))


def is_shell_command_allowed(command: str, allow_list: list[str] | None) -> bool:
    """쉘 명령이 허용 목록에 있는지 확인하십시오.

    허용 목록은 명령의 첫 번째 토큰(실행 파일 이름)과 일치합니다. 이를 통해 ls, cat, grep 등과 같은 읽기 전용 명령을 자동 승인할 수
    있습니다.

    `allow_list`이 `SHELL_ALLOW_ALL` 센티널인 경우 비어 있지 않은 모든 명령은 무조건 승인됩니다. 위험한 패턴 검사는 건너뜁니다.

    보안: 일반 허용 목록의 경우 이 기능은 허용 목록을 우회할 수 있는 주입 공격을 방지하기 위해 구문 분석 전에 위험한 셸 패턴(명령 대체, 리디렉션,
    프로세스 대체 등)이 포함된 명령을 거부합니다.

Args:
        command: 확인할 전체 쉘 명령입니다.
        allow_list: 허용되는 명령 이름 목록(예: `["ls", "cat", "grep"]`), 모든 명령을 허용하는
                    `SHELL_ALLOW_ALL` 센티널 또는 `None`.

Returns:
        명령이 허용되면 `True`, 그렇지 않으면 `False`입니다.

    """
    if not allow_list or not command or not command.strip():
        return False

    # SHELL_ALLOW_ALL sentinel — skip pattern and token checks
    if isinstance(allow_list, _ShellAllowAll):
        return True

    # SECURITY: Check for dangerous patterns BEFORE any parsing
    # This prevents injection attacks like: ls "$(rm -rf /)"
    if contains_dangerous_patterns(command):
        return False

    allow_set = set(allow_list)

    # Extract the first command token
    # Handle pipes and other shell operators by checking each command in the pipeline
    # Split by compound operators first (&&, ||), then single-char operators (|, ;).
    # Note: standalone & (background) is blocked by contains_dangerous_patterns above.
    segments = re.split(r"&&|\|\||[|;]", command)

    # Track if we found at least one valid command
    found_command = False

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        try:
            # Try to parse as shell command to extract the executable name
            tokens = shlex.split(segment)
            if tokens:
                found_command = True
                cmd_name = tokens[0]
                # Check if this command is in the allow set
                if cmd_name not in allow_set:
                    return False
        except ValueError:
            # If we can't parse it, be conservative and require approval
            return False

    # All segments are allowed (and we found at least one command)
    return found_command


def get_langsmith_project_name() -> str | None:
    """추적이 구성된 경우 LangSmith 프로젝트 이름을 확인합니다.

    필수 API 키 및 추적 환경 변수를 확인합니다. 둘 다 존재하는 경우 우선순위가
    `settings.deepagents_langchain_project`(`DEEPAGENTS_CLI_LANGSMITH_PROJECT`에서), 환경에서
    `LANGSMITH_PROJECT`(참고: `DEEPAGENTS_CLI_LANGSMITH_PROJECT`과 일치하도록 부트스트랩 시간에 이미
    재정의되었을 수 있음), `'deepagents-cli'`로 프로젝트 이름을 확인합니다.

Returns:
        LangSmith 추적이 활성화된 경우 프로젝트 이름 문자열이고, 그렇지 않은 경우 없음입니다.

    """
    from deepagents_cli.model_config import resolve_env_var

    langsmith_key = resolve_env_var("LANGSMITH_API_KEY") or resolve_env_var(
        "LANGCHAIN_API_KEY"
    )
    langsmith_tracing = resolve_env_var("LANGSMITH_TRACING") or resolve_env_var(
        "LANGCHAIN_TRACING_V2"
    )
    if not (langsmith_key and langsmith_tracing):
        return None

    return (
        _get_settings().deepagents_langchain_project
        or os.environ.get("LANGSMITH_PROJECT")
        or "deepagents-cli"
    )


def fetch_langsmith_project_url(project_name: str) -> str | None:
    """LangSmith 클라이언트를 통해 LangSmith 프로젝트 URL을 가져옵니다.

    성공적인 결과는 모듈 수준에서 캐시되므로 반복 호출로 인해 추가 네트워크 요청이 발생하지 않습니다.

    네트워크 호출은 `_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS`의 하드 타임아웃으로 데몬 스레드에서 실행되므로 이 함수는
    LangSmith에 연결할 수 없는 경우에도 최대 해당 기간 동안 호출 스레드를 차단합니다.

    `langsmith` 패키지 누락, 네트워크 오류, 잘못된 프로젝트 이름, 클라이언트 초기화 문제 또는 시간 초과 등 오류가 발생하면 None(디버그
    로그와 함께)을 반환합니다.

Args:
        project_name: 조회할 LangSmith 프로젝트 이름입니다.

Returns:
        프로젝트 URL 문자열이 발견되면, 그렇지 않으면 없음.

    """
    global _langsmith_url_cache  # noqa: PLW0603  # Module-level cache requires global statement

    if _langsmith_url_cache is not None:
        cached_name, cached_url = _langsmith_url_cache
        if cached_name == project_name:
            return cached_url
        # Different project name — fall through to fetch.

    try:
        from langsmith import Client
    except ImportError:
        logger.debug(
            "Could not fetch LangSmith project URL for '%s'",
            project_name,
            exc_info=True,
        )
        return None

    result: str | None = None
    lookup_error: Exception | None = None
    done = threading.Event()

    def _lookup_url() -> None:
        nonlocal result, lookup_error
        try:
            from deepagents_cli.model_config import resolve_env_var

            # Explicit api_key because Client() reads os.environ directly
            # and doesn't know about the DEEPAGENTS_CLI_ prefix.
            api_key = resolve_env_var("LANGSMITH_API_KEY") or resolve_env_var(
                "LANGCHAIN_API_KEY"
            )
            project = Client(api_key=api_key).read_project(project_name=project_name)
            result = project.url or None
        except Exception as exc:  # noqa: BLE001  # LangSmith SDK error types are not stable
            lookup_error = exc
        finally:
            done.set()

    thread = threading.Thread(target=_lookup_url, daemon=True)
    thread.start()

    if not done.wait(_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS):
        logger.debug(
            "Timed out fetching LangSmith project URL for '%s' after %.1fs",
            project_name,
            _LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS,
        )
        return None

    if lookup_error is not None:
        logger.debug(
            "Could not fetch LangSmith project URL for '%s'",
            project_name,
            exc_info=(
                type(lookup_error),
                lookup_error,
                lookup_error.__traceback__,
            ),
        )
        return None

    if result is not None:
        _langsmith_url_cache = (project_name, result)
    return result


def build_langsmith_thread_url(thread_id: str) -> str | None:
    """추적이 구성된 경우 전체 LangSmith 스레드 URL을 작성하십시오.

    `get_langsmith_project_name` 및 `fetch_langsmith_project_url`을 하나의 편리한 도우미로 결합합니다.

Args:
        thread_id: URL을 구축할 스레드 식별자입니다.

Returns:
        전체 스레드 URL 문자열 또는 사용할 수 없는 경우 `None`(LangSmith는
            구성되었거나 프로젝트 URL을 확인할 수 없습니다.)

    """
    project_name = get_langsmith_project_name()
    if not project_name:
        return None

    project_url = fetch_langsmith_project_url(project_name)
    if not project_url:
        return None

    return f"{project_url.rstrip('/')}/t/{thread_id}?utm_source=deepagents-cli"


def reset_langsmith_url_cache() -> None:
    """LangSmith URL 캐시를 재설정합니다(테스트용)."""
    global _langsmith_url_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _langsmith_url_cache = None


def get_default_coding_instructions() -> str:
    """기본 코딩 에이전트 지침을 받으세요.

    이는 에이전트가 수정할 수 없는 변경 불가능한 기본 명령어입니다. 장기 메모리(AGENTS.md)는 미들웨어에 의해 별도로 처리됩니다.

Returns:
        문자열로 된 기본 에이전트 지침입니다.

    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def detect_provider(model_name: str) -> str | None:
    """모델 이름에서 공급자를 자동 감지합니다.

    다음을 수행하려면 `init_chat_model`을 호출하기 **전에** 공급자를 확인해야 하기 때문에 LangChain의
    `_attempt_infer_model_provider` 하위 집합을 의도적으로 복제합니다.

    1. 다음과 같은 공급자별 kwargs(API 기본 URL, 헤더 등)를 구축합니다.
       `init_chat_model` *에* 전달되었습니다.
    2. 사용자에게 친숙한 오류를 표시하기 위해 자격 증명을 조기에 검증합니다.

Args:
        model_name: 공급자를 감지할 모델 이름입니다.

Returns:
        제공업체 이름(openai, anthropic, google_genai, google_vertexai,
            nvidia) 또는 이름만으로는 공급자를 확인할 수 없는 경우 `None`입니다.

    """
    model_lower = model_name.lower()

    if model_lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"

    if model_lower.startswith("claude"):
        s = _get_settings()
        if not s.has_anthropic and s.has_vertex_ai:
            return "google_vertexai"
        return "anthropic"

    if model_lower.startswith("gemini"):
        s = _get_settings()
        if s.has_vertex_ai and not s.has_google:
            return "google_vertexai"
        return "google_genai"

    if model_lower.startswith(("nemotron", "nvidia/")):
        return "nvidia"

    return None


def _get_default_model_spec() -> str:
    """사용 가능한 자격 증명을 기반으로 기본 모델 사양을 가져옵니다.

    순서대로 확인합니다.

    1. 구성 파일의 `[models].default`(사용자의 의도적인 기본 설정) 2. 구성 파일의 `[models].recent`(마지막
    `/model` 스위치). 3. 사용 가능한 API 자격 증명을 기반으로 자동 감지합니다.

Returns:
        Model specification in provider: 모델 형식.

Raises:
        ModelConfigError: 자격 증명이 구성되지 않은 경우.

    """
    from deepagents_cli.model_config import ModelConfig, ModelConfigError

    config = ModelConfig.load()
    if config.default_model:
        return config.default_model

    if config.recent_model:
        return config.recent_model

    s = _get_settings()
    if s.has_openai:
        return "openai:gpt-5.2"
    if s.has_anthropic:
        return "anthropic:claude-sonnet-4-6"
    if s.has_google:
        return "google_genai:gemini-3.1-pro-preview"
    if s.has_vertex_ai:
        return "google_vertexai:gemini-3.1-pro-preview"
    if s.has_nvidia:
        return "nvidia:nvidia/nemotron-3-super-120b-a12b"

    msg = (
        "No credentials configured. Please set one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, "
        "GOOGLE_CLOUD_PROJECT, or NVIDIA_API_KEY"
    )
    raise ModelConfigError(msg)


_OPENROUTER_APP_URL = "https://pypi.org/project/deepagents-cli/"
"""OpenRouter 특성에 대한 기본 `app_url`(`HTTP-Referer`에 매핑).

자세한 내용은 https://openrouter.ai/docs/app-attribution을 참조하세요.
"""

_OPENROUTER_APP_TITLE = "Deep Agents CLI"
"""OpenRouter 특성에 대한 기본 `app_title`(`X-Title`에 매핑)."""

_OPENROUTER_APP_CATEGORIES: list[str] = ["cli-agent"]
"""OpenRouter의 기본값은 `app_categories`(`X-OpenRouter-Categories`에 매핑됨)입니다."""


def _apply_openrouter_defaults(kwargs: dict[str, Any]) -> None:
    """기본 OpenRouter 속성 kwargs를 삽입합니다.

    구성에서 사용자가 제공한 값이 우선하도록 `setdefault`를 통해 `app_url` 및 `app_title`을 설정합니다. 이는
    `ChatOpenRouter`에서 앱 속성을 위해 전송하는 `HTTP-Referer` 및 `X-Title` 헤더에
    매핑됩니다(https://openrouter.ai/docs/app-attribution). 참조).

    사용자는 `~/.deepagents/config.toml`에서 공급자 전체 또는 모델별 값을 재정의할 수 있습니다.

    ```toml
    # Provider-wide
    [models.providers.openrouter.params]
    app_url = "https://myapp.com"
    app_title = "My App"

    # Per-model (shallow-merges on top of provider-wide)
    [models.providers.openrouter.params."openai/gpt-oss-120b"]
    app_title = "My App (GPT)"
    ```

Args:
        kwargs: 변경 가능한 kwargs는 제자리에서 업데이트하도록 지시합니다.

    """
    kwargs.setdefault("app_url", _OPENROUTER_APP_URL)
    kwargs.setdefault("app_title", _OPENROUTER_APP_TITLE)
    kwargs.setdefault("app_categories", _OPENROUTER_APP_CATEGORIES)


def _get_provider_kwargs(
    provider: str, *, model_name: str | None = None
) -> dict[str, Any]:
    """구성 파일에서 공급자별 kwargs를 가져옵니다.

    지정된 공급자에 대한 사용자의 `config.toml`에서 `base_url`, `api_key_env` 및 `params` 테이블을 읽습니다.

    `model_name`이 제공되면 `params` 하위 테이블의 모델별 재정의가 맨 위에 얕은 병합됩니다.

Args:
        provider: 제공자 이름(예: openai, anthropic, Fireworks, ollama)
        model_name: 모델별 재정의를 위한 선택적 모델 이름입니다.

Returns:
        공급자별 kwargs 사전입니다.

    """
    from deepagents_cli.model_config import ModelConfig

    config = ModelConfig.load()
    result: dict[str, Any] = config.get_kwargs(provider, model_name=model_name)
    base_url = config.get_base_url(provider)
    if base_url:
        result["base_url"] = base_url
    from deepagents_cli.model_config import PROVIDER_API_KEY_ENV, resolve_env_var

    api_key_env = config.get_api_key_env(provider)
    if not api_key_env:
        api_key_env = PROVIDER_API_KEY_ENV.get(provider)
        if api_key_env:
            logger.debug(
                "No api_key_env in config.toml for '%s';"
                " using hardcoded provider env var",
                provider,
            )
    if api_key_env:
        api_key = resolve_env_var(api_key_env)
        if api_key:
            result["api_key"] = api_key

    if provider == "openrouter":
        from deepagents._models import check_openrouter_version  # noqa: PLC2701

        check_openrouter_version()
        _apply_openrouter_defaults(result)

    return result


def _create_model_from_class(
    class_path: str,
    model_name: str,
    provider: str,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """사용자 정의 `BaseChatModel` 클래스를 가져오고 인스턴스화합니다.

Args:
        class_path: `module.path:ClassName` 형식의 정규화된 클래스입니다.
        model_name: `model` kwarg로 전달할 모델 식별자입니다.
        provider: 공급자 이름(오류 메시지용)
        kwargs: 생성자의 추가 키워드 인수입니다.

Returns:
        `BaseChatModel` 인스턴스화되었습니다.

Raises:
        ModelConfigError: 클래스를 가져올 수 없거나, `BaseChatModel` 하위 클래스가 아니거나, 인스턴스화에 실패하는
                          경우입니다.

    """
    from langchain_core.language_models import (
        BaseChatModel as _BaseChatModel,  # Runtime import; module level is typing only
    )

    from deepagents_cli.model_config import ModelConfigError

    if ":" not in class_path:
        msg = (
            f"Invalid class_path '{class_path}' for provider '{provider}': "
            "must be in module.path:ClassName format"
        )
        raise ModelConfigError(msg)

    module_path, class_name = class_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        msg = f"Could not import module '{module_path}' for provider '{provider}': {e}"
        raise ModelConfigError(msg) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        msg = (
            f"Class '{class_name}' not found in module '{module_path}' "
            f"for provider '{provider}'"
        )
        raise ModelConfigError(msg)

    if not (isinstance(cls, type) and issubclass(cls, _BaseChatModel)):
        msg = (
            f"'{class_path}' is not a BaseChatModel subclass (got {type(cls).__name__})"
        )
        raise ModelConfigError(msg)

    try:
        return cls(model=model_name, **kwargs)
    except Exception as e:
        msg = f"Failed to instantiate '{class_path}' for '{provider}:{model_name}': {e}"
        raise ModelConfigError(msg) from e


def _create_model_via_init(
    model_name: str,
    provider: str,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """langchain의 `init_chat_model`을 사용하여 모델을 만듭니다.

Args:
        model_name: 모델 식별자.
        provider: 공급자 이름(자동 감지를 위해 비어 있을 수 있음)
        kwargs: 추가 키워드 인수.

Returns:
        `BaseChatModel` 인스턴스화되었습니다.

Raises:
        ModelConfigError: 가져오기, 값 또는 런타임 오류 시.

    """
    from langchain.chat_models import init_chat_model

    from deepagents_cli.model_config import ModelConfigError

    try:
        if provider:
            return init_chat_model(model_name, model_provider=provider, **kwargs)
        return init_chat_model(model_name, **kwargs)
    except ImportError as e:
        import importlib.util

        package_map = {
            "anthropic": "langchain-anthropic",
            "openai": "langchain-openai",
            "google_genai": "langchain-google-genai",
            "google_vertexai": "langchain-google-vertexai",
            "nvidia": "langchain-nvidia-ai-endpoints",
        }
        package = package_map.get(provider, f"langchain-{provider}")
        # Convert pip package name to Python module name for import check.
        module_name = package.replace("-", "_")
        try:
            spec_found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            spec_found = False
        if spec_found:
            # Package is installed but an internal import failed — surface
            # the real error instead of the misleading "missing package" hint.
            msg = (
                f"Provider package '{package}' is installed but failed to "
                f"import for provider '{provider}': {e}"
            )
        else:
            msg = (
                f"Missing package for provider '{provider}'. "
                f"Install: pip install {package}"
            )
        raise ModelConfigError(msg) from e
    except (ValueError, TypeError) as e:
        spec = f"{provider}:{model_name}" if provider else model_name
        msg = f"Invalid model configuration for '{spec}': {e}"
        raise ModelConfigError(msg) from e
    except Exception as e:  # provider SDK auth/network errors
        spec = f"{provider}:{model_name}" if provider else model_name
        msg = f"Failed to initialize model '{spec}': {e}"
        raise ModelConfigError(msg) from e


@dataclass(frozen=True)
class ModelResult:
    """채팅 모델을 생성하고 모델을 해당 메타데이터와 번들링한 결과입니다.

    이를 통해 모델 생성과 설정 변형이 분리되므로 호출자는 메타데이터를 전역 설정으로 커밋할 시기를 결정할 수 있습니다.

Attributes:
        model: 인스턴스화된 채팅 모델.
        model_name: 모델명이 해결되었습니다.
        provider: 확인된 공급자 이름입니다.
        context_limit: 모델 프로필의 최대 입력 토큰 또는 `None`.
        unsupported_modalities: 모델 프로필에서 지원되는 것으로 표시되지 않은 입력 양식(예: `{"audio", "video"}`)

    """

    model: BaseChatModel
    model_name: str
    provider: str
    context_limit: int | None = None
    unsupported_modalities: frozenset[str] = frozenset()

    def apply_to_settings(self) -> None:
        """이 결과의 메타데이터를 전역 `settings`에 커밋합니다."""
        s = _get_settings()
        s.model_name = self.model_name
        s.model_provider = self.provider
        s.model_context_limit = self.context_limit
        s.model_unsupported_modalities = self.unsupported_modalities


def _apply_profile_overrides(
    model: BaseChatModel,
    overrides: dict[str, Any],
    model_name: str,
    *,
    label: str,
    raise_on_failure: bool = False,
) -> None:
    """`overrides`을(를) `model.profile`에 병합합니다.

    모델에 이미 사전 프로필이 있는 경우 재정의가 맨 위에 계층화되어 기존 키(예: `tool_calling`)가 변경되지 않고 유지됩니다.

Args:
        model: 프로필이 업데이트될 채팅 모델입니다.
        overrides: 프로필에 병합할 키/값 쌍입니다.
        model_name: 로그/오류 메시지에 사용되는 모델 이름입니다.
        label: 사람이 읽을 수 있는 메시지 소스 라벨(예: `"config.toml"`, `"CLI --profile-override"`)
        raise_on_failure: `True`인 경우 할당이 실패하면 경고를 기록하는 대신 `ModelConfigError`을 발생시킵니다.

Raises:
        ModelConfigError: `raise_on_failure`이(가) `True`이고 모델이 프로필 할당을 거부하는 경우.

    """
    from deepagents_cli.model_config import ModelConfigError

    logger.debug("Applying %s profile overrides: %s", label, overrides)
    profile = getattr(model, "profile", None)
    merged = {**profile, **overrides} if isinstance(profile, dict) else overrides
    try:
        model.profile = merged  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        if raise_on_failure:
            msg = (
                f"Could not apply {label} to model '{model_name}': {exc}. "
                f"The model may not support profile assignment."
            )
            raise ModelConfigError(msg) from exc
        logger.warning(
            "Could not apply %s profile overrides to model '%s': %s. "
            "Overrides will be ignored.",
            label,
            model_name,
            exc,
        )


def create_model(
    model_spec: str | None = None,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    profile_overrides: dict[str, Any] | None = None,
) -> ModelResult:
    """채팅 모델을 만듭니다.

    표준 공급자에 대해 `init_chat_model`을 사용하거나 공급자의 구성에 `class_path`이 있는 경우 사용자 지정
    `BaseChatModel` 하위 클래스를 가져옵니다.

    명시적인 공급자 선택을 위해 `provider:model` 형식(예: `'anthropic:claude-sonnet-4-5'`)을 지원하거나 자동
    감지를 위해 기본 모델 이름을 지원합니다.

Args:
        model_spec: `provider:model` 형식(예: `'anthropic:claude-sonnet-4-5'`,
                    `'openai:gpt-4o'`) 또는 자동 감지를 위한 모델 이름(예: `'claude-sonnet-4-5'`)의 모델
                    사양입니다.

                제공되지 않으면 환경 기반 기본값을 사용합니다.
        extra_kwargs: 모델 생성자에 전달할 추가 kwargs입니다.

            이는 구성 파일의 값보다 우선하여 가장 높은 우선순위를 갖습니다.
        profile_overrides: `--profile-override`의 추가 프로필 필드입니다.

            구성 파일 프로필 재정의 위에 병합됩니다(CLI가 우선).

Returns:
        모델과 해당 메타데이터가 포함된 `ModelResult`입니다.

Raises:
        ModelConfigError: 모델 이름으로 공급자를 확인할 수 없는 경우 필수 공급자 패키지가 설치되지 않았거나 자격 증명이 구성되지 않은
                          것입니다.

Examples:
        >>> model = create_model("anthropic:claude-sonnet-4-5")
        >>> model = create_model("openai:gpt-4o")
        >>> model = create_model("gpt-4o")  # Auto-detects openai
        >>> model = create_model()  # Uses environment defaults

    """
    from deepagents_cli.model_config import ModelConfig, ModelConfigError, ModelSpec

    if not model_spec:
        model_spec = _get_default_model_spec()

    # Parse provider:model syntax
    provider: str
    model_name: str
    parsed = ModelSpec.try_parse(model_spec)
    if parsed:
        # Explicit provider:model (e.g., "anthropic:claude-sonnet-4-5")
        provider, model_name = parsed.provider, parsed.model
    elif ":" in model_spec:
        # Contains colon but ModelSpec rejected it (empty provider or model)
        _, _, after = model_spec.partition(":")
        if after:
            # Leading colon (e.g., ":claude-opus-4-6") — treat as bare model name
            model_name = after
            provider = detect_provider(model_name) or ""
        else:
            msg = (
                f"Invalid model spec '{model_spec}': model name is required "
                "(e.g., 'anthropic:claude-sonnet-4-5' or 'claude-sonnet-4-5')"
            )
            raise ModelConfigError(msg)
    else:
        # Bare model name — auto-detect provider or let init_chat_model infer
        model_name = model_spec
        provider = detect_provider(model_spec) or ""

    # Provider-specific kwargs (with per-model overrides)
    kwargs = _get_provider_kwargs(provider, model_name=model_name)

    # CLI --model-params take highest priority
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # Check if this provider uses a custom BaseChatModel class
    config = ModelConfig.load()
    class_path = config.get_class_path(provider) if provider else None

    if class_path:
        model = _create_model_from_class(class_path, model_name, provider, kwargs)
    else:
        model = _create_model_via_init(model_name, provider, kwargs)

    resolved_provider = provider or getattr(model, "_model_provider", provider)

    # Apply profile overrides from config.toml (e.g., max_input_tokens)
    if provider:
        config_profile_overrides = config.get_profile_overrides(
            provider, model_name=model_name
        )
        if config_profile_overrides:
            _apply_profile_overrides(
                model,
                config_profile_overrides,
                model_name,
                label=f"config.toml (provider '{provider}')",
            )

    # CLI --profile-override takes highest priority (on top of config.toml)
    if profile_overrides:
        _apply_profile_overrides(
            model,
            profile_overrides,
            model_name,
            label="CLI --profile-override",
            raise_on_failure=True,
        )

    # Extract context limit and modality support from model profile
    context_limit: int | None = None
    unsupported_modalities: frozenset[str] = frozenset()
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict):
        if isinstance(profile.get("max_input_tokens"), int):
            context_limit = profile["max_input_tokens"]

        modality_keys = {
            "image_inputs": "image",
            "audio_inputs": "audio",
            "video_inputs": "video",
            "pdf_inputs": "pdf",
        }
        unsupported_modalities = frozenset(
            label for key, label in modality_keys.items() if profile.get(key) is False
        )

    return ModelResult(
        model=model,
        model_name=model_name,
        provider=resolved_provider,
        context_limit=context_limit,
        unsupported_modalities=unsupported_modalities,
    )


def validate_model_capabilities(model: BaseChatModel, model_name: str) -> None:
    """모델에 `deepagents`에 필요한 기능이 있는지 확인하십시오.

    모델의 프로필(사용 가능한 경우)을 확인하여 에이전트 기능에 필요한 도구 호출을 지원하는지 확인합니다. 프로파일이 없거나 상황 창이 제한된 모델에 대해
    경고를 표시합니다.

Args:
        model: 유효성을 검사할 인스턴스화된 모델입니다.
        model_name: 오류/경고 메시지의 모델 이름입니다.

Note:
        이 검증은 최선의 노력입니다. 프로필이 없는 모델은 경고와 함께 통과됩니다. 모델 프로필이 명시적으로 tool_calling=False를
        나타내는 경우 sys.exit(1)을 통해 종료됩니다.

    """
    console = _get_console()
    profile = getattr(model, "profile", None)

    if profile is None:
        # Model doesn't have profile data - warn but allow
        console.print(
            f"[dim][yellow]Note:[/yellow] No capability profile for "
            f"'{model_name}'. Cannot verify tool calling support.[/dim]"
        )
        return

    if not isinstance(profile, dict):
        return

    # Check required capability: tool_calling
    tool_calling = profile.get("tool_calling")
    if tool_calling is False:
        console.print(
            f"[bold red]Error:[/bold red] Model '{model_name}' "
            "does not support tool calling."
        )
        console.print(
            "\nDeep Agents requires tool calling for agent functionality. "
            "Please choose a model that supports tool calling."
        )
        console.print("\nSee MODELS.md for supported models.")
        sys.exit(1)

    # Warn about potentially limited context (< 8k tokens)
    max_input_tokens = profile.get("max_input_tokens")
    if max_input_tokens and max_input_tokens < 8000:  # noqa: PLR2004  # Model context window default
        console.print(
            f"[dim][yellow]Warning:[/yellow] Model '{model_name}' has limited context "
            f"({max_input_tokens:,} tokens). Agent performance may be affected.[/dim]"
        )


def _get_console() -> Console:
    """지연 초기화된 전역 `Console` 인스턴스를 반환합니다.

    콘솔 출력이 실제로 필요할 때까지 `rich.console` 가져오기를 연기합니다. 결과는 `globals()["console"]`에 캐시됩니다.

Returns:
        전역 Rich `Console` 싱글톤입니다.

    """
    cached = globals().get("console")
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = globals().get("console")
        if cached is not None:
            return cached
        from rich.console import Console

        inst = Console(highlight=False)
        globals()["console"] = inst
        return inst


def _get_settings() -> Settings:
    """지연 초기화된 전역 `Settings` 인스턴스를 반환합니다.

    설정을 구성하기 전에 부트스트랩이 실행되었는지 확인합니다. 결과는 `globals()["settings"]`에 캐시되므로 다른 모듈의 `from
    config import settings`을 포함한 후속 액세스는 즉시 해결됩니다.

Returns:
        전역 `Settings` 싱글톤.

    """
    cached = globals().get("settings")
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = globals().get("settings")
        if cached is not None:
            return cached
        _ensure_bootstrap()
        try:
            inst = Settings.from_environment(start_path=_bootstrap_start_path)
        except Exception:
            logger.exception(
                "Failed to initialize settings from environment (start_path=%s)",
                _bootstrap_start_path,
            )
            raise
        globals()["settings"] = inst
        return inst


def __getattr__(name: str) -> Settings | Console:
    """`settings` 및 `console`에 대한 지연 모듈 속성입니다.

    처음 액세스할 때까지 대규모 초기화를 연기합니다. 후속 액세스는 모듈 수준 속성에 직접 적중합니다(`__getattr__` 오버헤드 없음).

Returns:
        요청된 게으른 싱글톤입니다.

Raises:
        AttributeError: *name*이 느리게 제공되는 속성이 아닌 경우.

    """
    if name == "settings":
        return _get_settings()
    if name == "console":
        return _get_console()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
