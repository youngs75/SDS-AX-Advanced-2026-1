"""터미널 및 데스크톱 복사 작업 흐름을 위한 클립보드 도우미입니다.

CLI는 가능한 경우 OSC 52와 같은 터미널 친화적인 메커니즘을 선호하며 복사된 내용을 요약하는 사용자 대상 알림으로 대체됩니다.
"""

from __future__ import annotations

import base64
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from deepagents_cli.config import get_glyphs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from textual.app import App

_PREVIEW_MAX_LENGTH = 40


def _copy_osc52(text: str) -> None:
    """OSC 52 이스케이프 시퀀스를 사용하여 텍스트를 복사합니다(SSH/tmux에서 작동)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    osc52_seq = f"\033]52;c;{encoded}\a"
    if os.environ.get("TMUX"):
        osc52_seq = f"\033Ptmux;\033{osc52_seq}\033\\"

    with pathlib.Path("/dev/tty").open("w", encoding="utf-8") as tty:
        tty.write(osc52_seq)
        tty.flush()


def _shorten_preview(texts: list[str]) -> str:
    """알림 미리보기 텍스트를 줄입니다.

Returns:
        알림 표시에 적합한 단축된 미리보기 텍스트입니다.

    """
    glyphs = get_glyphs()
    dense_text = glyphs.newline.join(texts).replace("\n", glyphs.newline)
    if len(dense_text) > _PREVIEW_MAX_LENGTH:
        return f"{dense_text[: _PREVIEW_MAX_LENGTH - 1]}{glyphs.ellipsis}"
    return dense_text


def copy_selection_to_clipboard(app: App) -> None:
    """앱 위젯에서 선택한 텍스트를 클립보드로 복사합니다.

    이는 text_selection에 대한 모든 위젯을 쿼리하고 선택한 텍스트를 시스템 클립보드에 복사합니다.

    """
    selected_texts = []

    for widget in app.query("*"):
        if not hasattr(widget, "text_selection") or not widget.text_selection:
            continue

        selection = widget.text_selection

        if selection.end is None:
            continue

        try:
            result = widget.get_selection(selection)
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            logger.debug(
                "Failed to get selection from widget %s: %s",
                type(widget).__name__,
                e,
                exc_info=True,
            )
            continue

        if not result:
            continue

        selected_text, _ = result
        if selected_text.strip():
            selected_texts.append(selected_text)

    if not selected_texts:
        return

    combined_text = "\n".join(selected_texts)

    # Try multiple clipboard methods
    # Prefer pyperclip/app clipboard first (works reliably on local machines)
    # OSC 52 is last resort (for SSH/remote where native clipboard unavailable)
    copy_methods = [app.copy_to_clipboard]

    # Try pyperclip if available (preferred - uses pbcopy on macOS)
    try:
        import pyperclip

        copy_methods.insert(0, pyperclip.copy)
    except ImportError:
        pass

    # OSC 52 as fallback for remote/SSH sessions
    copy_methods.append(_copy_osc52)

    for copy_fn in copy_methods:
        try:
            copy_fn(combined_text)
            # Use markup=False to prevent copied text from being parsed as Rich markup
            app.notify(
                f'"{_shorten_preview(selected_texts)}" copied',
                severity="information",
                timeout=2,
                markup=False,
            )
        except (OSError, RuntimeError, TypeError) as e:
            logger.debug(
                "Clipboard copy method %s failed: %s",
                getattr(copy_fn, "__name__", repr(copy_fn)),
                e,
                exc_info=True,
            )
            continue
        else:
            return

    # If all methods fail, still notify but warn
    app.notify(
        "Failed to copy - no clipboard method available",
        severity="warning",
        timeout=3,
    )
