"""텍스트 위젯 내에서 안전한 링크 활성화를 처리합니다.

터미널 하이퍼링크 이스케이프 시퀀스가 ​​실행되기 전에 텍스트가 클릭을 가로채므로 이 모듈은 URL 안전 검사를 먼저 적용하는 동안 단일 클릭 링크 동작을
재현합니다.
"""

from __future__ import annotations

import logging
import webbrowser
from typing import TYPE_CHECKING

from deepagents_cli.unicode_security import check_url_safety, strip_dangerous_unicode

if TYPE_CHECKING:
    from textual.events import Click

logger = logging.getLogger(__name__)


def open_style_link(event: Click) -> None:
    """리치 링크 스타일이 있는 경우 클릭 시 URL을 엽니다.

    Rich `Style(link=...)`에는 OSC 8 터미널 하이퍼링크가 포함되어 있지만 Textual의 마우스 캡처는 터미널이 작동하기 전에
    일반적인 클릭을 차단합니다. 텍스트 클릭 이벤트를 직접 처리하여 마크다운 위젯의 링크 동작과 일치하는 한 번의 클릭으로 URL을 엽니다.

    안전 검사에 실패한 URL(예: 숨겨진 유니코드 또는 동형이의어 도메인 포함)은 차단되고 열리지 않습니다. 이벤트 버블과 경고가 기록되어 텍스트
    알림으로 표시됩니다.

    성공하면 이벤트가 중지되어 더 이상 거품이 발생하지 않습니다. 실패 시(예: 헤드리스 환경에서 사용할 수 있는 브라우저가 없는 경우) 오류는 디버그
    수준에 기록되고 이벤트는 정상적으로 버블링됩니다.

    Args:
        event: 검사할 텍스트 클릭 이벤트입니다.

    """
    url = event.style.link
    if not url:
        return

    safety = check_url_safety(url)
    if not safety.safe:
        detail = safety.warnings[0] if safety.warnings else "Suspicious URL"
        logger.warning("Blocked suspicious URL: %s (%s)", url, detail)
        try:
            app = getattr(event, "app", None)
            notify = getattr(app, "notify", None)
            if callable(notify):
                safe_url = strip_dangerous_unicode(url)
                notify(
                    f"Blocked suspicious URL: {safe_url}\n{detail}",
                    severity="warning",
                    markup=False,
                )
        except (AttributeError, TypeError):
            logger.debug("Could not send URL-blocked notification", exc_info=True)
        return

    try:
        webbrowser.open(url)
    except Exception:
        logger.debug("Could not open browser for URL: %s", url, exc_info=True)
        return
    event.stop()
