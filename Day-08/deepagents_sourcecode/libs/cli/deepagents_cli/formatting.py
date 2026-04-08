"""라이트플레이트 서식 파일입니다.

관계 프레임워크를 가져오기 위해 CLI의 어느 곳에서나 추가할 수 있도록 이 모듈을 적극적으로 활용하도록 유지하세요.
"""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """기간(초)을 사람이 이해할 수 있는 문자열로 형식화합니다.

Args:
        seconds: 기간(초)입니다.

Returns:
        `"5s"`, `"2.3s"`, `"5m 12s"` 또는 `"1h 23m 4s"`과 같은 형식의 문자열입니다.


    """
    rounded = round(seconds, 1)
    if rounded < 60:  # noqa: PLR2004
        if rounded % 1 == 0:
            return f"{int(rounded)}s"
        return f"{rounded:.1f}s"
    minutes, secs = divmod(int(rounded), 60)
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"
