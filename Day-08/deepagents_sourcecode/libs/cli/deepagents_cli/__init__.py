"""`deepagents-cli`에 대한 공개 패키지 진입점.

이 모듈은 경량 메타데이터를 적극적으로 노출하고 `cli_main`이 요청될 때만 전체 CLI 부트스트랩을 로드하여 패키지 가져오기 부작용을 작게 유지합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_cli._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "__version__",
    "cli_main",  # noqa: F822  # resolved lazily by __getattr__
]


def __getattr__(name: str) -> Callable[[], None]:
    """패키지 가져오기 시 `main.py` 로드를 방지하기 위해 `cli_main`에 대한 지연 가져오기.

    `main.py`은 `config` 또는 `widgets`과 같은 하위 모듈을 직접 가져올 때 필요하지 않은 `argparse`, 신호 처리 및 기타
    시작 기계를 가져옵니다.

Returns:
        요청된 호출 가능 항목입니다.

Raises:
        AttributeError: *name*이 느리게 제공되는 속성이 아닌 경우.

    """
    if name == "cli_main":
        from deepagents_cli.main import cli_main

        return cli_main
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
