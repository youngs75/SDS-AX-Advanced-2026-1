"""자세한 파일 기반 추적을 위한 공유 디버그 로깅 구성입니다.

`DEEPAGENTS_CLI_DEBUG` 환경 변수가 설정되면 스트리밍 또는 원격 통신을 처리하는 모듈에서 자세한 파일 기반 로깅을 활성화할 수 있습니다. 이
도우미는 설정을 중앙 집중화하므로 env-var 이름, 파일 경로 및 형식이 한 곳에서 정의됩니다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from deepagents_cli._env_vars import DEBUG, DEBUG_FILE


def configure_debug_logging(target: logging.Logger) -> None:
    """`DEEPAGENTS_CLI_DEBUG`이 설정된 경우 *대상*에 파일 처리기를 연결합니다.

    로그 파일의 기본값은 `'/tmp/deepagents_debug.log'`이지만 `DEEPAGENTS_CLI_DEBUG_FILE`로 재정의할 수
    있습니다. 핸들러는 여러 모듈이 세션 전체에서 동일한 로그 파일을 공유하도록 추가됩니다.

    `DEEPAGENTS_CLI_DEBUG`이 설정되지 않은 경우 아무 작업도 수행하지 않습니다.

Args:
        target: 구성할 로거입니다.

    """
    if not os.environ.get(DEBUG):
        return

    debug_path = Path(
        os.environ.get(
            DEBUG_FILE,
            "/tmp/deepagents_debug.log",  # noqa: S108
        )
    )
    try:
        handler = logging.FileHandler(str(debug_path), mode="a")
    except OSError as exc:
        import sys

        print(  # noqa: T201
            f"Warning: could not open debug log file {debug_path}: {exc}",
            file=sys.stderr,
        )
        return
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
