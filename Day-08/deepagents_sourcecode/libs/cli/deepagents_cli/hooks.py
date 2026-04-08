"""외부 도구 통합을 위한 경량 후크 디스패치.

`~/.deepagents/hooks.json`에서 후크 구성을 로드하고 stdin에서 JSON 페이로드와 일치하는 명령을 실행합니다. 하위 프로세스 작업이
백그라운드 스레드로 오프로드되므로 호출자의 이벤트 루프가 중단되지 않습니다. 실패는 기록되지만 호출자에게 표시되지는 않습니다.

구성 형식(`~/.deepagents/hooks.json`):

```json
{"hooks": [{"command": ["bash", "adapter.sh"], "events": ["session.start"]}]}
```

`events`이 생략되거나 비어 있으면 후크는 **모든** 이벤트를 수신합니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess  # noqa: S404
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

_hooks_config: list[dict[str, Any]] | None = None
"""캐시된 구성 — 첫 번째 발송 시 느리게 로드됩니다."""

_background_tasks: set[asyncio.Task[None]] = set()
"""GC를 방지하기 위해 실행 후 잊어버리는 작업에 대한 강력한 참조입니다."""


def _load_hooks() -> list[dict[str, Any]]:
    """구성 파일에서 후크 정의를 로드하고 캐시합니다.

Returns:
        파일이 없거나 형식이 잘못된 경우 빈 목록이 표시되어 정상적으로 작동합니다.
            실행은 중단되지 않습니다.

    """
    global _hooks_config  # noqa: PLW0603
    if _hooks_config is not None:
        return _hooks_config

    from deepagents_cli.model_config import DEFAULT_CONFIG_DIR

    hooks_path = DEFAULT_CONFIG_DIR / "hooks.json"

    if not hooks_path.is_file():
        _hooks_config = []
        return _hooks_config

    try:
        data = json.loads(hooks_path.read_text())
        if not isinstance(data, dict):
            logger.warning(
                "Hooks config at %s must be a JSON object, got %s",
                hooks_path,
                type(data).__name__,
            )
            _hooks_config = []
            return _hooks_config
        hooks = data.get("hooks", [])
        if not isinstance(hooks, list):
            logger.warning(
                "Hooks config 'hooks' key at %s must be a list, got %s",
                hooks_path,
                type(hooks).__name__,
            )
            _hooks_config = []
            return _hooks_config
        _hooks_config = hooks
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load hooks config from %s: %s", hooks_path, exc)
        _hooks_config = []

    return _hooks_config


def _run_single_hook(command: list[str], event: str, payload_bytes: bytes) -> None:
    """단일 후크 명령을 실행하여 JSON 페이로드를 stdin에 씁니다.

    시간 초과 시 자동으로 하위 프로세스를 종료하는 `subprocess.run`을 사용하여 좀비/고아 프로세스 누출을 방지합니다.

Args:
        command: 실행할 명령 및 인수입니다.
        event: 이벤트 이름(로깅용).
        payload_bytes: 명령의 stdin에 쓰기 위한 JSON 페이로드입니다.

    """
    try:
        subprocess.run(  # noqa: S603
            command,
            input=payload_bytes,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Hook command timed out (>5s) for event %s: %s", event, command)
    except (FileNotFoundError, PermissionError) as exc:
        logger.warning("Hook command failed for event %s: %s — %s", event, command, exc)
    except Exception:
        logger.debug(
            "Hook dispatch failed for event %s: %s",
            event,
            command,
            exc_info=True,
        )


def _dispatch_hook_sync(
    event: str, payload_bytes: bytes, hooks: list[dict[str, Any]]
) -> None:
    """스레드 풀을 통해 동시에 실행하여 일치하는 후크를 디스패치합니다.

    구성된 모든 후크를 반복하여 이벤트 필터가 일치하지 않거나 `command`이 누락/유효하지 않은 후크는 건너뜁니다. 일치하는 후크는 명령당 5초의
    제한 시간으로 동시에 실행됩니다. 오류는 후크별로 포착되어 전파되지 않고 기록됩니다.

Args:
        event: 점으로 구분된 이벤트 이름(예: `'session.start'`).
        payload_bytes: 각 명령의 stdin에 쓰기 위한 JSON 페이로드입니다.
        hooks: 구성 파일의 후크 정의 목록입니다.

    """
    matching: list[list[str]] = []
    for hook in hooks:
        command = hook.get("command")
        if not isinstance(command, list) or not command:
            continue

        events = hook.get("events")
        # Empty/missing events list means "subscribe to everything".
        if events and event not in events:
            continue

        matching.append(command)

    if not matching:
        return

    if len(matching) == 1:
        _run_single_hook(matching[0], event, payload_bytes)
        return

    with ThreadPoolExecutor(max_workers=len(matching)) as pool:
        futures = [
            pool.submit(_run_single_hook, cmd, event, payload_bytes) for cmd in matching
        ]
        for future in futures:
            future.result()


async def dispatch_hook(event: str, payload: dict[str, Any]) -> None:
    """stdin에서 JSON으로 직렬화된 `payload`을 사용하여 일치하는 후크 명령을 실행합니다.

    `event` 이름은 `"event"` 키 아래의 페이로드에 자동으로 삽입되므로 호출자가 이를 복제할 필요가 없습니다.

    차단 하위 프로세스 작업이 스레드로 오프로드되므로 호출자의 이벤트 루프가 중단되지 않습니다. 일치하는 후크는 동시에 실행되며 각각 5초의 시간 초과가
    적용됩니다. 오류는 기록되며 전파되지 않습니다.

Args:
        event: 점으로 구분된 이벤트 이름(예: `'session.start'`).
        payload: 명령의 stdin에 전송된 임의의 JSON 직렬화 가능 dict입니다.

    """
    try:
        hooks = _load_hooks()
        if not hooks:
            return

        payload_bytes = json.dumps({"event": event, **payload}).encode()
        await asyncio.to_thread(_dispatch_hook_sync, event, payload_bytes, hooks)
    except Exception:
        logger.warning(
            "Unexpected error in dispatch_hook for event %s",
            event,
            exc_info=True,
        )


def dispatch_hook_fire_and_forget(event: str, payload: dict[str, Any]) -> None:
    """강력한 참조를 사용하여 `dispatch_hook`을(를) 백그라운드 작업으로 예약하세요.

    작업이 완료되기 전에 가비지 수집되는 것을 방지하려면 베어 `create_task(dispatch_hook(...))` 대신 이것을 사용하십시오.

    이벤트 루프가 실행되는 동안 동기화 코드에서 호출하는 것이 안전합니다.

Args:
        event: 점으로 구분된 이벤트 이름(예: `'session.start'`).
        payload: 명령의 stdin에 전송된 임의의 JSON 직렬화 가능 dict입니다.

    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("No running event loop; skipping hook for %s", event)
        return
    task = loop.create_task(dispatch_hook(event, payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
