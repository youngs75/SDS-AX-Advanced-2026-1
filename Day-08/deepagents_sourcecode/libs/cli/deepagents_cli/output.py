"""CLI 하위 명령에 대한 기계 판독 가능 JSON 출력 도우미입니다.

이 모듈은 의도적으로 stdlib 전용으로 유지되므로 불필요한 종속성 트리를 가져오지 않고도 CLI 시작 경로에서 가져올 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Literal

OutputFormat = Literal["text", "json"]
"""CLI 하위 명령에 대해 허용되는 내부 출력 모드입니다."""


def add_json_output_arg(
    parser: argparse.ArgumentParser, *, default: OutputFormat | None = None
) -> None:
    """argparse 파서에 `--json` 플래그를 추가합니다.

Args:
        parser: 업데이트할 파서입니다.
        default: 이 구문 분석기의 기본 출력 형식입니다.

            상위 파서 값이 보존되도록 하위 파서에 `None`을 전달합니다.

    """
    if default is None:
        parser.add_argument(
            "--json",
            dest="output_format",
            action="store_const",
            const="json",
            default=argparse.SUPPRESS,
            help="Emit machine-readable JSON for this command",
        )
    else:
        parser.add_argument(
            "--json",
            dest="output_format",
            action="store_const",
            const="json",
            default=default,
            help="Emit machine-readable JSON for this command",
        )


def write_json(command: str, data: list | dict) -> None:
    """stdout에 JSON 봉투를 작성하고 플러시합니다.

    봉투는 안정적인 스키마가 포함된 한 줄짜리 JSON 객체입니다.

    ```json
    {"schema_version": 1, "command": "...", "data": ...}
    ```

Args:
        command: 자체 문서화 명령 이름(예: `'list'`, `'threads list'`).
        data: 페이로드 — 일반적으로 명령 나열을 위한 목록 또는 작업/정보 명령을 위한 사전입니다.

            `default=str`은 `Path` 및 `datetime` 개체가 오류 없이 직렬화되도록 사용됩니다.

    """
    envelope = {"schema_version": 1, "command": command, "data": data}
    sys.stdout.write(json.dumps(envelope, default=str) + "\n")
    sys.stdout.flush()
