"""프롬프트를 작성하거나 수정하기 위해 외부 편집기를 실행합니다.

여기의 도우미는 일반적인 편집기 규칙을 표준화하므로 CLI가 잘못 차단하지 않고 GUI 또는 터미널 편집기에 대규모 프롬프트 편집을 전달할 수 있습니다.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

GUI_WAIT_FLAG: dict[str, str] = {
    "code": "--wait",
    "cursor": "--wait",
    "zed": "--wait",
    "atom": "--wait",
    "subl": "-w",
    "windsurf": "--wait",
}
"""GUI 편집기 기본 이름을 차단 플래그에 매핑합니다."""

VIM_EDITORS = {"vi", "vim", "nvim"}
"""`-i NONE` 플래그를 받는 vim-family 편집기 기본 이름 집합입니다."""


def resolve_editor() -> list[str] | None:
    """환경에서 편집기 명령을 해결합니다.

    $VISUAL을 확인한 다음 $EDITOR를 확인한 다음 플랫폼 기본값으로 돌아갑니다.

Returns:
        토큰화된 명령 목록 또는 env var가 설정되었지만 이후 비어 있는 경우 `None`
            토큰화.

    """
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        if sys.platform == "win32":
            return ["notepad"]
        return ["vi"]
    tokens = shlex.split(editor)
    return tokens or None


def _prepare_command(cmd: list[str], filepath: str) -> list[str]:
    """적절한 플래그를 사용하여 전체 명령 목록을 작성하십시오.

    GUI 편집기에는 --wait/-w를 추가하고 vim-family 편집기에는 `-i NONE`을 추가합니다.

Returns:
        플래그와 파일 경로가 추가된 전체 명령 목록입니다.

    """
    cmd = list(cmd)  # copy
    exe = Path(cmd[0]).stem.lower()

    # Auto-inject wait flag for GUI editors
    if exe in GUI_WAIT_FLAG:
        flag = GUI_WAIT_FLAG[exe]
        if flag not in cmd:
            cmd.insert(1, flag)

    # Vim workaround: avoid viminfo errors in temp environments
    if exe in VIM_EDITORS and "-i" not in cmd:
        cmd.extend(["-i", "NONE"])

    cmd.append(filepath)
    return cmd


def open_in_editor(current_text: str) -> str | None:
    """외부 편집기에서 current_text를 엽니다.

    임시 .md 파일을 생성하고 편집기를 시작한 다음 결과를 다시 읽습니다.

Args:
        current_text: 편집기에 미리 채워질 텍스트입니다.

Returns:
        정규화된 줄 끝으로 편집된 텍스트 또는 편집자의 경우 `None`
            0이 아닌 상태로 종료되었거나 찾을 수 없거나 결과가 비어 있거나 공백만 있었습니다.

    """
    cmd = resolve_editor()
    if cmd is None:
        return None

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".md",
            prefix="deepagents-edit-",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(current_text)

        full_cmd = _prepare_command(cmd, tmp_path)

        # S603: editor command comes from user's own $EDITOR env var
        result = subprocess.run(  # noqa: S603
            full_cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "Editor exited with code %d: %s", result.returncode, full_cmd
            )
            return None

        edited = Path(tmp_path).read_text(encoding="utf-8")

        # Normalize line endings
        edited = edited.replace("\r\n", "\n").replace("\r", "\n")

        # Most editors append a final newline on save (POSIX convention).
        # Strip exactly one so the cursor lands on content, not a blank line,
        # while preserving any intentional trailing newlines the user added.
        edited = edited.removesuffix("\n")

        # Treat empty result as cancellation
        if not edited.strip():
            return None

    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Editor failed", exc_info=True)
        return None
    else:
        return edited
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
