"""채팅 입력 기록을 유지하고 탐색합니다.

히스토리 관리자는 디스크에 작은 추가 전용 명령 로그를 유지하고 채팅 입력 위젯에 맞춰 필터링된 이전/다음 탐색을 표시합니다.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class HistoryManager:
    """파일 영속성을 통해 명령 히스토리을 관리합니다.

        동시 안전을 위해 추가 전용 쓰기를 사용합니다. 여러 에이전트가 손상 없이 동일한 기록 파일에 안전하게 쓸 수 있습니다.
    """

    def __init__(self, history_file: Path, max_entries: int = 100) -> None:
        """히스토리 관리자를 초기화합니다.

        Args:
                    history_file: JSON-lines 기록 파일 경로
                    max_entries: 보관할 최대 항목 수
        """
        self.history_file = history_file
        self.max_entries = max_entries
        self._entries: list[str] = []
        self._current_index: int = -1
        self._temp_input: str = ""
        self._query: str = ""
        self._load_history()

    def _load_history(self) -> None:
        """파일에서 기록을 로드합니다."""
        if not self.history_file.exists():
            return

        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                entries = []
                for raw_line in f:
                    line = raw_line.rstrip("\n\r")
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        entry = line
                    entries.append(entry if isinstance(entry, str) else str(entry))
                self._entries = entries[-self.max_entries :]
        except (OSError, UnicodeDecodeError):
            logger.warning(
                "Failed to load history from %s; starting with empty history",
                self.history_file,
                exc_info=True,
            )
            self._entries = []

    def _append_to_file(self, text: str) -> None:
        """기록 파일에 단일 항목을 추가합니다(동시 안전)."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(text) + "\n")
        except OSError:
            logger.warning(
                "Failed to append history entry to %s",
                self.history_file,
                exc_info=True,
            )

    def _compact_history(self) -> None:
        """오래된 항목을 제거하려면 기록 파일을 다시 작성하십시오.

                재작성을 최소화하기 위해 항목이 max_entries의 2배를 초과하는 경우에만 호출됩니다.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning(
                "Failed to compact history file %s",
                self.history_file,
                exc_info=True,
            )

    def add(self, text: str) -> None:
        """기록에 명령을 추가합니다.

        Args:
                    text: 추가할 명령 텍스트
        """
        text = text.strip()
        # Skip empty or slash commands
        if not text or text.startswith("/"):
            return

        # Skip duplicates of the last entry
        if self._entries and self._entries[-1] == text:
            return

        self._entries.append(text)

        # Append to file (fast, concurrent-safe)
        self._append_to_file(text)

        # Compact only when we have 2x max entries (rare operation)
        if len(self._entries) > self.max_entries * 2:
            self._entries = self._entries[-self.max_entries :]
            self._compact_history()

        self.reset_navigation()

    def get_previous(self, current_input: str, *, query: str = "") -> str | None:
        """하위 문자열 쿼리와 일치하는 이전 기록 항목을 가져옵니다.

                쿼리는 탐색 세션의 첫 번째 호출(`_current_index == -1`인 경우)에서 캡처되고 `reset_navigation`까지 모든 후속
                호출에 재사용됩니다. 이후 호출에서 다른 값을 전달해도 아무런 효과가 없습니다.

        Args:
                    current_input: 현재 입력 텍스트. 탐색 세션의 첫 번째 호출에만 저장됩니다. 후속 호출에서는 무시됩니다.
                    query: 기록 항목과 일치시킬 하위 문자열입니다. 탐색 세션의 첫 번째 호출에서 한 번 캡처됩니다.

        Returns:
                    이전 일치 항목 또는 `None`.
        """
        if not self._entries:
            return None

        # Save current input and capture query on first navigation
        if self._current_index == -1:
            self._temp_input = current_input
            self._current_index = len(self._entries)
            self._query = query.strip().lower()

        # Search backwards for matching entry
        for i in range(self._current_index - 1, -1, -1):
            if not self._query or self._query in self._entries[i].lower():
                self._current_index = i
                return self._entries[i]

        return None

    def get_next(self) -> str | None:
        """저장된 쿼리와 일치하는 다음 기록 항목을 가져옵니다.

                가장 최근의 `get_previous` 호출로 캡처된 쿼리를 사용합니다.

        Returns:
                    일치하는 다음 항목 또는 최신 항목을 지난 경우 원래 입력
                        성냥.

                        `None` 현재 기록을 탐색하고 있지 않은 경우.
        """
        if self._current_index == -1:
            return None

        # Search forwards for matching entry
        for i in range(self._current_index + 1, len(self._entries)):
            if not self._query or self._query in self._entries[i].lower():
                self._current_index = i
                return self._entries[i]

        # Return to original input at the end
        result = self._temp_input
        self.reset_navigation()
        return result

    @property
    def in_history(self) -> bool:
        """현재 기록 항목을 탐색하는지 여부입니다."""
        return self._current_index >= 0

    def reset_navigation(self) -> None:
        """탐색 상태를 재설정합니다."""
        self._current_index = -1
        self._temp_input = ""
        self._query = ""
