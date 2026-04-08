"""@ 멘션 및 / 명령에 대한 자동 완성 시스템입니다.

이는 슬래시 명령(/) 및 파일 언급(@)에 대한 트리거 기반 완료를 처리하는 사용자 정의 구현입니다.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil

# S404: subprocess is required for git ls-files to get project file list
import subprocess  # noqa: S404
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from deepagents_cli.project_utils import find_project_root


def _get_git_executable() -> str | None:
    """shutdown.which()를 사용하여 git 실행 파일의 전체 경로를 가져옵니다.

    Returns:
        git 실행 파일의 전체 경로이거나, 찾을 수 없으면 None입니다.

    """
    return shutil.which("git")


if TYPE_CHECKING:
    from textual import events


class CompletionResult(StrEnum):
    """완료 시스템에서 주요 이벤트를 처리한 결과입니다."""

    IGNORED = "ignored"  # Key not handled, let default behavior proceed
    HANDLED = "handled"  # Key handled, prevent default
    SUBMIT = "submit"  # Key triggers submission (e.g., Enter on slash command)


class CompletionView(Protocol):
    """완료 제안을 표시할 수 있는 보기에 대한 프로토콜입니다."""

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        """완료 제안 팝업을 렌더링합니다.

        Args:
            suggestions: (레이블, 설명) 튜플 목록
            selected_index: 현재 선택된 항목의 인덱스

        """
        ...

    def clear_completion_suggestions(self) -> None:
        """완료 제안 팝업을 숨기거나 지웁니다."""
        ...

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        """입력의 텍스트를 처음부터 끝까지 교체로 바꿉니다.

        Args:
            start: 입력 텍스트의 시작 인덱스
            end: 입력 텍스트의 끝 인덱스
            replacement: 삽입할 텍스트

        """
        ...


class CompletionController(Protocol):
    """완료 컨트롤러를 위한 프로토콜입니다."""

    def can_handle(self, text: str, cursor_index: int) -> bool:
        """이 컨트롤러가 현재 입력 상태를 처리할 수 있는지 확인하세요."""
        ...

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """입력 텍스트가 변경되면 호출됩니다."""
        ...

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """주요 이벤트를 처리합니다. 이벤트가 처리된 방법을 반환합니다."""
        ...

    def reset(self) -> None:
        """완료 상태를 재설정/삭제합니다."""
        ...


# ============================================================================
# Slash Command Completion
# ============================================================================


MAX_SUGGESTIONS = 10
"""완료 팝업이 다루기 힘들지 않도록 UI 캡을 적용했습니다."""

_MIN_SLASH_FUZZY_SCORE = 25
"""슬래시 명령 퍼지 일치에 대한 최소 점수입니다."""

_MIN_DESC_SEARCH_LEN = 2
"""명령 설명을 검색하기 위한 최소 쿼리 길이(단일 문자 노이즈 방지)"""


class SlashCommandController:
    """/슬래시 명령 완료를 위한 컨트롤러입니다."""

    def __init__(
        self,
        commands: list[tuple[str, str, str]],
        view: CompletionView,
    ) -> None:
        """슬래시 명령 컨트롤러를 초기화합니다.

        Args:
            commands: `(command, description, hidden_keywords)` 튜플 목록입니다.
            view: 제안을 렌더링할 보기입니다.

        """
        self._commands = commands
        self._view = view
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0

    def update_commands(self, commands: list[tuple[str, str, str]]) -> None:
        """명령 목록을 바꾸고 제안을 재설정합니다.

        런타임 시 동적으로 검색된 기술 명령을 정적 명령 레지스트리와 병합하는 데 사용됩니다.

        Args:
            commands: `(command, description, hidden_keywords)` 튜플의 새 목록입니다.

        """
        self._commands = commands
        self.reset()

    @staticmethod
    def can_handle(text: str, cursor_index: int) -> bool:  # noqa: ARG004  # Required by AutocompleteProvider interface
        """/로 시작하는 입력을 처리합니다.

        Returns:
            텍스트가 명령을 나타내는 슬래시로 시작하면 참입니다.

        """
        return text.startswith("/")

    def reset(self) -> None:
        """명확한 제안."""
        if self._suggestions:
            self._suggestions.clear()
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    @staticmethod
    def _score_command(search: str, cmd: str, desc: str, keywords: str = "") -> float:
        """검색 문자열에 대해 명령의 점수를 매깁니다. 높을수록 더 잘 일치합니다.

        Args:
            search: 소문자 검색 문자열(앞에 `/` 제외)
            cmd: 명령 이름(예: `'/help'`).
            desc: 명령 설명 텍스트입니다.
            keywords: 일치를 위한 공백으로 구분된 숨겨진 키워드입니다.

        Returns:
            점수 값이 높을수록 더 나은 일치 품질을 나타냅니다.

        """
        if not search:
            return 0.0
        name = cmd.lstrip("/").lower()
        lower_desc = desc.lower()
        # Prefix match on command name — highest priority
        if name.startswith(search):
            return 200.0
        # Substring match on command name
        if search in name:
            return 150.0
        # Hidden keyword match — treated like a word-boundary description match
        if keywords and len(search) >= _MIN_DESC_SEARCH_LEN:
            for kw in keywords.lower().split():
                if kw.startswith(search) or search in kw:
                    return 120.0
        # Substring match on description (require ≥2 chars to avoid single-letter noise)
        if len(search) >= _MIN_DESC_SEARCH_LEN and search in lower_desc:
            idx = lower_desc.find(search)
            # Word-boundary bonus: match at start of description or after a space
            if idx == 0 or lower_desc[idx - 1] == " ":
                return 110.0
            return 90.0
        # Fuzzy match via SequenceMatcher on name + desc
        name_ratio = SequenceMatcher(None, search, name).ratio()
        desc_ratio = SequenceMatcher(None, search, lower_desc).ratio()
        best = max(name_ratio * 60, desc_ratio * 30)
        return best if best >= _MIN_SLASH_FUZZY_SCORE else 0.0

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """텍스트가 변경되면 제안을 업데이트합니다."""
        if cursor_index < 0 or cursor_index > len(text):
            self.reset()
            return

        if not self.can_handle(text, cursor_index):
            self.reset()
            return

        # Get the search string (text after /)
        search = text[1:cursor_index].lower()

        if not search:
            # No search text — show all commands (display only cmd + desc)
            suggestions = [(cmd, desc) for cmd, desc, _ in self._commands][
                :MAX_SUGGESTIONS
            ]
        else:
            # Score and filter commands using fuzzy matching
            scored = [
                (score, cmd, desc)
                for cmd, desc, kw in self._commands
                if (score := self._score_command(search, cmd, desc, kw)) > 0
            ]
            scored.sort(key=lambda x: -x[0])
            suggestions = [(cmd, desc) for _, cmd, desc in scored[:MAX_SUGGESTIONS]]

        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._view.render_completion_suggestions(
                self._suggestions, self._selected_index
            )
        else:
            self.reset()

    def on_key(
        self, event: events.Key, _text: str, cursor_index: int
    ) -> CompletionResult:
        """탐색 및 선택을 위한 주요 이벤트를 처리합니다.

        Returns:
            키가 처리된 방법을 나타내는 CompletionResult입니다.

        """
        if not self._suggestions:
            return CompletionResult.IGNORED

        match event.key:
            case "tab":
                if self._apply_selected_completion(cursor_index):
                    return CompletionResult.HANDLED
                return CompletionResult.IGNORED
            case "enter":
                if self._apply_selected_completion(cursor_index):
                    return CompletionResult.SUBMIT
                return CompletionResult.HANDLED
            case "down":
                self._move_selection(1)
                return CompletionResult.HANDLED
            case "up":
                self._move_selection(-1)
                return CompletionResult.HANDLED
            case "escape":
                self.reset()
                return CompletionResult.HANDLED
            case _:
                return CompletionResult.IGNORED

    def _move_selection(self, delta: int) -> None:
        """선택 항목을 위나 아래로 이동합니다."""
        if not self._suggestions:
            return
        count = len(self._suggestions)
        self._selected_index = (self._selected_index + delta) % count
        self._view.render_completion_suggestions(
            self._suggestions, self._selected_index
        )

    def _apply_selected_completion(self, cursor_index: int) -> bool:
        """현재 선택한 완성을 적용합니다.

        Returns:
            완성이 적용되면 True이고, 제안이 없으면 False입니다.

        """
        if not self._suggestions:
            return False

        command, _ = self._suggestions[self._selected_index]
        # Replace from start to cursor with the command
        self._view.replace_completion_range(0, cursor_index, command)
        self.reset()
        return True


# ============================================================================
# Fuzzy File Completion (from project root)
# ============================================================================

# Constants for fuzzy file completion
_MAX_FALLBACK_FILES = 1000
"""git이 아닌 glob 대체에 의해 반환된 파일에 대한 하드 캡입니다."""

_MIN_FUZZY_SCORE = 15
"""파일 완성 결과에 포함할 최소 점수입니다."""

_MIN_FUZZY_RATIO = 0.4
"""파일 이름만 퍼지 일치하는 SequenceMatcher 임계값입니다."""


def _get_project_files(root: Path) -> list[str]:
    """git ls-files를 사용하거나 glob으로 대체하여 프로젝트 파일을 가져옵니다.

    Returns:
        프로젝트 루트의 상대 파일 경로 목록입니다.

    """
    git_path = _get_git_executable()
    if git_path:
        try:
            # S603: git_path is validated via shutil.which(), args are hardcoded
            result = subprocess.run(  # noqa: S603
                [git_path, "ls-files"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                return [f for f in files if f]  # Filter empty strings
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Fallback: simple glob (limited depth to avoid slowness)
    files = []
    try:
        for pattern in ["*", "*/*", "*/*/*", "*/*/*/*"]:
            for p in root.glob(pattern):
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    files.append(p.relative_to(root).as_posix())
                if len(files) >= _MAX_FALLBACK_FILES:
                    break
            if len(files) >= _MAX_FALLBACK_FILES:
                break
    except OSError:
        pass
    return files


def _fuzzy_score(query: str, candidate: str) -> float:
    """쿼리에 대해 후보자의 점수를 매깁니다. 높을수록 더 잘 일치합니다.

    Returns:
        점수 값이 높을수록 더 나은 일치 품질을 나타냅니다.

    """
    query_lower = query.lower()
    # Normalize path separators for cross-platform support
    candidate_normalized = candidate.replace("\\", "/")
    candidate_lower = candidate_normalized.lower()

    # Extract filename for matching (prioritize filename over full path)
    filename = candidate_normalized.rsplit("/", 1)[-1].lower()
    filename_start = candidate_lower.rfind("/") + 1

    # Check filename first (higher priority)
    if query_lower in filename:
        idx = filename.find(query_lower)
        # Bonus for being at start of filename
        if idx == 0:
            return 150 + (1 / len(candidate))
        # Bonus for word boundary in filename
        if idx > 0 and filename[idx - 1] in "_-.":
            return 120 + (1 / len(candidate))
        return 100 + (1 / len(candidate))

    # Check full path
    if query_lower in candidate_lower:
        idx = candidate_lower.find(query_lower)
        # At start of filename
        if idx == filename_start:
            return 80 + (1 / len(candidate))
        # At word boundary in path
        if idx == 0 or candidate[idx - 1] in "/_-.":
            return 60 + (1 / len(candidate))
        return 40 + (1 / len(candidate))

    # Fuzzy match on filename only (more relevant)
    filename_ratio = SequenceMatcher(None, query_lower, filename).ratio()
    if filename_ratio > _MIN_FUZZY_RATIO:
        return filename_ratio * 30

    # Fallback: fuzzy on full path
    ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()
    return ratio * 15


def _is_dotpath(path: str) -> bool:
    """경로에 dotfiles/dotdirs(예: .github/...)가 포함되어 있는지 확인하세요.

    Returns:
        경로에 숨겨진 디렉터리나 파일이 포함되어 있으면 참입니다.

    """
    return any(part.startswith(".") for part in path.split("/"))


def _path_depth(path: str) -> int:
    """경로의 깊이(/구분 기호 수)를 가져옵니다.

    Returns:
        경로의 경로 구분 기호 수입니다.

    """
    return path.count("/")


def _fuzzy_search(
    query: str,
    candidates: list[str],
    limit: int = 10,
    *,
    include_dotfiles: bool = False,
) -> list[str]:
    """점수별로 정렬된 상위 일치 항목을 반환합니다.

    Args:
        query: 검색어
        candidates: 검색할 파일 경로 목록
        limit: 반환할 최대 결과
        include_dotfiles: 도트 파일 포함 여부(기본값 False)

    Returns:
        관련성 점수를 기준으로 정렬된 일치하는 파일 경로 목록입니다.

    """
    # Filter dotfiles unless explicitly searching for them
    filtered = (
        candidates
        if include_dotfiles
        else [c for c in candidates if not _is_dotpath(c)]
    )

    if not query:
        # Empty query: show root-level files first, sorted by depth then name
        sorted_files = sorted(filtered, key=lambda p: (_path_depth(p), p.lower()))
        return sorted_files[:limit]

    scored = [
        (score, c)
        for c in filtered
        if (score := _fuzzy_score(query, c)) >= _MIN_FUZZY_SCORE
    ]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


class FuzzyFileController:
    """프로젝트 루트에서 퍼지 일치를 사용하여 @ 파일 완성을 위한 컨트롤러입니다."""

    def __init__(
        self,
        view: CompletionView,
        cwd: Path | None = None,
    ) -> None:
        """퍼지 파일 컨트롤러를 초기화합니다.

        Args:
            view: 제안을 렌더링하기 위한 보기
            cwd: 프로젝트 루트를 찾을 시작 디렉터리

        """
        self._view = view
        self._cwd = cwd or Path.cwd()
        self._project_root = find_project_root(self._cwd) or self._cwd
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0
        self._file_cache: list[str] | None = None

    def _get_files(self) -> list[str]:
        """캐시된 파일 목록을 가져오거나 새로 고칩니다.

        Returns:
            프로젝트 파일 경로 목록입니다.

        """
        if self._file_cache is None:
            self._file_cache = _get_project_files(self._project_root)
        return self._file_cache

    def refresh_cache(self) -> None:
        """파일 캐시를 강제로 새로 고칩니다."""
        self._file_cache = None

    async def warm_cache(self) -> None:
        """이벤트 루프에서 파일 캐시를 미리 채웁니다."""
        if self._file_cache is not None:
            return
        # Best-effort; _get_files() falls back to sync on failure.
        with contextlib.suppress(Exception):
            self._file_cache = await asyncio.to_thread(
                _get_project_files, self._project_root
            )

    @staticmethod
    def can_handle(text: str, cursor_index: int) -> bool:
        """뒤에 공백이 없고 @가 포함된 입력을 처리합니다.

        Returns:
            커서가 @ 뒤에 있고 파일 언급 컨텍스트 내에 있으면 참입니다.

        """
        if cursor_index <= 0 or cursor_index > len(text):
            return False

        before_cursor = text[:cursor_index]
        if "@" not in before_cursor:
            return False

        at_index = before_cursor.rfind("@")
        if cursor_index <= at_index:
            return False

        # Fragment from @ to cursor must not contain spaces
        fragment = before_cursor[at_index:cursor_index]
        return bool(fragment) and " " not in fragment

    def reset(self) -> None:
        """명확한 제안."""
        if self._suggestions:
            self._suggestions.clear()
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """텍스트가 변경되면 제안을 업데이트합니다."""
        if not self.can_handle(text, cursor_index):
            self.reset()
            return

        before_cursor = text[:cursor_index]
        at_index = before_cursor.rfind("@")
        search = before_cursor[at_index + 1 :]

        suggestions = self._get_fuzzy_suggestions(search)

        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._view.render_completion_suggestions(
                self._suggestions, self._selected_index
            )
        else:
            self.reset()

    def _get_fuzzy_suggestions(self, search: str) -> list[tuple[str, str]]:
        """퍼지 파일 제안을 받으세요.

        Returns:
            일치하는 파일에 대한 (label, type_hint) 튜플 목록입니다.

        """
        files = self._get_files()
        # Include dotfiles only if query starts with "."
        include_dots = search.startswith(".")
        matches = _fuzzy_search(
            search, files, limit=MAX_SUGGESTIONS, include_dotfiles=include_dots
        )

        suggestions: list[tuple[str, str]] = []
        for path in matches:
            # Get file extension for type hint
            ext = Path(path).suffix.lower()
            type_hint = ext[1:] if ext else "file"
            suggestions.append((f"@{path}", type_hint))

        return suggestions

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """탐색 및 선택을 위한 주요 이벤트를 처리합니다.

        Returns:
            키가 처리된 방법을 나타내는 CompletionResult입니다.

        """
        if not self._suggestions:
            return CompletionResult.IGNORED

        match event.key:
            case "tab" | "enter":
                if self._apply_selected_completion(text, cursor_index):
                    return CompletionResult.HANDLED
                return CompletionResult.IGNORED
            case "down":
                self._move_selection(1)
                return CompletionResult.HANDLED
            case "up":
                self._move_selection(-1)
                return CompletionResult.HANDLED
            case "escape":
                self.reset()
                return CompletionResult.HANDLED
            case _:
                return CompletionResult.IGNORED

    def _move_selection(self, delta: int) -> None:
        """선택 항목을 위나 아래로 이동합니다."""
        if not self._suggestions:
            return
        count = len(self._suggestions)
        self._selected_index = (self._selected_index + delta) % count
        self._view.render_completion_suggestions(
            self._suggestions, self._selected_index
        )

    def _apply_selected_completion(self, text: str, cursor_index: int) -> bool:
        """현재 선택한 완성을 적용합니다.

        Returns:
            완성이 적용된 경우 True이고, 제안 사항이 없거나 잘못된 상태인 경우 False입니다.

        """
        if not self._suggestions:
            return False

        label, _ = self._suggestions[self._selected_index]
        before_cursor = text[:cursor_index]
        at_index = before_cursor.rfind("@")

        if at_index < 0:
            return False

        # Replace from @ to cursor with the completion
        self._view.replace_completion_range(at_index, cursor_index, label)
        self.reset()
        return True


# Keep old name as alias for backwards compatibility
PathCompletionController = FuzzyFileController


# ============================================================================
# Multi-Completion Manager
# ============================================================================


class MultiCompletionManager:
    """활성 컨트롤러에 위임하여 여러 완료 컨트롤러를 관리합니다."""

    def __init__(self, controllers: list[CompletionController]) -> None:
        """컨트롤러 목록으로 초기화합니다.

        Args:
            controllers: 완료 컨트롤러 목록(순서대로 확인)

        """
        self._controllers = controllers
        self._active: CompletionController | None = None

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        """적절한 컨트롤러를 활성화하여 텍스트 변경을 처리합니다."""
        # Find the first controller that can handle this input
        candidate = None
        for controller in self._controllers:
            if controller.can_handle(text, cursor_index):
                candidate = controller
                break

        # No controller can handle - reset if we had one active
        if candidate is None:
            if self._active is not None:
                self._active.reset()
                self._active = None
            return

        # Switch to new controller if different
        if candidate is not self._active:
            if self._active is not None:
                self._active.reset()
            self._active = candidate

        # Let the active controller process the change
        candidate.on_text_changed(text, cursor_index)

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        """활성 컨트롤러에 위임하여 키 이벤트를 처리합니다.

        Returns:
            활성 컨트롤러의 CompletionResult이거나 활성 컨트롤러가 없으면 무시됩니다.

        """
        if self._active is None:
            return CompletionResult.IGNORED
        return self._active.on_key(event, text, cursor_index)

    def reset(self) -> None:
        """모든 컨트롤러를 재설정합니다."""
        if self._active is not None:
            self._active.reset()
            self._active = None
