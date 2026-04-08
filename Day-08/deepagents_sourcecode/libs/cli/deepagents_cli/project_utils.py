"""프로젝트에 민감한 CLI 동작에 대한 프로젝트 컨텍스트를 해결합니다.

이 모듈은 MCP 검색, 신뢰 결정 및 서버 하위 프로세스와 같은 기능이 범위에 동의할 수 있도록 안정적인 사용자 및 프로젝트 경로를 도출합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_cli._env_vars import SERVER_ENV_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectContext:
    """프로젝트에 민감한 동작을 위한 명시적인 사용자/프로젝트 경로 컨텍스트입니다.

Attributes:
        user_cwd: CLI 호출의 신뢰할 수 있는 작업 디렉터리입니다.
        project_root: `user_cwd`에 대한 프로젝트 루트가 확인되었습니다(있는 경우).

    """

    user_cwd: Path
    project_root: Path | None = None

    def __post_init__(self) -> None:
        """경로 필드가 절대적인지 확인하십시오.

Raises:
            ValueError: `user_cwd` 또는 `project_root`이 절대적이지 않은 경우.

        """
        if not self.user_cwd.is_absolute():
            msg = f"user_cwd must be absolute, got {self.user_cwd!r}"
            raise ValueError(msg)
        if self.project_root is not None and not self.project_root.is_absolute():
            msg = f"project_root must be absolute, got {self.project_root!r}"
            raise ValueError(msg)

    @classmethod
    def from_user_cwd(cls, user_cwd: str | Path) -> ProjectContext:
        """명시적인 사용자 작업 디렉터리에서 프로젝트 컨텍스트를 빌드합니다.

Args:
            user_cwd: 사용자 호출 디렉터리.

Returns:
            프로젝트 컨텍스트가 해결되었습니다.

        """
        resolved_cwd = Path(user_cwd).expanduser().resolve()
        return cls(
            user_cwd=resolved_cwd,
            project_root=find_project_root(resolved_cwd),
        )

    def resolve_user_path(self, path: str | Path) -> Path:
        """명시적인 사용자 작업 디렉터리에 상대적인 경로를 확인합니다.

Args:
            path: 절대 또는 상대 사용자 대상 경로입니다.

Returns:
            절대 해결 경로.

        """
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.user_cwd / candidate).resolve()

    def project_agent_md_paths(self) -> list[Path]:
        """이 컨텍스트에 대한 프로젝트 수준 `AGENTS.md` 파일을 반환합니다."""
        if self.project_root is None:
            return []
        return find_project_agent_md(self.project_root)

    def project_skills_dir(self) -> Path | None:
        """프로젝트 `.deepagents/skills` 디렉터리가 있으면 반환합니다."""
        if self.project_root is None:
            return None
        return self.project_root / ".deepagents" / "skills"

    def project_agents_dir(self) -> Path | None:
        """프로젝트 `.deepagents/agents` 디렉터리가 있으면 반환합니다."""
        if self.project_root is None:
            return None
        return self.project_root / ".deepagents" / "agents"

    def project_agent_skills_dir(self) -> Path | None:
        """프로젝트 `.agents/skills` 디렉터리가 있으면 반환합니다."""
        if self.project_root is None:
            return None
        return self.project_root / ".agents" / "skills"


def get_server_project_context(
    env: Mapping[str, str] | None = None,
) -> ProjectContext | None:
    """환경 전송 데이터에서 서버 프로젝트 컨텍스트를 읽습니다.

Args:
        env: 읽을 환경 매핑입니다.

Returns:
        재구성된 프로젝트 컨텍스트 또는 서버 컨텍스트가 없는 경우 `None`입니다.

    """
    environment = os.environ if env is None else env
    raw_cwd = environment.get(f"{SERVER_ENV_PREFIX}CWD")
    if not raw_cwd:
        return None

    try:
        user_cwd = Path(raw_cwd).expanduser().resolve()
        raw_project_root = environment.get(f"{SERVER_ENV_PREFIX}PROJECT_ROOT")
        project_root = (
            Path(raw_project_root).expanduser().resolve()
            if raw_project_root
            else find_project_root(user_cwd)
        )
    except OSError:
        logger.warning(
            "Could not resolve server project context from CWD=%s",
            raw_cwd,
            exc_info=True,
        )
        return None

    return ProjectContext(user_cwd=user_cwd, project_root=project_root)


def find_project_root(start_path: str | Path | None = None) -> Path | None:
    """.git 디렉토리를 찾아 프로젝트 루트를 찾으세요.

    프로젝트 루트를 나타내는 .git 디렉터리를 찾기 위해 start_path(또는 cwd)에서 디렉터리 트리를 탐색합니다.

Args:
        start_path: 검색을 시작할 디렉터리입니다. 기본값은 현재 작업 디렉터리입니다.

Returns:
        발견되면 프로젝트 루트에 대한 경로이고, 그렇지 않으면 없음입니다.

    """
    current = Path(start_path or Path.cwd()).expanduser().resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def find_project_agent_md(project_root: Path) -> list[Path]:
    """프로젝트별 AGENTS.md 파일을 찾으세요.

    두 위치를 확인하고 존재하는 모든 항목을 반환합니다. 1. project_root/.deepagents/AGENTS.md 2.
    project_root/AGENTS.md

    두 파일이 모두 존재하는 경우 두 파일이 모두 로드되어 결합됩니다.

Args:
        project_root: 프로젝트 루트 디렉터리의 경로입니다.

Returns:
        기존 AGENTS.md 경로.

            파일이 둘 다 없으면 비어 있고, 하나만 있으면 항목이 하나이고, 두 위치에 모두 파일이 있으면 항목이 두 개입니다.

    """
    candidates = [
        project_root / ".deepagents" / "AGENTS.md",
        project_root / "AGENTS.md",
    ]
    paths: list[Path] = []
    for candidate in candidates:
        try:
            if candidate.exists():
                paths.append(candidate)
        except OSError:
            pass
    return paths
