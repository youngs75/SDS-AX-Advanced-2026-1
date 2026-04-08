"""CLI 명령용 스킬 로더.

이 모듈은 CLI 작업(목록, 생성, 정보, 삭제)을 위한 파일 시스템 기반 기술 검색을 제공합니다. deepagents.middleware.skills에서
사전 구축된 미들웨어 기능을 래핑하고 CLI 명령에 필요한 직접 파일 시스템 액세스에 맞게 조정합니다.

에이전트 내에서 미들웨어를 사용하려면 deepagents.middleware.skills.SkillsMiddleware를 직접 사용하세요.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

from deepagents.backends.filesystem import FilesystemBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
from deepagents.middleware.skills import (
    SkillMetadata,
    _list_skills as list_skills_from_backend,  # noqa: PLC2701  # Intentional access to internal skill listing
)

from deepagents_cli._version import __version__ as _cli_version

logger = logging.getLogger(__name__)


class ExtendedSkillMetadata(SkillMetadata):
    """CLI 디스플레이를 위한 확장된 기술 메타데이터에 소스 추적이 추가됩니다.

    Attributes:
        source: 스킬의 유래. `'built-in'`, `'user'`, `'project'` 또는 `'claude (experimental)'`
                중 하나입니다.

    """

    source: Literal["built-in", "user", "project", "claude (experimental)"]


# Re-export for CLI commands
__all__ = ["SkillMetadata", "list_skills", "load_skill_content"]


def list_skills(
    *,
    built_in_skills_dir: Path | None = None,
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
    user_agent_skills_dir: Path | None = None,
    project_agent_skills_dir: Path | None = None,
    user_claude_skills_dir: Path | None = None,
    project_claude_skills_dir: Path | None = None,
) -> list[ExtendedSkillMetadata]:
    """기본 제공, 사용자 및/또는 프로젝트 디렉터리의 기술을 나열합니다.

    이는 사전 구축된 미들웨어의 스킬 로딩 기능에 대한 CLI 관련 래퍼입니다. FilesystemBackend를 사용하여 로컬 디렉터리에서 기술을
    로드합니다.

    우선 순위(낮은 것에서 가장 높은 것까지): 0. `built_in_skills_dir`(`<package>/built_in_skills/`) 1.
    `user_skills_dir`(`~/.deepagents/{agent}/skills/`) 2.
    `user_agent_skills_dir`(`~/.agents/skills/`) 3.
    `project_skills_dir`(`.deepagents/skills/`) 4.
    `project_agent_skills_dir`(`.agents/skills/`) 5.
    `user_claude_skills_dir`(`~/.claude/skills/`, 실험적) 6.
    `project_claude_skills_dir`(`.claude/skills/`, 실험적)

    우선 순위가 높은 디렉터리의 기술은 동일한 이름을 가진 기술보다 우선합니다.

    Args:
        built_in_skills_dir: 패키지와 함께 제공되는 내장 기술의 경로입니다.
        user_skills_dir: `~/.deepagents/{agent}/skills/`에 대한 경로입니다.
        project_skills_dir: `.deepagents/skills/`에 대한 경로입니다.
        user_agent_skills_dir: `~/.agents/skills/`(별칭)에 대한 경로입니다.
        project_agent_skills_dir: `.agents/skills/`(별칭)에 대한 경로입니다.
        user_claude_skills_dir: `~/.claude/skills/` 경로(실험용).
        project_claude_skills_dir: `.claude/skills/` 경로(실험용).

    Returns:
        모든 소스의 스킬 메타데이터 목록을 더 높은 우선순위로 병합했습니다.
            이름이 충돌할 때 디렉터리가 우선순위를 갖습니다.

    """
    all_skills: dict[str, ExtendedSkillMetadata] = {}

    sources: list[tuple[Path | None, str, bool]] = [
        (built_in_skills_dir, "built-in", False),
        (user_skills_dir, "user", False),
        (user_agent_skills_dir, "user", False),
        (project_skills_dir, "project", False),
        (project_agent_skills_dir, "project", False),
        (user_claude_skills_dir, "claude (experimental)", True),
        (project_claude_skills_dir, "claude (experimental)", True),
    ]
    """우선순위 순서대로 소스(낮은 것부터 높은 것까지).

    각 튜플: `(directory, source label, is_experimental)`.

    각 소스는 개별적으로 시도/제외 보호되므로 액세스할 수 없는 단일 디렉토리가 나머지를 차단하지 않습니다.

    """

    for skill_dir, source_label, experimental in sources:
        if not skill_dir or not skill_dir.exists():
            continue
        try:
            backend = FilesystemBackend(root_dir=str(skill_dir))
            skills = list_skills_from_backend(backend=backend, source_path=".")
            if experimental and skills:
                logger.info(
                    "Discovered %d skill(s) from experimental Claude path: %s",
                    len(skills),
                    skill_dir,
                )
            for skill in skills:
                extra: dict[str, object] = {"source": source_label}
                if source_label == "built-in":
                    extra["metadata"] = {
                        **skill["metadata"],
                        "deepagents-cli-version": _cli_version,
                    }
                extended = cast("ExtendedSkillMetadata", {**skill, **extra})
                all_skills[skill["name"]] = extended
        except (OSError, KeyError, TypeError):
            logger.warning(
                "Could not load skills from %s",
                skill_dir,
                exc_info=True,
            )

    return list(all_skills.values())


def load_skill_content(
    skill_path: str,
    *,
    allowed_roots: Sequence[Path] = (),
) -> str | None:
    """기술에 대한 전체 원시 SKILL.md 콘텐츠를 읽어보세요.

    YAML 머리말을 포함한 전체 파일 콘텐츠를 반환합니다. 호출자는 필요한 경우 앞부분을 구문 분석하거나 제거할 책임이 있습니다.

    `allowed_roots`이 제공되면 확인된 경로는 최소한 하나의 루트 디렉터리에 속해야 합니다. 이렇게 하면 심볼릭 링크 순회가 알려진 기술
    디렉터리 외부의 파일을 읽는 것을 방지할 수 있습니다.

    Args:
        skill_path: SKILL.md 파일의 경로입니다(`SkillMetadata['path']`에서).
        allowed_roots: 해결된 경로가 스킬 루트 디렉터리에 포함되어야 합니다.

            호출자는 `Path.resolve()`을 통해 이를 미리 해결해야 합니다. 해결된 스킬 경로는 직접 비교되므로 해결되지 않은 루트는
            잘못된 격리 실패를 유발합니다.

            비어 있으면 포함이 확인되지 않습니다.

    Returns:
        SKILL.md 파일의 전체 텍스트 콘텐츠 또는 읽기 실패 시 `None`.

    Raises:
        PermissionError: 확인된 경로가 모두 `allowed_roots` 외부인 경우.

    """
    from pathlib import Path

    path = Path(skill_path).resolve()

    if allowed_roots and not any(path.is_relative_to(root) for root in allowed_roots):
        logger.warning(
            "Skill path %s is outside all allowed roots, refusing to read",
            skill_path,
        )
        from deepagents_cli._env_vars import EXTRA_SKILLS_DIRS

        msg = (
            f"Skill path {skill_path} resolves outside all allowed skill "
            "directories. If this is a symlink, add the target directory to "
            f"{EXTRA_SKILLS_DIRS} or [skills].extra_allowed_dirs "
            "in ~/.deepagents/config.toml."
        )
        raise PermissionError(msg)

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning(
            "Could not read skill content from %s", skill_path, exc_info=True
        )
        return None
