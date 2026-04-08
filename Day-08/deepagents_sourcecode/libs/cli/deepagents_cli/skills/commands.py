"""스킬 관리를 위한 CLI 명령입니다.

이러한 명령은 main.py를 통해 CLI에 등록됩니다. - deepagents Skill list [옵션] - deepagents Skill create
<이름> [옵션] - deepagents Skill Info <이름> [옵션] - deepagents Skill delete <이름> [옵션]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents.middleware.skills import SkillMetadata

    from deepagents_cli.output import OutputFormat

from deepagents_cli import theme

MAX_SKILL_NAME_LENGTH = 64


def _validate_name(name: str) -> tuple[bool, str]:
    """에이전트 기술 사양에 따라 이름을 확인하세요.

    요구 사항(https://agentskills.io/specification): - 최대 64자 - 유니코드 소문자 영숫자 및 하이픈만 - 하이픈으로
    시작하거나 끝날 수 없음 - 연속 하이픈 없음 - 경로 순회 시퀀스 없음

    유니코드 소문자 영숫자는 `c.isalpha() and c.islower()` 또는 `c.isdigit()`이 `True`를 반환하는 모든 문자를
    의미하며, 여기에는 악센트가 있는 라틴 문자(예: `'cafe'`, `'uber-tool'`) 및 기타 스크립트가 포함됩니다.  이는 SDK의
    `_validate_skill_name` 구현과 일치합니다.

    Args:
        name: 유효성을 검사할 이름입니다.

    Returns:
        (is_valid, error_message)의 튜플입니다. 유효한 경우 error_message는 비어 있습니다.

    """
    # Check for empty or whitespace-only names
    if not name or not name.strip():
        return False, "cannot be empty"

    # Check length (spec: max 64 chars)
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "cannot exceed 64 characters"

    # Check for path traversal sequences (CLI-specific; the SDK validates
    # against the directory name instead, but the CLI accepts user input
    # directly so we need explicit path-safety checks)
    if ".." in name or "/" in name or "\\" in name:
        return False, "cannot contain path components"

    # Structural hyphen checks
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return (
            False,
            "must be lowercase alphanumeric with single hyphens only",
        )

    # Character-by-character check (matches SDK's _validate_skill_name)
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return (
            False,
            "must be lowercase alphanumeric with single hyphens only",
        )

    return True, ""


def _validate_skill_path(skill_dir: Path, base_dir: Path) -> tuple[bool, str]:
    """해결된 기술 디렉터리가 기본 디렉터리 내에 있는지 확인합니다.

    Args:
        skill_dir: 유효성을 검사할 기술 디렉터리 경로
        base_dir: Skill_dir을 포함해야 하는 기본 기술 디렉터리

    Returns:
        (is_valid, error_message)의 튜플입니다. 유효한 경우 error_message는 비어 있습니다.

    """
    try:
        # Resolve both paths to their canonical form
        resolved_skill = skill_dir.resolve()
        resolved_base = base_dir.resolve()

        # Check if skill_dir is within base_dir
        if not resolved_skill.is_relative_to(resolved_base):
            return False, f"Skill directory must be within {base_dir}"
    except (OSError, RuntimeError) as e:
        return False, f"Invalid path: {e}"
    else:
        return True, ""


def _format_info_fields(skill: SkillMetadata) -> list[tuple[str, str]]:
    """표시할 비어 있지 않은 선택적 메타데이터 필드를 추출합니다.

    업스트림 `_parse_skill_metadata`은 빈/공백 라이선스 및 호환성 값을 `None`로 정규화하므로 아래의 진실성 확인만으로 충분합니다.

    Args:
        skill: 표시 필드를 추출할 기술 메타데이터입니다.

    Returns:
        비어 있지 않은 필드에 대한 (레이블, 값) 튜플의 순서가 지정된 목록입니다.
            Fields appear in order: 라이센스, 호환성, 허용된 도구,
            메타데이터.

    """
    fields: list[tuple[str, str]] = []
    license_val = skill.get("license")
    if license_val:
        fields.append(("License", license_val))
    compat_val = skill.get("compatibility")
    if compat_val:
        fields.append(("Compatibility", compat_val))
    if skill.get("allowed_tools"):
        fields.append(
            ("Allowed Tools", ", ".join(str(t) for t in skill["allowed_tools"]))
        )
    meta = skill.get("metadata")
    if meta and isinstance(meta, dict):
        formatted = ", ".join(f"{k}={v}" for k, v in meta.items())
        fields.append(("Metadata", formatted))
    return fields


def _list(
    agent: str, *, project: bool = False, output_format: OutputFormat = "text"
) -> None:
    """지정된 에이전트에게 사용 가능한 모든 기술을 나열합니다.

    Args:
        agent: 스킬에 대한 에이전트 식별자(기본값: 에이전트).
        project: True인 경우 프로젝트 기술만 표시합니다. False인 경우 모든 기술(사용자 + 프로젝트)을 표시합니다.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    """
    from rich.markup import escape as escape_markup

    from deepagents_cli.config import Settings, console, get_glyphs
    from deepagents_cli.skills.load import list_skills

    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    # If --project flag is used, only show project skills
    if project:
        if not project_skills_dir:
            if output_format == "json":
                from deepagents_cli.output import write_json

                write_json("skills list", [])
                return
            console.print("[yellow]Not in a project directory.[/yellow]")
            console.print(
                "[dim]Project skills require a .git directory "
                "in the project root.[/dim]",
                style=theme.MUTED,
            )
            return

        # Check both project skill directories
        has_deepagents_skills = project_skills_dir.exists() and any(
            project_skills_dir.iterdir()
        )
        has_agent_skills = (
            project_agent_skills_dir
            and project_agent_skills_dir.exists()
            and any(project_agent_skills_dir.iterdir())
        )

        if not has_deepagents_skills and not has_agent_skills:
            if output_format == "json":
                from deepagents_cli.output import write_json

                write_json("skills list", [])
                return
            console.print("[yellow]No project skills found.[/yellow]")
            console.print(
                f"[dim]Project skills will be created in {project_skills_dir}/ "
                "when you add them.[/dim]",
                style=theme.MUTED,
            )
            console.print(
                "\n[dim]Create a project skill:\n"
                "  deepagents skills create my-skill --project[/dim]",
                style=theme.MUTED,
            )
            return

        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )

        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json("skills list", [dict(s) for s in skills])
            return

        console.print("\n[bold]Project Skills:[/bold]\n", style=theme.PRIMARY)
    else:
        # Load skills from all directories (including built-in)
        skills = list_skills(
            built_in_skills_dir=settings.get_built_in_skills_dir(),
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )

        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json("skills list", [dict(s) for s in skills])
            return

        if not skills:
            console.print()
            console.print("[yellow]No skills found.[/yellow]")
            console.print()
            console.print(
                "[dim]Skills are loaded from these directories "
                "(highest precedence first):\n"
                "  1. .agents/skills/                 project skills\n"
                "  2. .deepagents/skills/             project skills (alias)\n"
                "  3. ~/.agents/skills/               user skills\n"
                "  4. ~/.deepagents/<agent>/skills/   user skills (alias)\n"
                "  5. <package>/built_in_skills/      built-in skills[/dim]",
                style=theme.MUTED,
            )
            console.print(
                "\n[dim]Create your first skill:\n"
                "  deepagents skills create my-skill[/dim]",
                style=theme.MUTED,
            )
            return

        console.print("\n[bold]Available Skills:[/bold]\n", style=theme.PRIMARY)

    # Group skills by source
    user_skills = [s for s in skills if s["source"] == "user"]
    project_skills_list = [s for s in skills if s["source"] == "project"]
    built_in_skills_list = [s for s in skills if s["source"] == "built-in"]

    # Show user skills
    if user_skills and not project:
        console.print("[bold cyan]User Skills:[/bold cyan]", style=theme.PRIMARY)
        bullet = get_glyphs().bullet
        for skill in user_skills:
            skill_path = Path(skill["path"])
            name = escape_markup(skill["name"])
            console.print(f"  {bullet} [bold]{name}[/bold]", style=theme.PRIMARY)
            console.print(
                f"    {escape_markup(str(skill_path.parent))}/",
                style=theme.MUTED,
            )
            console.print()
            console.print(
                f"    {escape_markup(skill['description'])}",
                style=theme.MUTED,
            )
            console.print()

    # Show project skills
    if project_skills_list:
        if not project and user_skills:
            console.print()
        console.print("[bold green]Project Skills:[/bold green]", style=theme.PRIMARY)
        bullet = get_glyphs().bullet
        for skill in project_skills_list:
            skill_path = Path(skill["path"])
            name = escape_markup(skill["name"])
            console.print(f"  {bullet} [bold]{name}[/bold]", style=theme.PRIMARY)
            console.print(
                f"    {escape_markup(str(skill_path.parent))}/",
                style=theme.MUTED,
            )
            console.print()
            console.print(
                f"    {escape_markup(skill['description'])}",
                style=theme.MUTED,
            )
            console.print()

    # Show built-in skills
    if built_in_skills_list and not project:
        if user_skills or project_skills_list:
            console.print()
        console.print(
            "[bold magenta]Built-in Skills:[/bold magenta]", style=theme.PRIMARY
        )
        bullet = get_glyphs().bullet
        for skill in built_in_skills_list:
            name = escape_markup(skill["name"])
            console.print(f"  {bullet} [bold]{name}[/bold]", style=theme.PRIMARY)
            console.print()
            console.print(
                f"    {escape_markup(skill['description'])}",
                style=theme.MUTED,
            )
            console.print()


def _generate_template(skill_name: str) -> str:
    """새 기술에 대한 `SKILL.md` 템플릿을 생성합니다.

    템플릿은 에이전트 기술 사양(https://agentskills.io/specification) 및 기술 생성자 지침:

    - 설명에는 "사용 시기" 트리거 정보가 포함됩니다(본문 아님) - 본문에는 스킬 트리거 이후 로드된 지침만 포함됩니다.

    Args:
        skill_name: 스킬의 이름(머리말과 제목에 사용됨)

    Returns:
        YAML 머리말과 마크다운 본문으로 `SKILL.md` 콘텐츠를 완성하세요.

    """
    title = skill_name.title().replace("-", " ")
    description = (
        "TODO: Explain what this skill does and when to use it. "
        "Include specific triggers — scenarios, file types, or phrases "
        "that should activate this skill. Example: 'Create and edit PDF "
        "documents. Use when the user asks to merge, split, fill, or "
        "annotate PDF files.'"
    )
    return f"""---
name: {skill_name}
description: "{description}"
# (Warning: SKILL.md files exceeding 10 MB are silently skipped at load time.)
# Optional fields per Agent Skills spec:
# license: Apache-2.0
# compatibility: Designed for Deep Agents CLI
# metadata:
#   author: your-org
#   version: "1.0"
# allowed-tools: Bash(git:*) Read
---

# {title}

## Overview

[TODO: 1-2 sentences explaining what this skill enables]

## Instructions

### Step 1: [First Action]
[Explain what to do first]

### Step 2: [Second Action]
[Explain what to do next]

### Step 3: [Final Action]
[Explain how to complete the task]

## Best Practices

- [Best practice 1]
- [Best practice 2]
- [Best practice 3]

## Examples

### Example 1: [Scenario Name]

**User Request:** "[Example user request]"

**Approach:**
1. [Step-by-step breakdown]
2. [Using tools and commands]
3. [Expected outcome]
"""


def _create(
    skill_name: str,
    agent: str,
    project: bool = False,
    *,
    output_format: OutputFormat = "text",
) -> None:
    """템플릿 SKILL.md 파일을 사용하여 새 기술을 만듭니다.

    Args:
        skill_name: 생성할 스킬의 이름입니다.
        agent: 스킬에 대한 에이전트 식별자
        project: True인 경우 프로젝트 기술 디렉터리에 만듭니다. False인 경우 사용자 기술 디렉터리에 만듭니다.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    Raises:
        SystemExit: 스킬 이름이 유효하지 않거나 디렉터리를 생성할 수 없는 경우.

    """
    from deepagents_cli.config import Settings, console, get_glyphs

    # Validate skill name first (per Agent Skills spec)
    is_valid, error_msg = _validate_name(skill_name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] Invalid skill name: {error_msg}")
        console.print(
            "[dim]Per Agent Skills spec: names must be lowercase alphanumeric "
            "with hyphens only.\n"
            "Examples: web-research, code-review, data-analysis[/dim]",
            style=theme.MUTED,
        )
        raise SystemExit(1)

    # Determine target directory
    settings = Settings.from_environment()
    if project:
        if not settings.project_root:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            console.print(
                "[dim]Project skills require a .git directory "
                "in the project root.[/dim]",
                style=theme.MUTED,
            )
            raise SystemExit(1)
        skills_dir = settings.ensure_project_skills_dir()
        if skills_dir is None:
            console.print(
                "[bold red]Error:[/bold red] Could not create project skills directory."
            )
            raise SystemExit(1)
    else:
        skills_dir = settings.ensure_user_skills_dir(agent)

    skill_dir = skills_dir / skill_name

    # Validate the resolved path is within skills_dir
    is_valid_path, path_error = _validate_skill_path(skill_dir, skills_dir)
    if not is_valid_path:
        console.print(f"[bold red]Error:[/bold red] {path_error}")
        raise SystemExit(1)

    if skill_dir.exists():
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json(
                "skills create",
                {
                    "name": skill_name,
                    "path": str(skill_dir),
                    "project": project,
                    "already_existed": True,
                },
            )
            return
        console.print(
            f"Skill '{skill_name}' already exists at {skill_dir}",
            style=theme.MUTED,
        )
        return

    # Create skill directory
    skill_dir.mkdir(parents=True, exist_ok=True)

    template = _generate_template(skill_name)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(template)

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json(
            "skills create",
            {
                "name": skill_name,
                "path": str(skill_dir),
                "project": project,
            },
        )
        return

    checkmark = get_glyphs().checkmark
    console.print(
        f"\n[bold]{checkmark} Skill '{skill_name}' created successfully![/bold]",
        style=theme.PRIMARY,
    )
    console.print(f"Location: {skill_dir}\n", style=theme.MUTED)
    console.print(
        "[dim]Edit the SKILL.md file to customize:\n"
        "  1. Update the description in YAML frontmatter\n"
        "  2. Fill in the instructions and examples\n"
        "  3. Add any supporting files (scripts, configs, etc.)\n"
        "\n"
        f"  nano {skill_md}\n"
        "\n"
        "  See examples/skills/ in the deepagents-cli repo for example skills:\n"
        "   - web-research: Structured research workflow\n"
        "   - langgraph-docs: LangGraph documentation lookup\n"
        "\n"
        "   Copy an example:\n"
        "   cp -r examples/skills/web-research ~/.deepagents/agent/skills/\n",
        style=theme.MUTED,
    )


def _info(
    skill_name: str,
    *,
    agent: str = "agent",
    project: bool = False,
    output_format: OutputFormat = "text",
) -> None:
    """특정 기술에 대한 자세한 정보를 표시합니다.

    Args:
        skill_name: 정보를 표시할 스킬의 이름입니다.
        agent: 스킬에 대한 에이전트 식별자(기본값: 에이전트).
        project: True인 경우 프로젝트 기술에서만 검색합니다. False인 경우 사용자 및 프로젝트 기술을 모두 검색합니다.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    Raises:
        SystemExit: 스킬을 찾을 수 없거나 프로젝트 디렉터리에 없는 경우.

    """
    from rich.markup import escape as escape_markup

    from deepagents_cli.config import Settings, console
    from deepagents_cli.skills.load import list_skills

    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    # Load skills based on --project flag
    if project:
        if not project_skills_dir:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            raise SystemExit(1)
        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )
    else:
        skills = list_skills(
            built_in_skills_dir=settings.get_built_in_skills_dir(),
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )

    # Find the skill
    skill = next((s for s in skills if s["name"] == skill_name), None)

    if not skill:
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found.")
        console.print("\n[dim]Available skills:[/dim]", style=theme.MUTED)
        for s in skills:
            console.print(f"  - {s['name']}", style=theme.MUTED)
        raise SystemExit(1)

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json("skills info", dict(skill))
        return

    # Read the full SKILL.md file
    skill_path = Path(skill["path"])
    skill_content = skill_path.read_text(encoding="utf-8")

    # Determine source label
    source_labels = {
        "project": ("Project Skill", "green"),
        "user": ("User Skill", "cyan"),
        "built-in": ("Built-in Skill", "magenta"),
    }
    source_label, source_color = source_labels.get(skill["source"], ("Skill", "dim"))

    # Check if this project skill shadows a user skill with the same name.
    # This is a cosmetic hint — if the second list_skills() call fails
    # (e.g. permission error reading user dirs) we silently skip the warning
    # rather than crashing the entire `skills info` display.
    shadowed_user_skill = False
    if skill["source"] == "project" and not project:
        try:
            user_only = list_skills(
                user_skills_dir=user_skills_dir,
                project_skills_dir=None,
                user_agent_skills_dir=user_agent_skills_dir,
                project_agent_skills_dir=None,
            )
            shadowed_user_skill = any(s["name"] == skill_name for s in user_only)
        except Exception:  # noqa: BLE001, S110  # Shadow detection is cosmetic, safe to swallow
            pass

    console.print(
        f"\n[bold]Skill: {escape_markup(skill['name'])}[/bold] "
        f"[bold {source_color}]({source_label})[/bold {source_color}]\n",
        style=theme.PRIMARY,
    )
    if shadowed_user_skill:
        console.print(
            f"[yellow]Note: Overrides user skill '{escape_markup(skill_name)}' "
            "of the same name[/yellow]\n"
        )
    console.print(
        f"[bold]Location:[/bold] {escape_markup(str(skill_path.parent))}/\n",
        style=theme.MUTED,
    )
    console.print(
        f"[bold]Description:[/bold] {escape_markup(skill['description'])}\n",
        style=theme.MUTED,
    )

    # Show optional metadata fields
    for label, value in _format_info_fields(skill):
        console.print(
            f"[bold]{label}:[/bold] {escape_markup(value)}\n",
            style=theme.MUTED,
        )

    # List supporting files
    skill_dir = skill_path.parent
    supporting_files = [f for f in skill_dir.iterdir() if f.name != "SKILL.md"]

    if supporting_files:
        console.print("[bold]Supporting Files:[/bold]", style=theme.MUTED)
        for file in supporting_files:
            console.print(f"  - {escape_markup(file.name)}", style=theme.MUTED)
        console.print()

    # Show the full SKILL.md content
    console.print("[bold]Full SKILL.md Content:[/bold]\n", style=theme.PRIMARY)
    console.print(skill_content, style=theme.MUTED)
    console.print()


def _delete(
    skill_name: str,
    *,
    agent: str = "agent",
    project: bool = False,
    force: bool = False,
    dry_run: bool = False,
    output_format: OutputFormat = "text",
) -> None:
    """유효성 검사 및 선택적 사용자 확인 후 기술 디렉터리를 삭제합니다.

    기술 이름을 검증하고, 사용자 또는 프로젝트 디렉터리에서 기술을 찾고, 사용자에게 삭제를 확인하고(`force`이 `True`이 아닌 경우) 기술
    디렉터리를 반복적으로 제거합니다.

    Args:
        skill_name: 삭제할 스킬의 이름입니다.
        agent: 스킬에 대한 에이전트 식별자입니다.
        project: `True`인 경우 프로젝트 기술에서만 검색하세요.

            `False`인 경우 사용자 및 프로젝트 기술을 모두 검색하세요.
        force: `True`인 경우 확인 메시지를 건너뜁니다.
        dry_run: `True`인 경우 삭제하지 않고 제거할 항목을 인쇄합니다.
        output_format: 출력 형식 — `'text'`(Rich) 또는 `'json'`.

    Raises:
        SystemExit: 삭제에 실패하거나 안전 점검을 위반한 경우.

    """
    from rich.markup import escape as escape_markup

    from deepagents_cli.config import Settings, console, get_glyphs
    from deepagents_cli.skills.load import list_skills

    # Validate skill name first (per Agent Skills spec)
    is_valid, error_msg = _validate_name(skill_name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] Invalid skill name: {error_msg}")
        raise SystemExit(1)

    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    # Load skills based on --project flag
    if project:
        if not project_skills_dir:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            raise SystemExit(1)
        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )
    else:
        skills = list_skills(
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )

    # Find the skill
    skill = next((s for s in skills if s["name"] == skill_name), None)

    if not skill:
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found.")
        console.print("\n[dim]Available skills:[/dim]", style=theme.MUTED)
        for s in skills:
            source_tag = "[project]" if s["source"] == "project" else "[user]"
            console.print(f"  - {s['name']} {source_tag}", style=theme.MUTED)
        raise SystemExit(1)

    skill_path = Path(skill["path"])
    skill_dir = skill_path.parent

    # Validate the path is safe to delete
    base_dir = project_skills_dir if skill["source"] == "project" else user_skills_dir
    if not base_dir:
        console.print(
            "[bold red]Error:[/bold red] Cannot determine base skills directory. "
            "Refusing to delete."
        )
        raise SystemExit(1)
    is_valid_path, path_error = _validate_skill_path(skill_dir, base_dir)
    if not is_valid_path:
        console.print(f"[bold red]Error:[/bold red] {path_error}")
        raise SystemExit(1)

    if dry_run:
        if output_format == "json":
            from deepagents_cli.output import write_json

            write_json(
                "skills delete",
                {
                    "name": skill_name,
                    "path": str(skill_dir),
                    "dry_run": True,
                },
            )
            return
        console.print(
            f"Would delete skill '{skill_name}' at {skill_dir}",
        )
        console.print("No changes made.", style=theme.MUTED)
        return

    # Display confirmation summary (text mode only)
    if output_format != "json":
        source_label = "Project Skill" if skill["source"] == "project" else "User Skill"
        source_color = "green" if skill["source"] == "project" else "cyan"

        # Count files for the confirmation summary (display-only; a permission
        # error in a subdirectory should not abort the entire delete flow).
        try:
            file_count = sum(1 for f in skill_dir.rglob("*") if f.is_file())
        except OSError:
            file_count = -1

        console.print(
            f"\n[bold]Skill:[/bold] {escape_markup(skill_name)}"
            f" [bold {source_color}]({source_label})[/bold {source_color}]",
            style=theme.PRIMARY,
        )
        console.print(
            f"[bold]Location:[/bold] {escape_markup(str(skill_dir))}/",
            style=theme.MUTED,
        )
        if file_count >= 0:
            console.print(
                f"[bold]Files:[/bold] {file_count} file(s) will be deleted\n",
                style=theme.MUTED,
            )
        else:
            console.print(
                "[bold]Files:[/bold] (unable to count files)\n",
                style=theme.MUTED,
            )

    # Confirmation (skip in JSON mode — no interactive prompt)
    if not force and output_format != "json":
        console.print(
            "[yellow]Are you sure you want to delete this skill? (y/N)[/yellow] ",
            end="",
        )
        try:
            response = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return

        if response not in {"y", "yes"}:
            console.print("[dim]Cancelled.[/dim]")
            return

    # Re-validate immediately before deletion to narrow the TOCTOU window
    # (the user may have paused at the confirmation prompt).
    if skill_dir.is_symlink():
        console.print(
            "[bold red]Error:[/bold red] Skill directory is a symlink. "
            "Refusing to delete for safety."
        )
        raise SystemExit(1)

    is_valid_path, path_error = _validate_skill_path(skill_dir, base_dir)
    if not is_valid_path:
        console.print(f"[bold red]Error:[/bold red] {path_error}")
        raise SystemExit(1)

    # Delete the skill directory
    try:
        shutil.rmtree(skill_dir)
    except OSError as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to fully delete skill: {e}\n"
            f"[yellow]Warning:[/yellow] Some files may have been partially removed.\n"
            f"Please inspect: {skill_dir}/"
        )
        raise SystemExit(1) from e

    if output_format == "json":
        from deepagents_cli.output import write_json

        write_json(
            "skills delete",
            {
                "name": skill_name,
                "path": str(skill_dir),
                "deleted": True,
            },
        )
        return

    checkmark = get_glyphs().checkmark
    console.print(
        f"{checkmark} Skill '{skill_name}' deleted successfully!",
        style=theme.PRIMARY,
    )


def setup_skills_parser(
    subparsers: Any,  # noqa: ANN401  # argparse subparsers uses dynamic typing
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
    add_output_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    """모든 하위 명령을 사용하여 기술 하위 명령 구문 분석기를 설정합니다.

    각 하위 명령에는 전용 도움말 화면이 있으므로 `deepagents skills -h`은 전역 도움말이 아닌 기술별 도움말을 표시합니다.

    Args:
        subparsers: 기술 파서를 추가할 상위 하위 파서 개체입니다.
        make_help_action: 인수가 없는 도움말 호출 가능 항목을 허용하고 이에 연결된 argparse Action 클래스를 반환하는
                          팩토리입니다.
        add_output_args: 공유 `--json` 플래그를 추가하는 선택적 후크입니다.

    Returns:
        인수 처리를 위한 기술 하위 구문 분석기입니다.

    """

    # Lazy wrapper: defers ui import until the help action fires.
    def _lazy_help(fn_name: str) -> Callable[[], None]:
        def _show() -> None:
            from deepagents_cli import ui

            getattr(ui, fn_name)()

        return _show

    def help_parent(help_fn: Callable[[], None]) -> list[argparse.ArgumentParser]:
        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument("-h", "--help", action=make_help_action(help_fn))
        return [parent]

    skills_parser = subparsers.add_parser(
        "skills",
        help="Manage agent skills",
        description="Manage agent skills - list, create, view, and delete skills.",
        add_help=False,
        parents=help_parent(_lazy_help("show_skills_help")),
    )
    if add_output_args is not None:
        add_output_args(skills_parser)
    skills_subparsers = skills_parser.add_subparsers(
        dest="skills_command", help="Skills command"
    )

    # Skills list
    list_parser = skills_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List all available skills",
        description=(
            "List skills from all four skill directories "
            "(user, user alias, project, project alias)."
        ),
        add_help=False,
        parents=help_parent(_lazy_help("show_skills_list_help")),
    )
    if add_output_args is not None:
        add_output_args(list_parser)
    list_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    list_parser.add_argument(
        "--project",
        action="store_true",
        help="Show only project-level skills",
    )

    # Skills create
    create_parser = skills_subparsers.add_parser(
        "create",
        help="Create a new skill",
        description=(
            "Create a new skill with a template SKILL.md file. "
            "By default, skills are created in "
            "~/.deepagents/<agent>/skills/. "
            "Use --project to create in the project's "
            ".deepagents/skills/ directory."
        ),
        add_help=False,
        parents=help_parent(_lazy_help("show_skills_create_help")),
    )
    if add_output_args is not None:
        add_output_args(create_parser)
    create_parser.add_argument(
        "name",
        help="Name of the skill to create (e.g., web-research)",
    )
    create_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    create_parser.add_argument(
        "--project",
        action="store_true",
        help="Create skill in project directory instead of user directory",
    )

    # Skills info
    info_parser = skills_subparsers.add_parser(
        "info",
        help="Show detailed information about a skill",
        description="Show detailed information about a specific skill",
        add_help=False,
        parents=help_parent(_lazy_help("show_skills_info_help")),
    )
    if add_output_args is not None:
        add_output_args(info_parser)
    info_parser.add_argument("name", help="Name of the skill to show info for")
    info_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    info_parser.add_argument(
        "--project",
        action="store_true",
        help="Search only in project skills",
    )

    # Skills delete
    delete_parser = skills_subparsers.add_parser(
        "delete",
        help="Delete a skill",
        description="Delete a skill directory and all its contents",
        add_help=False,
        parents=help_parent(_lazy_help("show_skills_delete_help")),
    )
    if add_output_args is not None:
        add_output_args(delete_parser)
    delete_parser.add_argument("name", help="Name of the skill to delete")
    delete_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    delete_parser.add_argument(
        "--project",
        action="store_true",
        help="Search only in project skills",
    )
    delete_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    return skills_parser


def execute_skills_command(args: argparse.Namespace) -> None:
    """구문 분석된 인수를 기반으로 기술 하위 명령을 실행합니다.

    Args:
        args: Skill_command 속성을 사용하여 구문 분석된 명령줄 인수

    Raises:
        SystemExit: 에이전트 이름이 잘못된 경우.

    """
    from deepagents_cli.config import console

    # validate agent argument
    if args.agent:
        is_valid, error_msg = _validate_name(args.agent)
        if not is_valid:
            console.print(
                f"[bold red]Error:[/bold red] Invalid agent name: {error_msg}"
            )
            console.print(
                "[dim]Agent names must only contain letters, numbers, "
                "hyphens, and underscores.[/dim]",
                style=theme.MUTED,
            )
            raise SystemExit(1)

    output_format = getattr(args, "output_format", "text")

    # "ls" is an argparse alias for "list" — argparse stores the alias
    # as-is in the namespace, so we must match both values.
    if args.skills_command in {"list", "ls"}:
        _list(agent=args.agent, project=args.project, output_format=output_format)
    elif args.skills_command == "create":
        _create(
            args.name,
            agent=args.agent,
            project=args.project,
            output_format=output_format,
        )
    elif args.skills_command == "info":
        _info(
            args.name,
            agent=args.agent,
            project=args.project,
            output_format=output_format,
        )
    elif args.skills_command == "delete":
        _delete(
            args.name,
            agent=args.agent,
            project=args.project,
            force=args.force,
            dry_run=args.dry_run,
            output_format=output_format,
        )
    else:
        # No subcommand provided, show skills help screen
        from deepagents_cli.ui import show_skills_help

        show_skills_help()


__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
