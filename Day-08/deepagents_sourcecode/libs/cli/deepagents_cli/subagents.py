"""CLI용 하위 에이전트 로더.

파일 시스템에서 사용자 정의 하위 에이전트 정의를 로드합니다. 하위 에이전트는 Agents/ 디렉터리에 YAML 머리말이 있는 마크다운 파일로 정의됩니다.

디렉토리 구조:
    .deepagents/agents/{agent_name}/AGENTS.md

예제 파일(연구원/AGENTS.md):
    --- 이름: 연구원 설명: 콘텐츠 작성 전 웹에서 조사 주제 모델: anthropic:claude-haiku-4-5-20251001 ---

    당신은 웹 검색에 접근할 수 있는 연구 조교입니다.

    ## 프로세스 1. 관련 정보 검색 2. 결과를 명확하게 요약
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

import yaml

if TYPE_CHECKING:
    from pathlib import Path


class SubagentMetadata(TypedDict):
    """파일 시스템에서 로드된 사용자 정의 하위 에이전트에 대한 메타데이터입니다."""

    name: str
    """작업 도구와 함께 사용되는 하위 에이전트의 고유 식별자입니다."""

    description: str
    """이 하위 에이전트가 수행하는 작업입니다. 주 에이전트는 이를 사용하여 위임 시기를 결정합니다."""

    system_prompt: str
    """하위 에이전트에 대한 지침(markdown 파일의 본문)"""

    model: str | None
    """'provider:model-name' 형식의 선택적 모델 재정의."""

    source: str
    """이 하위 에이전트가 로드된 위치('사용자' 또는 '프로젝트')입니다."""

    path: str
    """하위 에이전트 정의 파일의 절대 경로입니다."""


def _parse_subagent_file(file_path: Path) -> SubagentMetadata | None:
    """YAML Frontmatter를 사용하여 하위 에이전트 마크다운 파일을 구문 분석합니다.

    파일에는 최소한 '이름' 및 '설명' 필드를 포함하는 YAML 머리말(---로 구분)이 있어야 합니다. 파일 본문이 system_prompt가 됩니다.

    Args:
        file_path: 마크다운 파일의 경로입니다.

    Returns:
        구문 분석이 성공하면 SubagentMetadata이고, 그렇지 않으면 없음입니다.

    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Extract YAML frontmatter (--- delimited)
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None

    # Validate frontmatter structure and required fields
    if not isinstance(frontmatter, dict):
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    model = frontmatter.get("model")

    # Validate types: name and description must be non-empty strings
    # model is optional but must be string if present
    name_valid = isinstance(name, str) and name
    description_valid = isinstance(description, str) and description
    model_valid = model is None or isinstance(model, str)

    if not (name_valid and description_valid and model_valid):
        return None

    return {
        "name": name,
        "description": description,
        "system_prompt": match.group(2).strip(),
        "model": model,
        "source": "",  # Set by caller
        "path": str(file_path),
    }


def _load_subagents_from_dir(
    agents_dir: Path, source: str
) -> dict[str, SubagentMetadata]:
    """디렉터리에서 하위 에이전트를 로드합니다.

    예상되는 구조: Agent_dir/{subagent_name}/AGENTS.md

    Args:
        agents_dir: 하위 에이전트 폴더가 포함된 디렉터리입니다.
        source: 소스 식별자('사용자' 또는 '프로젝트')입니다.

    Returns:
        하위 에이전트 이름을 메타데이터에 매핑하는 사전입니다.

    """
    subagents: dict[str, SubagentMetadata] = {}

    if not agents_dir.exists() or not agents_dir.is_dir():
        return subagents

    for folder in agents_dir.iterdir():
        if not folder.is_dir():
            continue

        # Look for {folder_name}/AGENTS.md
        subagent_file = folder / "AGENTS.md"
        if not subagent_file.exists():
            continue

        subagent = _parse_subagent_file(subagent_file)
        if subagent:
            subagent["source"] = source
            subagents[subagent["name"]] = subagent

    return subagents


def list_subagents(
    *,
    user_agents_dir: Path | None = None,
    project_agents_dir: Path | None = None,
) -> list[SubagentMetadata]:
    """사용자 및/또는 프로젝트 디렉터리의 하위 에이전트를 나열합니다.

    제공된 디렉터리에서 하위 에이전트 정의를 검색합니다. 프로젝트 하위 에이전트는 동일한 이름을 가진 사용자 하위 에이전트를 재정의합니다.

    Args:
        user_agents_dir: 사용자 수준 에이전트 디렉터리의 경로입니다.
        project_agents_dir: 프로젝트 수준 에이전트 디렉터리의 경로입니다.

    Returns:
        프로젝트 하위 에이전트가 우선적으로 적용되는 하위 에이전트 메타데이터 목록입니다.

    """
    all_subagents: dict[str, SubagentMetadata] = {}

    # Load user subagents first (lower priority)
    if user_agents_dir is not None:
        all_subagents.update(_load_subagents_from_dir(user_agents_dir, "user"))

    # Load project subagents second (override user)
    if project_agents_dir is not None:
        all_subagents.update(_load_subagents_from_dir(project_agents_dir, "project"))

    return list(all_subagents.values())
