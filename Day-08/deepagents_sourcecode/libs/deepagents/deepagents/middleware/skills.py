"""에이전트 스킬을 로드하고 시스템 프롬프트에 노출하는 미들웨어 모듈.

이 모듈은 Anthropic의 에이전트 스킬 패턴을 점진적 공개(progressive disclosure) 방식으로 구현하며,
백엔드 스토리지의 설정 가능한 소스에서 스킬을 로드합니다.

## 아키텍처

스킬은 하나 이상의 **소스(source)** — 백엔드 내 스킬이 구성된 경로 — 에서 로드됩니다.
소스는 순서대로 로드되며, 동일한 이름의 스킬이 있으면 나중 소스가 이전 소스를 덮어씁니다
(마지막 승리 원칙). 이를 통해 계층적 구성이 가능합니다: 기본 → 사용자 → 프로젝트 → 팀 스킬.

미들웨어는 백엔드 API만 사용하므로(직접 파일시스템 접근 없음), 다양한 스토리지 백엔드
(파일시스템, 상태, 원격 스토리지 등)에서 이식 가능합니다.

StateBackend(임시/인메모리) 사용 시:
```python
SkillsMiddleware(backend=StateBackend(), ...)
```

## 스킬 구조

각 스킬은 SKILL.md 파일을 포함하는 디렉토리입니다:

```
/skills/user/web-research/
├── SKILL.md          # 필수: YAML 프론트매터 + 마크다운 지시사항
└── helper.py         # 선택: 보조 파일
```

SKILL.md 형식:
```markdown
---
name: web-research
description: 체계적인 웹 리서치를 수행하기 위한 접근법
license: MIT
---

# Web Research Skill

## 사용 시점
- 사용자가 주제를 조사해달라고 요청할 때
...
```

## 스킬 메타데이터 (SkillMetadata)

Agent Skills 명세(https://agentskills.io/specification)에 따라 YAML 프론트매터에서 파싱:
- `name`: 스킬 식별자 (최대 64자, 소문자 영숫자와 하이픈)
- `description`: 스킬 설명 (최대 1024자)
- `path`: SKILL.md 파일의 백엔드 경로
- 선택: `license`, `compatibility`, `metadata`, `allowed_tools`

## 점진적 공개(Progressive Disclosure) 패턴

LLM에게 모든 스킬의 전체 내용을 즉시 주입하지 않고:
1. 시스템 프롬프트에는 스킬의 **이름과 설명**만 표시
2. LLM이 특정 스킬이 필요하다고 판단하면 **read_file로 SKILL.md를 읽어** 전체 지시사항 확인
3. 이를 통해 시스템 프롬프트의 토큰 사용을 최소화하면서도 스킬 접근성 유지

## 소스

소스는 단순히 백엔드 내 스킬 디렉토리 경로입니다.
소스 이름은 경로의 마지막 구성요소에서 추출됩니다 (예: "/skills/user/" → "user").

## 경로 규약

모든 경로는 `PurePosixPath`를 통한 POSIX 규약(슬래시)을 사용합니다:
- 백엔드 경로: "/skills/user/web-research/SKILL.md"
- 가상의, 플랫폼 독립적 경로
- 백엔드가 필요시 플랫폼별 변환을 처리

## 사용 예시

```python
from deepagents.backends.state import StateBackend
from deepagents.middleware.skills import SkillsMiddleware

middleware = SkillsMiddleware(
    backend=my_backend,
    sources=[
        "/skills/base/",     # 기본 스킬 (최저 우선순위)
        "/skills/user/",     # 사용자 스킬
        "/skills/project/",  # 프로젝트 스킬 (최고 우선순위)
    ],
)
```
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated

import yaml
from langchain.agents.middleware.types import PrivateStateAttr

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langgraph.prebuilt import ToolRuntime

from deepagents.backends.protocol import LsResult
from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

# 보안: SKILL.md 파일의 최대 크기 제한 — DoS 공격 방지 (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills 명세의 필드별 길이 제약 (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64         # 스킬 이름 최대 길이
MAX_SKILL_DESCRIPTION_LENGTH = 1024  # 스킬 설명 최대 길이
MAX_SKILL_COMPATIBILITY_LENGTH = 500  # 호환성 정보 최대 길이


class SkillMetadata(TypedDict):
    """Agent Skills 명세에 따른 스킬 메타데이터.

    SKILL.md 파일의 YAML 프론트매터에서 파싱된 스킬의 핵심 정보를 담습니다.
    이 정보는 시스템 프롬프트에 스킬 목록으로 표시되며, LLM이 어떤 스킬을
    사용할지 판단하는 데 사용됩니다.

    참조: https://agentskills.io/specification
    """

    path: str
    """SKILL.md 파일의 백엔드 경로. LLM이 read_file로 전체 내용을 읽을 때 사용."""

    name: str
    """스킬 식별자.

    Agent Skills 명세의 제약:
    - 1~64자
    - 유니코드 소문자 영숫자와 하이픈만 허용 (`a-z`와 `-`)
    - `-`로 시작하거나 끝날 수 없음
    - 연속 `--` 불가
    - SKILL.md를 포함하는 부모 디렉토리 이름과 일치해야 함
    """

    description: str
    """스킬 설명.

    Agent Skills 명세의 제약:
    - 1~1024자
    - 스킬이 무엇을 하는지와 언제 사용하는지를 모두 기술
    - 에이전트가 관련 작업을 식별하는 데 도움이 되는 키워드 포함 권장
    """

    license: str | None
    """라이선스 이름 또는 번들된 라이선스 파일 참조."""

    compatibility: str | None
    """환경 요구사항.

    Agent Skills 명세의 제약:
    - 제공 시 1~500자
    - 특정 호환성 요구사항이 있을 때만 포함
    - 대상 제품, 필요한 패키지 등을 명시 가능
    """

    metadata: dict[str, str]
    """추가 메타데이터를 위한 임의의 키-값 매핑.

    클라이언트가 명세에 정의되지 않은 추가 속성을 저장하는 데 사용합니다.
    충돌을 피하기 위해 고유한 키 이름을 사용하는 것을 권장합니다.
    """

    allowed_tools: list[str]
    """스킬이 사용을 권장하는 도구 이름 목록.

    주의: 이 기능은 실험적(experimental)입니다.

    Agent Skills 명세의 제약:
    - 공백으로 구분된 도구 이름 목록
    """


class SkillsState(AgentState):
    """스킬 미들웨어의 상태 스키마.

    에이전트 상태를 확장하여 로드된 스킬 메타데이터를 저장합니다.
    PrivateStateAttr로 표시되어 부모 에이전트에게 전파되지 않습니다.
    """

    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
    """설정된 소스에서 로드된 스킬 메타데이터 리스트. 부모 에이전트에게 전파되지 않음."""


class SkillsStateUpdate(TypedDict):
    """스킬 미들웨어의 상태 업데이트용 TypedDict.

    before_agent 훅에서 반환하여 스킬 메타데이터를 상태에 저장할 때 사용합니다.
    """

    skills_metadata: list[SkillMetadata]
    """상태에 병합할 스킬 메타데이터 리스트."""


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Agent Skills 명세에 따라 스킬 이름을 검증합니다.

    검증 규칙:
    - 1~64자
    - 유니코드 소문자 영숫자와 하이픈만 허용
    - `-`로 시작/끝 불가, 연속 `--` 불가
    - 부모 디렉토리 이름과 일치해야 함

    유니코드 소문자 영숫자는 `c.isalpha() and c.islower()` 또는 `c.isdigit()`가
    True인 문자를 의미합니다. 악센트 라틴 문자(예: 'café', 'über-tool')와
    다른 스크립트도 포함됩니다.

    Args:
        name: YAML 프론트매터의 스킬 이름.
        directory_name: 부모 디렉토리 이름.

    Returns:
        `(유효 여부, 오류_메시지)` 튜플. 유효하면 오류 메시지는 빈 문자열.
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, "name must be lowercase alphanumeric with single hyphens only"

    # 각 문자가 허용된 문자(소문자 영숫자 또는 하이픈)인지 검사
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return False, "name must be lowercase alphanumeric with single hyphens only"

    # 이름이 부모 디렉토리와 일치하는지 검증
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(  # noqa: C901
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """SKILL.md 내용에서 YAML 프론트매터를 파싱하여 스킬 메타데이터를 추출합니다.

    파싱 과정:
    1. 내용 크기가 MAX_SKILL_FILE_SIZE를 초과하는지 확인 (DoS 방지)
    2. `---` 구분자로 둘러싸인 YAML 프론트매터를 정규식으로 추출
    3. yaml.safe_load로 YAML을 딕셔너리로 파싱
    4. 필수 필드(name, description) 존재 확인
    5. 이름 형식 검증 (경고만, 호환성을 위해 로드는 계속)
    6. 설명 길이 제한 적용
    7. allowed-tools 파싱 (공백 구분, 쉼표 호환)
    8. 호환성 정보 길이 제한 적용

    Args:
        content: SKILL.md 파일의 전체 내용.
        skill_path: SKILL.md 파일 경로 (오류 메시지 및 메타데이터에 사용).
        directory_name: 스킬을 포함하는 부모 디렉토리 이름.

    Returns:
        파싱 성공 시 SkillMetadata, 파싱 실패 또는 유효성 오류 시 None.
    """
    # 보안: 파일 크기 제한 검사 (10MB 초과 시 거부)
    if len(content) > MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: content too large (%d bytes)", skill_path, len(content))
        return None

    # --- 구분자 사이의 YAML 프론트매터를 정규식으로 매칭
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter found", skill_path)
        return None

    frontmatter_str = match.group(1)

    # YAML을 safe_load로 파싱 (중첩 구조 지원, 코드 실행 방지)
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s", skill_path, e)
        return None

    # 프론트매터가 딕셔너리인지 확인 (리스트나 스칼라가 아닌지)
    if not isinstance(frontmatter_data, dict):
        logger.warning("Skipping %s: frontmatter is not a mapping", skill_path)
        return None

    # 필수 필드 추출 및 존재 확인
    name = str(frontmatter_data.get("name", "")).strip()
    description = str(frontmatter_data.get("description", "")).strip()
    if not name or not description:
        logger.warning("Skipping %s: missing required 'name' or 'description'", skill_path)
        return None

    # 이름 형식 검증 (경고만 — 하위 호환성을 위해 로드는 계속)
    is_valid, error = _validate_skill_name(str(name), directory_name)
    if not is_valid:
        logger.warning(
            "Skill '%s' in %s does not follow Agent Skills specification: %s. Consider renaming for spec compliance.",
            name,
            skill_path,
            error,
        )

    # 설명 길이 제한 적용
    description_str = description
    if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
        logger.warning(
            "Description exceeds %d characters in %s, truncating",
            MAX_SKILL_DESCRIPTION_LENGTH,
            skill_path,
        )
        description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

    # allowed-tools 파싱
    # 공백으로 구분된 도구 이름 목록. Claude Code에서 생성된 스킬의 쉼표도 호환 처리.
    raw_tools = frontmatter_data.get("allowed-tools")
    if isinstance(raw_tools, str):
        allowed_tools = [
            t.strip(",")  # Claude Code 호환을 위해 쉼표 제거
            for t in raw_tools.split()
            if t.strip(",")
        ]
    else:
        if raw_tools is not None:
            logger.warning(
                "Ignoring non-string 'allowed-tools' in %s (got %s)",
                skill_path,
                type(raw_tools).__name__,
            )
        allowed_tools = []

    # 호환성 정보 파싱 및 길이 제한
    compatibility_str = str(frontmatter_data.get("compatibility", "")).strip() or None
    if compatibility_str and len(compatibility_str) > MAX_SKILL_COMPATIBILITY_LENGTH:
        logger.warning(
            "Compatibility exceeds %d characters in %s, truncating",
            MAX_SKILL_COMPATIBILITY_LENGTH,
            skill_path,
        )
        compatibility_str = compatibility_str[:MAX_SKILL_COMPATIBILITY_LENGTH]

    return SkillMetadata(
        name=str(name),
        description=description_str,
        path=skill_path,
        metadata=_validate_metadata(frontmatter_data.get("metadata", {}), skill_path),
        license=str(frontmatter_data.get("license", "")).strip() or None,
        compatibility=compatibility_str,
        allowed_tools=allowed_tools,
    )


def _validate_metadata(
    raw: object,
    skill_path: str,
) -> dict[str, str]:
    """YAML 프론트매터의 metadata 필드를 검증하고 정규화합니다.

    YAML `safe_load`는 metadata 키에 대해 어떤 타입이든 반환할 수 있습니다.
    이 함수는 `SkillMetadata`의 값이 항상 `dict[str, str]`이 되도록
    `str()`로 변환하고, 딕셔너리가 아닌 입력은 거부합니다.

    Args:
        raw: `frontmatter_data.get("metadata", {})`의 원시 값.
        skill_path: 경고 메시지에 사용할 SKILL.md 파일 경로.

    Returns:
        검증된 `dict[str, str]`.
    """
    if not isinstance(raw, dict):
        if raw:
            logger.warning(
                "Ignoring non-dict metadata in %s (got %s)",
                skill_path,
                type(raw).__name__,
            )
        return {}
    # 모든 키와 값을 str()로 변환하여 일관된 타입 보장
    return {str(k): str(v) for k, v in raw.items()}


def _format_skill_annotations(skill: SkillMetadata) -> str:
    """스킬의 선택적 필드에서 괄호 주석 문자열을 생성합니다.

    라이선스와 호환성 정보를 쉼표로 구분된 문자열로 결합하여
    시스템 프롬프트의 스킬 목록에 표시합니다.

    Args:
        skill: 주석을 추출할 스킬 메타데이터.

    Returns:
        `'License: MIT, Compatibility: Python 3.10+'` 같은 주석 문자열.
        두 필드 모두 없으면 빈 문자열.
    """
    parts: list[str] = []
    if skill.get("license"):
        parts.append(f"License: {skill['license']}")
    if skill.get("compatibility"):
        parts.append(f"Compatibility: {skill['compatibility']}")
    return ", ".join(parts)


def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """백엔드 소스에서 모든 스킬을 목록화합니다 (동기 버전).

    백엔드에서 SKILL.md 파일을 포함하는 하위 디렉토리를 스캔하고,
    내용을 다운로드하여 YAML 프론트매터를 파싱한 후 스킬 메타데이터를 반환합니다.

    처리 과정:
    1. source_path에서 ls로 디렉토리 목록 조회
    2. 각 디렉토리의 SKILL.md 경로를 생성
    3. download_files로 모든 SKILL.md를 배치 다운로드
    4. 각 파일의 YAML 프론트매터를 파싱하여 메타데이터 추출

    예상 구조:
    ```
    source_path/
    └── skill-name/
        ├── SKILL.md   # 필수
        └── helper.py  # 선택
    ```

    Args:
        backend: 파일 작업에 사용할 백엔드 인스턴스.
        source_path: 백엔드 내 스킬 디렉토리 경로.

    Returns:
        성공적으로 파싱된 SKILL.md 파일의 스킬 메타데이터 리스트.
    """
    skills: list[SkillMetadata] = []
    ls_result = backend.ls(source_path)
    items = ls_result.entries if isinstance(ls_result, LsResult) else ls_result

    # 스킬 디렉토리 찾기 (is_dir인 항목만)
    skill_dirs = []
    for item in items or []:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # 각 스킬 디렉토리에 대해 SKILL.md 경로 생성
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # PurePosixPath로 안전하고 표준화된 경로 연산
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    # 모든 SKILL.md를 배치 다운로드 (네트워크 왕복 최소화)
    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)

    # 다운로드된 각 SKILL.md 파싱
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # SKILL.md가 없는 디렉토리는 스킬이 아니므로 건너뜀
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # PurePosixPath로 디렉토리 이름 추출 (이름 검증에 사용)
        directory_name = PurePosixPath(skill_dir_path).name

        # YAML 프론트매터 파싱 및 메타데이터 추출
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


async def _alist_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """백엔드 소스에서 모든 스킬을 목록화합니다 (비동기 버전).

    동기 버전 `_list_skills`와 동일한 로직이지만, 비동기 백엔드 메서드를 사용합니다.

    Args:
        backend: 파일 작업에 사용할 백엔드 인스턴스.
        source_path: 백엔드 내 스킬 디렉토리 경로.

    Returns:
        성공적으로 파싱된 SKILL.md 파일의 스킬 메타데이터 리스트.
    """
    skills: list[SkillMetadata] = []
    ls_result = await backend.als(source_path)
    items = ls_result.entries if isinstance(ls_result, LsResult) else ls_result

    # 스킬 디렉토리 찾기
    skill_dirs = []
    for item in items or []:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # 각 디렉토리의 SKILL.md 경로 생성
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    # 비동기 배치 다운로드
    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = await backend.adownload_files(paths_to_download)

    # 다운로드된 각 SKILL.md 파싱
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        directory_name = PurePosixPath(skill_dir_path).name

        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


# LLM에게 주입되는 스킬 시스템 프롬프트 템플릿
# {skills_locations}: 스킬 소스 경로 목록
# {skills_list}: 이름, 설명, 경로가 포함된 스킬 목록
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only read full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Read the skill's full instructions**: Use the path shown in the skill list above
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include helper scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill with its path
2. Read the skill using the path shown
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware[SkillsState, ContextT, ResponseT]):
    """에이전트 스킬을 로드하고 시스템 프롬프트에 점진적으로 노출하는 미들웨어.

    백엔드 소스에서 스킬을 로드하고, 점진적 공개(progressive disclosure) 패턴으로
    시스템 프롬프트에 주입합니다 (메타데이터 우선, 전체 내용은 요청 시 제공).

    스킬은 소스 순서대로 로드되며, 동일 이름의 스킬은 나중 소스가 덮어씁니다.

    이 미들웨어의 생명주기:
        1. **before_agent** (최초 1회): 모든 소스에서 SKILL.md를 스캔하고 메타데이터 파싱
        2. **wrap_model_call** (매 LLM 호출): 스킬 목록과 사용법을 시스템 프롬프트에 주입

    사용 예시:
        ```python
        from deepagents.backends.filesystem import FilesystemBackend

        backend = FilesystemBackend(root_dir="/path/to/skills")
        middleware = SkillsMiddleware(
            backend=backend,
            sources=[
                "/path/to/skills/user/",
                "/path/to/skills/project/",
            ],
        )
        ```

    Args:
        backend: 파일 작업을 위한 백엔드 인스턴스.
        sources: 스킬 소스 경로 리스트. 소스 이름은 마지막 경로 구성요소에서 추출.
    """

    # 이 미들웨어가 에이전트 상태에 추가하는 필드 스키마
    state_schema = SkillsState

    def __init__(self, *, backend: BACKEND_TYPES, sources: list[str]) -> None:
        """스킬 미들웨어를 초기화합니다.

        Args:
            backend: 백엔드 인스턴스 (예: `StateBackend()`).
            sources: 스킬 소스 경로 리스트
                (예: `['/skills/user/', '/skills/project/']`).
        """
        self._backend = backend
        self.sources = sources
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _get_backend(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """백엔드 인스턴스를 해석합니다.

        인스턴스이면 그대로, callable(팩토리)이면 호출하여 생성합니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 팩토리 함수에 전달할 런타임 컨텍스트.
            config: 백엔드 팩토리에 전달할 Runnable 설정.

        Returns:
            해석된 백엔드 인스턴스.

        Raises:
            AssertionError: 팩토리가 None을 반환한 경우.
        """
        if callable(self._backend):
            # 팩토리 호출을 위해 인위적인 ToolRuntime 생성
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            backend = self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
            if backend is None:
                msg = "SkillsMiddleware requires a valid backend instance"
                raise AssertionError(msg)
            return backend

        return self._backend

    def _format_skills_locations(self) -> str:
        """시스템 프롬프트에 표시할 스킬 소스 위치를 포맷합니다.

        각 소스의 이름과 경로를 표시하며, 마지막 소스에는 "(higher priority)" 표시를 추가합니다.

        Returns:
            포맷된 스킬 소스 위치 문자열.
            예: "**User Skills**: `/skills/user/`\n**Project Skills**: `/skills/project/` (higher priority)"
        """
        locations = []

        for i, source_path in enumerate(self.sources):
            # 경로의 마지막 구성요소를 대문자로 시작하여 소스 이름으로 사용
            name = PurePosixPath(source_path.rstrip("/")).name.capitalize()
            # 마지막 소스(최고 우선순위)에 표시 추가
            suffix = " (higher priority)" if i == len(self.sources) - 1 else ""
            locations.append(f"**{name} Skills**: `{source_path}`{suffix}")

        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """시스템 프롬프트에 표시할 스킬 목록을 포맷합니다.

        각 스킬의 이름, 설명, 라이선스/호환성 주석, allowed_tools,
        그리고 전체 지시사항을 읽기 위한 SKILL.md 경로를 포함합니다.

        Args:
            skills: 포맷할 스킬 메타데이터 리스트.

        Returns:
            포맷된 스킬 목록 문자열. 스킬이 없으면 안내 메시지.
        """
        if not skills:
            paths = [f"{source_path}" for source_path in self.sources]
            return f"(No skills available yet. You can create skills in {' or '.join(paths)})"

        lines = []
        for skill in skills:
            # 라이선스/호환성 주석 생성
            annotations = _format_skill_annotations(skill)
            desc_line = f"- **{skill['name']}**: {skill['description']}"
            if annotations:
                desc_line += f" ({annotations})"
            lines.append(desc_line)
            # allowed_tools가 있으면 표시
            if skill["allowed_tools"]:
                lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
            # 전체 지시사항을 읽기 위한 경로 표시 (점진적 공개의 핵심)
            lines.append(f"  -> Read `{skill['path']}` for full instructions")

        return "\n".join(lines)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """모델 요청의 시스템 메시지에 스킬 문서를 주입합니다.

        상태에서 스킬 메타데이터를 읽어 포맷한 후, 시스템 메시지에 추가합니다.
        동기/비동기 wrap_model_call에서 공통으로 호출됩니다.

        Args:
            request: 수정할 모델 요청.

        Returns:
            스킬 문서가 시스템 메시지에 주입된 새 모델 요청.
        """
        skills_metadata = request.state.get("skills_metadata", [])
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # 템플릿에 스킬 위치와 목록을 삽입
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        new_system_message = append_to_system_message(request.system_message, skills_section)

        return request.override(system_message=new_system_message)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """에이전트 실행 전에 스킬 메타데이터를 로드합니다 (동기 버전).

        세션당 한 번만 모든 설정된 소스에서 스킬을 로드합니다.
        `skills_metadata`가 이미 상태에 존재하면(이전 턴 또는 체크포인트 세션) 건너뜁니다.

        스킬은 소스 순서대로 로드되며, 동일 이름의 스킬은 나중 소스가 덮어씁니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.
            config: Runnable 설정.

        Returns:
            `skills_metadata`가 채워진 상태 업데이트, 또는 이미 존재하면 None.
        """
        # 이미 로드되었으면 건너뜀 (빈 리스트여도 존재하면 건너뜀)
        if "skills_metadata" in state:
            return None

        # 백엔드 해석 (직접 인스턴스 또는 팩토리 함수 지원)
        backend = self._get_backend(state, runtime, config)

        # 이름 → 메타데이터 딕셔너리로 중복 제거 (나중 소스가 우선)
        all_skills: dict[str, SkillMetadata] = {}

        # 각 소스에서 순서대로 스킬 로드 (나중 소스가 동일 이름 덮어씀)
        for source_path in self.sources:
            source_skills = _list_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """에이전트 실행 전에 스킬 메타데이터를 로드합니다 (비동기 버전).

        동기 버전과 동일한 로직이지만, 비동기 백엔드 메서드를 사용합니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.
            config: Runnable 설정.

        Returns:
            `skills_metadata`가 채워진 상태 업데이트, 또는 이미 존재하면 None.
        """
        if "skills_metadata" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        for source_path in self.sources:
            source_skills = await _alist_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """시스템 프롬프트에 스킬 문서를 주입합니다 (동기 버전).

        매 LLM 호출 전에 사용 가능한 스킬 목록과 사용법을 시스템 메시지에 추가합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """시스템 프롬프트에 스킬 문서를 주입합니다 (비동기 버전).

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 비동기 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)


__all__ = ["SkillMetadata", "SkillsMiddleware"]
