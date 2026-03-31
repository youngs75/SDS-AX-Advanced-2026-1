# Agent Skills 종합 가이드

---

## 1. Agent Skills란?

**Agent Skills**는 Claude가 특정 작업에서 성능을 향상시키기 위해 **동적으로 로드하는 지침, 스크립트, 리소스의 번들**입니다. 기존의 프롬프트 엔지니어링이 "LLM에게 무엇을 말할까?"에 집중했다면, Skills는 "LLM이 작업 시점에 어떤 절차적 지식을 갖고 있어야 하는가?"를 시스템 수준에서 설계합니다.

### Skills가 해결하는 문제

| 문제 | Skills 이전 | Skills 이후 |
|------|------------|------------|
| 절차적 지식 부재 | 매번 프롬프트에 절차를 반복 작성 | SKILL.md에 한 번 정의, 자동 로드 |
| 컨텍스트 낭비 | 관련 없는 지식도 항상 로드 | 필요한 Skills만 동적 로드 (Progressive Loading) |
| 일관성 부족 | 매번 다른 프롬프트 → 다른 결과 | 표준화된 지침 → 일관된 결과 |
| 확장성 한계 | 시스템 프롬프트 크기 제한 | 무제한 리소스 (스크립트, 문서, 템플릿) |

### Skills의 실제 활용 사례

- 회사 브랜드 가이드라인에 맞는 문서 생성
- 조직별 워크플로우에 따른 데이터 분석
- 특정 코딩 컨벤션에 맞는 코드 생성
- 문서 변환 (DOCX, PDF, PPTX, XLSX)

---

## 2. Skills vs 다른 Claude Code 커스터마이징 옵션

> Anthropic Skilljar 커리큘럼 참고: [Introduction to Agent Skills](https://anthropic.skilljar.com/introduction-to-agent-skills)

Claude Code에는 동작을 커스터마이징하는 여러 메커니즘이 있습니다. Skills는 이 중 하나이며, 각각의 용도와 적용 범위가 다릅니다.

| 메커니즘 | 범위 | 로딩 방식 | 주요 용도 | 예시 |
|----------|------|----------|----------|------|
| **CLAUDE.md** | 프로젝트 전체 | 항상 로드 (세션 시작 시) | 프로젝트 규칙, 코딩 컨벤션, 빌드 명령어 | "테스트는 pytest로 실행", "커밋 메시지는 한국어로" |
| **Skills** | 특정 작업 | 동적 로드 (트리거 시) | 전문화된 절차, 도메인 지식 | PDF 변환, 코드 리뷰, 데이터 분석 |
| **Hooks** | 이벤트 기반 | 자동 실행 (이벤트 발생 시) | 자동화된 검증, 포맷팅 | "파일 저장 시 lint 실행", "커밋 전 테스트" |
| **Subagents** | 위임된 작업 | 명시적 호출 | 격리된 전문 작업 수행 | 보안 검토, 성능 분석 |

### 선택 가이드

```
"이 규칙이 모든 작업에 적용되어야 하나?"
  ├─ Yes → CLAUDE.md
  └─ No → "특정 작업 유형에서만 필요한가?"
           ├─ Yes → Skills
           └─ No → "특정 이벤트에 반응해야 하나?"
                    ├─ Yes → Hooks
                    └─ No → "격리된 실행이 필요한가?"
                             ├─ Yes → Subagents
                             └─ No → 직접 구현
```

### Skills와 CLAUDE.md의 핵심 차이

| | CLAUDE.md | Skills |
|---|---|---|
| **로딩** | 항상 전량 로드 | 필요할 때만 동적 로드 |
| **토큰 비용** | 매 호출마다 소비 | 트리거 시에만 소비 |
| **범위** | 프로젝트 전역 규칙 | 작업 특화 절차 |
| **크기** | 짧게 유지 권장 (~500 토큰) | 더 길어도 OK (500줄 이내) |
| **리소스** | 텍스트만 | 스크립트, 문서, 템플릿 번들 가능 |

> **실무 팁**: CLAUDE.md에 "이 프로젝트에서 PDF를 다룰 때는 pdf Skill을 사용하세요"라는 **포인터**만 두고, 실제 절차는 Skills에 정의하면 토큰 효율과 유지보수 모두 좋습니다.

---

## 3. Skills의 기술 아키텍처

### 2.1 핵심 개념: Prompt Expansion

Skills는 전통적인 **Tool Call**(함수 호출 → 결과 반환)이 아닌 **Prompt Expansion**(컨텍스트 확장) 방식으로 동작합니다.

```
전통적 Tool Call:
  Agent → 함수 호출 → 결과 반환 (동기적, 데이터 반환)

Skills Prompt Expansion:
  Agent → Skill 트리거 감지 → SKILL.md 로드 → 컨텍스트에 주입
  (비동기적, 지식/지침 주입)
```

**구체적 실행 흐름:**

1. **Skill 트리거**: 사용자 메시지가 Skill의 `description`과 매치
2. **Prompt Expansion**: Skill이 두 가지를 동시에 반환:
   - `newMessages`: Skill 지침을 'user' 역할 메시지로 주입
   - `contextModifier`: 권한, 모델, 토큰 예산 수정
3. **실행**: Claude가 확장된 컨텍스트로 작업 수행

### 2.2 3-Level Progressive Loading

Skills는 OS의 **Demand Paging**(요구 페이징)과 유사한 3단계 로딩 아키텍처를 사용합니다:

```
Level 1: 메타데이터 (항상 로드)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - name + description (~100 words)
  - 모든 Skills의 메타데이터가 항상 컨텍스트에 존재
  - 비유: OS Page Table

Level 2: SKILL.md 본문 (트리거 시 로드)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 마크다운 지침 (<500 lines 권장)
  - Skill이 활성화될 때만 로드
  - 비유: Demand Paging

Level 3: 번들 리소스 (필요시 로드)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - scripts/, references/, assets/
  - 스크립트는 로드 없이 실행 가능
  - 비유: Lazy Loading / Memory-Mapped Files
```

**토큰 효율성**: Level 1은 ~100 토큰, Level 2는 ~2,000-5,000 토큰, Level 3는 필요한 부분만 선택적 로드. 전체를 항상 로드하는 것 대비 **90%+ 토큰 절약**.

---

## 4. SKILL.md 형식

### 4.1 기본 구조

```markdown
---
name: my-skill-name
description: 이 스킬이 무엇을 하고 언제 사용하는지 명확하게 설명합니다.
---

# My Skill Name

[Claude가 이 스킬이 활성화될 때 따를 지침을 작성합니다]

## Examples
- 예시 사용법 1
- 예시 사용법 2

## Guidelines
- 가이드라인 1
- 가이드라인 2
```

### 4.2 YAML Frontmatter 필드

| 필드 | 필수 | 설명 |
|------|------|------|
| `name` | 필수 | 스킬 고유 식별자 (소문자, 하이픈 구분) |
| `description` | 필수 | 스킬 설명 + 트리거 조건. **이것이 주요 트리거 메커니즘** |
| `compatibility` | 선택 | 필요한 도구, 의존성 |
| `license` | 선택 | 라이선스 정보 |
| `allowed-tools` | 선택 | 스킬이 사용할 수 있는 도구 제한 |
| `metadata` | 선택 | 운영 파라미터, 정책 등 커스텀 메타데이터 |

### 4.3 디렉토리 구조

```
skill-name/
├── SKILL.md              (필수 — 스킬 정의)
│   ├── YAML frontmatter  (name, description 필수)
│   └── Markdown 지침
└── Bundled Resources     (선택)
    ├── scripts/          — 결정적/반복 작업용 실행 코드
    ├── references/       — 필요시 컨텍스트에 로드되는 문서
    └── assets/           — 출력에 사용되는 파일 (템플릿, 아이콘, 폰트)
```

### 4.4 도메인별 정리 패턴

스킬이 여러 도메인/프레임워크를 지원할 때:

```
cloud-deploy/
├── SKILL.md          (워크플로우 + 선택 로직)
└── references/
    ├── aws.md        ← 필요할 때만 로드
    ├── gcp.md
    └── azure.md
```

Claude는 관련 reference 파일만 선택적으로 읽습니다.

---

## 5. Skills 작성 가이드

### 5.1 Description 작성 — 트리거의 핵심

`description`은 **Skills의 트리거 메커니즘**입니다. Claude는 이 필드를 기반으로 어떤 Skill을 활성화할지 결정합니다.

**나쁜 예시:**
```yaml
description: 대시보드를 만드는 방법
```

**좋은 예시:**
```yaml
description: 내부 데이터를 표시하기 위한 간단한 대시보드 구축 방법.
  사용자가 대시보드, 데이터 시각화, 내부 메트릭스를 언급하거나
  어떤 종류의 데이터를 표시하려 할 때 이 스킬을 사용하세요.
```

> **팁**: Claude는 Skills를 "과소 트리거(undertrigger)"하는 경향이 있으므로, description을 약간 "적극적"으로 작성하세요.

### 5.2 지침 작성 원칙

1. **명령형(imperative) 사용**: "~하세요", "~합니다" 형태
2. **이유 설명**: 강제적 "MUST" 대신 왜 중요한지 설명
3. **일반화**: 특정 예시에 너무 좁히지 않고 일반적 패턴으로
4. **길이 관리**: SKILL.md 본문 500줄 이내 권장. 초과 시 references/로 분리

### 5.3 출력 형식 정의 패턴

```markdown
## 보고서 구조
반드시 이 템플릿을 따르세요:

# [제목]
## 요약
## 핵심 발견
## 권장 사항
```

### 5.4 예시 포함 패턴

```markdown
## 커밋 메시지 형식

**예시 1:**
Input: JWT 토큰 기반 사용자 인증 추가
Output: feat(auth): JWT 기반 인증 구현

**예시 2:**
Input: 로그인 페이지 CSS 오류 수정
Output: fix(ui): 로그인 페이지 스타일시트 수정
```

---

## 6. 고급 설정

### 11.1 allowed-tools: 도구 접근 제한

`allowed-tools` frontmatter 필드로 Skill이 사용할 수 있는 도구를 명시적으로 제한합니다. 이는 **최소 권한 원칙(Principle of Least Privilege)**을 구현합니다.

```yaml
---
name: safe-data-reader
description: 읽기 전용 데이터 분석. 파일 수정이나 삭제를 하지 않습니다.
allowed-tools:
  - read_file
  - grep
  - ls
  - search_knowledge_base
---
```

**효과**: 이 Skill이 활성화된 동안 Claude는 `write_file`, `edit_file`, `delete_file` 등을 호출할 수 없습니다.

### 11.2 Scripts: 컨텍스트를 소모하지 않는 실행

`scripts/` 디렉토리의 코드는 **컨텍스트에 로드되지 않고 직접 실행**됩니다. 이는 Level 3 리소스의 핵심 특성입니다.

```
# 컨텍스트에 로드됨 (토큰 소비):
references/aws-guide.md  → read_file로 읽으면 컨텍스트에 추가

# 컨텍스트에 로드되지 않음 (토큰 미소비):
scripts/validate.py      → Bash로 실행, 결과만 반환
```

**활용 패턴:**
```markdown
## 검증 절차
데이터 검증이 필요하면 아래 스크립트를 실행하세요:
`python -m scripts.quick_validate <input_file>`
결과를 읽고 사용자에게 보고하세요.
```

이 패턴으로 수백 줄의 검증 로직을 컨텍스트에 넣지 않고도 실행할 수 있습니다.

---

## 7. 팀 배포 및 공유

### 7.1 배포 방법

| 방법 | 적합한 경우 | 명령어 |
|------|------------|--------|
| **Git 저장소 커밋** | 프로젝트 팀 공유 | `git add skills/ && git commit` |
| **Plugin으로 배포** | 조직 전체 배포 | `/plugin marketplace add <repo>` |
| **Claude.ai 업로드** | 개인 사용 | Settings → Skills → Upload |
| **.skill 패키징** | 외부 배포 | `python -m scripts.package_skill <path>` |

### 7.2 Plugin 마켓플레이스 등록

```bash
# 1. 저장소에 .claude-plugin/plugin.json 추가
{
  "name": "my-skills",
  "description": "우리 팀의 커스텀 Skills 모음",
  "author": { "name": "Team Name" }
}

# 2. Claude Code에서 마켓플레이스 등록
/plugin marketplace add <github-org>/<repo>

# 3. 팀원이 설치
/plugin install <skill-name>@<marketplace-name>
```

---

## 8. Subagent 통합 패턴

Skills를 **커스텀 Subagent에 연결**하면, 격리된 전문가 위임이 가능합니다.

### 8.1 개념: Skills + Subagent = 전문가 에이전트

```
Orchestrator Agent
  ├─ Skill: "project-rules" (CLAUDE.md 역할)
  │
  ├─ SubAgent: "security-reviewer"
  │   └─ Skill: "security-audit" (보안 체크리스트)
  │
  └─ SubAgent: "code-reviewer"
      └─ Skill: "code-review" (리뷰 가이드라인)
```

각 SubAgent는 자신만의 Skill을 로드하여 전문화됩니다. 메인 에이전트의 컨텍스트는 오염되지 않습니다 (**Context Quarantine**).

### 8.2 DeepAgent에서의 구현

```python
from deepagents.middleware.subagents import SubAgent

security_subagent = SubAgent(
    name="security-reviewer",
    description="보안 취약점 검토",
    system_prompt=load_skill("security-audit"),  # Skill을 시스템 프롬프트로 주입
    tools=[grep_tool, read_file_tool],
)

orchestrator = create_deep_agent(
    model=llm,
    subagents=[security_subagent],
    system_prompt="보안 검토가 필요하면 security-reviewer에게 위임하세요.",
)
```

---

## 9. 트러블슈팅 가이드

### 9.1 Skill이 트리거되지 않을 때

| 증상 | 원인 | 해결 |
|------|------|------|
| Skill이 전혀 활성화되지 않음 | `description`이 너무 좁거나 모호함 | description을 더 적극적으로 작성. 트리거 키워드를 명시적으로 나열 |
| 다른 Skill이 대신 활성화됨 | description 간 충돌 | 각 Skill의 description 범위를 명확히 구분 |
| 간헐적으로만 트리거됨 | description에 핵심 키워드 누락 | "이 스킬은 X, Y, Z를 언급할 때 사용하세요" 명시 |

**진단 방법:**
```
1. Skill 메타데이터가 로드되었는지 확인 (Level 1)
2. 사용자 메시지와 description의 키워드 매칭 확인
3. description을 더 구체적으로 수정 후 재테스트
```

### 9.2 우선순위 충돌

여러 Skills의 description이 겹칠 때 발생합니다.

**해결 전략:**
- 각 Skill의 description에 **배타적 트리거 조건** 명시
- "X를 할 때는 이 스킬, Y를 할 때는 저 스킬" 형태로 구분
- 범용 Skill보다 특화 Skill이 우선되도록 description 구체화

### 9.3 런타임 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| 스크립트 실행 실패 | Python 의존성 미설치 | `requirements.txt` 또는 `pyproject.toml`에 의존성 명시 |
| reference 파일 못 찾음 | 경로 오류 | SKILL.md에서 상대 경로 확인 (`references/guide.md`) |
| allowed-tools 위반 | 필요한 도구가 허용 목록에 없음 | `allowed-tools`에 누락된 도구 추가 |
| YAML 파싱 에러 | frontmatter 문법 오류 | `---` 구분자 확인, YAML 유효성 검사 |

### 9.4 성능 최적화

| 문제 | 원인 | 해결 |
|------|------|------|
| 응답 속도 저하 | SKILL.md가 너무 큼 | 500줄 이내로 유지, 초과분은 references/로 분리 |
| 토큰 비용 증가 | 불필요한 reference 로드 | "필요할 때만 읽으세요" 지침을 SKILL.md에 명시 |
| 지침 무시 | SKILL.md가 복잡하여 Claude가 핵심을 놓침 | 핵심 지침을 상단에 배치, 구조를 단순화 |

---

## 10. Skills 생태계

### 10.1 공식 Skills 카테고리 (anthropics/skills 저장소)

| 카테고리 | 포함 스킬 | 라이선스 |
|----------|----------|----------|
| **Creative & Design** | 아트, 음악, 디자인 | Apache 2.0 |
| **Development & Technical** | 테스팅, MCP 서버 생성, 코딩 | Apache 2.0 |
| **Enterprise & Communication** | 비즈니스 워크플로우, 브랜딩 | Apache 2.0 |
| **Document Skills** | DOCX, PDF, PPTX, XLSX 변환 | Source-Available |

> 저장소: [github.com/anthropics/skills](https://github.com/anthropics/skills) (106k+ stars)

### 10.2 Skills 사용 환경

| 환경 | 사용 방법 |
|------|----------|
| **Claude Code** | `/plugin marketplace add anthropics/skills` → `/plugin install <skill>` |
| **Claude.ai** | 유료 플랜에서 Settings → Skills 업로드 |
| **Claude API** | Skills API로 프로그래밍 방식 적용 |

### 10.3 Skills + MCP 통합

Skills는 MCP(Model Context Protocol)와 결합하여 더 강력해집니다:

```
Skills = "무엇을 알아야 하는가" (지식, 절차, 정책)
MCP   = "어떻게 실행하는가" (외부 도구, 서비스 연결)
```

**통합 예시:**
- `code-review` Skill + MCP GitHub Server → 코드 리뷰 자동화
- `data-analysis` Skill + MCP Database Server → DB 직접 쿼리
- `document-gen` Skill + MCP File Server → 문서 자동 생성

---

## 11. skill-creator: Skills를 만드는 Skill

Day-03/skills/ 디렉토리에 포함된 `skill-creator`는 **Anthropic 공식 스킬**로, Skills의 전체 라이프사이클을 관리합니다.

### 11.1 skill-creator 워크플로우

```
1. 의도 파악 (Capture Intent)
   └─ 4가지 핵심 질문으로 스킬 요구사항 정의

2. 인터뷰 & 리서치 (Interview and Research)
   └─ 엣지 케이스, 입출력 형식, 의존성 확인

3. SKILL.md 작성 (Write the Skill)
   └─ frontmatter + 지침 + 예시 작성

4. 테스트 실행 (Run Evals)
   └─ with-skill vs baseline 비교 실행

5. 평가 & 개선 (Evaluate & Improve)
   └─ 정량 벤치마크 + 정성 피드백 → 스킬 개선

6. 패키징 (Package)
   └─ .skill 파일로 배포 가능하게 패키징
```

### 11.2 skill-creator 디렉토리 구조

```
skill-creator/
├── SKILL.md (32KB)           — 메인 스킬 지침
├── agents/                   — 서브에이전트 정의
│   ├── analyzer.md           — 벤치마크 분석
│   ├── comparator.md         — 블라인드 A/B 비교
│   └── grader.md             — assertion 채점
├── scripts/                  — 자동화 스크립트
│   ├── run_eval.py           — 평가 실행
│   ├── aggregate_benchmark.py — 벤치마크 집계
│   ├── run_loop.py           — description 최적화 루프
│   ├── package_skill.py      — .skill 패키징
│   ├── quick_validate.py     — 빠른 검증
│   └── ...
├── eval-viewer/              — 결과 시각화
│   ├── viewer.html           — SPA 뷰어 (44KB)
│   └── generate_review.py    — 리뷰 UI 생성
├── references/
│   └── schemas.md            — JSON 스키마 정의
└── assets/
    └── eval_review.html      — 평가 리뷰 템플릿
```

### 11.3 skill-creator가 보여주는 고급 패턴

`skill-creator`는 단순한 SKILL.md가 아니라 Agent Skills의 **모든 고급 패턴**을 활용하는 참조 구현입니다:

1. **Progressive Disclosure**: 32KB SKILL.md(Level 2) + references/(Level 3) 분리
2. **서브에이전트 위임**: agents/ 디렉토리의 3개 전문 에이전트
3. **스크립트 실행**: scripts/로 결정적 작업 (벤치마킹, 패키징) 분리
4. **자체 개선 루프**: description 최적화를 위한 5회 반복 루프 (run_loop.py)

---

## 12. Day-03 Skills 디렉토리 현황

```
Day-03/skills/
├── account-recovery/     — 계정 복구 스킬 (교육용 샘플)
│   └── SKILL.md
├── kb-search/            — KB 검색 스킬 (교육용 샘플)
│   └── SKILL.md
└── skill-creator/        — Anthropic 공식 스킬 생성기
    ├── SKILL.md
    ├── agents/ (3개)
    ├── scripts/ (9개)
    ├── eval-viewer/ (2개)
    ├── references/ (1개)
    └── assets/ (1개)
```

---

## 13. 참고 자료

| 자료 | URL |
|------|-----|
| Anthropic Skills GitHub | https://github.com/anthropics/skills |
| Claude Skills 사용법 | https://support.claude.com/en/articles/12512180-using-skills-in-claude |
| Skills API 가이드 | https://docs.claude.com/en/api/skills-guide |
| Claude Skills 기술 분석 보고서 | Day-03/Claude Skills 기술 분석 보고서.pdf |
| Building Effective Agents | https://www.anthropic.com/engineering/building-effective-agents |

---

> **면책 조항**: 이 가이드에 포함된 Skills는 데모 및 교육 목적으로 제공됩니다. 실제 Claude의 구현과 동작은 다를 수 있습니다. 중요한 작업에 의존하기 전에 자체 환경에서 충분히 테스트하세요.
