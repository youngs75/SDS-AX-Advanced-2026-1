# Repository Guidelines

## 프로젝트 개요
SDS AX Advanced 2026-1 교육과정 (2026-04-01 ~ 04-10) 저장소.
최종 산출물: AI Assistant Coding Agent Harness

## 프로젝트 구조

```
SDS-AX-Advanced-2026-1/
├── Day-01/                    # Day 1 교육 자료 및 실습
├── Day-02/                    # Day 2 교육 자료 및 실습
├── Day-03/                    # Day 3 교육 자료 및 실습 (GraphRAG 등)
├── Day-04/                    # Day 4 교육 자료 및 실습 (A2A, MCP)
│   ├── a2a/                   # 참조 구현 (교육용)
│   └── youngs75_a2a/          # 자체 구현 — A2A 멀티에이전트 프레임워크
│       ├── core/              #   도메인 무관 프레임워크
│       ├── a2a/               #   A2A 프로토콜 통합
│       ├── agents/            #   에이전트 구현체 (SimpleReAct, DeepResearch, DeepResearchA2A)
│       ├── docker/            #   프로덕션 배포 (MCP 3개 + Agent 3개 = 6 컨테이너)
│       ├── tests/             #   테스트 및 데모
│       ├── utils/             #   유틸리티
│       ├── REPORT.md          #   개발완료 보고서
│       └── CODING_AGENT_DESIGN.md  # 논문 인사이트 기반 Coding Agent 설계 가이드
├── pyproject.toml             # 프로젝트 의존성 (uv 기반)
├── uv.lock                    # 의존성 잠금 파일
├── AGENTS.md                  # 이 파일 — AI와 기여자가 따를 규칙 문서
└── .ai/sessions/              # 세션 인수인계 파일 저장 위치
```

기능 추가 시 Day-NN/ 또는 youngs75_a2a/ 하위에 배치합니다.
규칙이 여러 곳에 흩어져 있어도 기준 문서는 항상 `AGENTS.md`로 통일합니다.

## 커뮤니케이션 규칙
사용자와의 모든 소통은 항상 한국어로 진행합니다. 작업 전 계획 공유, 진행 상황 보고, 작업 후 결과 요약도 모두 한국어로 작성합니다. 코드 주석은 기존 스타일을 따르되, 새로 작성할 때는 한국어를 우선 사용합니다.

## 세션 파일 명명 규칙
세션 파일은 `.ai/sessions/session-YYYY-MM-DD-NNNN.md` 형식을 사용합니다.

- `YYYY-MM-DD`: 세션 당일 날짜
- `NNNN`: 같은 날짜 내 순번 (`0001`부터 시작)
- 같은 날짜 파일이 있으면 가장 큰 번호에 `+1`을 적용합니다.

## Resume 규칙
사용자가 `resume` 또는 `이어서`라고 요청하면 가장 최근 세션 파일을 찾아 이어서 작업합니다.

- `.ai/sessions/`에서 명명 규칙에 맞는 파일만 후보로 봅니다.
- 가장 최신 날짜를 우선 선택하고, 같은 날짜면 가장 큰 순번을 선택합니다.
- 초기 컨텍스트에 파일이 없어 보여도 실제 파일 시스템을 다시 확인합니다.
- 세션 파일 조회 또는 읽기가 샌드박스 제한으로 실패하면, `.ai/sessions/` 확인과 대상 파일 읽기에 필요한 최소 범위에서 권한 상승을 요청한 뒤 즉시 재시도합니다.
- 권한 상승이 필요한 이유는 세션 복구를 위한 실제 파일 시스템 확인임을 사용자에게 짧게 알립니다.
- 선택한 세션 파일은 전체를 읽습니다.
- 사용자에게 이전 작업 내용과 다음 할 일을 한국어로 간단히 브리핑합니다.

## Handoff 규칙
새 세션 파일은 사용자가 명시적으로 종료를 요청한 경우에만 생성합니다. 허용 트리거 예시는 `handoff`, `정리해줘`, `세션 저장`, `종료하자`, `세션 종료`입니다.

- 저장 위치는 항상 `.ai/sessions/`입니다.
- 기존 `session-*.md` 파일은 절대 수정하지 않습니다.
- 자동 저장이나 단계별 저장은 하지 않습니다.
- 새 파일에는 프로젝트 개요, 최근 작업 내역, 현재 상태, 다음 단계, 중요 참고사항을 포함합니다.
- 저장 후 사용자에게 생성된 파일 경로를 알립니다.

## 개발 및 검증 규칙

### 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv sync

# 환경변수 로드 (Day별 .env 파일)
export $(grep -v '^#' youngs75_a2a/.env | xargs)
```

### youngs75_a2a 테스트
```bash
cd Day-04

# Step 1: 프레임워크 자체 (외부 의존 없음)
python -m youngs75_a2a.tests.test_step1_no_llm

# Step 2: LLM 연동 (API 키 필요)
python -m youngs75_a2a.tests.test_step2_with_llm

# Step 3: 전체 파이프라인 (MCP 서버 필요)
python -m youngs75_a2a.tests.test_step3_full_pipeline

# Docker 배포 및 E2E
cd youngs75_a2a/docker && docker compose up -d
cd ../.. && python -m youngs75_a2a.tests.test_docker_e2e
```

구현 후에는 로그나 실행 결과로 정상 동작을 확인한 뒤에만 완료를 보고합니다.

## 커밋 및 PR 규칙
Conventional Commits 형식을 권장합니다. 예시: `feat: add coding agent scaffold`

- `origin`: 강사 리포지토리 (HyunjunJeon/SDS-AX-Advanced-2026-1)
- `mine`: 본인 리포지토리 (youngs75/SDS-AX-Advanced-2026-1)
- 강사 리포 동기화: `git fetch origin && git merge origin/main`
- Push는 항상 `mine`으로: `git push mine main`
- `.env` 파일, `.db` 파일, `.claude/` 디렉토리는 커밋하지 않습니다.
- PR에는 변경 목적, 검증 방법, 보류 이슈를 포함합니다.

## 주요 기술 스택
- **A2A SDK** 0.3.25 — Agent-to-Agent 프로토콜
- **LangGraph** 1.1.4 — 상태 그래프 기반 에이전트 오케스트레이션
- **LangChain Core** 1.2.23 — LLM 추상화
- **MCP** (langchain-mcp-adapters 0.2.2) — Model Context Protocol 도구 연동
- **Pydantic** 2.12.5 — 설정/스키마 검증
- **Starlette** + **Uvicorn** — ASGI 서버
- **Docker Compose** — 프로덕션 배포
