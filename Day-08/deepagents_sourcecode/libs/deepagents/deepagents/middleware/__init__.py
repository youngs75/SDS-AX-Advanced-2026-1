"""Deep Agents 에이전트를 위한 미들웨어 패키지.

## 개요

LLM은 두 가지 경로를 통해 도구(tool)를 전달받습니다:

1. **SDK 미들웨어** (이 패키지) -- 모든 SDK 사용자가 자동으로 얻게 되는
   도구, 시스템 프롬프트 주입, 요청 가로채기(intercept) 기능.
2. **소비자(Consumer) 제공 도구** -- `create_deep_agent()`의 `tools` 매개변수를 통해
   전달되는 일반 호출 가능(callable) 함수. CLI는 이 경로를 경량의 소비자별
   도구에 사용합니다.

두 경로 모두 `create_deep_agent()`에 의해 LLM이 보는 최종 도구 세트로 병합됩니다.

## 왜 일반 도구 대신 미들웨어를 사용하는가?

미들웨어는 `AgentMiddleware`를 상속하고, **모든 LLM 요청 전에 가로채는**
`wrap_model_call()` 훅을 오버라이드합니다. 이를 통해 미들웨어는 다음이 가능합니다:

* **도구를 동적으로 필터링** -- 예: `FilesystemMiddleware`는 백엔드가 지원하지 않을 경우
  호출 시점에 `execute` 도구를 제거합니다.
* **시스템 프롬프트 컨텍스트 주입** -- 예: `MemoryMiddleware`와 `SkillsMiddleware`는
  매 호출마다 관련 지시사항을 시스템 메시지에 주입하여 LLM이 자신이 제공하는
  도구의 사용법을 알 수 있게 합니다.
* **메시지 변환** -- 예: `SummarizationMiddleware`는 토큰을 세고, 오래된 도구 인자를
  절삭(truncate)하며, 컨텍스트 윈도우가 차면 히스토리를 요약으로 대체합니다.
* **턴(turn) 간 상태 유지** -- 미들웨어는 에이전트 턴에 걸쳐 지속되는 타입화된
  상태(state) 딕셔너리를 읽고 쓸 수 있습니다 (예: 요약 이벤트).

`tools=[]` 리스트의 일반 도구 함수는 이 중 어떤 것도 할 수 없습니다 --
그것은 LLM *에 의해* 호출될 뿐, LLM 호출 *전에* 호출되지 않습니다.

## 각 경로의 사용 시점

**미들웨어**를 사용해야 하는 경우:

* 호출마다 시스템 프롬프트나 도구 목록을 수정해야 할 때
* 턴 간 상태를 추적해야 할 때
* 모든 SDK 소비자가 사용할 수 있어야 할 때 (CLI만이 아닌)

**일반 도구**를 사용해야 하는 경우:

* 함수가 상태 없이(stateless) 자기 완결적(self-contained)일 때
* 시스템 프롬프트나 요청 수정이 필요 없을 때
* 도구가 단일 소비자에 한정될 때 (예: CLI 전용)
"""

# === 비동기 서브에이전트 미들웨어 ===
# 원격 Agent Protocol 서버에서 백그라운드 태스크로 실행되는 비동기 서브에이전트 지원
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware

# === 파일시스템 미들웨어 ===
# ls, read_file, write_file, edit_file, glob, grep, execute 등 파일 조작 도구 제공
from deepagents.middleware.filesystem import FilesystemMiddleware

# === 메모리 미들웨어 ===
# AGENTS.md 파일에서 프로젝트별 컨텍스트/지시사항을 로드하여 시스템 프롬프트에 주입
from deepagents.middleware.memory import MemoryMiddleware

# === 스킬 미들웨어 ===
# SKILL.md 파일에서 스킬 메타데이터를 로드하고 점진적 공개(progressive disclosure) 패턴으로 노출
from deepagents.middleware.skills import SkillsMiddleware

# === 동기 서브에이전트 미들웨어 ===
# `task` 도구를 통해 동기적으로 서브에이전트를 생성하고 실행 결과를 반환
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

# === 요약(Summarization) 미들웨어 ===
# 자동/수동 대화 요약 및 컨텍스트 압축 기능 제공
from deepagents.middleware.summarization import (
    SummarizationMiddleware,         # 자동 요약 미들웨어 (토큰 임계값 초과 시 자동 실행)
    SummarizationToolMiddleware,     # compact_conversation 도구를 통한 수동 요약 미들웨어
    create_summarization_tool_middleware,  # 모델 인식 기본값으로 두 미들웨어를 함께 생성하는 팩토리 함수
)

# 패키지의 공개 API 목록
# 이 리스트에 포함된 이름들만 `from deepagents.middleware import *` 시 노출됩니다.
__all__ = [
    "AsyncSubAgent",                        # 비동기 서브에이전트 설정 TypedDict
    "AsyncSubAgentMiddleware",              # 비동기 서브에이전트 미들웨어 클래스
    "CompiledSubAgent",                     # 사전 컴파일된 서브에이전트 설정 TypedDict
    "FilesystemMiddleware",                 # 파일시스템 도구 미들웨어 클래스
    "MemoryMiddleware",                     # 메모리(AGENTS.md) 미들웨어 클래스
    "SkillsMiddleware",                     # 스킬(SKILL.md) 미들웨어 클래스
    "SubAgent",                             # 동기 서브에이전트 설정 TypedDict
    "SubAgentMiddleware",                   # 동기 서브에이전트 미들웨어 클래스
    "SummarizationMiddleware",              # 자동 요약 미들웨어 클래스
    "SummarizationToolMiddleware",          # 수동 요약 도구 미들웨어 클래스
    "create_summarization_tool_middleware",  # 요약 미들웨어 팩토리 함수
]
