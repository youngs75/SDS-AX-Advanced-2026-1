# AI Agent Frameworks Current Trends: LangGraph vs Deep Agents

## Executive Summary
AI 에이전트 프레임워크의 최근 흐름은 두 갈래로 수렴한다. 하나는 **상태 기반 오케스트레이션(stateful orchestration)** 이고, 다른 하나는 **장기 과제 중심의 심층형 에이전트(deep agents)** 이다. LangGraph는 복잡한 분기, 반복, 체크포인트, 인간 승인, 멀티에이전트 조정이 필요한 운영형 워크플로에 강하고, Deep Agents는 리서치·코딩·문서화처럼 길고 복잡한 작업을 계획-분해-위임-기억-검증하는 하니스(harness) 중심 접근에 강하다. [1][2]

핵심적으로, **LangGraph는 “에이전트를 운영 가능한 시스템으로 만드는 오케스트레이션 레이어”**, **Deep Agents는 “장기 작업을 잘 수행하도록 설계된 실행 하니스와 패턴”**에 가깝다. 따라서 둘은 경쟁 관계라기보다 서로 다른 추상화 계층을 대표한다. [1][2]

## Current Trends in AI Agent Frameworks
최근 AI 에이전트 프레임워크 시장에서 반복적으로 보이는 트렌드는 다음과 같다.

1. **Stateful execution**: 단발성 tool-calling보다 상태와 실행 이력을 보존하는 구조가 중요해졌다.
2. **Human-in-the-loop**: 승인, 검토, 중단/재개가 가능한 통제형 워크플로가 중요해졌다.
3. **Multi-agent orchestration**: supervisor-worker 또는 분업형 구조가 널리 쓰인다.
4. **Long-context externalization**: 긴 컨텍스트를 프롬프트에 직접 넣기보다 파일, 메모리, 검색, 스킬로 외부화한다.
5. **Observability and evaluation**: trace, replay, sandbox, benchmark 같은 운영 체계가 프레임워크 선택의 핵심이 되었다.
6. **Harness engineering**: 모델 성능만이 아니라 도구, 프롬프트, 메모리, 검증 루프를 어떻게 설계하는지가 중요해졌다. [1][2]

이 흐름은 단순한 챗봇 제작 도구에서, 실제 업무를 안정적으로 수행하는 **production-grade agent systems**로 이동하고 있음을 보여준다.

## LangGraph: What It Is
LangGraph는 상태를 가진 그래프 기반 에이전트 오케스트레이션 프레임워크다. 단순 체인보다 그래프 구조를 통해 분기, 반복, 루프, 재시도, 중단/재개를 표현하기 쉽다. 공식 문서가 강조하는 핵심은 다음과 같다.

- **Stateful orchestration**
- **Durable execution**
- **Checkpointing / persistence**
- **Human-in-the-loop**
- **Subgraphs**
- **Streaming and interrupts** [1]

즉, LangGraph는 “무슨 답을 낼 것인가”보다 “그 답에 도달하는 과정을 어떻게 안정적으로 운영할 것인가”를 다루는 프레임워크다.

## LangGraph: Recent Direction
LangGraph의 최근 방향은 저수준 그래프 엔진에서 생산용 에이전트 플랫폼으로 확장되는 것이다.

- **Hierarchical multi-agent** 지원 강화
- **Supervisor / Swarm** 같은 멀티에이전트 패턴의 공식화
- **Command**를 통한 동적 흐름 제어
- **Memory / persistence** 강화
- **Observability 및 deployment 연계** 강화 [1][3]

이 방향은 기업 환경에서 필요한 복구성, 감사 가능성, 승인 절차, 장기 실행을 더 잘 지원하려는 흐름으로 해석할 수 있다.

## LangGraph: Strengths
### 1) 복잡한 워크플로에 강함
반복, 분기, 예외 처리, 승인, 재실행이 있는 워크플로를 표현하기 좋다.

### 2) 운영성에 강함
체크포인트와 durable execution 덕분에 실패 복구와 재개가 가능하다. [1]

### 3) 제어 가능성이 높음
흐름을 명시적으로 설계할 수 있어 규제 산업이나 내부 승인 프로세스에 유리하다.

### 4) 멀티에이전트 확장성
supervisor-worker, swarm, subgraph 패턴으로 확장 가능하다. [3]

## LangGraph: Limitations
### 1) 학습 난이도
추상화가 낮아 설계 부담이 크다.

### 2) 단순 작업에는 과함
간단한 tool-calling 챗봇에는 더 가벼운 접근이 적합할 수 있다.

### 3) 상태 설계 비용
state schema, reducer, checkpoint store, interrupts를 잘 설계해야 한다.

## Deep Agents: What It Is
Deep Agents는 장기 과제를 수행하도록 설계된 심층형 에이전트 하니스와 패턴을 가리킨다. LangChain 문서상 Deep Agents는 다음 구성요소를 중심으로 설명된다.

- **Planning and task decomposition**
- **Subagents**
- **Filesystem-backed context**
- **Long-running tool use**
- **Context engineering**
- **Skills / file-based knowledge loading**
- **Harness / CLI / customization** [2]

중요한 점은 Deep Agents가 단순한 모델 래퍼가 아니라, 장기 실행 작업을 위한 **실행 환경 전체**를 설계 대상으로 본다는 점이다.

## Deep Agents: Recent Direction
최근 Deep Agents의 방향은 다음과 같다.

- **Deep research**와 **async coding** 같은 장기 작업 중심 사용 사례 확대
- **Filesystem-based context engineering** 강화
- **Skills / AGENTS.md** 같은 파일 기반 능력 로딩
- **Harness engineering**과 평가 체계 중시
- **Subagents**를 통한 역할 분리 강화 [2]

즉, Deep Agents는 “더 똑똑한 단일 응답기”가 아니라 “길고 복잡한 작업을 잘 완주하는 운영형 에이전트”로 진화하고 있다.

## Deep Agents: Strengths
### 1) 장기 작업에 최적화
리서치, 코드 수정, 문서 작성, 검증 같은 작업에 적합하다.

### 2) 컨텍스트 외부화
파일시스템과 메모리 구조를 활용해 긴 작업의 상태를 관리한다. [2]

### 3) 역할 분리
subagent를 통해 연구, 실행, 검증을 분리할 수 있다.

### 4) 실무 친화성
실제 업무에서 필요한 filesystem, shell, search, write, eval 같은 도구 활용에 잘 맞는다.

## Deep Agents: Limitations
### 1) 복잡성
하니스, 툴, 메모리, 서브에이전트 조정이 필요하다.

### 2) 품질 변동성
계획 실패, 잘못된 컨텍스트 로딩, subagent 선택 오류가 발생할 수 있다.

### 3) 평가 어려움
open-ended task는 정답 정의가 어렵다.

## Direct Comparison
| 항목 | LangGraph | Deep Agents |
|---|---|---|
| 핵심 추상화 | 상태 기반 그래프 오케스트레이션 | 장기 과제형 에이전트 하니스 |
| 주된 문제 | 복잡한 워크플로 운영 | 긴 작업의 완주와 컨텍스트 관리 |
| 상태 관리 | checkpoint, persistence, interrupts | filesystem, memory, task decomposition |
| 멀티에이전트 | supervisor, swarm, subgraph | subagents, 역할 분리 |
| human-in-the-loop | 강함 | 가능하지만 중심 초점은 아님 |
| 운영성 | 매우 강함 | 중간 이상, harness 중심 |
| 개발 난이도 | 높음 | 중간~높음 |
| 적합한 사용 사례 | 승인형 워크플로, RAG orchestration, long-running workflows | deep research, coding agent, 문서/지식 작업 |
| 생태계 포지션 | orchestration layer | execution harness / agent workflow pattern |

## When to Choose Which
### LangGraph가 더 적합한 경우
- 승인/검토/중단/재개가 중요한 경우
- 상태가 복잡하고, 분기와 반복이 많은 경우
- 엔터프라이즈 워크플로와 결합해야 하는 경우
- 운영, 감사, 추적이 중요한 경우

### Deep Agents가 더 적합한 경우
- 리서치, 코딩, 문서화처럼 오래 걸리는 작업
- 파일 기반 컨텍스트 관리가 중요한 경우
- subagent로 업무를 분해하는 것이 효과적인 경우
- 실제 작업 완주율과 실행 하니스 품질이 중요한 경우

## Enterprise Implications
기업 관점에서 두 프레임워크는 서로 다른 가치가 있다.

- **LangGraph**는 거버넌스, 감사 가능성, 복구성, 통제력을 제공한다.
- **Deep Agents**는 장기 업무 자동화, 생산성 향상, 컨텍스트 엔지니어링을 제공한다.

실무적으로는 다음 전략이 현실적이다.

1. 단순한 작업은 lighter-weight agent로 시작한다.
2. 승인, 상태, 복구가 중요해지면 LangGraph를 도입한다.
3. 긴 작업, 리서치, 코드 작업이 중요해지면 Deep Agents 스타일 하니스를 사용한다.
4. 둘을 결합해, LangGraph로 상위 워크플로를 제어하고 Deep Agents 스타일 실행 단위를 하위 태스크로 두는 방식도 가능하다. [1][2]

## Bottom Line
AI 에이전트 프레임워크의 현재 동향은 **단일 대화형 에이전트**에서 **상태 기반, 장기 실행, 멀티에이전트, 관측 가능한 운영 시스템**으로 이동하고 있다. 그중 LangGraph는 **운영형 오케스트레이션의 대표주자**, Deep Agents는 **장기 과제형 실행 하니스의 대표주자**로 볼 수 있다. [1][2]

## Sources
[1] LangGraph official documentation and reference pages: https://docs.langchain.com/oss/python/langgraph/overview ; https://docs.langchain.com/oss/python/langgraph/durable-execution ; https://docs.langchain.com/oss/python/langgraph/interrupts ; https://docs.langchain.com/oss/python/langgraph/use-subgraphs ; https://reference.langchain.com/python/langgraph-supervisor ; https://reference.langchain.com/python/langgraph-swarm ; https://docs.langchain.com/oss/python/langgraph/graph-api

[2] Deep Agents official documentation and related pages: https://docs.langchain.com/oss/python/deepagents/overview ; https://docs.langchain.com/oss/python/deepagents/cli/overview ; https://docs.langchain.com/oss/python/deepagents/harness ; https://docs.langchain.com/oss/python/deepagents/customization ; https://docs.langchain.com/oss/python/deepagents/subagents ; https://docs.langchain.com/oss/python/deepagents/context-engineering

[3] LangChain release and product pages: https://docs.langchain.com/oss/python/releases/langchain-v1 ; https://docs.langchain.com/oss/python/langgraph/overview
