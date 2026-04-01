# AI 에이전트 프레임워크 동향: LangGraph vs Deep Agents

## 요약
AI 에이전트 프레임워크의 최근 흐름은 **자율성 극대화**보다 **제어 가능성, 지속 실행, 관측성, 인간 개입, 재현성**으로 이동하고 있다. 이 흐름에서 **LangGraph**는 상태 기반 그래프 오케스트레이션과 프로덕션 운영 기능으로, **Deep Agents**는 장기 작업·서브에이전트·파일시스템 중심의 agent harness로 각각 다른 위치를 차지한다. 둘은 경쟁 관계이기도 하지만, 실제로는 **런타임( LangGraph ) vs 작업 실행 하니스(Deep Agents)**로 보아야 정확하다 [1][2].

## 현재 동향
- 에이전트 시장은 데모 중심의 “완전 자율형”에서 벗어나, 실제 업무에 필요한 **장기 실행, 실패 복구, 승인 절차, 추적 가능성**을 중시하는 방향으로 재편되고 있다 [3].
- 프레임워크 경쟁도 “누가 더 자율적인가”보다 “누가 더 안전하게 운영 가능한가”로 이동했다.
- 이 맥락에서 LangGraph는 **stateful workflow / durable execution / human-in-the-loop**를 전면에 내세우고, Deep Agents는 **복잡한 작업을 오래 유지하며 끝까지 수행하는 harness**로 포지셔닝된다 [1][2].

## LangGraph 개요
LangGraph는 에이전트 흐름을 **그래프 기반 상태 머신**처럼 다루는 프레임워크다. 핵심은 다음과 같다.

- **명시적 상태(state)** 관리
- **checkpointing / persistence**
- **interrupts**를 통한 human-in-the-loop
- **durable execution**
- **streaming**
- **time travel**(체크포인트 단위 재실행/복구)
- **subgraph / multi-agent orchestration**

이는 단순한 LLM wrapper가 아니라, 에이전트 실행을 제어하는 **control plane**에 가깝다 [1].

### 강점
- 복잡한 분기와 반복이 많은 업무에 적합
- 실패 복구와 재개가 쉬움
- 승인/검토가 필요한 업무에 강함
- LangSmith와 결합 시 관측·평가·배포 체계가 강화됨 [4][5]

### 한계
- 학습 곡선이 있음
- 상태/전이/예외 설계를 잘 해야 함
- 단순 챗봇에는 과할 수 있음
- 인프라 복잡도가 높아질 수 있음

## Deep Agents 개요
Deep Agents는 LangChain 문맥에서 **agent harness**로 정의되며, 장기 작업을 안정적으로 수행하기 위한 상위 실행 계층이다 [2]. 공식 문서에서 강조되는 요소는 다음과 같다.

- **planning / task decomposition**
- **long-running tasks**
- **filesystem-backed state**
- **subagent spawning**
- **long-term memory**
- **skills** 기반 모듈화

즉, Deep Agents는 범용 오케스트레이션 프레임워크라기보다, **복잡한 작업을 구조화해서 끝내는 실행 하니스**에 가깝다 [2].

### 강점
- 장기 작업에 강함
- 컨텍스트 오염을 줄이기 좋음
- 서브에이전트로 역할 분리 가능
- 파일/메모리/스킬로 작업 지식 축적에 유리

### 한계
- 설계 복잡도가 높음
- 도구/파일/권한 관리가 필요함
- 장기 실행 구조라 비용과 지연이 늘 수 있음
- 완전 자율보다 제어된 자율성에 적합

## LangGraph vs Deep Agents 비교

| 항목 | LangGraph | Deep Agents |
|---|---|---|
| 위치 | 에이전트 런타임 / 오케스트레이션 프레임워크 | agent harness / 장기 작업 실행 계층 |
| 핵심 철학 | 상태 기반 제어, 재현성, 운영 안정성 | 복잡한 작업을 오래 유지하며 완료 |
| 구조 | 그래프, 노드/엣지, 체크포인트, interrupts | 메인 에이전트 + 서브에이전트, 파일시스템, 메모리, 스킬 |
| 강점 | 프로덕션 운영, HITL, 복구성 | 장기 작업, 컨텍스트 관리, 작업 분해 |
| 약점 | 설계 복잡성, 러닝커브 | 운영/권한 복잡성, 비용 증가 |
| 적합한 팀 | 규제/고신뢰 업무, 복잡한 워크플로 팀 | 리서치, 코딩, 문서 자동화, 장기 작업 팀 |
| 대표 사용 사례 | 고객지원, DevOps, 금융, 헬스케어, 사내 코파일럿 | 리서치, 코딩 에이전트, 브라우저 작업, 문서 생성 |

## 어떤 상황에 무엇을 선택할까

### LangGraph가 더 적합한 경우
- 승인/감사/중단-재개가 중요한 업무
- 외부 시스템 연동이 많은 복잡한 workflow
- 운영 가시성과 재현성이 중요한 조직
- 여러 분기와 예외 경로를 명시적으로 관리해야 하는 경우

### Deep Agents가 더 적합한 경우
- 긴 조사, 문서 생성, 코드 작업처럼 **하나의 작업을 오래 끌고 가며 마무리**해야 하는 경우
- 서브에이전트 분업이 유효한 경우
- 파일 기반 상태와 메모리 축적이 중요한 경우
- 도메인별 능력을 스킬로 분리하고 싶은 경우

## 생태계와 채택 동향
- LangGraph는 LangChain/LangSmith 생태계와 결합해 **관측, 평가, 배포**까지 포함한 운영 스택을 형성하고 있다 [4][5].
- Deep Agents는 공식 문서상 **long-running tasks, filesystem backends, skills, subagents**를 지원하는 harness로 설명되며, 장기 작업 중심의 에이전트 설계 흐름을 반영한다 [2].
- 공개 사례는 LangGraph 쪽이 더 많고 가시적이다. 다만 공식 사례가 곧 시장 점유를 의미하는 것은 아니며, 대표 사례 존재와 광범위 채택은 구분해서 봐야 한다 [6][7].

## 결론
현재의 에이전트 프레임워크 트렌드는 “더 자율적인 에이전트”가 아니라 **더 통제 가능하고, 복구 가능하며, 운영 가능한 에이전트**로 향하고 있다. 이 기준에서:

- **LangGraph**는 복잡한 업무를 안정적으로 운영하기 위한 **제어층**
- **Deep Agents**는 장기 작업을 수행하기 위한 **실행 하니스**

로 구분할 수 있다. 실무적으로는 둘 중 하나만 고르는 문제가 아니라, **LangGraph로 워크플로를 제어하고, Deep Agents 스타일의 장기 작업/서브에이전트 패턴을 그 안에 얹는 조합**도 유효하다.

## 출처
[1] LangGraph 공식 개요 문서: https://docs.langchain.com/oss/python/langgraph/overview
[2] Deep Agents 공식 개요/하니스 문서: https://docs.langchain.com/oss/python/deepagents/overview , https://docs.langchain.com/oss/python/deepagents/harness
[3] LangChain State of AI Agents Report: https://www.langchain.com/stateofaiagents
[4] LangGraph Persistence / Durable Execution 문서: https://docs.langchain.com/oss/python/langgraph/persistence , https://docs.langchain.com/oss/python/langgraph/durable-execution
[5] LangSmith Deployment 문서: https://docs.langchain.com/langsmith/deployment
[6] LangGraph case studies: https://docs.langchain.com/oss/python/langgraph/case-studies
[7] LangChain blog, Top 5 LangGraph Agents in Production 2024: https://blog.langchain.com/top-5-langgraph-agents-in-production-2024/
