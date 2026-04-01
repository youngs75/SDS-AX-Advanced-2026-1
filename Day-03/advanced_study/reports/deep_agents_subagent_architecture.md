# Deep Agents의 서브에이전트 아키텍처 조사 보고서

## 개요
LangChain의 Deep Agents는 복잡한 장기 작업을 수행하기 위해 계획(Planning), 파일시스템, 서브에이전트(subagents)를 결합한 에이전트 실행 프레임워크로 볼 수 있다. 핵심 아이디어는 메인 에이전트가 모든 일을 직접 처리하지 않고, 전문화된 서브에이전트에 작업을 위임해 컨텍스트를 분리하고 병렬성을 확보하는 것이다. [1][2][4]

## 서브에이전트의 역할
서브에이전트는 메인 에이전트가 호출하는 전문화된 하위 실행 단위다. 일반적인 도구(tool)와 달리, 각 서브에이전트는 자체적인 설명, 시스템 프롬프트, 도구 집합, 때로는 별도 모델을 가질 수 있다. 이 때문에 서브에이전트는 단순 함수 호출보다 더 큰 자율성을 갖는 “작은 에이전트”에 가깝다. [2][3][6]

주요 목적은 다음과 같다.
- 작업 분해: 복잡한 요청을 작은 하위 작업으로 나눈다. [1][5]
- 컨텍스트 격리: 대량 자료 조사나 긴 추론을 메인 컨텍스트에서 분리한다. [4][5]
- 전문화: 조사, 분석, SQL, 코드 작업 등 역할별로 최적화한다. [2][3]
- 병렬 처리: 독립적인 하위 작업을 동시에 실행한다. [5][6]

## 대표 아키텍처 패턴

### 1) 계획(Planning) + 작업 분해
Deep Agents는 즉시 실행보다 먼저 계획을 세우는 방식에 가깝다. 일반적으로 요청 해석 → 하위 과제 분해 → TODO/계획 작성 → 실행 → 집계의 순서로 동작한다. 공식 문서도 planning tool을 핵심 능력으로 강조한다. [1][5]

### 2) 위임(Delegation)
메인 에이전트는 특정 업무를 서브에이전트에 위임한다. 예를 들어, 한 서브에이전트는 웹 조사에 집중하고, 다른 서브에이전트는 데이터 분석이나 SQL 질의에 집중할 수 있다. [2][6]

### 3) 병렬 실행(Parallel execution)
서로 독립적인 하위 작업은 병렬 서브에이전트로 처리할 수 있다. 예를 들어, 여러 출처를 동시에 조사하거나 서로 다른 관점의 비교 분석을 진행할 수 있다. Deep research 예제는 이러한 병렬 패턴을 보여준다. [5]

### 4) 검증(Verification)
서브에이전트 결과는 메인 에이전트가 다시 확인하는 구조가 일반적이다. 충돌하는 결과를 비교하거나, 추가 도구 호출을 통해 교차검증할 수 있다. 이는 서브에이전트가 제공하는 결과를 “그대로 믿기보다” 후속 검토하는 패턴이다. [4][5][6]

### 5) 집계(Aggregation)
메인 에이전트는 여러 서브에이전트의 결과를 종합해 최종 답변을 생성한다. 즉, 서브에이전트는 부분 결과를 만들고, 메인 에이전트가 이를 조립하는 계층 구조다. [5][6]

## 구현 요소와 공식 예시
공식 문서와 저장소에는 Deep Agents를 구성하는 주요 요소가 소개된다.
- Python의 `create_deep_agent` API [3]
- 서브에이전트 개념인 `SubAgent`와 `CompiledSubAgent` [3]
- 서브에이전트별로 다른 도구/프롬프트/모델을 부여하는 구성 [2][3]
- 기존 LangGraph 그래프를 서브에이전트로 래핑하는 방식 [3]

또한 deep research 예제는 병렬 subagents와 built-in planning tool을 함께 사용하는 대표 사례로 제시된다. [5][8]

## 장점
1. 복잡한 작업에 강하다: 장기적, 다단계, 다출처 작업에 적합하다. [1][5]
2. 컨텍스트 격리를 제공한다: 메인 에이전트의 컨텍스트 오염을 줄인다. [4]
3. 전문화를 쉽게 구현한다: 역할별 서브에이전트를 구성할 수 있다. [2][3]
4. 병렬성을 활용할 수 있다: 독립 과제를 동시에 처리할 수 있다. [5]
5. 기존 LangGraph 자산을 재사용할 수 있다: pre-compiled graph를 서브에이전트로 넣을 수 있다. [3]

## 한계
1. 오케스트레이션 복잡도가 높아진다: 위임 기준, 라우팅, 집계 로직이 필요하다. [6]
2. 잘못된 서브에이전트 선택 문제가 생길 수 있다. [2]
3. 모델 호출 수가 늘어 비용이 증가할 수 있다. [6]
4. 검증 책임이 사라지지 않는다: 결과를 메인 에이전트가 재검토해야 한다. [4][5]

## 다른 프레임워크와의 위치
Deep Agents는 단순한 단일 에이전트보다 의견이 강한(opinionated) 상위 레벨 프레임워크에 가깝다. LangGraph와는 경쟁 관계라기보다 상보적 관계로 볼 수 있고, Deep Agents는 계획·파일시스템·서브에이전트를 더 쉽게 쓰게 해주는 실행 하네스(harness)에 가깝다. 비교 문서는 Claude Agent SDK, Codex와의 차이도 다루지만, 일부 비교는 문서의 직접적 명제라기보다 포지셔닝 해석으로 보는 것이 안전하다. [7][8]

## 결론
Deep Agents의 서브에이전트 아키텍처는 “메인 에이전트 + 전문화된 하위 에이전트 + 계획 + 파일시스템”으로 요약할 수 있다. 핵심 패턴은 계획, 위임, 병렬 실행, 검증, 집계이며, 이 구조는 복잡한 조사·분석·코딩 업무에 특히 적합하다. 다만 성능과 유연성만큼 오케스트레이션 복잡도와 검증 비용도 함께 증가한다. [1][4][5][6]

## 출처
[1] LangChain Docs — Deep Agents overview  
https://docs.langchain.com/oss/python/deepagents/overview

[2] LangChain Docs — Subagents (JavaScript)  
https://docs.langchain.com/oss/javascript/deepagents/subagents

[3] LangChain Reference — Subagents / Agent / CompiledSubAgent / create_deep_agent  
https://reference.langchain.com/python/deepagents/subagents  
https://reference.langchain.com/python/deepagents/agent

[4] LangChain Docs — Context engineering in Deep Agents  
https://docs.langchain.com/oss/python/deepagents/context-engineering

[5] LangChain Docs — Build a deep research agent  
https://docs.langchain.com/oss/python/deepagents/deep-research

[6] LangChain Docs — Multi-agent / Subagents pattern  
https://docs.langchain.com/oss/python/langchain/multi-agent  
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

[7] LangChain Docs — Comparison with Claude Agent SDK and Codex  
https://docs.langchain.com/oss/python/deepagents/comparison

[8] GitHub — langchain-ai/deepagents  
https://github.com/langchain-ai/deepagents
