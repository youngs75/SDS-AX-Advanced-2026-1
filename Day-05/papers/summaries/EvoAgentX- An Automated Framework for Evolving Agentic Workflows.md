# EvoAgentX: An Automated Framework for Evolving Agentic Workflows

## 1. 기본 정보 (Basic Information)

| 항목 | 내용 |
|------|------|
| **제목** | EvoAgentX: An Automated Framework for Evolving Agentic Workflows |
| **저자** | Yingxu Wang, Siwei Liu, Jinyuan Fang, Zaiqiao Meng |
| **소속** | Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), University of Aberdeen, University of Glasgow |
| **출판 정보** | arXiv:2507.03616v2, 2025년 9월 23일 |
| **페이지 수** | 13페이지 (본문 + 부록) |
| **키워드** | Multi-Agent Systems, Workflow Optimization, TextGrad, AFlow, MIPRO, Evolutionary Optimization, LLM Agents |

---

## 2. 한 줄 요약 (One-line Summary)

멀티 에이전트 워크플로우의 자동 생성, 실행, 진화적 최적화를 지원하는 오픈소스 플랫폼 EvoAgentX를 제안하며, TextGrad, AFlow, MIPRO 세 가지 최적화 알고리즘을 통합하여 HotPotQA에서 7.44%, MBPP에서 10.00%, MATH에서 10.00%, GAIA에서 최대 20.00%의 성능 향상을 달성한다.

---

## 3. 초록 요약 (Abstract Summary)

멀티 에이전트 시스템(MAS)은 대규모 언어 모델과 전문 도구를 조율하여 복잡한 태스크를 해결하는 강력한 패러다임으로 부상했다. 그러나 기존 MAS 프레임워크들은 수동 워크플로우 설정에 의존하며, 동적 진화와 성능 최적화를 위한 네이티브 지원이 부족하다. 또한 MAS 최적화 알고리즘들이 통합된 프레임워크에 결합되어 있지 않다.

이 논문은 **EvoAgentX**를 제안한다. 이것은 멀티 에이전트 워크플로우의 자동 생성, 실행, 진화적 최적화를 위한 오픈소스 플랫폼이다. 핵심 특징은 다음과 같다:
- **5계층 모듈러 아키텍처**: Basic Components, Agent, Workflow, Evolving, Evaluation
- **3가지 최적화 알고리즘 통합**: TextGrad, AFlow, MIPRO
- **다양한 벤치마크에서 일관된 성능 향상**: HotPotQA F1 7.44% 향상, MBPP pass@1 10.00% 향상, MATH 정확도 10.00% 향상, GAIA 전체 정확도 최대 20.00% 향상

---

## 4. 상세 내용 정리 (Detailed Content Summary)

### Section 1: Introduction (서론)

멀티 에이전트 시스템(MAS)은 계획, 추론, 코드 생성 등 각기 다른 역량을 가진 여러 에이전트를 조율하여 복잡한 문제를 해결하는 패러다임이다. MAS는 다중 홉 질의응답, 소프트웨어 엔지니어링 자동화, 코드 생성, 수학 문제 풀이, 대화 시스템 등에 폭넓게 활용되고 있다.

그러나 현재 MAS 프레임워크들의 두 가지 핵심 한계를 지적한다:

1. **수동 워크플로우 설계 의존**: CrewAI, CAMEL AI, LangGraph 등 기존 프레임워크는 에이전트 역할 정의, 태스크 분해 전략 설계, 상호작용 패턴 설계, 실행 워크플로우 설정 등을 수동으로 수행해야 한다. 이는 확장성과 적응성을 제한한다.

2. **동적 진화 메커니즘 부족**: DSPy, TextGrad 등 일부 최적화 방법이 존재하지만, 단편적이며 통합 플랫폼에 결합되어 있지 않아 실무자가 다양한 태스크에 일관되게 적용하기 어렵다.

EvoAgentX는 이 두 가지 문제를 해결하기 위해 다음 세 가지 핵심 기여를 제공한다:
- 높은 수준의 목표 설명에서 멀티 에이전트 워크플로우를 자동 생성
- TextGrad, AFlow, MIPRO를 통합한 진화적 워크플로우 최적화
- 내장 벤치마크 및 표준화된 평가 메트릭 제공

### Section 2: Related Work (관련 연구)

#### 2.1 Multi-Agent Systems

CAMEL AI, CrewAI, LangGraph 등 범용 MAS 프레임워크의 발전을 검토한다. 이들은 에이전트 역할 정의, 통신 프로토콜, 실행 로직 설계를 지원하지만, 대부분 수작업 워크플로우에 의존한다. EvoAgentX는 높은 수준의 태스크 설명에서 자동 워크플로우 생성을 도입하여 이 한계를 극복한다.

#### 2.2 Multi-Agent Optimization

멀티 에이전트 최적화 연구의 발전을 추적한다:
- **프롬프트 최적화**: DSPy, TextGrad
- **동적 토폴로지 조정**: DyLAN, Captain Agent
- **토폴로지 검색 및 최적화**: AutoFlow, GPTSwarm (강화학습 활용)
- **선호 학습 기반**: ScoreFlow, G-Designer
- **통합 최적화**: ADAS, FlowReasoner, MaAS (프롬프트와 토폴로지 동시 최적화)

EvoAgentX는 이들을 단일 엔드투엔드 시스템으로 통합하는 플랫폼을 제공한다.

### Section 3: System Design (시스템 설계)

EvoAgentX의 핵심인 5계층 모듈러 아키텍처를 상세히 설명한다.

#### 3.1 Basic Component Layer (기본 컴포넌트 계층)

시스템의 안정성, 확장성, 확장 가능성을 보장하는 기반 인프라를 제공한다:
- **설정 관리**: YAML/JSON 구조화 파일에서 시스템 파라미터 검증
- **로깅**: 시스템 이벤트 및 성능 메트릭 추적
- **파일 관리**: 워크플로우 상태 및 에이전트 체크포인트 관리
- **저장소 관리**: 캐싱 및 체크포인팅을 포함한 영구/임시 저장소 지원
- **LLM 통합**: OpenRouter, LiteLLM을 통한 다양한 LLM 연동

#### 3.2 Agent Layer (에이전트 계층)

모듈러 지능 엔티티의 구축을 담당한다. 각 에이전트는 다음과 같이 구성된다:

```
ai = <LLMi, Memi, {Acti(j)}j=1..M>
```

- **LLM**: 고수준 추론, 응답 생성, 컨텍스트 해석 담당
- **Memory (Memi)**: 메모리 모듈
- **Actions (Acti)**: 특정 태스크를 캡슐화하는 행동 모듈 (요약, 검색, API 호출 등)

각 액션은 프롬프트 템플릿, 입출력 형식, 선택적 도구 통합으로 구성된다.

#### 3.3 Workflow Layer (워크플로우 계층)

멀티 에이전트 워크플로우의 구성, 조율, 실행을 지원한다. 워크플로우는 방향 그래프로 모델링된다:

```
W = (V, E)
```

- **V**: 태스크 노드 집합 (PENDING, RUNNING, COMPLETED, FAILED 상태)
- **E**: 의존성과 데이터 흐름을 인코딩하는 방향 간선

두 가지 워크플로우 유형을 지원한다:
- **WorkFlowGraph**: 복잡한 태스크 그래프를 명시적으로 정의 (커스텀 노드, 간선, 조건 분기, 병렬 실행)
- **SequentialWorkFlowGraph**: 태스크 입출력 의존성을 기반으로 연결을 자동 추론 (빠른 프로토타이핑에 적합)

#### 3.4 Evolving Layer (진화 계층)

EvoAgentX의 핵심 차별화 요소로, 세 가지 최적화기(Optimizer)로 구성된다:

**1. Agent Optimizer (에이전트 최적화기)**

에이전트의 프롬프트 템플릿, 도구 설정, 행동 전략을 반복적으로 개선한다:
```
(Prompt(t+1), theta(t+1)) = O_agent(Prompt(t), theta(t), E)
```
- **TextGrad**: 그래디언트 기반 프롬프트 튜닝
- **MIPRO**: 인컨텍스트 학습 및 선호도 기반 개선

**2. Workflow Optimizer (워크플로우 최적화기)**

태스크 분해 및 실행 흐름을 개선한다:
```
W(t+1) = O_workflow(W(t), E)
```
- **SEW**: 자기 진화 에이전트 워크플로우
- **AFlow**: 노드 순서 변경, 의존성 수정, 대안적 실행 전략 탐색

**3. Memory Optimizer (메모리 최적화기)**

현재 개발 중이며, 선택적 보유, 동적 가지치기, 우선순위 기반 검색을 위한 구조화된 메모리 모듈을 제공하는 것이 목표이다:
```
M(t+1) = O_memory(M(t), E)
```

#### 3.5 Evaluation Layer (평가 계층)

두 가지 상호보완적 컴포넌트로 체계적 성능 평가를 제공한다:
1. **Task-specific Evaluator**: 도메인 관련 메트릭으로 정량 평가 (ground truth 대비 비교)
2. **LLM-based Evaluator**: 정성적 평가, 일관성 검사, 동적 기준 평가

### Section 4: Experiments (실험)

#### 4.1 Evolution Algorithms (진화 알고리즘)

세 가지 벤치마크에서 세 가지 최적화 알고리즘을 평가한다:

| 벤치마크 | 원본 | TextGrad | AFlow | MIPRO |
|----------|------|----------|-------|-------|
| HotPotQA (F1%) | 63.58 | **71.02** | 65.09 | 69.16 |
| MBPP (Pass@1%) | 69.00 | 71.00 | **79.00** | 68.00 |
| MATH (Solve%) | 66.00 | **76.00** | 71.00 | 72.30 |

핵심 발견:
- **TextGrad**가 다중 홉 추론(HotPotQA)과 수학 추론(MATH)에서 가장 큰 향상을 달성
- **AFlow**가 코드 생성(MBPP)에서 최고 성능 달성 (69% -> 79%)
- 각 최적화 알고리즘이 태스크별로 다른 강점을 보임

#### 4.2 Applications (실제 응용)

GAIA 벤치마크에서 두 가지 오픈소스 프레임워크를 최적화한 결과:

- **Open Deep Research**: 전체 정확도 18.41% 향상 (Level 1: +20.00%, Level 2: +8.71%, Level 3: +7.69%)
- **OWL Agent**: 전체 정확도 20.00% 향상 (Level 1: +28.57%, Level 2: +10.00%, Level 3: +100.00%)

이 결과는 EvoAgentX가 기존 멀티 에이전트 시스템의 성능을 자동 프롬프트 및 토폴로지 최적화를 통해 크게 향상시킬 수 있음을 입증한다.

#### 4.3 Case Study (사례 연구)

세 가지 최적화 알고리즘의 실제 작동 과정을 보여준다:

- **AFlow 워크플로우 최적화**: 단순한 단일 에이전트 수학 풀이에서 문제 분석, Python 코드 생성, 앙상블 기반 솔루션 개선을 포함하는 다중 에이전트 워크플로우로 자동 확장
- **TextGrad 프롬프트 최적화**: "Answer the math question" 수준의 단순 프롬프트가 문제 복잡도 평가, 관련 수학 원리 적용, 단계별 검증, 시각적 보조 활용 등을 포함하는 상세한 구조화된 프롬프트로 발전
- **MIPRO 프롬프트 최적화**: 단순 프롬프트를 중간 단계, 관련 수학 개념 설명, 풀이 과정 분해를 포함하는 프롬프트로 개선하며, 인컨텍스트 예시까지 자동 추가

### Section 5: Conclusion (결론)

EvoAgentX는 수동 워크플로우 설계의 필요성을 제거하고 동적, 태스크 특화 최적화를 지원함으로써 기존 프레임워크의 핵심 한계를 해결한다. 향후 계획으로 더 많은 최적화 알고리즘 확장, 도구 통합 강화, 장기 메모리 기능 추가, 그리고 MASS, AlphaEvolve, Darwin Godel Machine 등 고급 진화 전략 탐색을 제시한다.

---

## 5. 핵심 포인트 (Key Points)

1. **5계층 모듈러 아키텍처**: Basic Components -> Agent -> Workflow -> Evolving -> Evaluation의 계층적 구조로 관심사를 분리하여 확장성과 유연성을 확보한다. 특히 Evolving 계층이 핵심 차별화 요소이다.

2. **3가지 최적화 알고리즘 통합**: TextGrad(그래디언트 기반 프롬프트 최적화), AFlow(워크플로우 토폴로지 최적화), MIPRO(인컨텍스트 학습 최적화)를 하나의 플랫폼에 통합하여 에이전트 프롬프트, 도구 설정, 워크플로우 구조를 반복적으로 개선한다.

3. **자동 워크플로우 생성**: 높은 수준의 목표 설명에서 멀티 에이전트 워크플로우를 자동으로 생성하여 수동 설계의 부담을 크게 줄인다. SequentialWorkFlowGraph를 통한 빠른 프로토타이핑이 가능하다.

4. **태스크별 최적화 알고리즘 차별화**: TextGrad는 추론 태스크(HotPotQA, MATH)에, AFlow는 코드 생성(MBPP)에 강점을 보여, 태스크 특성에 따른 최적화 알고리즘 선택이 중요함을 보여준다.

5. **기존 시스템 성능 향상**: GAIA 벤치마크에서 Open Deep Research(+18.41%)와 OWL Agent(+20.00%)의 기존 성능을 자동 최적화만으로 크게 향상시킨 실용적 효과를 입증한다.

6. **메모리 최적화기의 미완성**: 에이전트 최적화기와 워크플로우 최적화기는 구현되었으나, 메모리 최적화기는 아직 개발 중으로, 선택적 보유, 동적 가지치기, 우선순위 기반 검색 기능이 향후 추가될 예정이다.

---

## 6. 핵심 인용구 (Key Quotes)

1. **"Multi-agent systems (MAS) have emerged as a powerful paradigm for orchestrating large language models (LLMs) and specialized tools to collaboratively address complex tasks."**
   - (Abstract) -- 멀티 에이전트 시스템이 복잡한 태스크 해결을 위한 강력한 패러다임으로 부상했음을 선언한다.

2. **"However, existing MAS frameworks often require manual workflow configuration and lack native support for dynamic evolution and performance optimization."**
   - (Abstract) -- 기존 MAS 프레임워크의 핵심 한계인 수동 설정 의존과 동적 진화 지원 부재를 지적한다.

3. **"EvoAgentX employs a modular architecture consisting of five core layers: the basic components, agent, workflow, evolving, and evaluation layers."**
   - (Abstract) -- EvoAgentX의 핵심 아키텍처인 5계층 구조를 명시한다.

4. **"EvoAgentX consistently achieves significant performance improvements, including a 7.44% increase in HotPotQA F1, a 10.00% improvement in MBPP pass@1, a 10.00% gain in MATH solve accuracy, and an overall accuracy improvement of up to 20.00% on GAIA."**
   - (Abstract) -- 다양한 벤치마크에서의 구체적 성능 향상 수치를 제시하여 플랫폼의 효과를 실증한다.

5. **"This reliance limits scalability and usability, especially when adapting workflows to new tasks or domains."**
   - (Section 1, Introduction) -- 수작업 설정에 대한 의존이 확장성과 사용성을 제한한다는 핵심 동기를 설명한다.

6. **"The central feature of EvoAgentX is the evolving layer, which seamlessly integrates three state-of-the-art optimization algorithms."**
   - (Section 3, System Design) -- Evolving 계층이 EvoAgentX의 핵심 차별화 요소임을 강조한다.

7. **"OWL achieves an overall accuracy improvement of 20.00%, driven by gains of 28.57% at Level 1, 10.00% at Level 2, and a remarkable 100.00% at Level 3."**
   - (Section 4.2, Applications) -- GAIA 벤치마크에서의 인상적인 성능 향상을 보여준다.

8. **"In the future, we will extend EvoAgentX with more optimization algorithms, richer tool integration, and long-term memory to further enhance agent adaptability and contextual awareness."**
   - (Section 5, Conclusion) -- 장기 메모리 추가, 더 많은 최적화 알고리즘 확장 등 향후 로드맵을 제시한다.

---

## 7. 의의 및 시사점 (Significance and Implications)

### 학술적 의의

EvoAgentX는 멀티 에이전트 워크플로우의 **자동 생성과 진화적 최적화**를 단일 통합 플랫폼으로 제공하는 최초의 시도 중 하나이다. 기존에 분산되어 있던 TextGrad, AFlow, MIPRO 등의 최적화 알고리즘을 하나의 프레임워크에 통합함으로써, 연구자들이 다양한 최적화 전략을 일관되게 비교하고 적용할 수 있는 기반을 마련했다. 특히 에이전트 수준, 워크플로우 수준, 메모리 수준의 세 가지 최적화 축을 형식화한 점은 향후 MAS 최적화 연구의 체계적 프레임워크를 제공한다.

### 실무적 시사점

1. **자동 워크플로우 생성**: 높은 수준의 태스크 설명만으로 멀티 에이전트 워크플로우를 자동 생성할 수 있어, MAS 구축의 진입 장벽을 크게 낮춘다. SequentialWorkFlowGraph 등의 간소화된 API를 통해 빠른 프로토타이핑이 가능하다.

2. **기존 시스템 향상**: GAIA 벤치마크에서 Open Deep Research와 OWL의 성능을 자동 최적화만으로 최대 20% 향상시킨 결과는, 기존 MAS 시스템에 EvoAgentX를 적용하여 즉각적인 성능 향상을 얻을 수 있음을 보여준다.

3. **태스크별 최적화 전략**: TextGrad가 추론 태스크에, AFlow가 코드 생성에 강점을 보이는 결과는, 실무자가 태스크 특성에 따라 적절한 최적화 알고리즘을 선택해야 함을 시사한다.

### 한계 및 미래 전망

메모리 최적화기가 아직 개발 중이라는 점은 현재의 주요 한계이다. 메모리 최적화가 완성되면 에이전트의 장기적 적응성과 컨텍스트 인식이 더욱 향상될 것으로 기대된다. 또한 MASS, AlphaEvolve, Darwin Godel Machine 등 고급 진화 전략의 통합은 EvoAgentX의 최적화 역량을 한층 강화할 것이다. 오픈소스로 공개(https://github.com/EvoAgentX/EvoAgentX)되어 있어 커뮤니티 기반의 지속적 발전이 기대된다.
