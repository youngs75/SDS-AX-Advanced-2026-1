# Memory in the Age of AI Agents: A Survey - Forms, Functions, and Dynamics

## 1. 기본 정보 (Basic Information)

| 항목 | 내용 |
|------|------|
| **제목** | Memory in the Age of AI Agents: A Survey - Forms, Functions, and Dynamics |
| **저자** | Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Junfeng Fang, Yuting Zheng, Yuxuan Liang, Yue Yu, Yong Li, Fei Wang, Mengyue Yang, Jun Wang |
| **소속** | National University of Singapore (NUS), Renmin University of China (RUC), Fudan University, Peking University, Nanyang Technological University (NTU), Hong Kong University of Science and Technology, University College London 등 |
| **출판 정보** | arXiv:2512.13564v2, 2026년 1월 13일 |
| **페이지 수** | 107페이지 (본문 약 76페이지 + 참고문헌 약 31페이지) |
| **키워드** | Agent Memory, Memory Taxonomy, Forms-Functions-Dynamics, LLM Agents, Token-level Memory, Parametric Memory, Latent Memory |

---

## 2. 한 줄 요약 (One-line Summary)

AI 에이전트 메모리 시스템을 "Forms(형태)-Functions(기능)-Dynamics(동역학)"의 3차원 분류 체계로 체계화한 최초의 포괄적 서베이 논문으로, 기존의 단기/장기 메모리 이분법을 넘어 500편 이상의 문헌을 분석하여 에이전트 메모리의 설계, 목적, 운영 메커니즘에 대한 통합적 프레임워크를 제시한다.

---

## 3. 초록 요약 (Abstract Summary)

이 논문은 파운데이션 모델 기반 에이전트의 핵심 역량으로 부상한 **메모리 시스템**에 대한 체계적 서베이를 제공한다. 저자들은 기존의 장기/단기 메모리 분류법이 현대 에이전트 메모리 시스템의 다양성과 역동성을 포착하기에 불충분하다고 주장하며, 세 가지 상호보완적 차원으로 구성된 새로운 분류 체계를 제안한다:

- **Forms (형태)**: 메모리가 어떤 구조로 저장되는가 -- Token-level, Parametric, Latent
- **Functions (기능)**: 에이전트가 왜 메모리를 필요로 하는가 -- Factual, Experiential, Working Memory
- **Dynamics (동역학)**: 메모리가 어떻게 운영되고 진화하는가 -- Formation, Evolution, Retrieval

이 프레임워크를 통해 500편 이상의 관련 연구를 분석하고, 메모리 관련 리소스와 벤치마크를 정리하며, 7가지 미래 연구 방향을 제시한다.

---

## 4. 상세 내용 정리 (Detailed Content Summary)

### Section 1: Introduction (서론)

AI 에이전트의 발전과 함께 메모리 시스템의 중요성이 급격히 증가하고 있다. LLM이 범용 텍스트 처리기에서 자율적, 목표 지향적 에이전트로 전환되는 것은 근본적인 패러다임 전환이며, 이 전환의 핵심에 메모리가 있다. 기존의 단기/장기 메모리 이분법은 현대 에이전트 메모리의 복잡성을 충분히 반영하지 못하므로, 저자들은 세 가지 차원(Forms, Functions, Dynamics)의 새로운 분류 체계를 제안한다.

논문의 핵심 기여는 다음과 같다:
1. 기존 분류법의 한계를 극복하는 3차원 분류 체계(Forms-Functions-Dynamics) 제안
2. 500편 이상의 최신 연구를 체계적으로 분석 및 분류
3. 메모리 관련 리소스, 프레임워크, 벤치마크 종합 정리
4. 7가지 미래 연구 방향(Positions and Frontiers) 제시

### Section 2: Preliminaries (사전 지식)

에이전트와 메모리 시스템의 형식적 정의를 제공한다. 에이전트는 환경과 상호작용하면서 목표를 달성하는 시스템이며, 메모리는 에이전트의 핵심 구성요소이다.

기존 단기/장기 메모리 구분의 한계를 지적한다: **"Short-term and long-term memory phenomena therefore emerge not from discrete architectural modules but from the temporal patterns with which formation, evolution, and retrieval are engaged."** 즉, 단기/장기 메모리 현상은 별도의 아키텍처 모듈이 아니라 메모리의 형성, 진화, 검색이 수행되는 시간적 패턴에서 나타난다.

### Section 3: Forms - 메모리의 형태 (What Does Memory Look Like?)

메모리가 어떤 형태로 저장되는지를 다루며, 크게 세 가지로 분류한다.

#### 3.1 Token-level Memory (토큰 수준 메모리)

인간이 읽을 수 있는 텍스트 형태로 저장되는 메모리로, 가장 직관적이고 해석 가능한 형태이다. 차원에 따라 세 가지로 세분화된다:

- **1D Flat Memory**: 평면적 리스트 구조. 자연어 텍스트 스니펫이나 키-값 쌍으로 저장. MemGPT, MemoryBank, mem0 등이 대표적. 구현이 간단하지만 정보량이 증가하면 검색 효율이 저하된다.
- **2D Planar Memory**: 노드 간 링크를 설정하여 관계를 표현하는 구조. 지식 그래프, 테이블 등이 포함. HippoRAG, GraphRAG 등이 대표적. 관계형 쿼리에 강하나 계층적 저장 메커니즘이 없다는 한계가 있다.
- **3D Hierarchical Memory**: 계층적으로 정보를 구성하는 구조. 피라미드형(HiAgent, GraphRAG)과 다층형(H-Mem, HippoRAG 2)이 있다. 다차원적 검색이 가능하나 구조의 복잡성으로 인한 검색 효율 문제가 있다.

#### 3.2 Parametric Memory (파라메트릭 메모리)

모델의 파라미터에 직접 정보를 저장하는 방식이다.

- **Internal Parametric Memory**: 모델의 원래 파라미터(가중치, 편향) 내에 인코딩. Character-LM, SELF-PARAM, ROME 등. 추가 추론 오버헤드가 없으나 업데이트 시 재훈련이 필요하고 망각 문제가 있다.
- **External Parametric Memory**: 어댑터, LoRA 모듈 등 보조 파라미터에 저장. MLP-Memory, K-Adapter, WISE, MemLoRA 등. 모듈식 업데이트가 가능하나 내부 표현과의 통합 효과에 한계가 있다.

#### 3.3 Latent Memory (잠재 메모리)

KV 캐시, 활성화, 은닉 상태 등 모델 내부 표현에 암묵적으로 저장되는 메모리이다.

- **Generate**: 보조 모델이나 모듈이 잠재 표현을 생성. Gist tokens, Titans, MemGen 등.
- **Reuse**: 이전 계산의 내부 활성화(주로 KV 캐시)를 직접 재사용. Memorizing Transformers, LONGMEM 등.
- **Transform**: 기존 잠재 상태를 선택, 집계, 압축하여 변환. Scissorhands, SnapKV, PyramidKV, H2O 등.

#### 3.4 Adaptation (적응 - 적합한 메모리 유형 선택)

각 메모리 유형의 특성과 적합한 시나리오를 정리한다:
- **Token-level**: 다중 턴 대화, 추천 시스템, 고위험 도메인(법률, 금융, 의료) -- 해석 가능성과 감사 가능성이 중요한 경우
- **Parametric**: 역할 연기, 수학적 추론, 정형화된 행동 -- 일반화와 추상적 패턴이 중요한 경우
- **Latent**: 멀티모달 아키텍처, 엣지 배포, 프라이버시 민감 도메인 -- 효율성과 밀도가 중요한 경우

### Section 4: Functions - 에이전트가 메모리를 필요로 하는 이유 (Why Agents Need Memory?)

**"The transition from large language models as general-purpose, stateless text processors to autonomous, goal-directed agents is not merely an incremental step but a fundamental paradigm shift."**

세 가지 기능적 분류를 제시한다:

#### 4.1 Factual Memory (사실적 메모리)

에이전트의 선언적 지식 기반으로, 일관성(Consistency), 응집성(Coherence), 적응성(Adaptability)을 보장한다.

- **User Factual Memory (사용자 사실 메모리)**: 사용자의 정체성, 선호도, 과거 약속 등을 저장. 대화 일관성(Dialogue Coherence)과 목표 일관성(Goal Consistency) 유지에 핵심적 역할.
  - 대화 일관성: MemGPT, TiM, MemoryBank, mem0 등이 선택적 보유와 의미적 추상화 전략을 활용
  - 목표 일관성: RecurrentGPT, A-Mem, MEMENTO 등이 태스크 상태를 동적으로 추적

- **Environment Factual Memory (환경 사실 메모리)**: 외부 세계의 문서, 코드베이스, 도구 등에 대한 사실 저장.
  - 지식 지속성(Knowledge Persistence): HippoRAG, MemTree, LMLM 등이 외부 지식을 구조화
  - 공유 접근(Shared Access): MetaGPT, GameGPT, Generative Agents 등이 다중 에이전트 간 메모리 공유

#### 4.2 Experiential Memory (경험적 메모리)

**"Experiential memory serves as a foundation for continual learning and self-evolution in the era of experience."**

에이전트의 절차적, 전략적 지식으로, 지속적 학습과 자기 진화를 가능하게 한다.

- **Case-based Memory (사례 기반 메모리)**: 최소한으로 처리된 과거 에피소드를 저장. 궤적(Trajectories)과 해결책(Solutions)으로 구분. ExpeL, Synapse, JARVIS-1 등이 대표적.
- **Strategy-based Memory (전략 기반 메모리)**: 전이 가능한 추론 패턴, 워크플로우, 통찰을 추출. Insights(H2R, R2D2), Workflows(AWM, Agent KB), Patterns(Buffer of Thoughts, ReasoningBank)로 세분화.
- **Skill-based Memory (스킬 기반 메모리)**: 실행 가능한 절차적 역량을 캡슐화. Code Snippets(Voyager), Functions/Scripts(CREATOR, SkillWeaver), APIs(Gorilla, ToolLLM), MCPs(Alita, Alita-G)로 분류.
- **Hybrid Memory (하이브리드 메모리)**: 여러 형태의 경험적 메모리를 통합. ExpeL, Agent KB, G-Memory, Memp 등.

#### 4.3 Working Memory (작업 메모리)

용량 제한적이며 동적으로 제어되는 스크래치패드로, 현재 태스크 수행을 위한 활성 컨텍스트 관리를 담당한다.

- **Single-turn Working Memory (단일 턴 작업 메모리)**: 대규모 즉각적 입력의 처리에 초점.
  - 입력 응축(Input Condensation): LLMLingua, Gist, AutoCompressors 등이 하드/소프트/하이브리드 압축 적용
  - 관찰 추상화(Observation Abstraction): Synapse, VideoAgent 등이 고차원 관찰을 구조화된 표현으로 변환

- **Multi-turn Working Memory (다중 턴 작업 메모리)**: 시간적 상태 유지에 초점.
  - 상태 통합(State Consolidation): MEM1, MemAgent, ReSum 등이 RL을 활용한 상태 압축
  - 계층적 접기(Hierarchical Folding): HiAgent, Context-Folding, AgentFold 등이 하위 목표 기반 궤적 분해
  - 인지 계획(Cognitive Planning): PRIME, Agent-S, KARMA 등이 외부화된 계획을 작업 메모리의 핵심으로 사용

### Section 5: Dynamics - 메모리의 운영과 진화 (How Memory Operates and Evolves?)

메모리 시스템이 정적 저장소에서 동적 관리 시스템으로 패러다임 전환하는 과정을 기술한다.

#### 5.1 Memory Formation (메모리 형성)

원시 컨텍스트를 정보 밀도 높은 지식으로 인코딩하는 과정. 다섯 가지 유형으로 분류:

1. **Semantic Summarization (의미적 요약)**: 증분적(Incremental -- MemGPT, Mem0, Mem1) 및 분할적(Partitioned -- MemoryBank, ReadAgent) 방식. 글로벌 의미 정보 보존에 초점.
2. **Knowledge Distillation (지식 증류)**: 사실적 메모리 증류(TiM, RMM)와 경험적 메모리 증류(ExpeL, H2R, Mem-alpha). 재사용 가능한 지식 추출.
3. **Structured Construction (구조화된 구성)**: Entity-Level(KGT, GraphRAG, Zep)과 Chunk-Level(RAPTOR, MemTree, A-MEM). 다중 홉 추론 지원.
4. **Latent Representation (잠재 표현)**: 텍스트 잠재 표현(MemoryLLM, M+, MemGen)과 멀티모달 잠재 표현(CoMEM, KARMA). 기계 네이티브 포맷 인코딩.
5. **Parametric Internalization (파라메트릭 내재화)**: 지식 내재화(MEND, ROME, MEMIT)와 역량 내재화(SFT, DPO, GRPO). 외부 메모리를 모델 가중치에 통합.

#### 5.2 Memory Evolution (메모리 진화)

새롭게 추출된 메모리를 기존 메모리와 통합하는 동적 진화 과정:

1. **Consolidation (통합)**: 단편적 단기 흔적을 일관된 장기 스키마로 재구성.
   - Local Consolidation: RMM의 top-K 유사 후보 매칭
   - Cluster-level Fusion: PREMem의 클러스터 간 융합
   - Global Integration: Matrix, AgentFold의 전역적 통합
2. **Updating (업데이트)**: 새 정보와 기존 메모리 간 충돌 해결.
   - External Memory Update: MemGPT의 규칙 기반에서 Zep의 시간 주석, Mem-alpha의 정책 학습으로 진화
   - Model Editing: ROME의 인과 추적, MEMIT의 대량 편집
3. **Forgetting (망각)**: 오래되거나 중복된 정보의 의도적 제거.
   - Time-based: MemGPT의 가장 오래된 메시지 제거, MAICC의 소프트 망각
   - Frequency-based: XMem의 LFU 정책, MemOS의 LRU 전략
   - Importance-driven: TiM, MemTool의 LLM 기반 중요도 평가

#### 5.3 Memory Retrieval (메모리 검색)

구축된 메모리 뱅크에서 관련 지식을 효율적으로 검색하는 과정:

1. **Retrieval Timing and Intent (검색 시기와 의도)**: 자동화된 검색 시기(MemGPT의 LLM 기반 판단, MemGen의 잠재적 트리거) 및 자동화된 검색 의도(AgentRR의 동적 전환)
2. **Query Construction (쿼리 구성)**: 분해(Decomposition -- PRIME, Agent KB)와 재작성(Rewriting -- HyDE, MemoRAG)
3. **Retrieval Strategies (검색 전략)**: 어휘 검색, 의미 검색, 그래프 검색, 하이브리드 검색
4. **Post-Retrieval Processing (후처리)**: 재순위화, 필터링, 집계

### Section 6: Resources and Benchmarks (리소스 및 벤치마크)

에이전트 메모리 연구를 위한 프레임워크(MemGPT/Letta, mem0, LangChain, LangGraph 등)와 벤치마크(LoCoMo, LongMemEval, LOCOBENCH 등)를 종합적으로 정리한다.

### Section 7: Positions and Frontiers (입장과 미래 방향)

7가지 핵심 연구 방향을 제시한다:

1. **Automated Memory Architecture Design**: 메모리 아키텍처의 자동 설계
2. **RL-Enhanced Memory Systems**: 강화학습을 통한 메모리 시스템 최적화
3. **Multimodal Memory**: 멀티모달 입력을 통합하는 메모리 시스템
4. **Multi-Agent Shared Memory**: 다중 에이전트 간 공유 메모리
5. **Trustworthiness in Memory**: 메모리 시스템의 신뢰성, 프라이버시, 안전성
6. **World Models as Memory**: 세계 모델을 메모리로 활용
7. **Connections to Human Cognition**: 인간 인지 과학과의 연결

---

## 5. 핵심 포인트 (Key Points)

1. **새로운 3차원 분류 체계**: 기존의 단기/장기 메모리 이분법을 넘어 Forms(형태), Functions(기능), Dynamics(동역학)의 세 축으로 에이전트 메모리를 체계적으로 분류하는 최초의 프레임워크를 제안한다.

2. **Token-level 메모리의 3단계 차원 분류**: 1D(Flat) -> 2D(Planar) -> 3D(Hierarchical)로 토큰 수준 메모리를 세분화하여, 단순 리스트에서 지식 그래프, 나아가 계층적 구조까지의 스펙트럼을 명확히 구분한다.

3. **경험적 메모리의 추상화 스펙트럼**: Case-based(원시 궤적) -> Strategy-based(전략/워크플로우) -> Skill-based(실행 가능한 코드/API/MCP)로 경험적 지식의 추상화 수준이 증가하며, 이들이 상호보완적으로 작동하는 하이브리드 설계의 중요성을 강조한다.

4. **메모리 형성의 5가지 연산**: Semantic Summarization, Knowledge Distillation, Structured Construction, Latent Representation, Parametric Internalization -- 이 다섯 가지 메모리 형성 방법은 상호 배타적이지 않으며 하나의 시스템 내에서 통합될 수 있다.

5. **동적 메모리 생명주기**: Formation(형성) -> Evolution(진화) -> Retrieval(검색)의 상호 연결된 순환 과정으로 메모리가 운영되며, 이 세 프로세스가 에이전트의 지속적 학습과 자기 진화를 가능하게 한다.

6. **메모리 진화의 세 메커니즘**: Consolidation(통합)은 단편적 메모리를 일반화된 통찰로 합성하고, Updating(업데이트)은 충돌을 해결하며, Forgetting(망각)은 불필요한 정보를 제거한다. 이 세 메커니즘이 함께 메모리의 일반화, 정확성, 시의성을 유지한다.

7. **Stability-Plasticity 딜레마**: 메모리 업데이트에서 기존 지식을 덮어쓸 시점과 새 정보를 잡음으로 취급할 시점을 결정하는 안정성-가소성 딜레마가 핵심 과제로 남아있다.

8. **7가지 미래 연구 프론티어**: 자동화된 메모리 아키텍처 설계, RL 강화 메모리 시스템, 멀티모달 메모리, 다중 에이전트 공유 메모리, 신뢰성, 세계 모델, 인간 인지 연결 등 광범위한 미래 방향을 제시한다.

---

## 6. 핵심 인용구 (Key Quotes)

1. **"Memory has emerged, and will continue to remain, a core capability of foundation model-based agents."**
   - (Abstract, p.1) -- 메모리가 파운데이션 모델 기반 에이전트의 핵심 역량으로 부상했으며 계속 그러할 것이라는 논문의 근본적 전제를 밝힌다.

2. **"Traditional taxonomies such as long/short-term memory have proven insufficient to capture the diversity and dynamics of contemporary agent memory systems."**
   - (Abstract, p.1) -- 기존 분류법의 한계를 명확히 지적하고 새로운 분류 체계의 필요성을 역설한다.

3. **"Short-term and long-term memory phenomena therefore emerge not from discrete architectural modules but from the temporal patterns with which formation, evolution, and retrieval are engaged."**
   - (Section 2.2, p.8) -- 단기/장기 메모리가 별도의 아키텍처 모듈이 아닌 시간적 패턴에서 발현된다는 핵심 통찰을 제공한다.

4. **"This transition from large language models as general-purpose, stateless text processors to autonomous, goal-directed agents is not merely an incremental step but a fundamental paradigm shift."**
   - (Section 4, p.31) -- LLM에서 에이전트로의 전환이 근본적 패러다임 전환임을 강조하며, 메모리의 필수성을 논증한다.

5. **"Experiential memory serves as a foundation for continual learning and self-evolution in the era of experience."**
   - (Section 4.2, p.37) -- 경험적 메모리가 에이전트의 지속적 학습과 자기 진화의 기반이라는 점을 명확히 한다.

6. **"Memory formation is not independent of the preceding sections. Depending on the task type, the memory formation process selectively extracts different architectural memories described in Section 3 to fulfill the corresponding functions outlined in Section 4."**
   - (Section 5.1, p.47) -- Forms, Functions, Dynamics가 독립적이 아닌 상호 연결된 체계임을 보여준다.

7. **"From an implementation standpoint, memory updating focuses on resolving conflicts and revising knowledge triggered by the arrival of new memories, whereas memory consolidation emphasizes the integration and abstraction of new and existing knowledge."**
   - (Section 5.2.2, p.58) -- 메모리 업데이트와 통합의 구분을 명확히 한다.

8. **"Time-based decay reflects the natural temporal fading of memory, frequency-based forgetting ensures efficient access to frequently used memories, and importance-driven forgetting introduces semantic discernment."**
   - (Section 5.2.3, p.59) -- 세 가지 망각 메커니즘이 각각 어떻게 메모리의 시의성, 효율성, 의미적 관련성을 유지하는지 설명한다.

---

## 7. 의의 및 시사점 (Significance and Implications)

### 학술적 의의

이 서베이는 AI 에이전트 메모리 연구에 대한 **최초의 포괄적이고 체계적인 분류 프레임워크**를 제안한다. 500편 이상의 문헌을 Forms-Functions-Dynamics의 3차원으로 분류함으로써, 연구자들이 메모리 시스템의 설계 공간을 명확히 이해하고 탐색할 수 있는 개념적 지도를 제공한다. 특히 기존의 단기/장기 메모리 이분법이 건축적 모듈이 아닌 시간적 패턴에서 발현된다는 통찰은 메모리 시스템 설계에 대한 근본적 사고 전환을 촉진한다.

### 실무적 시사점

1. **메모리 유형 선택 가이드라인**: 각 메모리 형태(Token-level, Parametric, Latent)의 특성과 적합한 시나리오를 명확히 정리하여, 실무자가 태스크 요구사항에 맞는 메모리 시스템을 선택할 수 있는 실용적 지침을 제공한다.

2. **동적 메모리 관리의 필요성**: 정적 저장소를 넘어 Formation-Evolution-Retrieval의 순환적 생명주기를 통한 동적 메모리 관리가 에이전트의 지속적 학습과 자기 진화에 필수적임을 보여준다.

3. **하이브리드 접근의 중요성**: 단일 메모리 유형보다 여러 유형을 통합하는 하이브리드 설계가 더 효과적이라는 점을 여러 연구 사례를 통해 입증한다.

### 미래 전망

이 논문이 제시하는 7가지 프론티어(자동 메모리 아키텍처 설계, RL 강화 메모리, 멀티모달 메모리, 다중 에이전트 공유 메모리, 신뢰성, 세계 모델, 인간 인지 연결)는 향후 에이전트 메모리 연구의 로드맵을 제공한다. 특히 강화학습을 통한 메모리 형성 및 검색 최적화(Mem1, MemAgent의 GRPO/PPO 활용), 그리고 MCP(Model Context Protocol) 기반 스킬 메모리의 표준화는 에이전트 생태계의 확장에 직접적 영향을 미칠 것으로 예상된다.
