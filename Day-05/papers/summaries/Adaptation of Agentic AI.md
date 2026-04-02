# Adaptation of Agentic AI

## 기본 정보
- **제목**: Adaptation of Agentic AI
- **저자**: Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, Jiawei Han
- **소속**: UIUC, Stanford, Princeton, Harvard, UC Berkeley, Caltech, UW, UCSD, Georgia Tech, Northwestern, TAMU, MGH, Keiji AI, Unity (29명 이상의 저자, 14개 이상 기관)
- **발표일**: 2025년 12월 22일 (arXiv:2512.16301v2)
- **학회/저널**: arXiv preprint (cs.AI)
- **페이지 수**: 67페이지 (309개 참고문헌)
- **GitHub**: https://github.com/pat-jj/Awesome-Adaptation-of-Agentic-AI

## 한줄 요약
에이전틱 AI 시스템의 적응(adaptation) 전략을 에이전트 적응(A1, A2)과 도구 적응(T1, T2)의 4가지 패러다임으로 체계적으로 분류하고, 각 패러다임의 기술적 방법론, 트레이드오프, 응용 분야를 포괄적으로 분석한 최초의 통합 프레임워크 서베이 논문이다.

## 초록 (Abstract) 요약
최첨단 에이전틱 AI 시스템은 파운데이션 모델 기반으로 계획, 추론, 외부 도구와의 상호작용을 통해 점점 더 복잡하고 전문화된 과제를 수행한다. 이 논문은 급속히 확장되는 연구 환경을 에이전트 적응(agent adaptation)과 도구 적응(tool adaptation)을 아우르는 체계적 프레임워크로 통합한다. 이를 다시 4가지 패러다임으로 세분화한다:
- **A1**: Tool Execution Signaled Agent Adaptation (도구 실행 결과 신호 기반 에이전트 적응)
- **A2**: Agent Output Signaled Agent Adaptation (에이전트 출력 신호 기반 에이전트 적응)
- **T1**: Agent-Agnostic Tool Adaptation (에이전트 비의존적 도구 적응)
- **T2**: Agent-Supervised Tool Adaptation (에이전트 감독 도구 적응)

이 프레임워크는 적응 전략의 설계 공간을 명확히 하고, 트레이드오프를 명시적으로 드러내며, 시스템 설계 시 전략 선택에 대한 실용적 가이드를 제공한다.

## 상세 내용

### 1. 서론 및 동기 (Section 1)
에이전틱 AI 시스템은 환경을 인식하고, 외부 도구를 호출하며, 메모리를 관리하고, 복잡한 과제를 완수하기 위해 다단계 계획을 실행하는 자율 AI 시스템이다. 그러나 현재 시스템은 여전히 도구 사용의 불안정성, 장기적 계획 수립의 한계, 도메인 특화 추론의 격차, 미탐험 환경으로의 일반화 부족 등의 문제에 직면해 있다. 이러한 한계는 파운데이션 모델이 특정 과제나 실세계 시나리오에 특화되기 위한 추가적 적응(adaptation)이 필요하다는 점을 보여준다.

이 논문은 에이전틱 AI 시스템의 적응에 관한 최초의 포괄적 서베이로서, 시스템 구성요소가 어떻게 수정되어 현재의 한계를 극복하는지 체계적으로 분석한다.

### 2. 배경 (Section 2)

#### 2.1 에이전틱 AI 시스템의 구성요소
에이전틱 AI 시스템의 핵심에는 파운데이션 모델(LLM 또는 멀티모달 모델)이 추론 및 제어 센터 역할을 한다. 추가 구성요소는 다음과 같다:
- **계획 모듈(Planning Module)**: 복잡한 목표를 실행 가능한 단계로 분해. 정적 계획(Chain-of-Thought, Tree-of-Thought)과 동적 계획(ReAct, Reflexion) 방식이 있다.
- **도구 사용(Tool Use)**: 웹 검색 엔진, API, 코드 실행 환경, MCP(Model Context Protocol), 브라우저 자동화 프레임워크 등 외부 자원과의 상호작용.
- **메모리 모듈(Memory Module)**: 단기 메모리(현재 작업 컨텍스트)와 장기 메모리(세션 간 지속되는 재사용 가능한 지식). RAG 메커니즘을 통해 저장된 지식을 검색하여 추론에 통합.

#### 2.2 적응 방법론
- **프롬프트 엔지니어링(Prompt Engineering)**: 모델 파라미터를 수정하지 않고 입력 프롬프트를 통해 행동을 유도하는 경량 적응 방법. CAMEL, AutoGen, MetaGen, ChatDev 등에서 활용.
- **파인튜닝(Fine-Tuning)**: 과제 특화 데이터에 대해 모델 내부 파라미터를 업데이트. SFT(Supervised Fine-Tuning), DPO(Direct Preference Optimization), PPO(Proximal Policy Optimization), GRPO(Group Relative Policy Optimization) 등의 훈련 패러다임 포함. LoRA 등 PEFT(Parameter-Efficient Fine-Tuning) 방법도 활용.

### 3. 적응 패러다임 개요 (Section 3)

#### 3.1 수학적 표기법
- **에이전트(A)**: 파라미터 theta로 매개변수화된 핵심 추론/의사결정 모델
- **도구(T)**: 리트리버, 플래너, 실행기, 시뮬레이터 등 외부 호출 가능 컴포넌트 집합 (메모리 모듈도 포함)
- **오프라인 데이터(D)**: 정렬 참조 또는 감독 소스
- **환경(E)**: 에이전트/도구가 상호작용하여 피드백을 받는 외부 환경
- **목적함수 O(.)**: 적응 과정에서 최적화되는 성능/정렬 품질 함수

#### 3.2 4가지 적응 패러다임

**A1: Tool Execution Signaled Agent Adaptation**
- 에이전트가 도구를 호출하고, 도구 실행 결과(코드 실행 성공, 검색 관련성 점수 등)를 기반으로 에이전트를 최적화
- 최적화 목표: A* = arg max_A O_tool(A, T)
- SFT 방식: 성공적인 도구 호출 궤적을 모방 학습
- RL 방식: 도구 실행 결과에서 보상 R = O_tool(y)를 받아 정책 최적화

**A2: Agent Output Signaled Agent Adaptation**
- 에이전트의 최종 출력 품질(정답 정확도 등)을 기반으로 에이전트를 최적화
- 최적화 목표: A* = arg max_A O_agent(A, T)
- 도구 없이(DeepSeek-R1, Kimi-1.5) 또는 도구와 함께(Search-R1, ReSearch) 적용 가능
- SFT에서는 A1 스타일의 도구 호출 감독과 A2 스타일의 최종 출력 감독을 결합해야 효과적

**T1: Agent-Agnostic Tool Adaptation**
- 에이전트를 고정(frozen)하고 도구만 독립적으로 최적화
- 최적화 목표: T* = arg max_T O_tool(T)
- 강력한 클로즈드소스 API(GPT, Claude, Gemini)를 파인튜닝할 수 없을 때 자연스러운 선택
- 표준 모델 훈련(지도학습, 대조학습, 강화학습)으로 환원

**T2: Agent-Supervised Tool Adaptation**
- 에이전트를 고정하되, 에이전트 출력을 감독 신호로 사용하여 도구를 최적화
- 최적화 목표: T* = arg max_T O_agent(A, T)
- 품질 가중 훈련, 출력 일관성 훈련, RL 기반 최적화 가능
- 메모리 모듈도 T2의 특수 사례: M <- Update(M, o)

### 4. 에이전트 적응 (Section 4)

#### 4.1 A1: 도구 실행 결과를 신호로 한 적응

**초기 SFT/Off-Policy 방법들의 진화:**
1. **Golden Answer와의 정렬**: Toolformer(NeurIPS 2023), TRICE(NAACL 2024), ToolAlpaca - 정답 또는 전문가 궤적과의 정확도 기반 학습
2. **Golden Format과의 정렬**: Gorilla(NeurIPS 2024), ToolFlow(NAACL 2025) - AST 기반 구조적 정확성, 그래프 기반 도구 관계 모델링
3. **직접적 도구 실행과의 정렬**: CodeAct(ICML 2024), NExT(ICML 2024), ToolLLM, AutoTools(WWW 2025) - 검증 가능한 실행 결과에서 직접 학습

**RLVR(Reinforcement Learning with Verifiable Reward) 기반 방법들:**
- **웹 검색/정보 검색**: DeepRetrieval(COLM 2025)이 쿼리 재작성을 MDP로 공식화, 검색 메트릭(Recall@K, NDCG)을 보상으로 사용. 문헌 검색에서 이전 SOTA 대비 약 3배의 리콜 향상(65.1% vs 24.7%). ReZero, Orion 등이 후속 확장.
- **코드 기반 도구**: LeDex(NeurIPS 2024), RLEF(ICML 2025), Code-R1, R1-Code-Interpreter 등
- **형식 정리 증명**: AlphaProof(Nature 2025), DeepSeek-Prover-V2(ICLR 2025), Kimina-Prover 등 - 증명 검증기의 단계별 의미론적 검증이 밀도 높은 보상 신호 제공
- **멀티 도구 추론**: Router-R1(NeurIPS 2025), FTRL, Tool-N1, WebGen-Agent 등

#### 4.2 A2: 에이전트 출력을 신호로 한 적응

**도구 없는 적응 (w/o Tools):**
- **DeepSeek-R1 프레임워크** (Nature 2025): RLVR로 LLM의 추론 능력을 효과적으로 강화. 수학/코드 생성 등 결정론적 정확성 신호가 있는 영역에 초점.
- **Self-Refine** (NeurIPS 2023): 같은 LM이 생성자와 비평자 역할을 동시에 수행하는 반복 정제 프레임워크
- **TextGrad** (Nature 2025): 자연어 비평을 "텍스트 그래디언트"로 사용하는 범용 자기 개선 프레임워크. GPT-4o의 LeetCode-Hard 코드 정확도를 26%에서 36%로 향상.
- **SCoRe** (ICLR 2025): 다중 턴 온라인 강화학습으로 자기 수정 능력 학습

**도구 포함 적응 (w/ Tools):**
- **검색 기반**: R1-Searcher, Search-R1(COLM 2025), ReSearch(NeurIPS 2025) 등이 R1 패러다임을 확장하여 LLM이 다중 턴 추론 중 자율적으로 검색 쿼리 생성/정제. ReSearch는 9-22% 절대 향상 달성.
- **코드/실행 기반**: CodePRM(ACL 2025), ReTool 등이 실시간 코드 실행을 RL 롤아웃에 통합
- **범용 멀티 도구**: Agent Lightning, Self-Challenging Agents, VerlTool 등

### 5. 도구 적응 (Section 5)

#### 5.1 T1: 에이전트 비의존적 도구 적응

**기초 시스템 및 아키텍처:**
- **Neural Operators** (JMLR): 에이전트 비의존적 도구 학습의 초기 사례로, 무한 차원 함수 공간 간의 매핑을 학습하는 미분 가능 대리 모델
- **HuggingGPT** (NeurIPS 2023): ChatGPT가 파인튜닝 없이 HuggingFace Hub의 1000+ ML 모델을 지휘하는 오케스트레이션 패러다임 개척
- **ViperGPT** (ICCV 2023): 코드 생성을 오케스트레이션 메커니즘으로 도입. Python 함수를 통한 유연한 도구 합성
- **SciToolAgent** (Nature Computational Science 2025): 지식 그래프 기반으로 500+ 과학 도구를 조직
- **MCP(Model Context Protocol)**: 에이전트가 외부 시스템과 인터페이스하는 방식을 통합하는 개방형 표준. 컨텍스트 사용량을 98% 이상 줄이면서 완전한 합성성 유지.

**도구 카테고리:**
- 비전 모델(CLIP, SAM), 음성/오디오(Whisper), 코드 실행(CodeAct), 검색/리트리벌(DPR, ColBERT, Contriever), 과학 도구(AlphaFold2, ESMFold) 등

#### 5.2 T2: 에이전트 감독 도구 적응

**초기 방법론:**
- **REPLUG** (NAACL 2024): 고정 LM의 퍼플렉시티 감소를 리트리버 훈련 신호로 사용
- **AAR** (ACL 2023): 증강 인식 검색으로 LM 유래 선호도 쌍을 대조 손실로 학습
- **BGM**: T5-XXL "브릿지 모델"이 고정 리트리버와 고정 생성기 사이에서 LLM 친화적 컨텍스트로 변환. HotpotQA에서 38% 상대적 향상(35.6% vs 25.8%).

**Subagent-as-Tool (2025년의 패러다임 전환):**

1. **에이전틱 검색기(Agentic Searcher)**:
   - **s3** (EMNLP 2025): 경량 7B "검색기"를 훈련. GBR(Gain Beyond RAG) 보상 사용. 2.4k 훈련 샘플로 58.9% 평균 정확도 달성 - Search-R1(A2, 170k 샘플) 대비 70배 적은 데이터, 33배 빠른 훈련. 의료 QA에서도 76.6% vs 71.8%로 더 높은 일반화.
   - **QAgent**: 2단계 훈련으로 보상 해킹 방지

2. **메모리 구축 서브에이전트**:
   - **Mem-alpha**: Qwen3-4B 컨트롤러가 고정 백엔드를 위한 3부분 외부 메모리를 RL로 관리
   - **AutoGraph-R1**: KG 구축을 T2 서브에이전트로 최적화

3. **메타인지 계획자/오케스트레이터**:
   - **AI-SearchPlanner**: 다목적 최적화(효과성 + 효율성)로 검색 전략 계획
   - **AgentFlow**: 고정 전문가 모듈의 플래너만 훈련. 7B AgentFlow가 GAIA에서 33.1% 달성(GPT-4 초과)
   - **Matryoshka Pilot** (NeurIPS 2025): 소형 화이트박스 LLM이 대형 블랙박스 LLM을 제어

4. **자기 진화 서브에이전트**:
   - **R-Zero**: Solver와 Challenger 역할을 번갈아 학습
   - **MAE**: Proposer, Solver, Judge의 3중 아키텍처

#### 5.2.3 에이전틱 메모리
- **Memento**: 고정 GPT-4.1 플래너 + 학습 가능한 에피소딕 사례 메모리. GAIA 검증에서 87.88%(1위), SimpleQA에서 95.0% 달성.
- **Dynamic Cheatsheet**: 블랙박스 LM을 위한 "지속적, 진화하는 메모리" 프레임워크
- **ToolkenGPT** (NeurIPS 2023): 도구를 고정 LLM 어휘의 학습 가능한 토큰 임베딩으로 표현. 234개 도구 추가에 약 1M 파라미터만 필요.

### 6. 적응 패러다임 비교 (Section 6)

비교 축은 4가지: (1) 비용/유연성, (2) 데이터 효율성, (3) 일반화 능력, (4) 모듈성/시스템 진화

| 패러다임 | 적응 대상 | 감독 신호 | 비용/유연성 | 모듈성 |
|---------|----------|----------|-----------|--------|
| A1 | 에이전트 정책 | 도구 실행 결과 | 높은 비용, 높은 파라미터 유연성 | 단일체적, 과적합 위험 |
| A2 | 에이전트 정책 | 에이전트 출력 | 높은 비용, 높은 파라미터 유연성 | 단일체적, 망각 위험 |
| T1 | 외부 도구 | 에이전트 독립 | 낮은 비용, 높은 시스템 유연성 | 높음 (Plug-and-Play) |
| T2 | 외부 도구 | 고정 에이전트 출력 | 낮은 비용, 높은 시스템 유연성 | 높음 (공생적, 망각 없음) |

**A1 vs A2:**
- A1: 도구 메카닉 최적화 (인과적, 즉각적, 세밀한 보상). DeepRetrieval에서 리콜 3배 향상.
- A2: 도구 전략 최적화 (전체적, 희소한, 고수준 보상). ReSearch에서 9-22% 절대 향상.

**T2의 데이터 효율성 우위:**
- s3(T2)가 Search-R1(A2) 대비 70배 적은 데이터로 비슷하거나 우수한 성능
- T2 서브에이전트는 절차적 기술만 학습하면 되므로 훨씬 효율적

**"졸업(Graduation)" 수명주기**: A1/A2로 훈련된 에이전트가 고정되어 T1 도구로 재배포되는 발전 경로. 예: DeepRetrieval -> 고정 -> T1 검색 도구.

### 7. 응용 분야 (Section 7)

#### 7.1 딥 리서치 (Deep Research)
OpenAI, Google, Perplexity 등의 AI 검색 시스템에서 에이전트가 자율적으로 연구를 수행. A2 스타일 RL로 다중 턴 검색 최적화, T2 스타일로 검색 계획자 훈련.

#### 7.2 소프트웨어 개발
SWE-Bench에서 실세계 소프트웨어 엔지니어링 과제 해결. SWE-Grep이 A1 RL로 훈련된 후 T1 코드 검색 도구로 졸업.

#### 7.3 컴퓨터 사용 (Computer Use)
GUI 에이전트가 실세계 컴퓨터 인터페이스를 탐색. WebGenAgent가 스크린샷 피드백으로 웹사이트 코드 생성.

#### 7.4 약물 발견 및 개발
DrugAgent, TxGemma 등이 에이전틱 AI를 약물 발견 파이프라인에 적용.

### 8. 미래 기회 (Section 8)

#### 8.1 공동 적응 (Co-Adaptation)
에이전트와 도구가 동시에 적응하는 통합 프레임워크. 안정적인 공동 학습 동역학 보장이 과제.

#### 8.2 지속적 적응 (Continual Adaptation)
도구와 환경이 지속적으로 변화하는 비정상적(non-stationary) 설정에서의 적응.

#### 8.3 안전한 적응 (Safe Adaptation)
적응 과정에서 안전 제약을 유지. 보상 해킹, 분포 이동, 도구 오용 방지 메커니즘 필요.

#### 8.4 효율적 적응 (Efficient Adaptation)
대규모 모델 재훈련의 계산 비용 절감. LoRA 등 PEFT, 커리큘럼 학습, 효율적 보상 설계 등이 유망.

## 핵심 키 포인트

1. **4가지 적응 패러다임의 통합 분류 체계**: "무엇이 적응되는가"(에이전트 vs 도구)와 "어떤 신호가 사용되는가"(도구 실행 vs 에이전트 출력)의 두 축으로 A1, A2, T1, T2 도출

2. **T2의 압도적 데이터 효율성**: s3(T2)가 Search-R1(A2) 대비 70배 적은 데이터(2.4k vs 170k)와 33배 빠른 훈련으로 비슷하거나 우수한 성능 달성

3. **RLVR의 부상**: 검증 가능한 보상을 활용한 강화학습이 A1 방법의 핵심 진화 방향

4. **Subagent-as-Tool 패러다임**: 반응적 도구에서 능동적 서브에이전트로의 전환. 검색, 메모리, 계획, 오케스트레이션 등 다양한 인지 기능을 별도 학습 가능 모듈로 분리

5. **졸업 수명주기(Graduation Lifecycle)**: A1/A2로 훈련된 에이전트가 T1 도구로 재배포되는 발전 경로

6. **공생적 적응(Symbiotic Adaptation)**: 고정 호스트 에이전트가 감독 신호를 제공하고, 적응형 서브에이전트가 정보를 변환/필터링하는 공생 관계

7. **모듈성의 실용적 우위**: T1/T2는 개별 도구를 독립적으로 교체/추가 가능. A1/A2는 재앙적 망각 위험

## 주요 인용 (Key Quotes)

1. **적응의 필요성에 대해**:
   > "Current agentic AI systems still struggle with challenges such as unreliable tool use, limited long-horizon planning, domain-specific reasoning gaps, robustness issues in real-world environments, and poor generalization to unexplored environments where the agent lacks prior interaction experience." (Section 1, p.2)

2. **프레임워크의 핵심 설계 원리**:
   > "This framework clarifies the underlying design space, highlights the trade-offs between different adaptation strategies, and provides practical guidance for choosing or transitioning between paradigms based on supervision signals, task requirements, and system-level constraints." (Section 1, p.2)

3. **T2 패러다임의 개념적 전환**:
   > "The T2 paradigm represents a profound conceptual inversion in how we approach adaptation in agentic systems. Rather than asking 'how can we modify the agent to better use its tools?' (the A1/A2 question), T2 asks: 'how can we modify the tools to better serve a fixed agent?'" (Section 5.2, p.28)

4. **공생적 적응의 정의**:
   > "The T2 paradigm exploits this asymmetry, achieving what we term symbiotic adaptation: the frozen agent provides high-quality supervision signals derived from its vast pre-trained knowledge, while the tools learn to translate, filter, and present information in exactly the form the agent finds most useful." (Section 5.2, p.28-29)

5. **s3의 데이터 효율성에 대해**:
   > "s3 achieves 58.9% average generation accuracy with only 2.4k training samples -- 70x less data than Search-R1 (an A2-style agent requiring 170k examples) and 33x faster wall-clock training time." (Section 5.2.2, p.33)

6. **A1과 A2의 신호 차이**:
   > "Tool-execution signals (A1) are grounded, causal, and process-oriented... Agent-output signals (A2) are holistic, flexible, and outcome-oriented. Rewards are assigned to the agent's final outputs... while this allows for end-to-end task optimization, relying solely on terminal signals can make the agent vulnerable to shortcut learning." (Section 6.2, p.38-39)

7. **MCP의 역할**:
   > "MCP represents a scalable T1-style tool adaptation infrastructure that decouples execution from inference, while the code-execution mode bridges toward T2-style optimization by dynamically improving efficiency under frozen agents." (Section 5.1.1, p.27)

8. **서브에이전트의 성숙**:
   > "The consistent lesson is that decoupling tool training from generator training, while enabling tools to adapt to one another, yields systems that are more data-efficient, modular, generalizable, and robust than monolithic alternatives." (Section 5.2.2, p.35)

## 시사점 및 의의

### 학술적 의의
1. **최초의 통합 프레임워크**: 에이전틱 AI 적응에 대한 최초의 체계적 분류 체계를 제공하여, 향후 연구의 위치 설정에 활용 가능한 개념적 기반 마련
2. **트레이드오프의 명시화**: 비용, 유연성, 데이터 효율성, 일반화, 모듈성 등의 축을 따라 4가지 패러다임의 장단점을 체계적으로 비교

### 산업적 시사점
1. **클로즈드소스 모델 활용 전략**: GPT, Claude 등을 직접 파인튜닝할 수 없는 기업에게 T1/T2 적응이 효과적인 대안임을 입증
2. **비용 효율적 시스템 설계**: T2 방식의 경량 서브에이전트(7B) 훈련이 대규모 에이전트 전체 훈련보다 70배 데이터 효율적이라는 실증적 증거 제공
3. **모듈형 시스템 아키텍처**: 도구별 독립적 업그레이드가 가능한 T1/T2 접근이 프로덕션 환경의 지속적 시스템 개선에 유리

### 향후 연구 방향
1. **에이전트-도구 공동 적응**: 현재 개별 최적화에서 동시 적응으로의 전환이 시스템 성능 극대화의 핵심
2. **지속적 적응**: API 변화, 라이브러리 업데이트 등 비정상적 환경에서의 지속적 학습
3. **안전한 적응**: 보상 해킹 방지, 도구 오용 방지 등의 안전성 메커니즘 개발
4. **효율적 적응**: 계산 비용과 적응 품질 간의 균형 최적화
