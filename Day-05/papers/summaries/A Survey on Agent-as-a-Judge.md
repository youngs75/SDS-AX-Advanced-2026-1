# A Survey on Agent-as-a-Judge

## 기본 정보
- **저자**: Runyang You*, Hongru Cai*, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li (corresponding), Wenjie Li
- **발행일/출처**: 2026년 1월 8일, arXiv:2601.05111v1 [cs.CL] / The Hong Kong Polytechnic University, University of Cambridge, Shandong Jianzhu University, Huawei Technologies
- **페이지 수**: 15페이지
- **키워드**: Agent-as-a-Judge, LLM-as-a-Judge, Agentic Evaluation, Multi-Agent Collaboration, Tool Integration, Planning, Memory, AI Evaluation

## 한줄 요약
> 이 논문은 LLM-as-a-Judge에서 Agent-as-a-Judge로의 패러다임 전환을 최초로 종합적으로 조사하며, 에이전틱 평가 시스템의 방법론, 응용 분야, 미래 방향을 체계적 분류학으로 정리한다.

## 초록 (Abstract)
LLM-as-a-Judge는 대규모 언어 모델을 활용한 확장 가능한 평가로 AI 평가를 혁신했지만, 평가 대상이 점점 복잡해지고 전문화됨에 따라 내재된 편향, 단일 패스의 얕은 추론, 실세계 관찰에 대한 검증 불가능 등의 한계에 직면했다. 이러한 한계가 Agent-as-a-Judge로의 전환을 촉발시켰으며, 에이전틱 평가자들은 계획(planning), 도구 기반 검증(tool-augmented verification), 다중 에이전트 협업(multi-agent collaboration), 영속적 메모리(persistent memory)를 활용하여 더 견고하고 검증 가능한 평가를 수행한다. 본 서베이는 이 진화를 추적하는 최초의 종합적 조사이다.

## 상세 내용

### 1. 서론 (Introduction)
LLM-as-a-Judge 패러다임은 인간 평가의 확장성 한계와 전통 지표의 의미론적 둔감성을 극복하기 위해 등장했다. 그러나 세 가지 핵심 한계가 있다:
1. **편향**: 단일 패스 평가자는 장황함이나 자체 출력 패턴을 선호하는 파라메트릭 편향에 취약
2. **수동적 관찰**: 정적 LLM 판단자는 실세계 피드백에 반응하지 못하고 언어적 패턴에 기반한 평가만 수행하여 "환각된 정확성(hallucinated correctness)"을 생성
3. **인지적 과부하**: 다면적 평가 루브릭을 단일 추론 단계에서 모두 평가하려 하면 세밀한 뉘앙스를 반영하지 못하는 조잡한 점수가 발생

Agent-as-a-Judge는 이를 극복하기 위해 복잡한 목표를 하위 과제로 분해하고, 다중 에이전트 협업으로 편향을 완화하며, 도구 기반 증거 수집으로 평가를 그라운딩하고, 중간 상태를 유지하여 세밀한 평가를 수행한다.

### 2. 진화: LLM-as-a-Judge에서 Agent-as-a-Judge로 (Evolution)

#### 2.1 LLM-as-a-Judge
Zheng et al.이 MT-Bench를 통해 접근법을 형식화했고, G-Eval은 NLG에서 CoT 프롬프팅을 활용했으며, Prometheus는 전문 튜닝으로 세밀한 평가를 유도했고, JudgeLM은 파인튜닝으로 더 견고한 평가자를 개발했다.

#### 2.2 패러다임 전환의 세 차원
- **견고성 진화: 모놀리식에서 분산으로**: 전문화된 분산 에이전트가 자율적 의사결정을 통해 협업하며, 전문 사전 지식의 주입이 가능
- **검증 진화: 직관에서 실행으로**: 정적 판단에서 외부 환경과의 상호작용으로 전환. 코드 인터프리터, 정리 증명기, 검색 도구를 활용한 객관적 검증
- **세분화 진화: 전역에서 세밀로**: 단일 패스 평가에서 자율적이고 계층적인 추론으로 전환. 과제별 루브릭 동적 생성 및 독립적 컴포넌트 평가

#### 2.3 Agent-as-a-Judge의 발전 단계
1. **절차적(Procedural)**: 사전 정의된 워크플로우나 고정 하위 에이전트 간 구조화된 토론. 복잡한 판단이 가능하지만 새로운 평가 시나리오에 적응 불가
2. **반응적(Reactive)**: 중간 피드백에 기반한 적응적 의사결정. 외부 도구나 하위 에이전트를 조건부로 호출하지만 고정된 의사결정 공간 내에서만 동작
3. **자기진화(Self-Evolving)**: 운영 중 내부 컴포넌트를 정제하는 높은 자율성. 즉석 평가 루브릭 합성, 학습된 교훈으로 메모리 업데이트

### 3. 방법론 (Methodologies)

#### 3.1 다중 에이전트 협업 (Multi-Agent Collaboration)
두 가지 토폴로지:
- **집단적 합의(Collective Consensus)**: 수평적 토론 메커니즘. ChatEval(법정 토론), M-MAD(기계번역), Multi-agent-as-judge(도메인 전문가 생성)
- **과제 분해(Task Decomposition)**: 분할 정복 전략. CAFES/GEMA-Score(순차적 단계), SAGEval(메타 평가자), HiMATE(트리 구조), AGENT-X(적응적 라우터)

#### 3.2 계획 (Planning)
- **워크플로우 오케스트레이션**: 정적 분해(MATEval)에서 동적 다라운드 계획(Evaluation Agent)까지
- **루브릭 발견**: 평가 기준의 자율적 발견과 정제. EvalAgents(웹 검색), AGENT-X(적응적 라우터), ARJudge(반복적 질문 생성), OnlineRubrics(RL과 통합)

#### 3.3 도구 통합 (Tool Integration)
- **증거 수집**: 코드 실행 피드백(Agent-as-a-Judge, CodeVisionary), 시각적 모델(Evaluation Agent), 문서 접근(ARM-Thinker)
- **정확성 검증**: 형식적 정리 증명(HERMES), 프로그래매틱/심볼릭 검증기(VerifiAgent), 검색 엔진+파이썬 인터프리터(Agentic RM)

#### 3.4 메모리와 개인화 (Memory and Personalization)
- **중간 상태**: 다단계 평가에서 중간 상태를 유지하여 조건부 라우팅과 적응적 의사결정 지원
- **개인화된 컨텍스트**: 사용자 선호, 평가 기준, 과거 피드백을 유지하여 일관된 평가. RLPA와 SynthesizeMe는 사용자 페르소나를 구축하고 유지

#### 3.5 최적화 패러다임 (Optimization Paradigms)
- **훈련 시간 최적화**: SFT(SynthesizeMe), RL(TIR-Judge, ARM-Thinker)
- **추론 시간 최적화**: 사전 정의된 파이프라인(Evaluation Agent, HERMES) 또는 적응적 행동(Multi-Agent LLM Judge, SAGEval)

### 4. 응용 분야 (Application)

#### 4.1 일반 도메인
- **수학 및 코드**: HERMES(형식적 증명), VerifiAgent(도구 기반 검증), Popper(통제된 반증)
- **사실 확인**: FACT-AUDIT(다중 에이전트 루프), NarrativeFactScore(캐릭터 수준 지식 표현)
- **멀티모달 및 비전**: CIGEval(특화 도구), ARM-Thinker(이미지 검사)
- **대화 및 상호작용**: IntellAgent(대화 벤치마크), ESC-Judge(감정 지원), PSYCHE(정신과 환자 프로필)
- **오픈엔디드 응답**: 다양한 관점의 역할극 기반 평가

#### 4.2 전문 도메인
- **의료**: MAJ-Eval(다중 평가자 페르소나 토론), GEMA-Score(에이전트 협업 기반 점수화), AI Hospital(다중 에이전트 시뮬레이터)
- **법률**: AgentsCourt(적대적 토론), SAMVAD(사법 합의 시뮬레이션)
- **금융**: FinResearchBench(논리 트리 기반), SAEA(에이전트 궤적 감사), M-SAEA(다중 에이전트 실패 추적)
- **교육**: Grade-Like-Human(단계적 채점), AutoSCORE(구조적 컴포넌트 인식), GradeOpt(반복적 채점 지침 정제)

### 5. 논의 (Discussion)

#### 5.1 과제
- **계산 비용**: 훈련(RL)과 추론(다중 추론 단계, 도구 호출, 다중 에이전트 조정) 모두에서 비용 증가
- **지연 시간**: 순차적 추론, 외부 도구 호출, 다중 에이전트 통신으로 인한 지연 증가. 실시간 설정에서 문제
- **안전성**: 도구 접근이 프롬프트 주입, 도구 남용의 공격 면을 확대. 다중 에이전트 간 안전하지 않은 행동 전파 위험
- **프라이버시**: 영속적 메모리나 개인화된 평가에서 민감한 데이터 유출 위험. 의료, 법률, 교육 등 전문 도메인에서 특히 우려

#### 5.2 미래 방향
- **개인화**: 정적 평가 기준에서 능동적 메모리 관리로 진화. 선호도 등록, 업데이트, 오래된 피드백 제거를 자율적으로 결정
- **일반화**: 사전 정의된 루브릭에서 동적 발견 및 적응으로. 컨텍스트 인식 루브릭 생성, 적응적 다세분화 점수화
- **상호작용성**: 수동 관찰에서 능동적 상호작용으로. 과제 복잡성 자율 확대, 인간-에이전트 협업 교정
- **최적화**: 추론 시간 엔지니어링에서 훈련 기반 최적화로. 개별 역량(RL) 및 학습된 조정(다중 에이전트 합동 목표)
- **진정한 자율성을 향하여**: 고정 프로토콜을 초월한 자기 주도적 적응, 능동적 컨텍스트 큐레이션, 지속적 자기 정제

### 6. 결론 (Conclusion)
Agent-as-a-Judge의 최초 종합 서베이를 제공한다. 다중 에이전트 협업, 자율 계획, 도구 통합, 메모리 등의 에이전틱 역량이 나이브한 LLM 판단자의 한계를 극복하여 더 견고하고 검증 가능한 판단을 일반 및 전문 도메인에서 제공함을 보였다. 미래에는 개인화, 일반화, 최적화를 우선하여 진화하는 AI 환경에 지속적으로 적응하는 진정한 자율 평가자를 실현해야 한다.

## 핵심 키 포인트
1. **패러다임 전환**: LLM-as-a-Judge의 세 가지 한계(편향, 수동적 관찰, 인지적 과부하)가 Agent-as-a-Judge로의 전환을 촉발했다.
2. **3단계 발전**: 절차적(Procedural) -> 반응적(Reactive) -> 자기진화(Self-Evolving)의 발전 단계를 식별했다.
3. **5가지 핵심 방법론**: 다중 에이전트 협업, 계획, 도구 통합, 메모리/개인화, 최적화 패러다임으로 체계화했다.
4. **도구 기반 검증**: 직관에서 실행으로의 전환이 가장 중요한 차별점이다. 코드 실행, 정리 증명, 검색 도구를 통한 객관적 검증이 핵심이다.
5. **폭넓은 응용**: 수학/코드부터 의료, 법률, 금융, 교육까지 다양한 도메인에서 적용된다.
6. **실용적 과제**: 계산 비용, 지연 시간, 안전성, 프라이버시 등 실제 배포에서의 과제가 상당하다.
7. **미래 방향**: 진정한 자율성을 향해 자기 주도적 적응, 능동적 메모리 관리, 훈련 기반 최적화가 필요하다.

## 주요 인용 (Key Quotes)

> "LLM-as-a-Judge has revolutionized AI evaluation by leveraging large language models for scalable assessments. However, as evaluands become increasingly complex, specialized, and multi-step, the reliability of LLM-as-a-Judge has become constrained by inherent biases, shallow single-pass reasoning, and the inability to verify assessments against real-world observations." (Abstract, p.1)

> "This has catalyzed the transition to Agent-as-a-Judge, where agentic judges employ planning, tool-augmented verification, multi-agent collaboration, and persistent memory to enable more robust, verifiable, and nuanced evaluations." (Abstract, p.1)

> "Static LLM judges are fundamentally passive observers, unable to react to real-world feedback. They assess answers based on linguistic plausibility -- how correct a response looks -- without verification or evidence collection, leading to 'hallucinated correctness' in complex tasks." (Section 2.2, p.3)

> "Agent-as-a-Judge bridges this reality gap by replacing intuition with execution." (Section 2.2, p.3)

> "The next generation of judge agents must transcend fixed protocols to become genuinely agentic entities capable of self-directed adaptation, active context curation, and continuous self-refinement." (Section 5.2, p.10)

> "Rather than ad-hoc inference collaboration, agents should be trained with joint objectives to intrinsically learn effective communication and consensus strategies." (Section 5.2, p.10)

## 시사점 및 의의
이 서베이는 AI 평가의 미래를 조망하는 데 매우 중요한 기여를 한다. Agent-as-a-Judge는 단순히 평가 방법의 개선이 아니라, AI 시스템의 품질 보증과 지속적 개선을 위한 인프라적 전환을 의미한다. AgentOps 관점에서 이 연구는 다음과 같은 핵심 시사점을 제공한다: (1) 에이전트의 출력 평가가 단순 지표가 아닌 에이전틱 시스템으로 수행되어야 하며, (2) 도구 기반 검증이 "환각된 정확성"을 방지하는 핵심이고, (3) 다중 에이전트 토론이 단일 평가자의 편향을 효과적으로 완화하며, (4) 메모리와 개인화가 시간에 걸친 일관된 평가를 가능하게 한다. 다만, 계산 비용과 지연 시간이 실제 배포에서의 주요 병목이 될 수 있으므로, 평가의 깊이와 실용성 사이의 균형을 고려해야 한다.
