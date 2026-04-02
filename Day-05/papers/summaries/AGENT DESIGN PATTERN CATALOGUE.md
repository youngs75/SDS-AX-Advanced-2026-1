# Agent Design Pattern Catalogue: A Collection of Architectural Patterns for Foundation Model Based Agents

## 기본 정보
- **저자**: Yue Liu, Sin Kit Lo, Qinghua Lu, Liming Zhu, Dehai Zhao, Xiwei Xu, Stefan Harrer, Jon Whittle (Data61, CSIRO, Australia)
- **발행일/출처**: 2024년 11월 7일, arXiv:2405.10467v4 [cs.AI]
- **페이지 수**: 38페이지
- **키워드**: Foundation Model, AI Agent, Design Pattern, Software Architecture, Multi-Agent System, Goal-Seeking, Plan Generation, LLM Agent

## 한줄 요약
> 이 논문은 체계적 문헌 검토를 기반으로 파운데이션 모델 기반 에이전트 설계를 위한 18개의 아키텍처 패턴을 수집하고, 각 패턴의 컨텍스트, 문제, 힘(forces), 해결책, 결과(장단점), 실제 사용 사례를 분석하며, 패턴 선택을 위한 의사결정 모델을 제안하는 포괄적인 패턴 카탈로그이다.

## 초록 (Abstract)
파운데이션 모델(FM) 기반 생성적 AI는 뛰어난 추론 및 언어 처리 능력을 활용하여 사용자의 목표를 능동적이고 자율적으로 추구하는 에이전트 개발을 촉진한다. 그러나 목표 탐색(도구적 목표 및 계획 생성 포함)의 도전과제(환각, 추론 과정의 설명 가능성, 복잡한 책임 소재 등)를 고려한 에이전트 설계를 안내할 체계적 지식이 부족하다. 이 문제를 해결하기 위해 체계적 문헌 검토를 수행하여 최신 FM 기반 에이전트와 광범위한 생태계를 이해했다. 본 논문은 이전 문헌 검토의 결과물로서 컨텍스트, 힘, 트레이드오프 분석을 포함한 18개 아키텍처 패턴으로 구성된 패턴 카탈로그를 제시하고, 패턴 선택을 위한 의사결정 모델을 제안한다.

## 상세 내용

### 1. 서론 (Introduction)
FM 기반 자율 에이전트(예: AutoGPT, BabyAGI)는 사용자의 광범위한 목표를 실행 가능한 작업 집합으로 분해하고 작업 실행을 조율하는 능력으로 급성장하고 있다. 그러나 실무자들이 FM 기반 에이전트를 설계하고 구현하는 데는 가파른 학습 곡선이 존재한다.

저자들은 FM 기반 에이전트 개발의 주요 도전과제를 다음과 같이 정리한다:
- **부정확한 응답**: 에이전트가 복잡한 작업을 완전히 이해하고 실행하기 어려워 부정확한 응답 가능성 존재. 장기 계획에서 단계 간 의존성으로 인해 약간의 편차도 전체 성공률에 큰 영향
- **불완전한 명세**: 사용자가 제한된 컨텍스트, 모호한 목표, 불명확한 지시를 제공하여 추론 과정과 응답 생성에 과소명세 발생
- **설명 가능성 부족**: 에이전트와 FM의 정교한 내부 아키텍처로 인해 "블랙박스"가 되어, 추론 단계를 해석하기 어렵고 신뢰성에 영향
- **복잡한 책임 소재**: 다양한 이해관계자, FM 기반 에이전트, 비에이전트 AI 모델, 비AI 소프트웨어 간의 상호작용으로 책임과 책무가 여러 개체에 얽힘

본 논문의 기여:
1. 실제 에이전트 구현을 위한 설계 솔루션 풀로서의 아키텍처 패턴 컬렉션
2. 아키텍처 패턴이 주석된 FM 기반 에이전트 생태계
3. 각 패턴의 적용 컨텍스트, 문제, 장단점, 실제 사용 사례의 정밀 분석
4. 합리적 설계 결정을 위한 의사결정 모델

### 2. 배경 및 관련 연구 (Background & Related Work)
ChatGPT의 출시(2022년 11월)가 2개월 만에 1억 사용자를 돌파하며 FM과 GenAI 제품 개발 경쟁을 촉발했다. 다양한 연구들이 에이전트 아키텍처의 특정 구성요소나 체계에 초점을 맞추었으나(예: MemGPT의 메모리 설계, Andrew Ng의 반성/도구사용/계획/멀티에이전트 협업), 아키텍처 설계에 대한 **전체적 관점(holistic view)**이 부족하다. 기존 프레임워크들은 고수준 구성요소만 나열하며 소프트웨어 구성요소, 관계, 속성에 대한 명시적 식별이 없는 경우가 많다.

### 3. 방법론 (Methodology)
체계적 문헌 검토(SLR)를 수행하여 57개 연구를 포함했다. 사전 설정된 기준에 따라 논문을 선별하고 전방/후방 스노우볼링을 수행했다. SLR 결과를 바탕으로 패턴 분석을 심화하고, 그레이 리터러처와 실제 응용을 추가로 검토하여 실제 사용 사례를 식별했다.

### 4. 패턴 카탈로그 (Pattern Catalogue) - 18개 패턴 상세

**4.1 Passive Goal Creator (수동 목표 생성기)**
대화 인터페이스를 통해 사용자의 명시적 목표를 분석한다. 사용자가 직접 컨텍스트와 문제를 제공하며, 메모리에서 관련 정보를 검색하여 목표를 결정한다. 장점으로 상호작용성, 목표 탐색, 효율성이 있고, 단점으로 불명확한 입력 시 추론 불확실성이 있다. HuggingGPT가 대표적 사용 사례다.

**4.2 Proactive Goal Creator (능동 목표 생성기)**
대화 외에도 카메라, 스크린샷 등 감지기를 통해 사용자의 환경 정보를 다중모달로 포착하여 목표를 예측한다. 장애인 사용자의 접근성을 보장하며, GestureGPT(손 제스처 해석), ProAgent(다른 에이전트의 행동 관찰 및 의도 추론)가 대표 사례다.

**4.3 Prompt/Response Optimiser (프롬프트/응답 최적화기)**
사전 정의된 제약과 사양에 따라 프롬프트와 응답을 정제한다. 표준화, 목표 정렬, 상호운용성을 보장하며, LangChain의 프롬프트 템플릿, Amazon Bedrock의 프롬프트 구성, Dialogflow가 대표 사례다.

**4.4 Retrieval Augmented Generation (검색 증강 생성)**
벡터 데이터베이스의 매개변수화된 지식으로 에이전트의 지식 공백을 채운다. 사전 학습이나 파인튜닝 없이 로컬 데이터의 프라이버시를 보존하면서 최신 정보를 제공한다. 그래프 RAG, 페더레이티드 RAG 등 다양한 변형이 존재한다.

**4.5 One-Shot Model Querying (일회성 모델 질의)**
FM을 한 번만 질의하여 계획의 모든 필요 단계를 생성한다. 비용 효율적이고 단순하지만, 복잡한 작업에서는 과도한 단순화와 설명 가능성 부족의 단점이 있다. CoT와 Zero-shot-CoT가 이 패턴의 예시다.

**4.6 Incremental Model Querying (점진적 모델 질의)**
계획 생성의 각 단계에서 FM을 질의하여 단계별 추론을 수행한다. 보충 컨텍스트 제공, 추론 확실성 향상, 설명 가능성을 제공하지만, 여러 번의 상호작용으로 비용이 증가한다. HuggingGPT, EcoAssistant, ReWOO가 대표 사례다.

**4.7 Single-Path Plan Generator (단일 경로 계획 생성기)**
사용자 목표 달성을 위한 중간 단계의 선형적 생성을 조율한다. Chain-of-Thought(CoT)가 대표적이며, 추론 확실성과 일관성을 제공하지만 유연성이 제한된다. LlamaIndex의 ReAct Agent, ThinkGPT가 사용 사례다.

**4.8 Multi-Path Plan Generator (다중 경로 계획 생성기)**
각 중간 단계에서 다수의 선택지를 생성한다. Tree-of-Thoughts가 대표적이며, 사용자 선호도를 각 단계에서 반영하여 맞춤형 전략을 제공한다. AutoGPT, Gemini, GPT-4의 ToT 구현이 대표 사례다.

**4.9 Self-Reflection (자기 반성)**
에이전트가 자체적으로 계획과 추론 과정에 대한 피드백을 생성하고 개선 지침을 제공한다. 추론 확실성, 설명 가능성, 지속적 개선의 장점이 있으나, 자체 평가의 정확성에 의존한다. Reflexion, Generative Agents가 대표 사례다.

**4.10 Cross-Reflection (교차 반성)**
다른 에이전트나 FM을 사용하여 생성된 계획을 평가하고 개선한다. 제한된 능력의 에이전트가 스스로 반성하기 어려울 때 유용하다. 포용성과 확장성이 장점이나, 공정성 보존과 복잡한 책임 소재가 단점이다. XAgent, ChatDev가 대표 사례다.

**4.11 Human Reflection (인간 반성)**
사용자나 인간 전문가로부터 피드백을 수집하여 계획을 개선한다. 인간 선호도 정렬과 이의 제기 가능성(contestability)이 핵심 장점이다. Inner Monologue(로봇 시스템), Ma et al.의 사용자-에이전트 숙의 연구가 대표 사례다.

**4.12 Voting-based Cooperation (투표 기반 협력)**
에이전트들이 자유롭게 의견을 표명하고 투표를 통해 합의에 도달한다. 공정성, 책임 소재, 집단 지성이 장점이나, 특정 에이전트가 다수 결정권을 확보하는 중앙집중화가 위험이다. Hamilton(법원 시뮬레이션), ChatEval이 대표 사례다.

**4.13 Role-based Cooperation (역할 기반 협력)**
에이전트에게 다양한 역할(계획자, 할당자, 작업자, 생성자)을 부여하고 역할에 따라 결정을 확정한다. 분업, 장애 내성, 확장성이 장점이다. MetaGPT(건축가, PM, 엔지니어 역할), MedAgents(의료 전문가 역할), Mixture-of-Agents가 대표 사례다.

**4.14 Debate-based Cooperation (토론 기반 협력)**
에이전트들이 탈중앙화 방식으로 초기 응답을 전파하고 합의에 도달할 때까지 토론한다. 적응성, 설명 가능성, 비판적 사고가 장점이다. crewAI, Du et al.의 멀티에이전트 토론 연구가 대표 사례다.

**4.15 Multimodal Guardrails (다중모달 가드레일)**
FM의 입출력을 제어하여 사용자 요구사항, 윤리 기준, 법률에 부합하도록 한다. 견고성, 안전성, 표준 정렬이 장점이다. NVIDIA NeMo Guardrails, Meta Llama Guard, Guardrails AI가 대표 사례다.

**4.16 Tool/Agent Registry (도구/에이전트 레지스트리)**
다양한 에이전트와 도구를 선택할 수 있는 통합 소스를 유지한다. 발견 가능성, 효율성, 도구 적절성, 확장성이 장점이나, 벤더 종속과 단일 장애점이 위험이다. GPTStore, VOYAGER, OpenAgents가 대표 사례다.

**4.17 Agent Adapter (에이전트 어댑터)**
에이전트와 외부 도구를 연결하는 인터페이스를 제공한다. 도구 매뉴얼을 검색하여 인터페이스를 학습하고 에이전트 출력을 요구 형식으로 변환한다. AutoGen, Apple Intelligence, Semantic Kernel, SWE-agent가 대표 사례다.

**4.18 Agent Evaluator (에이전트 평가기)**
설계 시점과 런타임 모두에서 에이전트의 성능을 평가한다. 시나리오 기반 요구사항, 메트릭, 예상 출력을 정의하여 평가 파이프라인을 구축한다. UK AI Safety Institute의 Inspect, DeepEval, Promptfoo, Ragas가 대표 사례다.

### 5. 패턴 카탈로그에서 얻은 교훈 (Lessons Learned)

**5.1 의사결정 모델**
18개 패턴의 선택 과정을 시각화하는 의사결정 모델(Figure 19)을 제안한다. 각 설계 문제에 대해 결정이 해당 솔루션 공간에 매핑되며, 같은 솔루션 공간 내 패턴들은 상호 보완 관계다. 주요 결정 분기점: 환경 컨텍스트 포착 가능 여부 -> 프롬프트 최적화 필요 여부 -> 외부 데이터 저장소 사용 여부 -> FM 다회 질의 여부 -> 계획에서 다중 선택지 여부 -> 계획 반성 여부 -> 다중 에이전트 협력 여부 -> FM 입출력 제어 여부 -> 외부 도구/에이전트 사용 여부 -> 에이전트 성능 평가 여부.

**5.2 패턴 적용 사례**
- Universal Task Assistant: 모바일 UI 화면 캡처 및 요소 감지에 능동 목표 생성기 적용
- Prompt Sapper: 수동 목표 생성기, 프롬프트/응답 최적화기, 도구/에이전트 레지스트리 적용
- Shamsujjoha et al.: 다중모달 가드레일의 분류체계 개발
- Xia et al.: 시스템 수준/구성요소 수준 AI 시스템 평가 프레임워크

**5.3 논의**
- **기존 패턴과의 통합**: 책임 있는 AI 패턴과 결합하여 에이전트의 신뢰성 보장 가능. 블록체인 스마트 컨트랙트를 투표 기반 협력에 활용 가능
- **규제 및 표준 준수**: EU AI Act, NIST AI 위험 관리 프레임워크, ISO AI 관리 시스템 표준과의 매핑 필요
- **FM 기반 에이전트 평가**: 패턴의 장단점이 대부분 소프트웨어 품질 속성이며, 세밀한 메트릭과 루브릭의 정량화가 필요

### 6. 결론 (Conclusion)
FM 기반 에이전트가 다양한 도메인에서 비즈니스 프로세스의 지능화 및 자동화를 위해 증가하는 관심을 받고 있다. 본 연구는 18개 패턴의 힘, 해결책, 트레이드오프를 정밀하게 분석하여 실무자들에게 에이전트 설계 및 개발을 위한 포괄적 지침을 제공한다. 향후 연구에서는 기존 패턴과의 결합을 통한 에이전트 신뢰성 보존과 FM 기반 에이전트 관련 아키텍처 결정을 추가 탐구할 예정이다.

## 핵심 키 포인트
1. **18개 아키텍처 패턴 체계화**: 목표 생성(2개), 프롬프트/응답 최적화(1개), 지식 검색(1개), 모델 질의(2개), 계획 생성(2개), 반성(3개), 협력(3개), 가드레일(1개), 도구 관리(2개), 평가(1개)의 범주로 FM 기반 에이전트 설계의 전체 스펙트럼을 체계적으로 포괄한다.
2. **패턴 간 관계 분석**: 대안 패턴(예: Passive vs Proactive Goal Creator, One-shot vs Incremental Model Querying)과 보완 패턴(예: Plan Generator + Reflection, Cooperation + Tool Registry) 관계를 명확히 정의하여 조합 설계를 지원한다.
3. **의사결정 모델**: 이진 결정 트리 형태로 18개 패턴의 선택 과정을 시각화하여, 실무자가 요구사항에 따라 적절한 패턴을 체계적으로 선택할 수 있도록 한다.
4. **소프트웨어 품질 속성 기반 트레이드오프**: 각 패턴의 장단점을 추론 확실성, 설명 가능성, 효율성, 확장성, 공정성, 책임 소재 등 소프트웨어 품질 속성으로 분석하여, 설계 결정의 근거를 제공한다.
5. **멀티에이전트 협력의 세 가지 체계**: 투표 기반(공정성/집단지성), 역할 기반(분업/확장성), 토론 기반(적응성/비판적사고) 협력을 명확히 구분하고 각각의 적용 시나리오를 제시한다.
6. **실제 사용 사례 매핑**: AutoGPT, MetaGPT, LangChain, HuggingGPT, Apple Intelligence, NVIDIA NeMo 등 다양한 실제 시스템과 패턴을 매핑하여 이론과 실무의 연결고리를 제공한다.

## 주요 인용 (Key Quotes)
> "There is a lack of systematic knowledge to guide practitioners in designing the agents considering challenges of goal-seeking, such as hallucinations inherent in foundation models, explainability of reasoning process, complex accountability, etc." (Abstract, p.1)

> "Agents often struggle to fully comprehend and execute complex tasks, leading to the potential for inaccurate responses. This challenge may be intensified by the inherent reasoning uncertainties during plan generation and action procedures." (Section 1, p.1)

> "The accountability process is complicated due to the interactions between various stakeholders, FM-based agents, non-agent AI models, and non-AI software applications within the overall ecosystem. Highly autonomous agents may delegate or even create other agents or tools for certain tasks." (Section 1, p.2)

> "Frameworks that only list the high-level components supporting their functionality usually lack system-level thinking, with no explicit identification of software components, relationships among them, and their properties." (Section 2, p.3)

> "Reflection is an optimisation process formalised to iteratively review and refine the reasoning process and generated contents of the agent." (Section 4.9, p.18)

> "Guardrails can be applied as an intermediate layer between the foundation model and all other components in a compound AI system." (Section 4.15, p.26)

> "Agent adapter can help invoke and manage these interfaces by converting the agent messages into required format or content, and vice versa. In particular, the adapter can retrieve tool manual or tutorial from datastore, to acquire available interfaces and learn the usage." (Section 4.17, p.29)

> "Preserving the alignment of agents with both international and domestic regulations and standards should be noted as a fundamental factor for developers to provide agent services in different countries and regions." (Section 5.3, p.33)

## 시사점 및 의의
이 논문은 FM 기반 에이전트 설계에 대한 최초의 포괄적인 아키텍처 패턴 카탈로그로서 여러 중요한 시사점을 제공한다.

첫째, **소프트웨어 공학적 관점에서 에이전트 설계를 체계화**했다는 점이 가장 큰 의의이다. 기존의 에이전트 연구가 개별 기법이나 시스템에 초점을 맞추었던 것과 달리, 이 연구는 재사용 가능한 아키텍처 패턴이라는 소프트웨어 공학의 검증된 개념을 적용하여, 실무자들이 체계적으로 에이전트를 설계할 수 있는 프레임워크를 제공한다.

둘째, **트레이드오프 분석이 실무적으로 매우 유용**하다. 각 패턴의 장단점을 소프트웨어 품질 속성으로 명확히 정의함으로써, 아키텍트가 요구사항(효율성 vs 설명 가능성, 확장성 vs 오버헤드 등)에 따라 근거 있는 설계 결정을 내릴 수 있다.

셋째, **멀티에이전트 협력의 세 가지 패턴(투표/역할/토론)**은 현재 급성장하는 멀티에이전트 시스템 설계에 즉시 적용 가능한 실용적 가이드라인이다. 특히 각 협력 방식의 책임 소재(accountability) 분석은 AI 거버넌스 관점에서 중요한 통찰을 제공한다.

넷째, 이 패턴 카탈로그는 **AI 규제(EU AI Act, NIST 등)와의 연계 가능성**을 열어놓아, 규제 준수가 중요해지는 시대에 에이전트 개발의 표준화된 접근법을 제시한다. 향후 패턴과 규제 요구사항의 매핑 연구는 실무적으로 매우 가치 있을 것이다.
