# Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling

## 기본 정보
- **저자**: Gaurav Kumar (Stanford AI Professional Program), Annu Rana (IESE EMBA Program)
- **발행일/출처**: 2025년 12월 16일, arXiv:2512.14474v1 [cs.AI]
- **페이지 수**: 17페이지
- **키워드**: Model-First Reasoning, LLM Agents, Hallucination, Problem Modeling, Chain-of-Thought, ReAct, Planning, Constraint Satisfaction

## 한줄 요약
> 이 논문은 LLM 기반 에이전트의 계획 실패가 추론 능력 부족이 아니라 명시적 문제 표현의 부재에서 비롯된다고 주장하며, 추론 전에 문제 모델을 먼저 구축하는 2단계 패러다임(MFR)을 제안하여 제약 위반을 크게 줄인다.

## 초록 (Abstract)
LLM은 Chain-of-Thought(CoT)과 ReAct 같은 프롬프팅 전략으로 인상적인 추론 능력을 보여주지만, 복잡한 다단계 계획 과제에서 제약 위반, 일관성 없는 상태 추적, 취약한 솔루션을 빈번하게 보인다. 저자들은 이러한 실패가 추론 자체의 결함이 아니라 명시적 문제 표현의 부재에서 발생한다고 주장한다. Model-First Reasoning(MFR)은 LLM이 먼저 문제의 구조적 모델(엔티티, 상태 변수, 행동의 전제조건과 효과, 제약 조건)을 명시적으로 구축한 후, 이 모델에 기반하여 추론을 수행하는 2단계 패러다임이다. 다양한 제약 기반 계획 도메인에서의 실험을 통해 MFR이 CoT나 ReAct 대비 제약 위반을 크게 줄이고, 장기 일관성을 개선하며, 더 해석 가능한 솔루션을 생성함을 보인다.

## 상세 내용

### 1. 서론 (Introduction)
LLM 기반 에이전트는 계획, 문제 해결, 복잡한 환경과의 상호작용에서 자율적 에이전트로 활용되고 있다. 그러나 정확성이 여러 단계에 걸친 일관적인 내부 상태 유지, 다중 제약 조건 준수, 암묵적 가정 회피에 달려 있는 도메인에서 높은 실패율을 보인다. 의료 스케줄링, 자원 할당, 절차적 실행 등이 그 예이다.

#### 1.1 암묵적 추론의 한계
- **CoT의 한계**: 단계별 설명을 생성하지만, 문제의 엔티티, 상태 변수, 제약 조건을 명시적으로 정의하지 않는다. 상태가 모델의 잠재 표현과 자연어 출력에 암묵적으로 추적되어 드리프트, 누락, 모순에 취약하다.
- **ReAct의 한계**: 추론과 행동을 인터리빙하지만, 상태 추적이 여전히 비공식적이고 자유형 텍스트에 분산되어 있다. 관찰이 추론되기보다 가정되며, 제약이 전역적으로 거의 강제되지 않는다.

#### 1.2 모델 기반 추론으로서의 추론
인간의 추론은 과학, 공학, 일상적 문제 해결에서 근본적으로 모델 기반이다. 과학적 탐구는 관련 엔티티, 변수, 지배 법칙을 정의한 후 추론한다. 고전적 AI 계획 시스템(STRIPS, PDDL)도 명시적 도메인 모델에서 추론을 수행한다. 그러나 LLM 에이전트는 모델링과 추론을 하나의 생성 프로세스로 합쳐, 기저 구조가 암묵적이고 불안정하다.

#### 1.3 Model-First Reasoning 제안
이러한 관찰에 동기부여되어, 문제 표현과 추론을 명시적으로 분리하는 MFR 패러다임을 제안한다. MFR에서 모델은 솔루션 생성 전에 명시적 문제 모델을 구축하도록 지시받는다.

### 2. 배경 및 관련 연구 (Background and Related Work)

#### 2.1 Chain-of-Thought 추론
CoT는 중간 추론 단계를 생성하여 성능을 향상시키지만, 문제의 구조를 명시적으로 정의하지 않는다. "추론 방법"을 개선하지만 "추론 대상"은 개선하지 않는다.

#### 2.2 ReAct와 에이전틱 추론
ReAct는 추론과 행동의 인터리빙으로 적응성을 향상시키지만, 형식적 문제 표현을 도입하지 않는다. 추론, 행동, 상태 추적이 단일 생성 프로세스에 얽혀 있다.

#### 2.3 고전적 AI 계획과 명시적 모델
STRIPS, PDDL 같은 고전적 시스템은 문제 정의와 문제 해결을 명시적으로 분리한다. 객체, 상태 변수, 전제조건/효과를 가진 행동, 목표 조건과 제약을 정의한다. MFR은 이 전통에서 개념적 영감을 받되, 모델 자체를 LLM이 구축한다는 점이 다르다.

#### 2.4 심적 모델과 인지적 관점
인지과학에서 심적 모델(mental models)은 인간 추론의 핵심으로 오래 강조되어 왔다. 오류는 논리적 추론의 실패보다 불완전하거나 부정확한 모델에서 자주 발생한다. LLM은 자신의 내부 표현을 외부화하도록 요구받지 않는데, MFR은 이 간극을 해소한다.

### 3. Model-First Reasoning 방법론

#### 3.1 개요
자연어 문제 설명이 주어지면, MFR은 두 단계로 진행된다:
1. **모델 구축(Model Construction)**: 엔티티, 상태 변수, 전제조건과 효과를 가진 행동, 제약 조건을 명시적으로 정의
2. **추론 및 계획(Reasoning and Planning)**: 구축된 모델만을 사용하여 솔루션 생성

#### 3.2 1단계: 모델 구축
- 엔티티(Entities): 문제에 관련된 객체나 에이전트
- 상태 변수(State Variables): 시간에 따라 변할 수 있는 엔티티의 속성
- 행동(Actions): 상태를 수정하는 허용된 연산, 전제조건과 효과 포함
- 제약 조건(Constraints): 항상 준수해야 하는 불변량, 규칙, 제한

모델은 자연어, 반구조화된 텍스트, 또는 유사 형식 표기법으로 표현 가능하다. 고정된 형식주의를 요구하지 않아 유연성을 유지한다.

#### 3.3 2단계: 모델 기반 추론
구축된 모델에 의해 추론이 명시적으로 제약된다:
- 행동은 명시된 전제조건을 준수해야 한다
- 상태 전이는 정의된 효과와 일관되어야 한다
- 제약 조건은 매 단계에서 충족되어야 한다

#### 3.4 프롬프트 구조
아키텍처 변경이나 파인튜닝 없이 간단한 프롬프트 기반 기법으로 구현 가능하다.

#### 3.5 MFR이 효과적인 이유
- 잠재 상태 추적에 대한 의존도 감소
- 명시되지 않은 가정 방지
- 장기 일관성 향상
- 인간 및 자동 검증 가능

#### 3.6 기존 패러다임과의 관계
MFR은 CoT(추론 단계 내에서 결합 가능), ReAct(모델을 영속적 상태로 활용 가능)와 상호보완적이다.

### 4. 실험 설정 (Experimental Setup)
- **비교 전략**: CoT, ReAct, MFR 세 가지
- **과제**: 다단계 약물 스케줄링, 시간적 의존성이 있는 경로 계획, 순차적 제약이 있는 자원 할당, 논리 퍼즐, 절차적 합성 등
- **실행 모델**: ChatGPT, Gemini, Claude 등 복수의 LLM에서 독립 실행
- **평가 기준**: 제약 만족(Constraint Satisfaction), 암묵적 가정(Implicit Assumptions), 구조적 명확성(Structural Clarity)

### 5. 결과 및 분석 (Results and Analysis)

| 추론 전략 | 제약 위반 | 암묵적 가정 | 구조적 명확성 |
|----------|---------|-----------|------------|
| CoT | 중간 | 빈번 | 낮음 |
| ReAct | 중간-낮음 | 간헐적 | 중간 |
| MFR | 낮음 | 드묾 | 높음 |

- **CoT 분석**: 유창한 단계별 추론을 생성하지만, 핵심 중간 상태 누락, 미명시 행동/가정 도입, 전역 일관성 실패가 빈번했다.
- **ReAct 분석**: 사고와 행동의 인터리빙으로 국소 추론이 개선되었지만, 관찰이 때때로 추론이 아닌 가정되었고, 전역 제약이 일관적으로 강제되지 않았다.
- **MFR 분석**: 명시적 제약 그라운딩, 암묵적 가정 감소, 구조적 명확성 향상의 세 가지 뚜렷한 이점을 보였다.

### 6. 논의 (Discussion)
- 표현적 실패(representational failures)가 LLM 추론 실패의 주된 원인이라는 가설을 결과가 검증한다.
- MFR의 모델 구축은 일종의 소프트 심볼릭 그라운딩(soft symbolic grounding)으로 기능한다.
- 과제 복잡성이 중요: MFR의 이점은 고제약, 다단계 계획 과제에서 가장 두드러진다.
- 명시적 모델은 검사, 검증, 디버그가 더 용이하여 신뢰성을 높인다.
- 토큰 오버헤드 증가라는 트레이드오프가 있지만, 정확성과 검증 가능성 향상으로 상쇄된다.

### 7. 결론 (Conclusion)
MFR은 문제 모델링과 추론을 분리하여 LLM 기반 에이전트의 계획 오류와 환각을 근본적으로 줄인다. 환각과 계획 오류는 추론적(inferential) 결함이 아니라 표현적(representational) 결함이며, 명시적 모델링을 기초 단계로 만들면 AI 에이전트의 신뢰성, 해석 가능성, 신뢰도가 향상된다.

## 핵심 키 포인트
1. **문제는 추론이 아니라 표현**: LLM의 계획 실패는 추론 능력이 아닌 명시적 문제 표현의 부재에서 기인한다.
2. **2단계 분리**: 문제 모델 구축과 추론을 명시적으로 분리하는 것이 핵심이다.
3. **프롬프트만으로 구현 가능**: 아키텍처 변경, 외부 심볼릭 솔버, 추가 훈련 없이 프롬프팅만으로 적용 가능하다.
4. **소프트 심볼릭 그라운딩**: 엄격한 형식 언어를 부과하지 않으면서도 추론을 안정화하는 구조를 제공한다.
5. **환각의 재정의**: 환각은 단순히 거짓 진술의 생성이 아니라, 명확히 정의된 문제 공간 모델 없이 수행된 추론의 증상이다.
6. **기존 패러다임과 상호보완적**: CoT, ReAct와 대체가 아닌 보완 관계로, 기존 프레임워크에 통합 가능하다.
7. **제약 집중적 과제에서 가장 효과적**: 의료 스케줄링, 자원 할당 등 정확성이 중요한 도메인에서 특히 유용하다.

## 주요 인용 (Key Quotes)

> "We argue that many of these failures arise not from deficiencies in reasoning itself, but from the absence of an explicit problem representation." (Abstract, p.1)

> "Hallucination is not merely the generation of false statements. Rather, it is a symptom of reasoning performed without a clearly defined model of the problem space." (Section 1.2, p.3)

> "CoT improves how models reason, but not what they reason over." (Section 2.1, p.5)

> "Reasoning does not create structure; it operates on structure. When structure is implicit or unstable, reasoning becomes unreliable." (Section 2.4, p.6)

> "Model-First Reasoning functions as a form of soft symbolic grounding. It does not impose formal symbolic constraints, but introduces enough structure to stabilize reasoning in complex planning tasks." (Section 3.5, p.10)

> "Many observed LLM planning failures are fundamentally representational rather than inferential, and that explicit problem modeling should be viewed as a foundational component of reliable, agentic AI systems." (Abstract, p.1)

> "Our findings reframe hallucination and planning errors in LLMs as primarily representational rather than inferential." (Section 7, p.16)

## 시사점 및 의의
이 연구는 LLM 에이전트의 실패 원인에 대한 새로운 관점을 제시한다는 점에서 중요하다. 기존에는 추론 능력 자체를 향상시키는 데 초점이 맞추어져 있었지만, MFR은 "무엇에 대해 추론하는가"라는 표현의 문제가 더 근본적임을 주장한다. 이는 프롬프트 설계 차원에서 즉시 적용 가능한 실용적 시사점을 가진다. AgentOps 관점에서, 제약 조건이 많은 기업 자동화 시나리오(스케줄링, 자원 할당, 규정 준수 등)에서 MFR을 적용하면 에이전트의 신뢰성을 크게 향상시킬 수 있다. 다만, 실험이 질적 평가에 기반하고 대규모 벤치마크가 아닌 선택된 예제에 대한 것이라는 한계가 있어, 후속 정량적 검증이 필요하다.
