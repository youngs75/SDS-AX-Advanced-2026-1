# InfiAgent: An Infinite-Horizon Framework for General-Purpose Autonomous Agents

## 기본 정보
- **저자**: Chenglin Yu (The University of Hong Kong), Yuchen Wang, Songmiao Wang, Hongxia Yang, Ming Li* (The Hong Kong Polytechnic University)
- **발행일/출처**: 2026년 1월 6일, arXiv:2601.03204v1 [cs.AI]
- **페이지 수**: 10페이지
- **키워드**: LLM Agents, Infinite-Horizon Reasoning, File-Centric State Management, Autonomous Research, Multi-Agent Collaboration, Context Window, State Externalization

## 한줄 요약
> 이 논문은 LLM 에이전트의 장기 과제 수행 시 무제한 컨텍스트 성장과 누적 오류 문제를 해결하기 위해, 영속적 상태를 파일 시스템에 외부화하고 추론 컨텍스트를 엄격히 제한하는 범용 프레임워크 InfiAgent를 제안하며, 20B 오픈소스 모델로 대규모 독점 시스템과 경쟁하는 성과를 보여준다.

## 초록 (Abstract)
LLM 에이전트는 추론과 도구 사용이 가능하지만, 무제한 컨텍스트 성장과 누적 오류로 인해 장기 과제에서 자주 실패한다. 컨텍스트 압축이나 검색 증강 프롬프팅 같은 일반적 해결책은 정보 충실도와 추론 안정성 간의 트레이드오프를 도입한다. InfiAgent는 영속적 상태를 파일 중심 상태 추상화(file-centric state abstraction)에 외부화하여, 과제 기간에 관계없이 에이전트의 추론 컨텍스트를 엄격히 제한하는 범용 프레임워크를 제시한다. DeepResearch 벤치마크와 80편 논문 문헌 리뷰 과제에서, 과제별 파인튜닝 없이 20B 오픈소스 모델을 사용한 InfiAgent가 더 큰 독점 시스템과 경쟁하며, 컨텍스트 중심 베이스라인보다 현저히 높은 장기 커버리지를 유지한다.

## 상세 내용

### 1. 서론 (Introduction)
LLM은 자율 에이전트로 점점 더 배포되고 있지만, 장기 과제 수평선(long task horizons)에서 취약하다. 핵심 문제는 에이전트 상태의 표현과 유지 방식에 있다. 대부분의 현재 프레임워크는 LLM 프롬프트를 상태의 주요 운반체로 암묵적으로 취급하여, 대화 히스토리, 도구 추적, 중간 계획, 부분 결과를 컨텍스트 윈도우에 직접 축적한다. 과제 기간이 늘어남에 따라 이 설계는 무제한 컨텍스트 성장을 초래하고, 잘라내기, 요약, 휴리스틱 검색에 의존하게 되며, 정보 손실, 무관한 토큰의 간섭, 초기 오류에 대한 민감성 증가라는 실패 모드를 야기한다.

RAG와 긴 컨텍스트 모델은 부분적으로 이를 완화하지만, 장기적 과제 상태를 즉각적 추론 컨텍스트와 여전히 얽어놓아 LLM에 증가하는 인지 부하를 부과한다. 이를 "상태의 환상(illusion of state)"이라 부르며, 컨텍스트 길이를 단순히 확장하는 것으로는 장기 안정성 문제를 근본적으로 해결할 수 없다.

MAKER처럼 극단적 과제 분해를 통한 무한 실행이 가능한 접근법도 있지만, 고도로 구조화된 도메인에 특화되어 있어 과학 연구 같은 개방형 도메인에는 적용이 제한된다.

**InfiAgent의 핵심 주장**: 안정적인 장기 행동을 달성하려면 영속적 과제 상태와 제한된 추론 컨텍스트를 명시적으로 분리해야 한다.

### 2. 관련 연구 (Related Work)

#### 2.1 다중 에이전트 시스템 아키텍처
기존 피어-투-피어 협업 모델은 간단한 조정에는 효과적이지만 복잡한 계층적 과제 분해에서는 어려움을 겪는다. InfiAgent는 엄격한 구성적 제약과 파일 중심 상태를 통해 안정성을 강제한다.

#### 2.2 다중 에이전트 시스템의 과제 분해
목표 지향, 제약 기반, 학습 기반 분해 등 다양한 접근법이 있으나, 장기 수평선에서의 시스템 안정성 보장이 부족하다. InfiAgent는 엄격한 부모-자식 제어를 유지하는 재귀적 DAG 기반 분해를 사용한다.

#### 2.3 긴 컨텍스트와 자율 연구 에이전트
"상태의 환상" 문제가 Shojaee et al.에 의해 강조되었으며, 컨텍스트가 채워짐에 따라 성능이 저하된다. InfiAgent의 "제로 컨텍스트 압축" 접근법이 이러한 컨텍스트 중심 방법에 대한 견고한 대안을 제공한다.

### 3. 파일 중심 상태의 형식화 (Formalizing File-Centric State)

#### 3.1 상태 조건부 의사결정 과정
에이전트 실행을 이산 시간 단계의 상태 조건부 과정으로 정의한다. 기존 컨텍스트 중심 설계에서는 ct = <o1, a1, ..., ot-1, at-1, ot>로 모든 과제 상태가 단일 시퀀스에 얽혀 있어, 정보 유지와 추론 안정성 간의 본질적 트레이드오프가 발생한다.

#### 3.2 영속적 상태 외부화
장기 메모리를 제한된 추론 컨텍스트에서 분리하기 위해 명시적 영속적 상태를 도입한다:
- St = Ft (파일과 구조화된 아티팩트의 집합)
- 상태 전이: Ft+1 = T(Ft, at) (파일 생성, 수정, 삭제)
- Ft는 컨텍스트 윈도우 제약에 종속되지 않고 과제 복잡성과 기간에 따라 성장 가능

#### 3.3 제한된 추론 컨텍스트 재구성
매 단계에서 제한된 추론 컨텍스트를 재구성한다:
- c_bounded_t = g(Ft, at-k:t-1)
- k는 작은 상수(예: k=10)로, |c_bounded_t| = O(1) (과제 수평선에 대해 상수)
- 요약 기반 접근과 달리 권위적 상태에서 어떤 정보도 폐기되지 않음
- 관련성은 매 단계에서 상태 검사를 통해 동적으로 결정

#### 3.4 비교 및 함의
컨텍스트 중심 에이전트는 모든 과제 상태를 프롬프트에 직접 인코딩하고, RAG 에이전트는 부분적으로 메모리를 외부화하지만 검색된 텍스트를 다시 컨텍스트에 주입한다. InfiAgent의 파일 중심 추상화는 영속적 상태를 일급 객체(first-class object)로 취급하여, 무제한 컨텍스트 성장을 제거하고 장기 아티팩트와 단기 의사결정 간 간섭을 줄인다.

### 4. InfiAgent 프레임워크

#### 4.1 파일 중심 상태 관리와 과제 메모리
- 각 과제에 전용 워크스페이스 디렉토리 할당 (계획, 중간 아티팩트, 도구 출력, 검증 로그 저장)
- 고정 길이 최근 행동 버퍼 + 워크스페이스 스냅샷으로 추론 컨텍스트 구성
- **주기적 상태 통합(Periodic State Consolidation)**: 고정 간격마다 고수준 계획과 진행 마커를 업데이트하고 추론 컨텍스트를 새로고침

#### 4.2 다단계 에이전트 계층
트리 구조 계층(DAG):
- **Level 3 (Alpha Agent)**: 오케스트레이터 -- 고수준 계획 및 하위 과제 분해
- **Level 2 (Domain Agents)**: 전문가 -- 코더, 데이터 수집, 논문 작성 등
- **Level 1 (Atomic Agents)**: 원자적 도구 실행 -- 웹 검색, 파일 I/O 등

Agent-as-a-Tool 패턴: 상위 에이전트가 하위 에이전트를 호출 가능 도구로 사용하여 "도구 호출 혼란(tool calling chaos)"을 방지한다.

#### 4.3 외부 주의 파이프라인 (External Attention Pipeline)
- 대량 정보(예: 80편 논문)를 컨텍스트 팽창 없이 처리
- 문서를 컨텍스트에 로드하지 않고, 전문 도구(answer_from_pdf)가 임시 격리 LLM 프로세스를 실행하여 추출된 답변만 반환
- 응용 계층 주의 헤드(attention head)로 기능하여, 대규모 외부 데이터에서 관련 정보만 선택

### 5. 실험 (Experiments)

#### 5.1 DeepResearch 벤치마크
- **설정**: 20B 파라미터 오픈소스 모델(gpt-oss-20b), 과제별 파인튜닝 없음
- **전체 성능**: 41.45점으로, 상당히 큰 모델에 의존하는 시스템 대비 유리한 효율성 프론티어에 위치
- **구성 요소별 분석**: 지시 따르기(instruction following)와 가독성(readability)에서 특히 우수 -- 명시적 파일 중심 상태와 구조화된 실행 파이프라인에 기인
- 통찰력(insight)과 포괄성(comprehensiveness)도 더 큰 모델과 경쟁적

#### 5.2 장기 문헌 리뷰 과제
- **설정**: 80편 학술 논문 읽기, 요약, 관련성 점수 부여
- **평가 지표**: 커버리지(coverage) -- 내용 기반 요약을 생성한 논문 수

| 설정 | 모델 | 최대 | 최소 | 평균 |
|------|------|-----|-----|------|
| InfiAgent | GPT-OSS-20B | 80 | 15 | 67.1 |
| InfiAgent | Gemini-3-Flash | 80 | 80 | 80.0 |
| InfiAgent | Claude-4.5-Sonnet | 80 | 80 | 80.0 |
| Claude Code | Claude-4.5-Sonnet | 80 | 11 | 29.1 |
| Cursor | Claude-4.5-Sonnet | 5 | 0 | 1.0 |
| **절제: 파일 상태 없음** | GPT-OSS-20B | 7 | 1 | 3.2 |
| **절제: 파일 상태 없음** | Claude-4.5-Sonnet | 77 | 11 | 27.7 |

- InfiAgent는 강한 백본 모델에서 80편 전체를 일관되게 처리하고, 20B 모델에서도 상당한 커버리지 유지
- 베이스라인 에이전트들은 현저히 낮은 평균 커버리지와 높은 분산을 보임 -- 조기 종료, 항목 건너뛰기, 제목만 재진술하는 요약 생성
- **절제 분석**: 파일 중심 상태를 제거하고 압축된 긴 컨텍스트 프롬프트에 의존하면 모든 모델에서 커버리지가 대폭 감소하고 실행 간 분산 증가. 더 강한 백본에서도 마찬가지 -- 긴 컨텍스트만으로는 신뢰할 수 있는 장기 실행을 대체할 수 없음

#### 5.3 사례 연구: 실세계 응용 및 전문가 블라인드 리뷰
InfiHelper(InfiAgent 기반 구현):
- **계산 생물학**: 세포외 기질 단백질 조성 시뮬레이션 및 상호작용 예측
- **물류 운영**: 복잡한 제약 하의 자동 교대 스케줄링
- **학술 연구**: 문헌 리뷰부터 원고 작성까지 엔드투엔드 자동화
- 전문가 블라인드 리뷰에서 **인간 수준 품질** 판정 -- 학술 컨퍼런스 수용 기준을 충족하는 것으로 평가

### 6. 논의 (Discussion)
- **명시적 상태 외부화가 해결하지 못하는 것**: 기저 언어 모델의 추론 능력 자체를 향상시키지 않는다. 잘못된 중간 결론이 영속적 상태에 기록되어 전파될 수 있다.
- **긴 컨텍스트는 영속적 상태의 대체물이 아니다**: 절제 결과에서 큰 컨텍스트 윈도우 모델에서도 파일 중심 상태를 압축된 긴 컨텍스트로 대체하면 커버리지와 안정성이 대폭 저하됨을 확인
- **효율성과 지연 트레이드오프**: 상태 외부화와 계층적 실행이 단일 패스 에이전트 대비 추가 오버헤드를 도입하며, 실시간 응용보다는 장기 실행 지식 집약적 워크플로우에 적합
- **범위와 일반화**: 연구 지향 과제에 집중한 평가로, 반응적 대화, 구현체 상호작용, 빠르게 변화하는 환경에서의 검증이 필요
- **더 넓은 함의**: 장기 에이전트의 일반적 설계 원칙 제시 -- 영속적 과제 상태는 LLM의 제한된 추론 컨텍스트와 구별되는 일급 객체로 취급되어야 한다

### 7. 한계 (Limitations)
1. 다단계 에이전트 계층의 지연 오버헤드
2. 작은 모델(20B)에서의 환각 누적 위험
3. 상태 일관성을 위한 엄격한 직렬 실행으로 병렬 처리 불가

### 8. 결론 (Conclusion)
InfiAgent는 컨텍스트 기반 에이전트에서 파일 중심 에이전트로의 패러다임 전환을 제안한다. 파일 시스템에 상태를 외부화하고 엄격한 다단계 계층을 사용하여 사실상 무한한 실행 시간과 높은 안정성을 달성한다. 훈련 없이(training-free) 20B 오픈소스 모델이 복잡한 연구 벤치마크에서 최첨단 독점 에이전트와 경쟁할 수 있음을 보여, 더 접근 가능하고 확장 가능한 자율 연구 시스템의 길을 연다.

## 핵심 키 포인트
1. **영속적 상태와 추론 컨텍스트의 분리**: 장기 과제 상태를 파일 시스템에 외부화하고, 추론 컨텍스트를 엄격히 제한(O(1))하여 무제한 컨텍스트 성장을 제거한다.
2. **파일 중심 상태 추상화**: 파일 시스템을 권위적이고 영속적인 과제 기록으로 활용하며, 매 단계에서 워크스페이스 스냅샷과 최근 k개 행동으로 컨텍스트를 재구성한다.
3. **계층적 에이전트 아키텍처**: 3단계 트리 구조(Alpha-Domain-Atomic)로 구조화된 과제 분해와 제어된 도구 호출을 강제한다.
4. **외부 주의 파이프라인**: 대량 문서를 격리된 프로세스에서 처리하여 메인 에이전트의 인지 부하를 최소화한다.
5. **긴 컨텍스트는 대체물이 아님**: 절제 실험에서 큰 컨텍스트 윈도우가 명시적 상태 외부화를 대체할 수 없음을 입증한다.
6. **소형 모델의 경쟁력**: 20B 모델이 아키텍처적 설계를 통해 훨씬 큰 독점 모델과 경쟁할 수 있음을 보여, 모델 규모와 아키텍처 설계의 상호보완성을 입증한다.
7. **훈련 없는 접근법**: 과제별 파인튜닝 없이 즉시 적용 가능하다.

## 주요 인용 (Key Quotes)

> "LLM agents can reason and use tools, but they often break down on long-horizon tasks due to unbounded context growth and accumulated errors." (Abstract, p.1)

> "We argue that achieving stable long-horizon behavior in LLM agents requires an explicit separation between persistent task state and bounded reasoning context." (Section 1, p.2)

> "Simply extending context length does not fundamentally resolve the long-horizon stability problem." (Section 1, p.1)

> "InfiAgent treats the file system as the authoritative and persistent record of the agent's actions, environment, and intermediate artifacts." (Section 1, p.2)

> "Long Context Is Not a Substitute for Persistent State. Even when using models with large context windows, replacing file-centric state with compressed long-context prompts leads to substantial degradation in coverage and increased variance across runs." (Section 6, p.8)

> "InfiAgent proposes a paradigm shift from context-based agents to file-centric agents." (Section 8, p.9)

> "Persistent task state should be treated as a first-class object, distinct from the bounded reasoning context of the language model." (Section 6, p.8)

## 시사점 및 의의
InfiAgent는 LLM 에이전트 아키텍처의 근본적 설계 원칙을 제시한다는 점에서 매우 중요하다. AgentOps 관점에서의 핵심 시사점: (1) 장기 실행 에이전트를 운영할 때 컨텍스트 윈도우만으로는 안정성을 보장할 수 없으며, 명시적 상태 외부화가 필수적이다. (2) 파일 중심 상태 관리는 에이전트 행동의 감사(audit), 디버깅, 재현을 용이하게 하여 프로덕션 운영에 매우 유리하다. (3) 계층적 에이전트 아키텍처가 플랫 다중 에이전트 시스템의 "도구 호출 혼란"을 방지한다. (4) 소형 오픈소스 모델도 적절한 아키텍처로 대규모 독점 모델과 경쟁할 수 있어, 비용 효율적인 에이전트 운영이 가능하다. Agent Drift 논문과 함께 읽으면 특히 유익한데, InfiAgent의 파일 중심 상태 관리가 Agent Drift에서 식별한 컨텍스트 윈도우 오염에 대한 직접적 해결책이 될 수 있기 때문이다.
