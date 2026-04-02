# TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs

## 기본 정보
- **저자**: Shuyi Liu, Yuming Shang, Xi Zhang* (corresponding)
- **발행일/출처**: 2025년 11월 13일, arXiv:2511.10375v1 [cs.CL] / Key Laboratory of Trustworthy Distributed Computing and Service (MoE), Beijing University of Posts and Telecommunications, China / AAAI 2026
- **페이지 수**: 12페이지
- **키워드**: Retrieval-Augmented Generation (RAG), Knowledge Graphs, Knowledge Conflicts, Entropy-based Filtering, Factual Consistency, Triple Extraction, LLM

## 한줄 요약
> 이 논문은 RAG 시스템에서 LLM의 내부 파라메트릭 지식과 검색된 외부 정보 간의 사실적 수준 충돌을 해결하기 위해, 지식 그래프를 활용한 최초의 프레임워크인 TruthfulRAG를 제안하며, 트리플 추출, 쿼리 기반 그래프 검색, 엔트로피 기반 필터링을 통해 기존 방법들을 일관되게 능가한다.

## 초록 (Abstract)
RAG는 검색 기반 방법과 생성 모델을 통합하여 LLM의 능력을 향상시키는 강력한 프레임워크로 부상했다. 그러나 외부 지식 저장소가 계속 확장되고 모델 내부의 파라메트릭 지식이 시대에 뒤처지면서, 검색된 외부 정보와 LLM의 내부 지식 간 충돌이 발생하여 생성 콘텐츠의 정확성과 신뢰성을 저해한다. 기존 충돌 해결 접근법은 토큰 수준이나 의미 수준에서 작동하여 단편적이고 부분적인 사실적 불일치 이해를 초래한다. 이를 해결하기 위해 TruthfulRAG를 제안한다 -- 지식 그래프를 활용하여 RAG 시스템에서 사실적 수준의 지식 충돌을 해결하는 최초의 프레임워크이다.

## 상세 내용

### 1. 서론 (Introduction)
LLM은 전문적, 프라이버시 민감, 시간 민감 지식을 효과적으로 처리하지 못한다. RAG는 외부 지식 검색을 통합하여 이를 보완하지만, 동적 외부 소스와 정적 파라메트릭 지식 간의 시간적 격차가 지식 충돌을 불가피하게 초래한다.

기존 충돌 해결 방법의 두 가지 유형:
1. **토큰 수준 방법**: 출력 토큰의 확률 분포를 조정하여 내부/외부 지식 간 선호를 관리 (CD2, ASTUTE RAG)
2. **의미 수준 방법**: 내부/외부 소스의 지식 세그먼트를 의미적으로 통합하고 정렬 (CK-PLUG, FaithfulRAG)

이 방법들은 단편적 데이터 표현에 의존하는 조잡한 전략으로, LLM이 복잡한 상호의존성과 세밀한 사실적 불일치를 정확히 포착하지 못하게 한다. TruthfulRAG는 구조화된 트리플 기반 지식 표현을 사용하여 이 한계를 극복한다.

### 2. 방법론 (Methodology)
TruthfulRAG는 세 가지 상호연결된 모듈로 구성된다:

#### 2.1 그래프 구축 (Graph Construction)
- 검색된 콘텐츠 C를 의미적으로 일관된 텍스트 세그먼트로 분할
- 각 세그먼트에서 구조화된 지식 트리플 T = (h, r, t)를 추출 (머리 엔티티, 관계, 꼬리 엔티티)
- 명시적 사실 진술과 암묵적 의미 관계를 모두 포착
- 집계된 트리플 세트로 지식 그래프 G = (E, R, Tall) 구축

#### 2.2 그래프 검색 (Graph Retrieval)
- 사용자 쿼리 q에서 핵심 요소(대상 엔티티, 관계, 의도 범주) 추출
- 의미 유사도 매칭으로 상위 k개의 관련 엔티티와 관계 식별
- 각 핵심 엔티티에서 2-hop 그래프 순회로 초기 추론 경로 수집
- 사실 인식 스코어링 메커니즘으로 경로 필터링:
  - Ref(p) = alpha * (엔티티 커버리지) + beta * (관계 커버리지)
- 핵심 추론 경로를 경로 시퀀스 + 엔티티 속성 + 관계 속성으로 컨텍스트화

#### 2.3 충돌 해결 (Conflict Resolution)
- 엔트로피 기반 모델 신뢰도 분석을 활용
- 두 조건 비교: (1) 순수 파라메트릭 생성 (외부 컨텍스트 없음), (2) 구조화된 추론 경로 포함 RAG
- 각 조건에서 엔트로피 계산:
  - H(P(ans|context)) = -(1/|l|) * sum of p_i * log2(p_i)
- 엔트로피 변화량으로 충돌 감지:
  - deltaH_p = H(P_aug) - H(P_param)
  - 양수값: 외부 지식이 불확실성 증가시킴 (잠재적 사실 불일치)
  - 음수값: 외부 지식이 내부 이해와 일치하여 불확실성 감소
- 임계값 tau를 초과하는 경로를 교정 경로(P_corrective)로 분류
- 교정 경로를 컨텍스트로 사용하여 최종 응답 생성

### 3. 실험 (Experiments)

#### 3.1 실험 설정
- **데이터셋**: FaithEval (논리 수준 충돌), MuSiQue (다중 홉 추론), SQuAD (사실 수준 충돌), RealtimeQA (시간적 충돌)
- **모델**: GPT-4o-mini, Qwen2.5-7B-Instruct, Mistral-7B-Instruct
- **베이스라인**: Direct Generation, Standard RAG, KRE (프롬프트 최적화), COIECD (디코딩 조작), FaithfulRAG (자기 반성)
- **평가 지표**: ACC (정확도), CPR (컨텍스트 정밀도 비율)

#### 3.2 주요 결과
- TruthfulRAG가 모든 백본 LLM에서 최고 평균 정확도와 최대 상대적 개선을 달성
- FaithEval 81.9%, MuSiQue 79.4%, RealtimeQA 85.0%에서 최고 성능
- Standard RAG 대비 3.6%~29.2% 개선
- GPT-4o-mini에서 평균 78.8%, Qwen2.5-7B에서 78.3%, Mistral-7B에서 81.3% 달성

#### 3.3 비충돌 컨텍스트에서의 성능
- MuSiQue-golden에서 93.2%(+3.3%), SQuAD-golden에서 98.3%(+0.4%)
- 충돌 해결뿐 아니라 비충돌 컨텍스트에서도 우수한 성능 유지
- KRE 같은 방법이 비충돌 시나리오에서 성능이 크게 저하되는 것과 대조적

#### 3.4 구조화된 추론 경로의 영향
- 자연어 컨텍스트 대비 구조화된 추론 경로가 정답에 대한 모델 신뢰도를 일관되게 향상
- 모든 데이터셋에서 더 높은 logprob 값을 보여, LLM이 구조화된 지식 표현으로 추론할 때 외부 지식에 대한 신뢰가 증가함을 입증

#### 3.5 절제 연구 (Ablation Study)
- **지식 그래프 제거 시**: 정확도는 소폭 향상되나 CPR이 현저히 하락 -- LLM이 자연스러운 컨텍스트에서 관련 정보를 효과적으로 추출하기 어려움을 시사
- **충돌 해결 제거 시**: CPR은 크게 향상되나 풍부한 구조화 지식이 중복 정보를 동시에 도입하여 정확도 개선이 제한적
- 두 모듈이 시너지적으로 기능하여 사실 정확도와 컨텍스트 정밀도를 모두 향상

### 4. 관련 연구 (Related Work)

#### 4.1 지식 충돌의 영향 분석
- Longpre et al.: 엔티티 기반 충돌에서 LLM이 파라메트릭 메모리에 의존하는 경향
- Chen et al.: 검색 기반 LLM이 높은 리콜에서 비파라메트릭 증거에 의존하지만 신뢰 점수가 불일치를 반영하지 못함
- Xie et al.: 지지/충돌 정보가 동시에 제시될 때 강한 확인 편향
- Tan et al.: 검색된 것보다 자체 생성 컨텍스트에 대한 체계적 편향

#### 4.2 지식 충돌 해결 방법
- **토큰 수준**: CD2(주의 가중치 조작), ASTUTE RAG(그래디언트 기반 귀인) -- 정밀하지만 계산 오버헤드와 의미 인식 부족
- **의미 수준**: CK-PLUG(어댑터 기반), FaithfulRAG(자기 반성) -- 표면적 충돌만 다루고 기저 사실 관계 미포착
- TruthfulRAG는 구조화된 트리플 기반 표현으로 사실 수준 충돌을 정밀하게 식별하고 해결

### 5. 추가 실험 (Additional Experiments)
- **하이퍼파라미터 견고성**: 통일 임계값(tau=1) 사용 시에도 모델별 임계값과 유사한 성능, 세밀한 튜닝에 의존하지 않음
- **통계적 유의성**: 10회 독립 실행에서 FaithfulRAG 대비 4개 데이터셋 모두 p<0.05
- **SOTA LLM 평가**: Gemini-2.5-Flash와 Qwen2.5-72B에서도 일관된 개선
- **계산 비용**: FaithfulRAG 대비 중등도의 계산 오버헤드이지만, 실용적 효율성과 컴팩트한 컨텍스트 표현 유지

### 6. 결론 (Conclusion)
TruthfulRAG는 지식 그래프를 활용하여 RAG 시스템의 사실 수준 충돌을 해결하는 최초의 프레임워크이다. 체계적 트리플 추출, 쿼리 인식 그래프 검색, 엔트로피 기반 필터링을 통합하여, 비구조화 컨텍스트를 구조화된 추론 경로로 변환하고, LLM의 외부 지식에 대한 신뢰도를 향상시키며 사실적 불일치를 효과적으로 완화한다. 지식 집약적 응용에서의 신뢰성과 정확성 향상에 중요한 시사점을 가진다.

## 핵심 키 포인트
1. **사실 수준 충돌 해결**: 기존 토큰/의미 수준이 아닌 사실 수준에서 지식 충돌을 해결하는 최초의 프레임워크이다.
2. **지식 그래프 활용**: 비구조화 텍스트를 구조화된 트리플로 변환하여, LLM이 복잡한 상호의존성을 정확히 포착할 수 있게 한다.
3. **엔트로피 기반 충돌 감지**: 파라메트릭 생성과 RAG 생성 간의 엔트로피 변화를 측정하여 충돌을 정밀하게 탐지한다.
4. **구조화된 추론 경로의 신뢰도 향상**: 자연어보다 구조화된 경로가 LLM의 외부 지식 신뢰도를 일관되게 높인다.
5. **비충돌 시나리오에서도 견고**: 충돌 해결뿐 아니라 비충돌 컨텍스트에서도 우수한 성능을 유지하여 범용 적용 가능하다.
6. **세 모듈의 시너지**: 그래프 구축, 그래프 검색, 충돌 해결이 시너지적으로 작동하여 정확도와 컨텍스트 정밀도를 모두 향상시킨다.
7. **하이퍼파라미터 견고성**: 통일 임계값에서도 성능이 유지되어 세밀한 튜닝 없이도 효과적이다.

## 주요 인용 (Key Quotes)

> "A critical challenge for RAG systems is resolving conflicts between retrieved external information and LLMs' internal knowledge, which can significantly compromise the accuracy and reliability of generated content." (Abstract, p.1)

> "Existing approaches to conflict resolution typically operate at the token or semantic level, often leading to fragmented and partial understanding of factual discrepancies between LLMs' knowledge and context, particularly in knowledge-intensive tasks." (Abstract, p.1)

> "We discover that constructing contexts through textual representations on structured triples can enhance the confidence of LLMs in external knowledge, thereby promoting trustworthy and reliable model reasoning." (Contributions, p.2)

> "Positive values of delta_H_p indicate that the retrieved external knowledge intensifies uncertainty in the LLM's reasoning, potentially indicating factual inconsistencies with its parametric knowledge, whereas negative values suggest that the retrieved knowledge aligns with the LLM's internal understanding." (Conflict Resolution, p.4)

> "Structured reasoning paths consistently lead to higher logprob values for correct answers compared to natural language contexts, indicating greater model confidence when reasoning with structured knowledge representations." (Results, p.7)

> "TruthfulRAG not only excels at resolving conflicting information but also maintains superior performance in non-conflicting contexts, thereby revealing its universal applicability and effectiveness." (Results, p.7)

## 시사점 및 의의
TruthfulRAG는 RAG 시스템의 핵심 취약점인 지식 충돌 문제에 대한 체계적이고 효과적인 해결책을 제시한다. AgentOps 관점에서 이 연구의 시사점은 다음과 같다: (1) RAG 기반 에이전트가 외부 지식을 활용할 때 지식 충돌은 불가피하며, 이를 사실 수준에서 해결하는 메커니즘이 필수적이다. (2) 지식 그래프 기반의 구조화된 표현이 자연어 기반 표현보다 LLM의 외부 지식 신뢰도를 높이므로, 에이전트 아키텍처에 지식 그래프 통합을 고려해야 한다. (3) 엔트로피 기반 불확실성 측정이 충돌 탐지의 효과적인 도구가 될 수 있어, 에이전트의 자체 모니터링과 품질 보증에 활용 가능하다. 다만, 지식 그래프 구축과 엔트로피 계산에 따른 추가 계산 비용은 실시간 응용에서의 트레이드오프로 고려되어야 한다.
