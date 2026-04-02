# KVzap: Fast, Adaptive, and Faithful KV Cache Pruning

## 기본 정보
- **저자**: Simon Jegou, Maximilian Jeblick (NVIDIA)
- **발행일/출처**: arXiv:2601.07891v1, 2026년 1월 12일
- **페이지 수**: 20페이지
- **키워드**: KV Cache, Pruning, Transformer, LLM Inference, Compression, KVzip, Attention, Long-context, Reasoning

## 한줄 요약
> 이 논문은 KVzip의 빠르고 입력 적응적인 근사인 KVzap을 소개하며, 경량 대리 모델이 은닉 상태로부터 KV 쌍의 중요도를 예측하여 프리필링과 디코딩 모두에서 2-4배의 KV 캐시 압축을 무시할 수 있는 정확도 손실로 달성한다.

## 초록 (Abstract)
트랜스포머 기반 언어 모델의 증가하는 컨텍스트 길이는 KV 캐시를 중요한 추론 병목으로 만들었다. 많은 KV 캐시 프루닝 방법이 제안되었지만, 속도-정확도 트레이드오프 때문에 주요 추론 엔진에 아직 채택되지 않았다. KVzap은 프리필링과 디코딩 모두에서 작동하는 KVzip의 빠르고 입력 적응적인 근사이다. Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B에서 장기 컨텍스트 및 추론 태스크 전반에 걸쳐, KVzap은 무시할 수 있는 정확도 손실로 2-4배 KV 캐시 압축을 달성하며 KVpress 리더보드에서 최신 성능을 달성한다.

## 상세 내용

### 1. 서론 (Introduction)
트랜스포머 어텐션에서 각 입력 토큰은 캐시에 저장되고 자기 회귀 생성 중에 재사용되는 키-값(KV) 벡터 쌍을 생성한다. KV 캐시의 형상은 (2, L, H, T, D)이며, L은 레이어 수, H는 헤드 수, T는 시퀀스 길이, D는 키/값 차원이다. 예를 들어 bfloat16 정밀도에서 Llama1-65B의 KV 캐시는 T=128k에서 335GB의 메모리를 필요로 한다.

기존의 KV 캐시 크기 축소를 위한 아키텍처 수정:
- **H축**: GQA(Grouped Query Attention) - 4x(Llama3), 12x(GLM 4.5), 16x(Qwen3-235B) 압축
- **D축**: MLA(Multi-head Latent Attention) - 4H/9 압축
- **L축**: 슬라이딩 윈도우 어텐션(2x GPT-OSS-120B, 6x Gemma3) 또는 상태 공간 모델(8x Jamba, 4x Kimi-Linear)

주목할 점은 **T축을 따른 널리 채택된 아키텍처 변경이 없다**는 것이다. 대부분의 T축 KV 캐시 압축 시도는 임시적(ad-hoc) 프루닝 방법에 의존한다.

H2O 이후 20개 이상의 KV 캐시 프루닝 방법이 구현되었지만, vLLM, SGLang, TRT-LLM 같은 주요 추론 엔진에 통합된 것은 없다. 각 솔루션이 다음 기준 중 하나 이상을 충족하지 못하기 때문이다:
1. **빠르고 경량**: 프루닝 오버헤드가 무시할 수 있어야 함
2. **단계 비의존적**: 프리필링(긴 컨텍스트)과 디코딩(추론 태스크) 모두에 적용 가능해야 함
3. **최적화 친화적**: FlashAttention2, PagedAttention 같은 커널과 호환되어야 함
4. **충실**: 어떤 태스크에서도 최소한의 정확도 저하를 유발해야 함

### 2. 방법론 (Method)

#### 2.1 KVzip
현재 KVpress 리더보드의 최신 기술. 복사-붙여넣기(copy-and-paste) 사전 태스크를 사용하여 가장 중요한 KV 쌍에 점수를 매긴다. 확장된 프롬프트를 구성하고, 각 헤드에서 위치 i의 KV 쌍은 반복된 프롬프트에 대한 최대 어텐션 가중치로 점수가 매겨진다:

s_i = max_{j in <prompt>} a_ji

**한계점**: (1) 입력보다 두 배 긴 확장 프롬프트에서 프리필링 필요 (매우 느림), (2) 디코딩 중 사용 불가 (추론 태스크에 부적합)

#### 2.2 KVzip+
KVzip 점수에 정규화 항을 통합한 개선 버전:

s+_i = max_{j in <prompt>} a_ji * ||W_O v_i|| / ||h_j||

여기서 a_ji * W_O * v_i는 토큰 i가 잔여 스트림에 기여하는 양을 나타낸다. 이 정규화는 어텐션 가중치뿐만 아니라 값 벡터의 실제 기여도를 고려한다.

#### 2.3 KVzap
KVzip의 한계를 해결하기 위해, 레이어별 대리 모델(선형 레이어 또는 2계층 MLP)을 학습시켜 입력 은닉 상태 h에서 직접 H개의 점수 log(s+)를 예측한다. 모델은 각 시퀀스 위치 t에서 독립적으로 작동: h_t를 R^H의 점수로 매핑한다.

**학습 데이터**: Nemotron-Pretraining-Dataset-sample에서 1.2M 학습 쌍(h, log(s+))을 KV 헤드당 샘플링. 영어, 다국어, 코드, 수학 텍스트를 포함하는 다양한 데이터셋.

**퇴출 정책의 핵심 차이**: KVzip이 고정 예산(예: 상위 50% 유지)을 사용하는 반면, KVzap은 **임계값 기반 프루닝**을 사용한다. 예측 점수가 고정 임계값 tau 이하인 KV 쌍을 폐기한다. 이는 KVzap을 **입력 적응적**으로 만든다: 복잡한 입력에는 더 많은 토큰을 유지하고, 중복된 입력에는 더 적은 토큰을 유지한다.

로컬 컨텍스트를 보존하기 위해 최근 w=128 토큰의 슬라이딩 윈도우를 유지한다.

### 3. 실험 (Experiments)

#### 3.1 KVzap 학습
- **대리 모델**: KVzap-Linear(선형 모델)과 KVzap-MLP(2계층 MLP)
- **R^2 성능** (검증 세트, KVzip+ 점수 대비):

| 모델 | Linear | MLP |
|---|---|---|
| Qwen3-8B | 0.671 | 0.711 |
| Llama-3.1-8B-Instruct | 0.743 | 0.772 |
| Qwen3-32B | 0.629 | 0.668 |

두 대리 모델 모두 0.60-0.80 범위의 R^2를 달성하여, 비싼 KVzip+ 점수가 은닉 상태로부터 근사될 수 있음을 보여준다.

#### 3.2 계산 및 메모리 오버헤드
KVzap은 무시할 수 있는 오버헤드를 추가한다: 모든 모델에서 선형 투영만 고려할 때 KVzap-MLP는 1.1% 이하, KVzap-Linear는 0.02% 이하의 상대적 계산 비용. 디코딩 중에는 KVzap의 추가 FLOP이 KV 캐시 검색에 의해 정지된 유휴 GPU 사이클을 효과적으로 활용한다.

#### 3.3 프리필링 및 디코딩 태스크
세 가지 벤치마크에서 평가:
- **RULER** (n=6500): 검색, 다중 홉 추적, 집계, QA의 장기 컨텍스트 벤치마크
- **LongBench** (n=4750): 단일/다중 문서 QA, 요약, 소수 샷 학습, 합성 태스크, 코드 완성
- **AIME25** (n=30): 올림피아드 수준 수학 추론

#### 3.4 RULER 결과
RULER 4k에서 KVzap은 Qwen3-8B과 Llama-3.1-8B-Instruct 모두에서 최신 결과를 달성하여 15개의 동시대 KV 캐시 프루닝 방법을 크게 능가한다. KVzap은 3-4배 압축까지 완벽한 정확도를 유지한다.

#### 3.5 LongBench 결과
RULER 결과를 반영하여, KVzap 모델은 2-3배 압축까지 거의 완벽한 정확도를 유지한다. 동일한 임계값 tau가 더 낮은 압축 비율을 생성하는데, 이는 RULER 샘플이 합성적이고 반복적인 반면 LongBench는 정보 밀도가 더 높은 실제 데이터로 구성되기 때문이다.

#### 3.6 AIME25 결과
KVzap-MLP는 KV 캐시의 50% 이상을 폐기하면서도 추론 정확도를 유지한다. 2배 이상의 압축 비율에서도 추론 정확도가 보존된다.

#### 3.7 적응적 압축
KVzap의 임계값 기반 프루닝이 태스크에 따라 자동으로 다른 압축 비율로 변환된다. 최적 KVzap 구성:

| 모델 | 유형 | 파라미터 | 평균 압축 |
|---|---|---|---|
| Qwen3-8B | MLP | 76M | 3.5x |
| Llama-3.1-8B | Linear | 1.1M | 3.0x |
| Qwen3-32B | MLP | 210M | 2.7x |

#### 3.8 절삭 실험 (Ablations)
- **임계값 vs 고정 Top-k**: 임계값 기반 프루닝이 헤드별 또는 레이어별 고정 비율 top-k 선택보다 우수하다. 프롬프트에 따라 최대 20%의 압축 비율 변동을 보여 적응성을 입증한다.
- **슬라이딩 윈도우**: 로컬 윈도우 없이(w=0) 정확도가 28.37%로 급락. w=128에서 62.51%로 회복, w=512에서는 추가 이점 없음(62.37%).

### 4. 논의 (Discussion)

**범위 및 일반화**: 32B 모델에서의 결과는 고무적이지만, 더 큰 오픈소스 모델(GLM 4.7, Qwen3-235B)과 희소 어텐션 아키텍처(DeepSeek V3.2)에서의 추가 검증이 필요하다.

**임시적 vs 종단간 학습**: KVzap은 학습 불필요(training-free)가 아니며, 대부분의 KV 캐시 프루닝 방법처럼 사후적(post-hoc) 추가이다. 장기적으로 Multi-Token Prediction이 임시적 추측 디코딩 기법을 대체하듯이, 종단간 프루닝 목적이 더 나은 성능을 낼 수 있다.

**구현 도전**: 압축을 실제 벽시계 속도 향상과 GPU 메모리 절약으로 전환하려면 신중한 엔지니어링이 필요하며, 이 연구에서는 탐구되지 않았다. KVzap은 헤드 간 불균일한 캐시 길이를 도입하여 가변 길이 블록을 처리하는 PagedAttention 커널이 필요하다.

### 부록 주요 내용

#### A. KVzap 모델 학습 상세
- Llama-3.1-8B-Instruct의 점수 분포가 Qwen 모델들보다 유의미하게 낮아 더 낮은 프루닝 임계값이 필요하다.
- KVzap-MLP가 모든 모델에서 일관되게 KVzap-Linear보다 높은 R^2를 달성한다.
- 두 대리 모델 모두 첫 번째 트랜스포머 레이어에서 성능이 떨어져, 토큰 임베딩만으로는 KVzip+ 점수를 추론하기 어렵다.
- 키와 값(k, v)에서 직접 점수를 예측하면 은닉 상태 h에서 예측하는 것보다 R^2가 엄격히 낮다.

#### B. 계산 오버헤드 분석
GQA 설정에서 KVzap-MLP와 KVzap-Linear의 FLOP을 수식으로 유도하여 최대 1.1%(MLP), 0.02%(Linear)의 오버헤드를 확인한다.

#### C. 상세 벤치마크 결과
RULER 4k의 13개 서브셋, LongBench의 21개 서브셋, AIME25의 4개 롤아웃에 대한 상세 결과를 제공한다.

## 핵심 키 포인트
1. KV 캐시는 LLM 추론의 지배적 병목이 되었으며, T축(시퀀스 길이)을 따른 널리 채택된 압축이 아직 없다.
2. 기존 KV 캐시 프루닝 방법이 주요 추론 엔진에 채택되지 못한 이유는 속도, 단계 비의존성, 최적화 호환성, 충실성의 4가지 기준을 동시에 만족하지 못하기 때문이다.
3. KVzap은 KVzip+의 점수를 은닉 상태로부터 예측하는 경량 대리 모델(선형 또는 MLP)로, 프리필링과 디코딩 모두에서 사용 가능하다.
4. 임계값 기반 프루닝이 고정 비율 프루닝보다 우수하며, 입력의 정보 밀도에 따라 자동으로 압축 비율을 조정한다.
5. 2-4배 KV 캐시 압축을 무시할 수 있는 정확도 손실(< 1% 오버헤드)로 달성한다.
6. 슬라이딩 윈도우(w=128)가 로컬 컨텍스트 보존에 필수적이다.
7. KVzap-MLP가 일반적으로 KVzap-Linear보다 우수하지만, Llama 모델에서는 Linear가 오히려 우수한 흥미로운 결과를 보인다.

## 주요 인용 (Key Quotes)
> "Growing context lengths in transformer-based language models have made the key-value (KV) cache a critical inference bottleneck." (Abstract, p.1)

> "While many KV cache pruning methods have been proposed, they have not yet been adopted in major inference engines due to speed-accuracy trade-offs." (Abstract, p.1)

> "Notably, no widely adopted architectural change compresses the KV cache along the T-axis." (Section 1, p.2)

> "Just as a reader does not pay equal attention to every word when understanding a sentence, not all tokens are equally important, and some need not occupy HL slots in KV cache memory." (Section 1, p.2)

> "KVzap uses thresholding, discarding KV pairs whose predicted score falls below a fixed threshold tau. This makes KVzap input-adaptive: it dynamically adapts the compression rate based on the prompt information density." (Section 2.3, p.3)

> "KVzap adds negligible overhead: across all models, its relative compute cost is bounded by 1.1% for KVzap-MLP and 0.02% for KVzap-Linear." (Section 3.2, p.4)

> "Despite these challenges, we believe KVzap's combination of simplicity, high compression ratios, and robust performance across tasks and models makes it a prime candidate for production deployment." (Section 4, p.8)

## 시사점 및 의의
이 논문은 LLM 추론 효율화의 핵심 과제인 KV 캐시 압축에서 중요한 진전을 보여준다:

1. **실용적 배포 가능성**: 기존 KV 캐시 프루닝 방법이 학술적 수준에 머물렀던 것과 달리, KVzap은 빠르고(무시할 수 있는 오버헤드), 단계 비의존적(프리필링+디코딩), 충실한(최소 정확도 손실) 방법을 제공하여 실제 추론 엔진에 통합될 가능성을 열었다.

2. **입력 적응적 압축**: 고정 비율이 아닌 임계값 기반 프루닝은 입력의 정보 밀도에 따라 자동으로 압축 비율을 조정하는 우아한 해법을 제공한다. 이는 합성적이고 반복적인 데이터(RULER)에서는 높은 압축을, 정보가 밀집된 실제 데이터(LongBench)에서는 보수적인 압축을 자동으로 적용한다.

3. **대리 학습의 효과성**: 비싼 KVzip+ 점수가 은닉 상태로부터의 간단한 대리 모델로 효과적으로 근사될 수 있다는 발견은 트랜스포머의 은닉 상태에 KV 쌍의 중요도 정보가 이미 암시적으로 인코딩되어 있음을 시사한다.

4. **추론 태스크 지원**: 디코딩 중에도 사용 가능하다는 점은 수천 토큰을 생성하는 추론 태스크(예: AIME25)에서 특히 가치가 있으며, 이는 기존 KVzip의 중요한 한계를 해결한다.

5. **LLM의 KV 캐시 비효율성 증거**: KVzap이 2-4배 압축에서도 정확도 손실이 거의 없다는 것은 현재의 LLM이 KV 캐시를 완전히 활용하지 못하고 있으며, 사용되지 않는 KV 쌍이 은닉 상태에서 쉽게 식별될 수 있음을 추가로 입증한다.
