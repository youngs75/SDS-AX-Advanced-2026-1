# Sparse and Dense Retrievers Learn Better Together: Joint Sparse-Dense Optimization for Text-Image Retrieval

## 기본 정보
- **저자**: Jonghyun Song (서울대학교), Youngjune Lee (NAVER), Gyu-Hwung Cho (NAVER), Ilhyeon Song (NAVER), Saehun Kim (NAVER), Yohan Jo (서울대학교, 교신저자)
- **발행일/출처**: 2025년 8월 22일, CIKM '25 (34th ACM International Conference on Information and Knowledge Management), arXiv:2508.16707v1 [cs.CL]
- **페이지 수**: 5페이지
- **키워드**: Learned Sparse Retrieval, Cross-modal Retrieval, Vision-Language Pre-training, Self-Knowledge Distillation, Text-Image Retrieval

## 한줄 요약
> 이 논문은 텍스트-이미지 검색에서 희소(sparse)와 밀집(dense) 표현을 통합 유사도 점수 기반의 양방향 자기 지식 증류(Self-Knowledge Distillation)를 통해 공동 최적화하여, 희소 검색기가 기존 희소 베이스라인을 능가하고 밀집 모델과 동등하거나 더 나은 성능을 달성하면서도 희소 모델의 해석 가능성과 효율성을 유지하는 간결하면서도 효과적인 프레임워크를 제안한다.

## 초록 (Abstract)
Vision-Language Pretrained(VLP) 모델은 밀집 표현 기반으로 텍스트-이미지 검색을 포함한 다중모달 작업에서 인상적인 성능을 달성했다. 한편, 학습된 희소 검색(Learned Sparse Retrieval, LSR)은 텍스트 전용 환경에서 역색인을 통한 빠른 용어 기반 조회의 해석 가능성과 효율성으로 주목받고 있다. 최근 연구가 LSR을 다중모달 영역으로 확장했지만, 계산 비용이 큰 대조적 사전학습에 의존하거나 동결된 밀집 모델로부터의 증류에 의존하여 상호 향상의 잠재력이 제한된다. 이러한 한계를 해결하기 위해, 밀집 및 희소 유사도의 가중합인 통합 유사도 점수를 공유 교사 신호로 사용하여 양방향 학습을 가능케 하는 자기 지식 증류 프레임워크를 제안한다. 효율성을 위해 밀집 인코더의 최종 레이어와 희소 프로젝션 헤드만 파인튜닝한다. MSCOCO와 Flickr30k 실험에서 우리의 희소 검색기가 기존 희소 베이스라인을 능가할 뿐 아니라 밀집 모델과 동등하거나 그 이상의 성능을 달성함을 보인다.

## 상세 내용

### 1. 서론 (Introduction)
CLIP 등 VLP 모델은 이미지와 텍스트를 공유 밀집 임베딩 공간에 투영하여 교차 모달 검색에 널리 채택되고 있다. 한편, 텍스트 검색 분야에서 학습된 희소 검색(SPLADE 등)이 사전학습 언어 모델의 밀집 표현을 어휘 공간 위의 희소 표현으로 변환하여, 명시적 어휘 특징을 보존하면서 역색인 인프라를 활용한 빠른 조회를 가능케 한다.

최근 이 아이디어를 교차 모달 환경으로 확장한 연구(VisualSparta, LexLIP, D2S)가 있지만 두 가지 핵심 한계가 존재한다:
1. **비용 문제**: 초기 접근법(VisualSparta, LexLIP)은 수백만 이미지-텍스트 쌍에서의 end-to-end 사전학습에 의존하여 계산 비용이 높다.
2. **단방향 증류의 한계**: D2S 같은 증류 기반 접근법은 동결된 밀집 인코더로부터 희소 프로젝션 헤드를 학습하지만, 동결된 밀집 표현은 이미 밀집 임베딩에 존재하는 정보만 표현하도록 제한하며, 희소 표현이 밀집 표현을 개선할 수 있는 보완적 신호가 활용되지 않는다.

**제안**: 양방향 자기 지식 증류를 통해 밀집과 희소 표현을 공동 최적화하는 간결하면서도 매우 효과적인 프레임워크를 제안한다. 밀집 및 희소 유사도의 가중합으로 계산되는 통합 유사도 점수를 교사 신호로 사용하여 양방향 증류를 수행한다. 효율성과 효과의 균형을 위해 희소 프로젝션 헤드와 밀집 인코더의 최종 레이어만 업데이트한다.

### 2. 방법 (Methods)

**2.1 작업 정의**: N개의 이미지-텍스트 쌍으로 구성된 다중모달 데이터셋에서, 텍스트 캡션을 쿼리로 사용하여 대응하는 이미지를 대규모 후보 풀에서 검색하는 텍스트-이미지 검색에 초점을 맞춘다.

**2.2 아키텍처**: 텍스트와 이미지를 모달리티별 인코더로 밀집 [CLS] 임베딩(h_t, h_i)으로 인코딩한다. 동일한 희소 프로젝션 헤드 f(MLP)가 이를 어휘 공간의 용어 중요도(z_t, z_i)로 변환하며, ReLU와 로그 변환을 적용한다. 프로젝션 헤드는 동결된 텍스트 인코더의 전치 단어 임베딩 행렬로 초기화하여 의미적으로 유의미한 토큰 수준 점수를 생성하도록 유도한다. 효율성을 위해 백본 VLP 인코더의 대부분 레이어를 동결하고, 밀집 인코더의 최종 레이어와 희소 프로젝션 헤드만 파인튜닝한다.

**2.3 자기 지식 증류(Self-Knowledge Distillation)**: 핵심 기술적 기여이다.

- **세 가지 유사도 점수**: 밀집 점수(s_dense), 희소 점수(s_sparse), 통합 점수(s_inter = w1*s_dense + w2*s_sparse)
- **대조적 손실**: InfoNCE 손실을 세 가지 점수 유형 모두에 대해 양방향(텍스트->이미지, 이미지->텍스트)으로 적용. 전체 대조적 손실은 각 점수별 손실의 가중합: L = lambda1*L_dense + lambda2*L_sparse + lambda3*L_inter
- **증류 손실**: 통합 점수 s_inter를 교사 신호로 사용하여 밀집과 희소 점수 모두를 감독. 교차 엔트로피 기반: L' = (L_distill(s_inter, s_dense) + L_distill(s_inter, s_sparse)) / 2
- **최종 학습 목표**: L_final = L + L' + eta_t * L1(z_t) + eta_i * L1(z_i), 여기서 L1은 희소성 정규화

이 통합 점수는 양쪽 표현의 보완적 강점을 동적으로 반영하는 소프트 교사 신호로, 밀집과 희소 표현이 개별 특성을 잃지 않으면서 서로로부터 학습할 수 있게 한다.

### 3. 실험 (Experiments)

**3.1 실험 설정**:
- 데이터: MSCOCO(113.2k/5k/5k), Flickr30k(29.8k/1k/1k), Karpathy 분할 사용
- 구현: BLIP과 ALBEF 체크포인트에서 시작. 200 에폭 학습, RTX 3090 GPU 1장에서 약 4시간
- 메트릭: Recall@{1, 5}, MRR@10

**3.2 결과**:

**희소 검색 베이스라인과의 비교 (Table 1)**:
- BLIP 기반 모델이 모든 희소 베이스라인을 일관되게 능가
- MSCOCO: R@1 57.6 (vs D2S BLIP 55.9), Flickr30k: R@1 82.0 (vs D2S BLIP 81.3)
- 같은 비전-텍스트 인코더에서 학습 시 양방향 자기 지식 증류가 단방향 증류보다 효과적임을 입증
- 모든 향상이 통계적으로 유의미 (paired t-test, p < 0.05)

**밀집 검색 베이스라인과의 비교 (Table 2)**:
- 희소 검색기(Ours Sparse)가 밀집 모델과 동등하거나 종종 능가
- 밀집 검색 적용 시(Ours Dense)도 원본 백본 대비 성능 향상: BLIP 기반 MSCOCO R@1 57.0 -> 58.7, R@5 82.0 -> 82.9
- 희소 표현 학습이 밀집 표현의 품질도 동시에 개선

**3.3 분석**:

**소거 연구 (Table 3, ALBEF 기반)**:
- 자기 지식 증류 제거(대조적 손실만 사용) 시 성능 하락: MSCOCO R@1 53.2 -> 51.9, Flickr30k R@1 78.6 -> 77.5
- 최종 레이어 동결 시 성능 하락: MSCOCO R@1 53.2 -> 52.1
- 두 구성요소 모두 결합 시 최고 성능, 밀집-희소 간 상호 학습에 밀집 임베딩의 적응이 중요

**효과-효율 트레이드오프 (Figure 2 Left)**:
- PEC(Probabilistic Expansion Control) 적용 시 FLOPs가 크게 감소하면서 경쟁력 있는 R@1 성능 유지
- 응용 제약에 따라 정확도 우선(PEC 미적용) 또는 효율 우선(PEC 적용) 선택 가능

**밀집-희소 가중치 분석 (Figure 2 Right)**:
- 희소 점수에 높은 가중치(w2) 부여 시 일반적으로 더 나은 R@1 성능
- 큰 w1 값은 성능 저하 경향
- 희소 점수 자체가 효과적인 학습 신호로 작용할 수 있음을 시사 -- 기존 연구가 간과한 측면

### 4. 결론 (Conclusion)
통합 유사도 점수를 통한 자기 지식 증류로 밀집과 희소 표현을 공동 최적화하는 희소 표현 학습 프레임워크를 제안했다. 희소 검색기가 표준 교차 모달 검색 벤치마크에서 최첨단 성능을 달성하여, 종종 완전 파인튜닝된 밀집 모델을 능가한다. 또한 같은 학습 전략이 밀집 모델의 성능도 원본 백본 이상으로 향상시켜 프레임워크의 일반적 적용 가능성을 보여준다.

## 핵심 키 포인트
1. **양방향 자기 지식 증류**: 통합 유사도 점수(밀집+희소의 가중합)를 교사 신호로 사용하여, 기존의 단방향(밀집->희소) 증류의 한계를 극복하고 양방향 학습을 실현한다.
2. **효율적 학습**: 최종 레이어와 희소 프로젝션 헤드만 파인튜닝하여, RTX 3090 한 장에서 약 4시간 만에 학습이 완료되는 경량 프레임워크이다.
3. **상호 향상 효과**: 희소 표현 학습이 밀집 표현의 성능도 동시에 향상시키는 "상호 향상(mutual enhancement)" 효과를 처음으로 입증했다.
4. **희소 모델의 밀집 모델 대등/능가**: 해석 가능성과 역색인 효율성을 유지하면서도 밀집 모델과 동등하거나 더 나은 검색 성능을 달성한다.
5. **희소 점수의 독립적 가치**: 가중치 분석에서 희소 점수에 높은 가중치를 부여할수록 성능이 향상되어, 희소 점수 자체가 효과적인 학습 신호임을 보여준다.

## 주요 인용 (Key Quotes)
> "Most existing methods rely on unidirectional distillation -- dense to sparse -- thus underutilizing their potential synergy." (Section 1, p.1)

> "This integrated score serves as a soft teacher signal that dynamically reflects the complementary strengths of both representations. It supervises both dense and sparse scores in the self-knowledge distillation process, allowing them to learn from each other without losing their individual characteristics." (Section 2.3, p.2)

> "A frozen dense encoder restricts the sparse projection head to expressing only the information already present in the dense embeddings. Furthermore, although sparse representations could provide complementary signals to refine dense ones, such feedback is rarely exploited." (Section 1, p.1)

> "Our sparse retriever not only outperforms existing sparse baselines, but also achieves performance comparable to -- or even surpassing -- its dense counterparts, while retaining the benefits of sparse models." (Abstract, p.1)

> "These results demonstrate that our method effectively transforms knowledge from pretrained dense encoders into sparse representations, while preserving competitive performance of dense models." (Section 3.2, p.3)

> "These results suggest that sparse scores can serve as effective learning signals in their own right -- an aspect that has been largely overlooked in prior work, which depends on dense-to-sparse distillation." (Section 3.3, p.4)

## 시사점 및 의의
이 논문은 정보 검색 분야에서 희소-밀집 표현의 관계를 재정의하는 중요한 기여를 한다.

첫째, **양방향 학습의 가치**를 실증적으로 입증했다. 기존 연구가 밀집에서 희소로의 단방향 지식 전달에만 집중했던 것과 달리, 희소 표현이 밀집 표현의 품질을 향상시킬 수 있다는 발견은 패러다임의 전환을 시사한다. 이는 텍스트 검색뿐 아니라 다양한 모달리티의 표현 학습에 적용 가능한 범용적 통찰이다.

둘째, **실용성이 매우 높다**. RTX 3090 한 장에서 4시간이면 학습이 완료되고, 기존 VLP 모델에 쉽게 적용 가능하며, 역색인 인프라를 활용한 빠른 검색이 가능하다. 특히 대규모 이미지 데이터베이스를 다루는 실제 서비스 환경에서 밀집 검색 대비 높은 효율성을 유지하면서 경쟁력 있는 정확도를 제공할 수 있다.

셋째, 서울대학교와 NAVER의 공동 연구로서, **한국 산학협력의 우수한 성과**를 보여준다. 간결하면서도 효과적인 방법론 설계와 철저한 소거 연구, 효율-효과 트레이드오프 분석 등 학술적 엄밀성과 실용적 관점을 모두 갖추고 있다.
