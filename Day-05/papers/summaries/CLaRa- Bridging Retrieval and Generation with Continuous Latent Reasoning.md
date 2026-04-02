# CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning

## 기본 정보
- **저자**: Jie He (Apple, University of Edinburgh), Richard He Bai, Sinead Williamson, Jeff Z. Pan, Navdeep Jaitly, Yizhe Zhang (Apple)
- **발행일/출처**: 2025년 11월 27일, arXiv:2511.18659v2 [cs.CL]
- **페이지 수**: 41페이지 (본문 14페이지 + 부록 27페이지)
- **키워드**: Retrieval-Augmented Generation (RAG), 문서 압축, 연속 잠재 추론, 공유 잠재 공간, 미분 가능 top-k, end-to-end 최적화, QA 벤치마크

## 한줄 요약
> 이 논문은 RAG 시스템에서 검색(retrieval)과 생성(generation)이 별도로 최적화되는 근본적 문제를 해결하기 위해, 문서를 압축된 연속 표현으로 인코딩하고 공유된 잠재 공간에서 검색과 생성을 통합적으로 end-to-end 학습하는 CLaRa 프레임워크를 제안한다.

## 초록 (Abstract)
검색 증강 생성(RAG)은 대형 언어 모델(LLM)을 외부 지식으로 강화하지만, 긴 컨텍스트와 검색-생성의 분리 최적화 문제에 여전히 시달린다. 본 연구에서는 임베딩 기반 압축과 공유된 연속 공간에서의 공동 최적화를 수행하는 통합 프레임워크인 CLaRa(Continuous Latent Reasoning)를 제안한다. 의미적으로 풍부하고 검색 가능한 압축 벡터를 얻기 위해, QA와 패러프레이즈 감독을 활용하는 키 보존 데이터 합성 프레임워크 SCP를 도입한다. CLaRa는 미분 가능한 top-k 추정기를 사용하여 재순위기와 생성기를 단일 언어 모델링 손실로 end-to-end 학습한다. 다수의 QA 벤치마크 실험에서 CLaRa가 최첨단 압축 및 재순위 성능을 달성하며, 텍스트 기반 파인튜닝 베이스라인을 종종 능가함을 보인다.

## 상세 내용

### 1. 서론 (Introduction)
RAG는 환각(hallucination)과 지식 노후화 같은 LLM의 핵심 약점을 완화하는 강력한 패러다임이지만, 대부분의 RAG 시스템은 근본적인 구조적 문제를 안고 있다: **검색과 생성이 별도로 최적화**된다는 것이다.

이 분리 설계는 두 가지 상호 연결된 도전을 초래한다:
1. **최적화 문제**: 문서 선택이 이산적(discrete)이므로, 생성기에서 검색기로 그래디언트가 역전파될 수 없어 공동 학습이 불가능하다.
2. **효율성 문제**: 밀집 검색기는 임베딩 공간에서 문서를 순위매기지만 생성기는 여전히 원시 텍스트를 소비하여, 아키텍처 불일치가 발생한다. 이는 일관되지 않은 표현 공간, 중복 텍스트 처리로 인한 추론 비용 증가, 인코딩 중복 등을 야기한다.

**핵심 통찰**: 문서를 한 번만 압축된 메모리 토큰 표현으로 인코딩하여 검색과 생성 모두에 활용하는 **공유된 연속 표현(Shared Continuous Representations)** 접근법을 제안한다. 연속 표현은 Straight-Through 추정을 통한 미분 가능 top-k 선택을 가능케 하여, 비효율적인 RL 샘플링 대신 그래디언트 하강법으로 검색기를 직접 업데이트할 수 있다.

### 2. SCP: 핵심 정보 보존 압축기 사전학습 (Salient Compressor Pretraining)
기존 방법들은 토큰 수준 재구성 손실로 문서 표현을 학습하지만, 이는 사소한 토큰별 재구성에 제한된 용량을 낭비할 수 있다. SCP는 의미적으로 중요한 정보를 추상화하고 소화하는 것에 초점을 맞춘다.

**2.1 의미 보존을 위한 가이드 데이터 합성**
200만 개의 Wikipedia-2021 문서에서 Qwen-32B를 사용하여 세 가지 보완적 감독 형태를 생성한다:
- **Simple QA**: 각 쌍이 단일 사실을 포착하여 세밀한 사실 보존을 장려
- **Complex QA**: 여러 사실을 통합하여 관계적 추론과 상위 수준 추상화를 촉진
- **Paraphrase**: 문장 구조를 재배열하여 표면 형태를 변경하면서 핵심 의미를 보존

QA 쌍은 사실 중심 감독(어떤 세부사항이 중요한지)을, 패러프레이즈는 표현 수준 압축(같은 내용을 더 효율적으로 재구성하는 방법)을 제공한다. 검증 및 재생성 과정을 통해 최대 10라운드까지 누락된 정보를 보충한다.

**2.2 압축기 사전학습**
PISCO를 따라 여러 LoRA 어댑터가 장착된 공유 기본 모델을 채택한다. 문서에 학습 가능한 메모리 토큰을 추가하고 압축기 LoRA만 활성화하여 압축 표현을 얻는다. 교차 엔트로피 손실(LCE)과 평균 제곱 오차 손실(LMSE)을 결합하여 압축 표현이 원본 문서의 의미를 충실히 반영하도록 한다.

### 3. CLaRa: 검색과 생성의 공동 학습 (Joint Training)
**프레임워크 개요**: 각 문서를 사전학습된 압축기로 밀집 임베딩으로 압축하고, 압축기는 동결(frozen)하여 오프라인 문서 인코딩을 허용한다. 쿼리 추론기(Query Reasoner)를 학습하여 쿼리를 문서와 동일한 공간에 표현한다. 다음 토큰 예측(NTP) 학습을 통해, 쿼리 추론기는 쿼리 의도를 인코딩할 뿐만 아니라 관련 문서 내용을 예측하는 법을 학습한다.

코사인 유사도로 쿼리와 문서 간 관련성을 판단하고, top-k 문서 압축 임베딩을 쿼리와 연결하여 생성기에 입력한다. 통합 언어 모델링 손실로 쿼리 추론기와 생성기를 동시에 업데이트한다.

**미분 가능 Top-k 선택**: 이산적 top-k 선택의 "끊어진 그래디언트" 문제를 Straight-Through (ST) 추정기로 해결한다. 순전파에서는 이산적 행동을 유지하면서 역전파에서는 부드러운 그래디언트 피드백을 허용하는 "소프트 렌즈" 역할을 한다.

**이론적 정당화**: 그래디언트 결합 분석을 통해 검색기가 두 가지 보완적 학습 신호를 받음을 증명한다:
1. p(d|x)와 p(y|x,d) 사이의 확률적 정렬을 통한 올바른 문서 순위 보상
2. 생성기 추론을 촉진하는 방식으로 문서를 표현하도록 하는 표현 수준 피드백

**사례 연구 - 쿼리 추론기 분석**: Logit Lens 기법으로 쿼리 추론기의 임베딩을 분석한 결과, 원래 쿼리에 없는 "NFL", "Oklahoma" 같은 토큰이 인코딩되어 있음을 발견했다. 이는 end-to-end 최적화가 쿼리 추론기에 추론 관련 지식을 암묵적으로 인코딩하게 함을 보여준다.

### 4. 실험 (Experiments)
**실험 설정**: NQ, HotpotQA, MuSiQue, 2WikiMultihopQA 등 4개 QA 벤치마크에서 Mistral-7B와 Phi-4B로 평가했다.

**4.2 압축 효과 평가** (Table 2):
- SCP가 모든 베이스라인을 일관되게 능가. 최고 소프트 압축 모델 PISCO 대비 평균 1.13%(Normal), 5.35%(Oracle) 향상
- 하드 압축 베이스라인 LLMLingual-2 대비 5.37%, 17.31% 향상
- 놀랍게도, 비압축 문서를 사용하는 텍스트 기반 BGE 검색 베이스라인도 능가 (Mistral-7B에서 평균 2.36%, Phi-4-mini에서 6.36% 향상)

**4.3 공동 학습 결과** (Table 3):
- Normal 설정에서 16-32배 압축률에서 최고 성능. CLaRa-Mistral-7B(16x)가 텍스트 기반 DRO-Mistral-7B를 능가 (NQ: 51.01->51.41 F1, 2Wiki: 43.65->47.18 F1)
- Oracle 설정에서 F1이 NQ, HotpotQA 모두 75% 초과
- 지시 튜닝 초기화가 Normal에서, 사전학습 초기화가 검색에서 각각 더 우수

**4.4 검색 성능**:
- 사전학습 초기화 CLaRa가 완전 감독 Sup-Instruct까지 능가
- HotpotQA(압축비 4)에서 Recall@5 96.21% 달성, 최강 감독 베이스라인 BGE-Reranker(85.93%) 대비 +10.28%
- 약한 생성 감독만으로 완전 감독 모델에 필적하거나 능가하는 검색 품질 달성

### 5. 소거 연구 (Ablation Study)
**사전학습 데이터 혼합**: SimpleQA나 Paraphrase 단독도 사전학습 없는 베이스라인을 능가. 복수 QA 유형 결합(SimpleQA+ComplexQA+Para)이 최고 성능 달성, 다양한 목적이 의미적 범위와 일반화를 강화함을 확인했다.

**MSE 손실 효과**: 개선은 평균 0.3-0.6포인트로 소폭이나 일관적. t-SNE 시각화로 MSE 손실이 압축 임베딩과 원본 문서 표현의 강한 중첩을 유도함을 확인했다.

### 6. 관련 연구 (Related Work)
**임베딩 기반/소프트 압축**: AutoCompressor, XRAG, PISCO 등 기존 연구들은 컨텍스트를 연속 표현으로 단축하지만, LLM과 독립적으로 학습되어 검색-생성 공동 최적화를 지원하지 않는다. 가장 관련된 연구인 Oscar는 쿼리 인식 압축 모델을 공동 학습하지만, 쿼리별 재압축이 필요하여 재사용 가능한 쿼리 독립 표현의 목표와 모순된다.

**검색-생성 End-to-End 최적화**: 강화학습 접근법은 불안정하고 계산 비용이 높으며, 미분 가능 재순위(Gumbel-softmax)는 매 단계 전체 문서를 처리하여 표현 불일치와 컨텍스트 길이 문제가 해결되지 않는다. CLaRa는 압축과 공동 학습을 독특하게 결합하여 이 문제들을 해결한다.

### 7. 결론 (Conclusion)
문서를 고품질 암묵적 표현으로 압축하여 RAG 시스템의 성능을 향상시키는 과제를 다루었다. 다양한 사전학습 목적(QA 쌍, 패러프레이즈)을 설계하여 압축기가 필수 의미 정보를 보존하도록 하고, 재순위와 생성 단계에서 문서 표현을 통합하는 효율적인 end-to-end 학습 프레임워크를 도입했다. 다수 QA 벤치마크 실험에서 임베딩 기반 컨텍스트 압축이 입력 길이와 계산 비용을 줄일 뿐만 아니라 검색과 생성 사이의 격차를 메워 더 통합적이고 의미적으로 일관된 RAG 패러다임을 가능케 함을 입증했다.

### 한계점 (Limitations)
- **압축기 일반화**: Wikipedia 데이터로만 사전학습되어, 도메인 적응 사전학습과 더 다양한 코퍼스 활용 필요
- **압축 표현에 대한 추론**: 다중 홉 또는 계획 기반 RAG 시스템에서 효율적 추론 메모리로 기능할 수 있는지 조사 필요
- **모델 규모**: 중간 규모 모델(Mistral-7B, Phi-4B)만 실험. 더 큰 모델이 더 높은 품질의 문서 표현을 생성할 수 있는지 미지수
- **암묵적 표현의 일반화**: 도구 학습 등 더 광범위한 과제로의 확장 가능성

## 핵심 키 포인트
1. **공유 연속 표현**: 문서를 한 번만 압축된 메모리 토큰 표현으로 인코딩하여 검색과 생성 모두에 활용함으로써, 기존 RAG의 이중 인코딩 비효율과 표현 공간 불일치를 해결한다.
2. **SCP 데이터 합성**: Simple QA, Complex QA, Paraphrase 세 가지 보완적 감독 신호를 합성하여 압축기가 사소한 토큰 재구성이 아닌 핵심 의미 정보를 보존하도록 학습시킨다.
3. **미분 가능 Top-k**: Straight-Through 추정기를 통해 이산적 문서 선택을 미분 가능하게 만들어, 생성기의 그래디언트가 검색기로 직접 역전파되는 진정한 end-to-end 학습을 실현한다.
4. **약한 감독의 강력함**: 명시적 관련성 레이블 없이 다음 토큰 예측 손실만으로 학습함에도, 완전 감독 검색 모델(BGE-Reranker)을 +10.28% 능가하는 검색 성능을 달성한다.
5. **압축이 오히려 성능 향상**: 잘 학습된 소프트 압축이 비압축 원시 텍스트 기반보다 더 나은 성능을 보이며, 이는 압축 표현이 불필요한 콘텐츠를 필터링하여 추론 관련 컨텍스트에 집중시키기 때문이다.
6. **쿼리 추론기의 암묵적 추론**: Logit Lens 분석에서 쿼리 추론기가 원래 쿼리에 없는 추론 관련 키워드(예: "NFL", "Oklahoma")를 암묵적으로 인코딩함이 발견되어, end-to-end 최적화가 추론 능력을 부여함을 보여준다.

## 주요 인용 (Key Quotes)
> "Most RAG systems suffer from a fundamental structural issue: retrieval and generation are optimized separately. Retrievers select documents based on surface-level similarity, while generators produce answers without providing feedback about what information is truly needed." (Section 1, p.1)

> "Instead of maintaining separate embeddings and raw text, we encode documents once into compact memory-token representations that serve both purposes." (Section 1, p.1)

> "This mechanism allows the retriever to learn which documents truly enhance answer generation rather than relying on surface-level similarity." (Section 1, p.2)

> "Surprisingly, our model exceeds the text-based w/ BGE retrieval baseline using uncompressed documents, with average gains of 2.36% on Mistral-7B and 6.36% on Phi-4-mini. This implies that well-trained soft compression can retain essential reasoning information while substantially reducing input length." (Section 4.2, p.8)

> "Remarkably, under the pretraining-initialized setup, CLaRa even surpasses the fully supervised Sup-Instruct using ground-truth relevance labels. On HotpotQA (compression ratio 4), it achieves a Recall@5 of 96.21%, exceeding the strongest supervised baseline BGE-Reranker (85.93%) by +10.28%." (Section 4.4, p.9)

> "This finding indicates that our end-to-end optimization enables the query reasoner to implicitly encode reasoning-relevant knowledge aligned with the gold evidence, thus enhancing retrieval accuracy and semantic alignment compared to baseline systems." (Section 3, p.6)

> "Notably, our method does not rely on any supervised data of annotated document relevance labels." (Section 4.4, p.9)

> "Embedding-based contextual compression not only reduces input length and computation cost but also bridges the gap between retrieval and generation, enabling a more unified and semantically coherent RAG paradigm." (Section 7, p.11)

## 시사점 및 의의
CLaRa는 RAG 연구에서 중요한 패러다임 전환을 제시한다. 기존 RAG 시스템의 근본적 한계인 검색-생성 분리 최적화 문제를 공유된 연속 잠재 공간에서의 통합 학습으로 해결함으로써, 세 가지 중요한 시사점을 제공한다.

첫째, **"압축이 곧 성능 향상"이라는 반직관적 결과**가 매우 의미 깊다. 정보를 줄이는 것이 오히려 성능을 높인다는 발견은, 원시 텍스트에 포함된 불필요한 정보가 LLM의 추론을 방해할 수 있음을 시사하며, 향후 RAG 시스템 설계에 중요한 방향을 제시한다.

둘째, **약한 감독(다음 토큰 예측)만으로 완전 감독 검색 모델을 능가하는 결과**는 실용적으로 매우 중요하다. 관련성 레이블 데이터를 수집하는 비용과 도메인 특수성 문제를 우회하면서도 더 나은 검색 품질을 달성할 수 있다는 것은, RAG 시스템의 도메인 적용 장벽을 크게 낮춘다.

셋째, **검색과 생성의 진정한 end-to-end 최적화**를 실현했다는 점에서, 향후 더 통합적이고 효율적인 지식 증강 언어 모델 개발의 토대를 마련했다. 특히 쿼리 추론기가 암묵적 추론 능력을 획득한다는 발견은, 검색 시스템이 단순한 매칭을 넘어 추론적 검색으로 진화할 수 있음을 보여준다.
