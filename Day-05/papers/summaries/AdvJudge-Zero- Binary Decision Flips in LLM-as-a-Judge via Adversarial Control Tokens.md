# AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens

## 기본 정보
- **저자**: Tung-Ling Li, Yuhao Wu, Hongliang Liu (Palo Alto Networks)
- **발행일/출처**: arXiv:2512.17375v1, 2025년 12월 19일 (Preprint)
- **페이지 수**: 41페이지
- **키워드**: LLM-as-a-Judge, Adversarial Control Tokens, Reward Hacking, Binary Decision Flipping, Post-training Vulnerability, RLHF, DPO, RLAIF

## 한줄 요약
> 이 논문은 LLM-as-a-Judge 시스템에서 짧은 제어 토큰 시퀀스가 이진 평가 결정(Yes/No)을 뒤집을 수 있는 취약점을 발견하고, 시드 패턴 없이 자동으로 이러한 토큰을 찾아내는 AdvJudge-Zero 방법을 제안하며, LoRA 기반 적대적 학습으로 이를 완화할 수 있음을 보여준다.

## 초록 (Abstract)
보상 모델과 LLM-as-a-Judge 시스템은 RLHF, DPO, RLAIF와 같은 현대 후처리(post-training) 파이프라인에서 핵심적인 역할을 한다. 이 논문은 이러한 평가 시스템이 반복적으로 나타나는 취약점을 가지고 있음을 보여준다: 낮은 퍼플렉시티의 짧은 제어 토큰 시퀀스가 많은 이진 평가를 올바른 "No" 판단에서 잘못된 "Yes" 판단으로 뒤집을 수 있다. 이는 마지막 레이어의 로짓 갭 F = z_no - z_yes를 조종함으로써 이루어진다. AdvJudge-Zero라는 방법은 모델 자체의 다음 토큰 분포와 빔 서치 탐색을 사용하여 다양한 제어 토큰 시퀀스를 처음부터(zero-seed) 발견한다. 실험적으로 이러한 토큰은 여러 오픈 웨이트 모델 패밀리(Qwen, Llama, Gemma)와 전문 평가 모델에서 매우 높은 거짓 양성률(false positive rate)을 유발하며, LoRA 기반 적대적 학습이 이러한 거짓 양성을 크게 줄이면서도 평가 품질을 유지할 수 있음을 보여준다.

## 상세 내용

### 1. 서론 (Introduction)
후처리(post-training)는 대규모 언어 모델을 유용한 어시스턴트로 변환하는 주요 방법이 되었다. 일반적인 파이프라인은 SFT, RLHF, DPO/RLAIF 등을 결합하며, 보상 모델과 LLM-as-a-Judge 시스템이 대부분의 평가 신호를 제공한다. 저자들은 이 평가 시스템의 이진 결정이 마지막 레이어 리드아웃의 좁은 고-이득 표면(narrow, high-gain surface)에 집중되어 있다는 점을 관찰하였다.

이 논문의 핵심 개념은 "이진 평가 뒤집기(binary evaluation flipping)"이다. 소규모의 신중하게 설계된 프롬프트 편집이 모델의 마지막 레이어 로짓 갭 F = z_no - z_yes를 0을 가로질러 이동시켜, 모델의 평가를 "No"에서 "Yes"로 전환시킨다. 기존 연구가 수동으로 큐레이션된 시드 패턴에 의존하는 것과 달리, AdvJudge-Zero는 모델 자체의 다음 토큰 분포를 사용하여 처음부터 다양한 제어 토큰 시퀀스를 발견한다.

저자들의 해석은 기하학적이다. 트랜스포머 본체가 입력 프롬프트를 최종 은닉 상태 h_L(X)로 매핑하고, 후처리가 리드아웃에 "게이트"를 설치하여 "No"와 "Yes" 응답을 분리한다. 이 게이트가 얕고(shallow) 높은 이득(high-gain)을 가지기 때문에, 작은 로컬 이동만으로도 다양한 입력에 대해 F의 부호를 바꿀 수 있다.

이 논문의 초점은 최악의 경우의 임의 문자열이 아니라, 후처리 과정에서 자연스럽게 나타나는 취약점에 있다. 포맷팅 마커나 구조적 구분자 같은 낮은 퍼플렉시티 토큰이 체계적으로 평가를 편향시킬 수 있으며, 정책 모델이 답변 품질이 아닌 보상을 위해 이런 패턴을 생산하게 수렴할 수 있다.

### 2. 배경 및 관련 연구 (Background and Related Work)
- **후처리 및 선호도 최적화**: RLHF, DPO, RLAIF가 사전 훈련된 모델을 적응시킨다. Master-RM, Weaver와 같은 보상 집계 방법과 GRPO, RLVR 같은 RL 변형이 초기 토큰을 스코어링하는 얕은 헤드를 사용한다.
- **보상 해킹과 평가자 견고성**: 보상 과최적화는 정책이 평가자의 편향(아첨, 장황함)을 악용하게 한다. ODIN, 적대적 학습, 제약된 RL 등의 완화책이 있다. 이 논문은 정책 샘플링 중 자연스럽게 나타나는 낮은 퍼플렉시티 제어 토큰이 후기 레이어 로짓 갭 조종을 통해 체계적으로 평가자를 편향시킴을 특성화한다.
- 기존의 GCG, PAIR 같은 그래디언트 기반 공격이 높은 퍼플렉시티 시퀀스를 생성하는 것과 달리, 이 연구는 자연스러운 RL 학습 역학에 관련된 모델 고유의 취약점을 타겟으로 한다.

### 3. 방법론 (Methods)

#### 3.1 설정 및 핵심 개념
- **프롬프트 구조**: 질문, 모델 생성 응답, 참조 답안을 고정 평가 템플릿으로 감싸는 이진 평가 태스크를 고려한다.
- **로짓 갭과 뒤집기**: 프롬프트 X에 대해 F(X) = logit("No") - logit("Yes")를 정의하고, F(X) < 0일 때 뒤집기가 발생한다.
- **AdvJudge-Zero**: 시드 패턴이나 사전 가정 없이 처음부터 발견을 시작한다. 길이 1~n의 짧은 제어 토큰 시퀀스 A를 찾아 뒤집기 확률을 최대화한다.

#### 3.2 기하학적 조종의 경험적 관찰
결정 경계가 마지막 레이어 은닉 상태에 대한 선형 리드아웃으로 구현된다:
F(h) = z_No(h) - z_Yes(h) ≈ (w_No - w_Yes)^T h + b

여기서 w_F = w_No - w_Yes가 고유한 "거부 방향(Refusal Direction)"을 나타낸다. 성공적인 공격은 ∆h가 w_F와 상당한 음의 투영(반정렬)을 가져야 한다.

**저순위 조종 가설(Low-Rank Steering Hypothesis)**: 효과적인 제어 토큰은 등방성으로 h를 교란하지 않고, 모델 내 공유된 저순위 "소프트 모드" 방향 u를 활용한다. PCA 분석 결과, 첫 번째 주성분(PC1)이 분산의 28-35%를 설명하며(등방성 랜덤 교란의 경우 약 0.03%), Z-점수가 -7.47(Qwen)과 -4.80(Llama)으로 제어 토큰이 랜덤 노이즈가 아니라 모델의 거부 경계에 체계적으로 대항하는 방향 성분을 주입함을 강력히 시사한다.

#### 3.3 AdvJudge-Zero: 제어 토큰 발견 알고리즘
- **생성 단계**: 모델 자체의 다음 토큰 분포를 사용하여 후보 시퀀스를 제안한다. TOP_K_SCHEDULE로 빔 크기를 제어하여 첫 토큰에 큰 k(예: 300)를 사용하고 점차 줄인다.
- **검증 단계**: 각 후보 시퀀스를 완전한 평가 프롬프트에 삽입하고 로짓 갭을 계산한다. 뒤집기를 유발하는 후보를 기록한다.
- **통계 집계**: 복제 횟수(duplication count)와 평균 로짓 갭으로 순위를 매긴다.

#### 3.4 제어 토큰 선택
발견된 대규모 풀에서 안정적이고 대표적인 소규모 세트를 선택한다:
- **복제 횟수**: 여러 프롬프트에서 일반화되는 토큰 선별
- **평균 No-Yes 로짓 갭**: 더 부정적인 값이 "Yes" 방향으로 더 강한 견인력을 나타냄

### 4. 실험 (Experiments)

#### 4.1 실험 설정
- **범용 모델**: LLaMA-3.2/3.3 (3B, 70B), Qwen2.5/3 (4B, 7B, 30B), Gemma-3 (4B) 등 6개 모델
- **전문 평가 모델**: Omni-Judge, Qwen2.5-7B-Instruct-RLVR, general-verifier, Master-RM 등 4개 모델
- **데이터셋**: AIME (수학 경시), MATH (경시 수학), Multi-subject RLVR (다과목 추론), GSM8K (초등 수학)
- 각 모델-데이터셋 쌍에 대해 길이 1~7의 제어 토큰 시퀀스를 생성하고 최대 10개의 상위 시퀀스를 선택

#### 4.2 주요 결과
- **높은 거짓 양성률(FPR)**: 발견된 제어 토큰이 여러 모델과 데이터셋에서 매우 높은 FPR을 유발한다. 토큰 길이가 증가할수록 FPR이 대체로 증가하는 경향을 보인다.
- **모델 간 일반화**: Qwen, Llama, Gemma 모델 패밀리 전반에서 제어 토큰이 효과적으로 작동한다.
- **토큰 유형 특성화**: 특수 문자, 마크다운, 포맷팅 토큰, 모델 고유 특수 토큰(예: `<|start_header_id|>`, `<|im_end|>`) 등 다양한 유형이 발견되었다.

#### 4.3 적대적 학습을 통한 완화
소규모 제어 토큰 증강 데이터셋에 대한 LoRA 기반 적대적 학습이 FPR을 크게 줄이면서도 평가 능력을 유지할 수 있음을 보여준다.

### 5. 발견된 제어 토큰 패턴 분석 (Appendix F)
각 모델별로 발견된 효과적인 제어 토큰을 상세히 보고한다:
- **Gemma-3-4b-it**: `<details>`, `**Final Answer:**`, `| Step |`, 마크다운 테이블 형식 등이 효과적
- **Llama-3.2-3B**: `<|start_header_id|>assistant<|end_header_id|>`, `ablytyped`, `scalablytyped` 같은 모델 특수 토큰과 의미 없는 문자열이 효과적
- **Llama-3.3-70B**: `oreferrer`, `HeadersHeightSizeMode`, 기타 코드 관련 토큰이 효과적
- **Qwen2.5-7B**: `/Instruction`, `<|im_end|>`, `user`, PHP 코드 시작 문자열 등이 효과적
- **Qwen3 모델들**: `Certainly,`, `Alright,`, `Indeed,` 같은 긍정적 시작어, JSON 구조 토큰, `<|im_start|>assistant` 등이 효과적

### 6. 프롬프트 템플릿 (Appendix B-E)
다양한 평가 모델용 프롬프트 템플릿을 제공한다:
- 일반 LLM Judge 템플릿: 질문, 풀이 과정, 참조 답안을 포함하며 YES/NO만 출력
- General-Verifier 템플릿: 단계별 추론 후 Final Decision 출력
- Omni-Judge 템플릿: 학생 답안 추출, 동치 판단, 정당화 포함
- Master-RM 템플릿: 최종 답안만 비교하여 YES/NO 출력

### 7. 한계 및 향후 연구 (Limitations and Future Work)
정확성 스타일 태스크와 거짓 양성 뒤집기에 초점을 맞추며, 유해 콘텐츠 유도나 안전 필터 우회를 시도하지 않는다. 공격 프롬프트나 제어 토큰 시퀀스를 악의적으로 사용될 수 있는 형태로 공개하지 않는다.

## 핵심 키 포인트
1. LLM-as-a-Judge 시스템의 이진 결정이 마지막 레이어의 얕은 고-이득 게이트에 의존하며, 이것이 체계적 취약점을 만든다.
2. AdvJudge-Zero는 시드 패턴 없이 모델 자체의 토큰 분포를 활용하여 제어 토큰을 자동 발견하는 최초의 방법이다.
3. 제어 토큰에 의한 은닉 상태 교란은 저순위 구조를 가지며, 모델의 거부 방향과 체계적으로 반정렬된다(Z-score: -7.47, -4.80).
4. 낮은 퍼플렉시티의 포맷팅/구조 토큰이 정책 모델이 실제로 생성할 수 있는 현실적인 보상 해킹 위험을 나타낸다.
5. Qwen, Llama, Gemma 등 다양한 모델 패밀리와 전문 평가 모델 전반에서 취약점이 일반화된다.
6. 발견된 제어 토큰 유형에는 특수 문자, 마크다운 포맷팅, 모델 특수 토큰, JSON 구조 등이 포함된다.
7. LoRA 기반 적대적 학습이 소규모 증강 데이터만으로도 FPR을 효과적으로 줄일 수 있다.
8. 이 취약점은 RLHF/DPO/RLAIF 파이프라인에서 보상 해킹의 현실적 위험을 나타낸다.

## 주요 인용 (Key Quotes)
> "short sequences of low-perplexity control tokens can flip many binary evaluations from correct 'No' judgments to incorrect 'Yes' judgments by steering the last-layer logit gap F = z_no - z_yes." (Abstract, p.1)

> "These control tokens are patterns that a policy model could plausibly generate during post-training, and thus represent realistic reward-hacking risks rather than worst-case adversarial strings." (Abstract, p.1)

> "the induced hidden-state perturbations concentrate in a low-rank 'soft mode' that is anti-aligned with the judge's refusal direction." (Abstract, p.1)

> "Post-training then installs a gate at the readout that separates 'No'-like and 'Yes'-like responses, in line with observations that safety alignment often acts only on the first few tokens." (Section 1, p.2)

> "Because this gate is shallow and high-gain, these local moves are often sufficient to change the sign of F for a wide range of inputs." (Section 1, p.2)

> "The first principal component (PC1) explains a substantial portion of the variance (28%-35%), consistent with a strong low-rank structure. For comparison, an isotropic random perturbation in R^d would allocate on the order of 1/d of the variance to any single direction; for d ~ 3000-4000, this is about 0.03%." (Section 3.2, p.4)

> "These Z-scores provide strong evidence that the control tokens do not act as random noise but inject a directional component that systematically opposes the model's refusal boundary." (Section 3.2, p.4)

> "In RLHF/DPO/RLAIF pipelines, low-perplexity tokens such as formatting markers or structural delimiters can systematically bias judge evaluations. When this happens, policies may converge to producing such patterns for reward rather than for answer quality." (Section 1, p.2)

## 시사점 및 의의
이 논문은 현대 LLM 후처리 파이프라인의 핵심 구성요소인 LLM-as-a-Judge 시스템의 근본적인 구조적 취약점을 밝혀냈다는 점에서 매우 중요하다. 특히 다음과 같은 시사점이 있다:

1. **보상 해킹의 현실적 위험**: 이론적 공격이 아닌, 정책 모델이 실제로 학습 과정에서 생산할 수 있는 낮은 퍼플렉시티 토큰이 평가를 조작할 수 있음을 보여줌으로써, RLHF/DPO/RLAIF 파이프라인의 실제 배포 시 보안 우려를 제기한다.

2. **기하학적 해석의 가치**: 마지막 레이어의 선형 리드아웃이라는 단순한 구조가 취약점의 원인임을 기하학적으로 설명함으로써, 근본적인 아키텍처 수준의 개선 방향을 제시한다.

3. **방어 가능성**: LoRA 기반 적대적 학습이 효과적인 완화책이 될 수 있음을 보여줌으로써, 실용적인 방어 전략을 제공한다.

4. **생태계 전반의 영향**: Qwen, Llama, Gemma 등 주요 오픈 웨이트 모델 패밀리 전반에서 취약점이 확인되어, 이 문제가 특정 모델이 아닌 현재의 후처리 패러다임 자체의 구조적 한계임을 시사한다.

5. **AI 안전성 연구에 대한 기여**: 제어 토큰 발견을 자동화하는 AdvJudge-Zero 방법은 향후 더 견고한 평가 시스템 설계를 위한 레드팀 도구로 활용될 수 있다.
