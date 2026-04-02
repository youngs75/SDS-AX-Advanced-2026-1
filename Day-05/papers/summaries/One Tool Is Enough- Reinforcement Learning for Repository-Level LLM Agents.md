# One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents

## 1. 기본 정보 (Basic Information)

| 항목 | 내용 |
|------|------|
| **제목** | One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents |
| **저자** | Zhaoxi Zhang, Yitong Duan, Yanzhi Zhang, Yiming Xu, Weikang Li, Jiahui Liang, Deguo Xia, Jizhou Huang, Jiyan He, Yunfang Wu |
| **소속** | Peking University, Zhongguancun Academy, Baidu Inc. |
| **출판 정보** | arXiv:2512.20957v4, 2026년 1월 8일 (Submitted to ICML 2026) |
| **페이지 수** | 21페이지 (본문 + 부록 + 사례 연구) |
| **키워드** | RepoNavigator, Issue Localization, Single Tool, Jump Tool, GRPO, Reinforcement Learning, SWE-bench, Language Server |

---

## 2. 한 줄 요약 (One-line Summary)

단일 "jump" 도구만을 사용하는 LLM 에이전트 RepoNavigator를 제안하여, 강화학습(GRPO)으로 사전학습 모델에서 직접 훈련함으로써 7B 모델이 14B 베이스라인을, 14B 모델이 32B 경쟁 모델을 능가하는 SOTA 레포지토리 수준 이슈 위치 추적 성능을 달성하며, 다중 도구 파이프라인 대비 단일 도구의 우월성을 이론적, 실험적으로 입증한다.

---

## 3. 초록 요약 (Abstract Summary)

대규모 오픈소스 소프트웨어(OSS) 레포지토리에서 수정이 필요한 파일과 함수를 찾는 것은 코드의 규모와 구조적 복잡성으로 인해 매우 어려운 과제이다. 기존 LLM 기반 방법들은 이를 레포지토리 수준 검색 태스크로 취급하며 여러 보조 도구에 의존하는데, 이는 코드 실행 로직을 간과하고 모델 제어를 복잡하게 만든다.

이 논문은 **RepoNavigator**를 제안한다. 핵심 설계 원리:
- **단일 도구(jump)**: 호출된 심볼의 정의로 점프하는 실행 인식 도구 하나만 사용
- **강화학습 직접 훈련**: 사전학습 모델에서 GRPO로 직접 훈련하며, 비공개 모델 증류(distillation) 없음
- **SOTA 성능**: 7B 모델이 14B 베이스라인을 능가하고, 14B 모델이 32B 경쟁 모델을 초과하며, 32B 모델은 Claude-3.7 같은 비공개 모델까지 능가

---

## 4. 상세 내용 정리 (Detailed Content Summary)

### Section 1: Introduction (서론)

LLM의 급속한 발전에도 불구하고, 대규모 OSS 레포지토리에서의 작업 능력은 여전히 제한적이다. SWE-BENCH가 현재 이 영역의 가장 포괄적인 벤치마크로 기능하고 있다.

기존 에이전트들의 핵심 한계를 세 가지로 정리한다:

1. **사전학습-에이전틱 상호작용 불일치**: 대부분의 LLM은 텍스트에 대해 사전학습되었으며, 도구 호출과 같은 에이전틱 상호작용 패턴에 노출되지 않았다. 퓨샷 프롬프팅만으로는 복잡한 다단계 도구 체이닝 행동을 학습하기 불충분하다.

2. **다중 도구 파이프라인의 복잡성**: 기존 위치 추적 에이전트(LocAgent, CoSIL, RepoSearcher 등)는 SearchClass, SearchMethods, GetImports 등 여러 도구에 의존한다. 이 도구들은 클래스, 함수 등의 고수준 추상화를 고려하지만, 코드가 실제로 실행되는 방식을 반영하지 않는다.

3. **비공개 모델 증류 의존**: RepoSearcher와 LocAgent 같은 기존 훈련된 에이전트들은 Claude-3.7-Sonnet 같은 비공개 모델로부터의 증류(distillation)에 의존한다.

RepoNavigator의 세 가지 핵심 기여:
1. 사전학습 모델에서 RL로 직접 훈련하는 최초의 레포 수준 위치 추적 에이전트 (비공개 모델 증류 불필요)
2. 실제 실행 의미론에 맞춘 점프(jump) 연산으로 레포지토리를 탐색하는 에이전트 설계
3. 다중 도구 파이프라인 대비 단일 통합 도구의 효율성과 제어 가능성 우위 입증

### Section 2: Related Works (관련 연구)

#### 2.1 Agentic Training (에이전틱 훈련)

LLM 에이전트의 도구 사용 훈련 방법론을 검토한다:
- **SFT 기반**: 더 강력한 LLM이 생성한 궤적으로 학생 모델 훈련 (교사 모델 필요)
- **RFT 기반**: 에이전트 자체의 다중 롤아웃에서 생성된 궤적 활용
- **RLVR(Agentic RL)**: 결과만으로 궤적을 검증하는 온폴리시 방법. 검색 엔진, Python 실행기, 계산기 등에서 우수한 결과

#### 2.2 Software Engineering Agents (소프트웨어 엔지니어링 에이전트)

SWE-bench 관련 에이전트 파이프라인들을 검토한다:
- **프레임워크 기반**: SWE-AGENT, OpenHands
- **워크플로우 기반**: Agentless (위치 추적 -> 수리 -> 검증 분해)
- **그래프 기반**: LocAgent (코드베이스를 이종 그래프로 구축), CoSIL (콜 그래프 동적 구성)
- **RL 훈련**: DeepSWE, SWE-Swiss

기존 에이전트들의 공통 한계: 레포지토리 내 구조적 관계(모듈, 클래스, 함수 간 교차 참조)를 간과하며, 다중 검색 도구에 의존하여 오류 전파를 증폭시킨다.

### Section 3: Method (방법론)

#### 3.1 Problem Formulation (문제 정의)

레포지토리 R = {f1, ..., fN}과 이슈 설명 q가 주어졌을 때, 관련 코드 영역 Y* = {(fi, gi,j)}를 출력하는 것이 목표이다. 각 시간 단계 t에서 에이전트는 추론 단계 rt, 도구 호출 at, 관찰 ot를 생성하여 궤적 tau를 형성한다. 목적 함수: max_theta E[R(tau)].

#### 3.2 Agent Architecture (에이전트 아키텍처)

단일 도구 설계로 다중 도구 조율 오버헤드를 회피한다. 각 단계에서 정책 pi_theta는 추론을 계속하거나 JSON 형식 도구 호출을 발행할지 결정한다. 루프는 "reason -> act -> observe" 패턴을 따른다.

#### 3.3 Jump: Symbol Resolution (점프: 심볼 해석)

**핵심 도구**인 jump는 언어 서버(Pyright)를 활용하여 Python 심볼의 정의를 결정론적 정적 분석으로 해석한다. 해석 과정:

1. **구문 분석(Syntactic Analysis)**: 소스 파일을 추상 구문 트리(AST)로 파싱. 심볼의 구문적 역할(이름, 속성 접근, 호출 표현식 등)에 따라 해석 전략 결정.

2. **렉시컬 스코프 해석(Lexical Scope Resolution)**: 이름 심볼 x에 대해 Python의 LEGB 규칙을 따라 스코프 체인 S = {local, enclosing, module, builtins}을 따라 후보 정의 검색.

3. **정적 타입 추론(Static Type Inference)**: 속성 심볼에 대해 타입 주석, 할당 흐름 분석, 함수 반환 타입, 스텁 파일(.pyi)을 사용하여 리시버 표현식의 타입을 추론하고 MRO(Method Resolution Order)에 따라 멤버 해석.

4. **임포트 의존성 그래프(Import Dependency Graph)**: 교차 파일 해석을 위해 Python의 모듈 로딩 의미론을 정적으로 에뮬레이트하는 임포트 의존성 그래프 구축. 재내보내기(re-exports)와 __all__ 기반 필터링 포함.

#### 3.4 Reasoning-Action Loop (추론-행동 루프)

이력 ht = (q, o1:t-1, a1:t-1)이 주어지면, 에이전트는 자연어 추론 단계 rt 또는 구조화된 도구 호출 at를 샘플링한다. 도구 호출은 제한된 디코딩(constrained decoding)을 통해 JSON 문법을 준수한다. 에이전트가 최종 위치 Y-hat을 예측할 때까지 루프가 계속된다.

#### 3.5 Reinforcement Learning (강화학습)

사전학습 모델에서 직접 검증 가능한 보상으로 RL 훈련한다. GRPO(Group Reference Policy Optimization)를 적용:

```
L_GRPO(theta) = E[(pi_theta(at|st) / pi_theta_old(at|st)) * A_hat - beta * D_KL(pi_theta_old || pi_theta)]
```

보상 함수는 두 부분으로 구성:
```
R(Y-hat, Y*, tau) = DICE(Y-hat, Y*) + S(tau)
```

- **DICE**: 예측 집합과 정답 집합의 집합 수준 유사도
- **S(tau)**: 궤적에서 추출한 도구 호출 성공률 (형식 오류, 존재하지 않는 심볼 등 실패 페널티)

### Section 4: Experiment (실험)

#### 4.1 실험 설정

- **훈련 데이터**: SWE-smith에서 필터링한 4,000개 샘플
- **검증**: SWE-bench Verified (인간 검증 데이터셋)
- **추가 테스트**: SWE-bench Pro (더 어려운 벤치마크, 일반화 평가)
- **기본 모델**: Qwen2.5-Instruct 시리즈 (7B, 14B, 32B)
- **메트릭**: Sample-F1과 IoU를 핵심 메트릭으로 사용 (recall/precision 단독은 공정한 비교 불가)
- **훈련 환경**: 7B는 8x A100-80G, 14B/32B는 16x A100-80G, 1 epoch, 배치 크기 128, 8회 롤아웃

#### 4.2 효과성 (Effectiveness)

SWE-bench Verified 결과 (핵심 수치):

| 모델 | 방법 | Function IoU | File IoU |
|------|------|-------------|----------|
| Qwen2.5-7B | RepoNavigator+GRPO | **26.43** | **32.30** |
| Qwen2.5-7B | RepoSearcher(Distill+GRPO) | 17.59 | 18.23 |
| Qwen2.5-14B | RepoNavigator+GRPO | **30.08** | **32.30** |
| Qwen2.5-14B | 모든 Training-Free 방법 중 최고 | 20.98(Agentless) | 22.08(Agentless) |
| Qwen2.5-32B | RepoNavigator+GRPO | **37.19** | **37.19** |
| Claude-3.7-Sonnet | RepoSearcher | 17.89 | 20.67 |

핵심 발견:
- **GRPO 훈련된 7B 모델이 14B 베이스라인을 능가**
- **14B 모델이 32B 경쟁 모델을 능가**
- **32B 모델이 Claude-3.7 기반 RepoSearcher를 능가**
- RepoSearcher(Claude-3.7 증류 + GRPO)보다 모든 메트릭에서 우수 (recall 제외)
- Training-free 상태에서도 14B 이상 모델은 기존 다중 도구 방법 대비 경쟁력 있는 성능

SWE-bench Pro (일반화 평가): SWE-bench Verified와 일관된 결과를 보여 일반화 능력 입증.

#### 4.3 훈련 전략 비교 (Training Strategy Comparison)

GRPO vs RFT-only vs RFT+GRPO 비교:
- **GRPO 직접 훈련이 RFT-only와 RFT+GRPO를 모두 능가**
- RFT 단계가 길어질수록 후속 GRPO의 개선 폭 감소
- 사전학습 모델이 충분히 강하지 않을 때만 SFT/RFT cold start가 효과적
- **하이브리드 보상(도구 호출 성공률 포함)이 순수 결과 보상보다 우수** -- 올바른 도구 호출 학습이 에이전틱 학습에 핵심

#### 4.4 도구 호출의 스케일링 법칙 (Scaling Law of Tool-Calling)

최대 도구 호출 턴 수를 변화시킨 실험:
- RL 훈련 전후 모두 도구 호출 턴 수가 증가할수록 성능 일관적으로 향상
- **도구 호출의 스케일링 법칙**을 실험적으로 검증

#### 4.5 이슈 해결에 대한 영향 (Influence on Issue Resolution)

Agentless의 수리(repair) 단계에 각 위치 추적 방법의 결과를 입력한 실험:

| 방법 | Function IoU(%) | Resolved(%) |
|------|----------------|-------------|
| Agentless | 5.28 | 10.12 |
| LocAgent | 2.65 | 13.01 |
| RepoNavigator | 12.00 | 14.74 |
| RepoNavigator+RL | 14.58 | **15.03** |

RepoNavigator의 위치 추적 결과가 최종 이슈 해결 성능을 가장 높게 향상시킴.

### Section 5: Discussion - Building Less yet More Capable Tools (더 적지만 더 유능한 도구 구축)

단일 도구 설계의 이론적 근거를 네 가지 관점에서 분석한다:

#### 5.1 행동 공간에 대한 영향

단일 도구를 유지하면 행동 공간과 관찰 공간이 해당 도구가 접근할 수 있는 범위로 제한된다. 추가 도구는 LLM이 사전학습에서 노출되지 않은 새로운 인터페이스를 도입하여 오류 가능성을 증가시킨다.

#### 5.2 도구 호출 성공률에 대한 영향

k개의 순차적 도구 호출이 필요한 태스크의 전체 성공률:
```
P_succ(k) = prod(pi, i=1..k)
```
각 단계가 추가적 실패 지점을 도입하므로, 하나의 다목적 도구로 태스크를 완료하는 것이 여러 좁은 범위 도구를 순차적으로 사용하는 것보다 더 신뢰성이 높다.

#### 5.3 예측 공간에 대한 영향

jump 도구의 접근 범위(access scope)는 진입점에서 재귀적으로 참조된 심볼을 해석하여 도달 가능한 모든 정의의 집합이다. 이 접근 범위는 전체 레포지토리 범위보다 크게 작으면서도, 이슈 해결에 필요한 모든 위치를 포함한다. 따라서 jump 도구의 IoU가 다중 도구(전체 레포지토리 범위)보다 높다.

#### 5.4 검증 (Verification)

도구 세트를 변경한 실험 (Qwen2.5-7B-Instruct 기준):

| Jump | GetClass | GetFunc | GetStruc | Function IoU |
|------|----------|---------|----------|-------------|
| O | O | O | O | 13.71 |
| O | O | O | X | 21.44 |
| O | X | X | O | 24.00 |
| O | X | X | X | **24.28** |

**추가 도구가 성능을 향상시키지 않으며 오히려 저하시킨다.** Jump 도구만 사용했을 때 가장 높은 성능을 기록한다.

### Section 6: Conclusion (결론)

RepoNavigator는 기존 다중 도구 패러다임에서 벗어나 단일 jump 도구로 심볼 해석을 수행하는 레포지토리 수준 이슈 위치 추적 에이전트이다. 도구 통합 GRPO를 통해 추론, 도구 호출, 예측 개선을 폐쇄 루프(closed-loop)로 학습하며, 비공개 교사 모델이나 증류 없이 엔드투엔드 최적화를 달성한다. 단일 강력한 도구가 강화학습과 결합되었을 때, 여러 좁은 범위 도구에 의존하는 기존 프레임워크보다 더 강건하고 신뢰성 높은 다단계 추론을 제공함을 이론적 분석과 실험으로 확인했다.

### Appendix (부록)

부록에는 베이스라인 에이전트들의 상세 설명, 도구 세트 비교 (Table 5: CoSIL 3개, LocAgent 3개, Orcaloca 5개, RepoSearcher 5개 도구 vs RepoNavigator 1개 도구), 실험 세부사항(하이퍼파라미터, 메트릭 정의, 데이터 누출 방지 조치), 그리고 astropy-12907 이슈에 대한 전체 궤적 사례 연구가 포함되어 있다.

사례 연구에서 RepoNavigator는 다음과 같은 과정을 통해 정확한 위치를 찾아낸다:
1. 진입점(test_separable.py)에서 separability_matrix 심볼 점프
2. _separable 함수로 점프하여 CompoundModel 처리 로직 분석
3. _operators 사전으로 점프하여 연산자-함수 매핑 확인
4. _cstack 함수로 점프하여 중첩 모델 조합 로직 분석
5. _coord_matrix 함수로 점프하여 최종 분석
6. _cstack이 수정 대상임을 정확히 식별 (F1: 1.0, IoU: 1.0)

---

## 5. 핵심 포인트 (Key Points)

1. **단일 도구 설계의 우월성**: 다중 도구(3-5개) 파이프라인 대비, 실행 로직에 기반한 단일 jump 도구가 행동 공간 축소, 오류 전파 감소, 접근 범위 최적화를 통해 더 높은 성능을 달성한다. 추가 도구(GetClass, GetFunc, GetStruc)는 오히려 성능을 저하시킨다.

2. **비공개 모델 증류 없는 RL 직접 훈련**: RepoSearcher가 Claude-3.7-Sonnet 증류를 cold start로 사용하는 것과 달리, RepoNavigator는 사전학습 모델에서 GRPO로 직접 훈련한다. GRPO 직접 훈련이 RFT-only와 RFT+GRPO를 모두 능가한다.

3. **모델 크기 대비 탁월한 성능**: 7B 모델이 14B 베이스라인을, 14B 모델이 32B 경쟁 모델을 능가하는 결과는 도구 설계와 훈련 방법의 중요성을 모델 크기보다 강조한다.

4. **도구 호출 스케일링 법칙**: 도구 호출 턴 수 증가에 따른 일관적 성능 향상은 도구 호출의 스케일링 법칙을 실험적으로 입증한다.

5. **하이브리드 보상의 효과**: 결과 보상(DICE)만 사용하는 것보다 도구 호출 성공률(S(tau))을 포함하는 하이브리드 보상이 더 높은 성능을 달성하며, 올바른 도구 호출 학습이 에이전틱 RL에서 핵심적임을 보여준다.

6. **실행 의미론 기반 도구 설계**: jump 도구는 컴파일 후 사라지는 고수준 추상화(클래스, 상속)가 아닌, 실제 코드 실행 흐름(순차 실행 + 점프)을 반영한다. 언어 서버(Pyright)를 활용한 결정론적 정적 분석으로 구현된다.

7. **위치 추적에서 이슈 해결까지**: RepoNavigator의 정확한 위치 추적 결과가 Agentless의 수리 단계와 결합되었을 때 최종 이슈 해결률을 가장 높게 향상시킴(15.03%)을 확인한다.

---

## 6. 핵심 인용구 (Key Quotes)

1. **"We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool -- jumping to the definition of a invoked symbol."**
   - (Abstract) -- 단일 실행 인식 도구를 사용하는 RepoNavigator의 핵심 설계를 소개한다.

2. **"integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization."**
   - (Abstract) -- 단일 도구와 RL 훈련의 결합이 효율적이고 확장 가능한 해결책임을 주장한다.

3. **"building less tools with more powerful and more ensembled functions is more effective than building multiple task-specific tools"**
   - (Section 5) -- 논문의 핵심 철학. 더 적지만 더 강력한 도구가 다수의 태스크 특화 도구보다 효과적이라는 원칙을 밝힌다.

4. **"High-level abstractions, such as classes or inheritance, disappear after compilation, leaving only sequential execution and jump operations."**
   - (Section 1, Introduction) -- 고수준 추상화가 컴파일 후 사라지고 순차 실행과 점프만 남는다는 통찰로, jump 도구 설계의 이론적 근거를 제공한다.

5. **"Since each step introduces an additional potential point of failure, the cumulative success rate typically decreases as the number of required tool calls increases."**
   - (Section 5.2) -- 다중 도구 사용 시 누적 성공률이 감소하는 이론적 근거를 제시한다. 순차적 도구 호출의 성공률 곱셈 효과를 설명.

6. **"the access scope produced by exhaustive jump traversal is guaranteed to contain all locations that must be modified to resolve the issue."**
   - (Section 5.3) -- jump 도구의 접근 범위가 이슈 해결에 필요한 모든 위치를 포함한다는 이론적 보장을 제시한다.

7. **"directly training with GRPO outperforms RFT-only and RFT+GRPO. Moreover, although RFT has acceptable performance, the more steps RFT proceeds, the less improvement GRPO makes after the cold start."**
   - (Section 4.3) -- GRPO 직접 훈련의 우월성과, RFT cold start가 오히려 후속 GRPO 개선을 제한한다는 반직관적 발견을 보고한다.

8. **"we are the first fully-automatic LLM agent, with no fixed workflow and no planetary prompt, and we are the first method trained directly from pretrained open-source LLMs without a close-source teacher model."**
   - (Appendix A) -- 고정된 워크플로우나 사전 정의된 프롬프트 없이 완전 자동으로 작동하며, 비공개 교사 모델 없이 오픈소스 LLM에서 직접 훈련된 최초의 방법임을 강조한다.

---

## 7. 의의 및 시사점 (Significance and Implications)

### 학술적 의의

RepoNavigator는 **LLM 에이전트의 도구 설계에 대한 근본적 재고**를 촉구한다. 기존의 "더 많은 도구 = 더 높은 성능"이라는 암묵적 가정에 도전하여, 단일 도구가 다중 도구보다 우월할 수 있음을 이론적(행동 공간 축소, 누적 오류 감소, 접근 범위 최적화)으로 분석하고 실험적으로 입증했다. 이는 에이전트 도구 설계의 패러다임 전환을 시사한다.

또한, 사전학습 모델에서 GRPO로 직접 훈련하여 비공개 모델 증류 없이 SOTA를 달성한 것은, 오픈소스 LLM 기반 에이전트 훈련의 새로운 가능성을 열어준다. RFT cold start가 오히려 후속 RL 개선을 제한한다는 발견은 에이전틱 RL 분야의 중요한 통찰이다.

### 실무적 시사점

1. **도구 설계 원칙**: 실무자들은 에이전트에 여러 보조 도구를 추가하기보다, 실행 로직에 근거한 소수의 강력한 도구를 설계하는 것이 더 효과적일 수 있다. "building less tools with more powerful and more ensembled functions"이라는 원칙은 에이전트 시스템 설계 전반에 적용 가능하다.

2. **소규모 모델의 경쟁력**: 7B 모델이 14B 베이스라인을 능가하는 결과는, 적절한 도구 설계와 RL 훈련을 통해 소규모 모델로도 대규모 모델에 필적하거나 초과하는 성능을 달성할 수 있음을 보여준다. 이는 배포 비용 절감에 직접적으로 기여한다.

3. **이슈 위치 추적의 중요성**: 이슈 위치를 정확히 파악하면 해결이 크게 쉬워진다는 점이 확인되었으며(Table 3), 위치 추적을 수리와 분리하는 파이프라인 설계의 유효성을 지지한다.

### 한계 및 미래 전망

현재 Python 레포지토리에만 평가되었다는 점이 주요 한계이다. 각 프로그래밍 언어는 고유한 언어 서버를 가지므로, C/C++, Java 등으로의 확장은 추가적인 언어 서버 구현을 필요로 한다. 또한 monkey patching이나 동적 임포트가 정적 분석 기반 도구의 성능을 저하시킬 수 있으나, 평가 데이터셋에서는 이러한 사례가 관찰되지 않았다. 향후 다양한 프로그래밍 언어로의 확장과 위치 추적을 넘어선 전체 이슈 해결 파이프라인에서의 RL 훈련이 유망한 연구 방향이다.
