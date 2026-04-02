# Arize AI Prompt Optimization 방법론 리서치

## 1. Arize AI의 접근 방식 개요

Arize AI는 LLM 프롬프트를 체계적으로 최적화하기 위한 **7-Level Prompt Optimization Framework**을 제시합니다.
이 방법론은 단순 프롬프트 작성 가이드를 넘어, **데이터 기반의 반복적 개선 루프(Evaluation-Driven Improvement Loop)**를 핵심으로 합니다.

- 오픈소스 플랫폼: **Phoenix** (https://github.com/Arize-ai/phoenix)
- 클라우드 버전: https://app.phoenix.arize.com
- 블로그: https://arize.com/blog/prompt-optimization

---

## 2. 7-Level Prompt Optimization Framework

### Level 1: Craft Specific Instructions (구체적 지시 작성)
- 모델의 역할(role), 과제(task), 제약(constraints), 출력 형식(response format), 예시(examples) 명시
- 예: "당신은 고객 서비스 담당자입니다. 항상 한국어로, 3문장 이내로 답하십시오."

### Level 2: Account for Uncertainty (불확실성 처리)
- Chain-of-Thought (CoT) 기법 활용
- "답변 전에 단계별 추론 과정을 서술하세요" → 정확도 향상

### Level 3: Get LLM Feedback (LLM 피드백 활용)
- **Meta-Prompting**: LLM 자체에게 기존 프롬프트를 비판하고 개선안 제안
- 성공/실패 사례를 분석하여 자동으로 개선된 프롬프트 생성

### Level 4: Use A/B Tests and Experiment (A/B 테스트)
- 다양한 use case를 커버하는 dataset 구축 및 체계적 테스트
- Phoenix의 **Experiments** 기능으로 동일 데이터셋에서 여러 프롬프트 변형 비교

### Level 5: Hire LLMs as Judge (LLM-as-Judge)
- 대규모 평가에서 LLM을 자동 심사관으로 활용
- Phoenix의 `create_classifier()` API로 평가자 정의

### Level 6: Annotate with Human Feedback (인간 피드백)
- 구조화된 인간 라벨링으로 Ground Truth 확보
- Phoenix UI에서 직접 Ground Truth 라벨 첨부

### Level 7: Build Self-Correcting AI (자기 교정 AI)
- Dynamic Prompt Updates: 데이터 기반 자동 프롬프트 업데이트
- **DSPy** 통합으로 자동 프롬프트 튜닝
- Fine-tuning: 고품질 예시 활용한 모델 미세조정

---

## 3. Phoenix 실제 구현: 5가지 최적화 기법

**출처**: https://github.com/Arize-ai/phoenix/blob/main/tutorials/prompts/prompt-optimization.ipynb

### 3.1 Few-Shot Examples
시스템 프롬프트에 10개의 샘플 입력-출력 쌍을 직접 삽입하여 패턴 학습 유도

### 3.2 Meta-Prompting
LLM을 사용하여 기존 프롬프트 개선안을 자동 생성
```
"이전 프롬프트의 성능을 개선할 새로운 프롬프트를 생성하세요.
성공 사례: [...]
실패 사례: [...]"
```

### 3.3 Prompt Gradient Optimization
- 프롬프트를 `text-embedding-ada-002`로 embedding 벡터 변환
- 성공 프롬프트와 실패 프롬프트의 평균 embedding 계산
- **Gradient 방향(avg_successful - avg_failed)**으로 이동하여 최적화
- 최적화된 embedding을 GPT를 통해 텍스트로 역변환

```
gradient = avg_embedding(successful_prompts) - avg_embedding(failed_prompts)
optimized_embedding = current_embedding + learning_rate * gradient
optimized_prompt = LLM.decode(optimized_embedding)
```

### 3.4 DSPy MIPROv2 Automated Tuning
- DSPy 프레임워크의 **MIPROv2 optimizer** 활용
- Ground Truth 라벨 기반 validation metric 정의
- Assertion-driven backtracking, example bootstrapping 전략
- LM Assertions로 제약 준수 최대 164% 향상

### 3.5 DSPy Multi-Model Strategy
- 프롬프트 생성: GPT-4o (고성능), Task 실행: GPT-3.5 Turbo (저비용)
- 성능과 비용의 균형 확보

---

## 4. DeepEval vs Arize AI 비교

| 비교 항목 | Arize AI (Phoenix) | DeepEval |
|---|---|---|
| **접근 방식** | Observability 중심 반복적 개선 플랫폼 | 테스트 프레임워크 중심 자동 최적화 |
| **자동 최적화** | DSPy MIPROv2 + Meta-Prompting + Gradient Optimization | 내장 GEPA 및 MIPROv2 알고리즘 |
| **자동화 수준** | 반자동 (도구 제공, 개발자가 루프 구성) | 완전 자동 (PromptOptimizer end-to-end) |
| **평가 생태계** | LLM-as-Judge + Pre-built Templates + 외부 통합 | 50개+ 내장 메트릭 |
| **Tracing** | OpenTelemetry 기반 풀 스택 | 기본적 LLM 트레이싱 |
| **UI/Playground** | 풍부한 Playground, 실험 대시보드 | CLI 중심, Confident AI 클라우드 |
| **고유 기법** | Prompt Gradient Optimization, Span Replay | GEPA (유전-파레토 알고리즘) |
| **MIPROv2** | DSPy 통한 간접 지원 | 내장 직접 지원 |

### 철학적 차이

**Arize AI**: "Observe → Evaluate → Experiment → Iterate"
- 데이터 기반 의사결정을 위한 **플랫폼** 접근
- 개발자가 관찰 데이터를 보고 점진적으로 개선

**DeepEval**: "Define Metrics → Run Optimizer → Get Result"
- 프롬프트 최적화를 **알고리즘 문제**로 접근
- 유전 알고리즘(GEPA)이나 서로게이트 탐색(MIPROv2)으로 자동 최적 프롬프트 탐색

---

## 5. Evaluation-Driven Improvement Loop

Arize AI의 핵심 개선 루프:

```
┌─────────────────────────────────────────────┐
│  [1] Tracing (추적)                          │
│  - OpenTelemetry 기반 LLM 호출 상세 로그 수집  │
│  - 25개+ 통합 (OpenAI, LangChain, DSPy 등)   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  [2] Evaluation (평가)                       │
│  - LLM-as-Judge 자동 품질 평가               │
│  - Pre-built: Relevance, Faithfulness 등     │
│  - create_classifier()로 커스텀 평가자        │
│  - Human Annotation으로 Ground Truth 확보    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  [3] Experimentation (실험)                  │
│  - 동일 데이터셋에서 프롬프트 변형 체계적 비교  │
│  - SDK: run_experiment(dataset, task, evals) │
│  - 평가 점수 기반 실패 패턴 분석              │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  [4] Evidence-Based Iteration (증거 기반 반복)│
│  - 실패 분석 결과 → 프롬프트 수정             │
│  - 수정된 프롬프트로 재실험 + 정량 검증        │
└──────────────────┬──────────────────────────┘
                   │
                   └──── [1]로 돌아감 (지속적 개선)
```

### Phoenix Experiments SDK 사용 예시

```python
import phoenix as px

# 1. 데이터셋 생성
dataset = client.upload_dataset(
    dataframe=df,
    dataset_name="my_eval_dataset",
    input_keys=["question"],
    output_keys=["expected_answer"]
)

# 2. Task 정의 (프롬프트 적용 함수)
def my_task(example):
    response = llm.generate(prompt_template.format(**example.inputs))
    return {"response": response}

# 3. Evaluator 정의
def accuracy_evaluator(example, task_output):
    return 1.0 if task_output["response"] == example.output["expected_answer"] else 0.0

# 4. 실험 실행
experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[accuracy_evaluator]
)
```

---

## 6. Day3 프로젝트와의 대응 관계

| Day3 프로젝트 구성 | Arize AI 대응 개념 |
|---|---|
| Loop 1: Golden Dataset 빌드 | Phoenix Datasets + Golden Dataset |
| Loop 2 Step 5: DeepEval 오프라인 평가 | Phoenix Evaluation (LLM-as-Judge) |
| Loop 2 Step 6: Langfuse 모니터링 | Phoenix Tracing (OpenTelemetry) |
| Loop 2 Step 8: Prompt Optimization | Phoenix Experiments + Meta-Prompting |
| Loop 3: Remediation Agent | Phoenix의 Evidence-Based Iteration |
| `langfuse_failed_samples.json` | Phoenix의 실패 패턴 분석 → 실험 재실행 |

---

## 7. 참고 URL

| 리소스 | URL |
|---|---|
| Phoenix GitHub | https://github.com/Arize-ai/phoenix |
| Phoenix 공식 문서 | https://arize.com/docs/phoenix |
| 프롬프트 최적화 블로그 | https://arize.com/blog/prompt-optimization |
| Prompt Optimization 노트북 | https://github.com/Arize-ai/phoenix/blob/main/tutorials/prompts/prompt-optimization.ipynb |
| DSPy 통합 블로그 | https://arize.com/blog/dspy/ |
| DeepEval Prompt Optimization | https://deepeval.com/docs/prompt-optimization-introduction |
