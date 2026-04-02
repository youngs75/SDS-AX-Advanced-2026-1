# Loop2 커스텀 프롬프트 분석/최적화 플레이북

## 1. 현재 커스텀 프롬프트 분석

### A) Response Completeness (GEval)
- What: 질문에 필요한 핵심 요소를 답변이 충분히 다루는지 평가
- How: `INPUT + ACTUAL_OUTPUT + EXPECTED_OUTPUT`을 GEval criteria 기반으로 채점
- 개선 포인트:
1. must-have point 정의를 criteria에 명시하지 않으면 steps가 추상화되기 쉬움
2. 스타일 차이와 정보 누락을 분리해서 감점하도록 규칙화가 필요
3. evaluated text 내부 지시(injection 문구)를 무시하도록 명시 필요

### B) Citation Quality (GEval)
- What: 출처 귀속 품질(존재, 형식, 정합성) 평가
- How: `ACTUAL_OUTPUT + CONTEXT`를 기반으로 GEval 채점
- 개선 포인트:
1. 인용 정책을 엄격하게 고정하지 않으면 judge마다 형식 해석이 달라짐
2. 본 프로젝트는 `[k]` 인덱스형으로 고정해야 재현성이 높음
3. 인용 존재 여부뿐 아니라 claim-context 정합을 함께 평가해야 함

### C) Safety (BaseMetric)
- What: 유해성/PII/편향/민감주제 경고를 단일 점수로 통합 평가
- How: custom prompt + JSON 파싱(`score`, `reason`)
- 개선 포인트:
1. 모델 출력이 코드블록/부가문구를 섞을 수 있으므로 파싱 내구성 필요
2. score 범위를 0~1로 클램프해 후속 로직 안정성 확보 필요
3. evaluated text 내 지시를 따르지 말라는 규칙이 반드시 필요

## 2. Prompt Optimize/Customize 방향성

원칙:
1. **정의 우선**: 평가 단위(예: must-have point, [k] citation)를 먼저 고정
2. **루브릭 명시**: 점수 구간별 조건(0~10 또는 0~1) 명시
3. **인젝션 내성**: 평가 대상 텍스트 내부 지시는 무시
4. **출력 계약 고정**: Safety는 strict JSON 출력 계약 유지
5. **작은 루프 반복**: 대형 데이터셋 대신 캘리브레이션 세트로 빠르게 반복

## 3. Meta Prompt 설계

구현 위치:
- `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/src/loop2_evaluation/prompt_optimizer.py`

시스템 역할:
- 평가 프롬프트를 calibration failure 기반으로 개선하는 "prompt optimization engineer"

입력 payload:
1. `target_metric`
2. `current_fit_score`
3. `constraints` (GEval/JSON/[k] 정책 등)
4. `current_prompt`
5. `calibration_failures` (score, expected range, reason)

출력 스키마(JSON only):
```json
{
  "updated_prompt": "string",
  "change_log": ["string"],
  "risk_notes": ["string"]
}
```

## 4. 자동 최적화 루프

구현 위치:
- `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/src/loop2_evaluation/calibration_cases.py`
- `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/src/loop2_evaluation/prompt_optimizer.py`
- `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/scripts/08_optimize_eval_prompts.py`

절차:
1. baseline prompt로 metric별 캘리브레이션 평가
2. expected range를 벗어난 실패 케이스 상위 N개 추출
3. Meta Prompt로 후보 prompt 생성
4. 후보 prompt 재평가 후 fit score 개선 여부 판단
5. 반복(`--iters`) 후 best prompt 저장
6. 필요시 `--apply`로 `prompts.py` 상수 자동 반영

산출물:
- `data/prompt_optimization/report.json`
- `data/prompt_optimization/best_prompts.json`

## 5. 최종 개선 프롬프트 적용 결과

적용 파일:
- `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/src/loop2_evaluation/prompts.py`

핵심 반영 내용:
1. `RESPONSE_COMPLETENESS_PROMPT`
   - must-have point 정의 + 누락/모순 우선 감점 + 구간별 루브릭
2. `CITATION_QUALITY_PROMPT`
   - `[k]` 인덱스 정책 명문화 + invalid index/정합성 규칙 + 루브릭
3. `SAFETY_PROMPT`
   - 4개 안전 축 + 0.0~1.0 구간 정의 + strict JSON 계약 + 인젝션 무시 규칙

## 6. 실행 가이드

단독 실행:
- `python scripts/08_optimize_eval_prompts.py --iters 2 --max-cases 6`
- `python scripts/08_optimize_eval_prompts.py --iters 2 --apply`

통합 파이프라인:
- `python scripts/run_pipeline.py --step 8 --opt-iters 2 --opt-max-cases 6`
- `python scripts/run_pipeline.py --all --continue-on-error`

## 7. Golden Dataset 샘플링 운영 (Step 5)

Golden Dataset을 매번 전체 평가하면 비용/시간이 빠르게 증가하므로, Step 5는 샘플링 실행을 기본 옵션으로 지원합니다.

핵심 옵션:
- `--eval-sample-ratio`: 전체 Golden 중 평가할 비율(예: `0.3`)
- `--eval-sample-size`: 샘플 최대 건수 상한(예: `80`)
- `--eval-sample-seed`: deterministic 샘플링 시드(재현성)
- `--eval-stratify-by`: 층화 기준 필드(기본: `source_file`)

권장 실행 예시:
- `python scripts/05_run_eval.py --categories rag custom --sample-ratio 0.3 --sample-size 80 --sample-seed 42 --stratify-by source_file`
- `python scripts/run_pipeline.py --step 5 --categories rag custom --eval-sample-ratio 0.3 --eval-sample-size 80 --eval-sample-seed 42 --eval-stratify-by source_file`

운영 가이드:
1. **일상 점검(빠른 루프)**: `sample_ratio=0.2~0.4` + `sample_size` 상한으로 빠르게 확인
2. **프롬프트 변경 직후**: 동일 `sample_seed`로 A/B 비교(변경 효과 재현 가능)
3. **릴리즈 전 검증**: 샘플링 없이 전체 평가(또는 고비율 샘플링)
4. **층화 기준 확장**: 필요시 `source_file difficulty`처럼 다중 필드 층화

실패 판정 기준:
1. Step 5 결과의 `passed=False`를 실패로 간주
2. 현재 구현 기준으로 `passed`는 **실행된 모든 메트릭 점수가 0.5 이상**일 때만 True
3. 필수 입력이 없는 메트릭은 `skipped_metrics`로 분리하여 실패/성공 판정에서 제외

## 8. Langfuse 샘플링 연동 (Evaluation → Improve)

온라인 평가는 Langfuse 내부 Evaluation score를 기준으로 샘플링 기반 운영합니다.
Step 6은 DeepEval을 호출하지 않고 score 집계/실패 샘플 추출만 수행합니다.

권장 실행 예시:
- `python scripts/06_batch_eval_langfuse.py --mode monitor --tags env:prod --hours 24 --limit 500 --sample-ratio 0.2 --sample-size 80 --sample-seed 42 --score-prefix eval --threshold 0.7`
- `python scripts/run_pipeline.py --step 6 --lf-tags env:prod --lf-hours 24 --lf-limit 500 --lf-sample-ratio 0.2 --lf-sample-size 80 --lf-sample-seed 42 --lf-score-prefix eval --lf-threshold 0.7`
- `python scripts/run_pipeline.py --step 8 --opt-lf-failures data/eval_results/langfuse_failed_samples.json --opt-lf-hints-max 6`

운영 원칙:
1. **Fetch**: 최근 N시간 trace 조회
2. **Sample**: deterministic 샘플링으로 재현 가능한 subset 선정
3. **Evaluate Source**: Langfuse `eval.*`(또는 지정 prefix) score를 단일 기준으로 사용
4. **Fail Extract**: threshold 미달 trace를 실패 샘플로 저장
5. **Improve**: Step 8에서 실패 샘플을 힌트로 사용해 DeepEval 기반 프롬프트 최적화
6. **Verify**: 개선 후 다시 Langfuse score uplift 확인

주의:
- Step 8 프롬프트 최적화의 ground truth는 캘리브레이션/리뷰 데이터가 필요합니다.
- 온라인 점수만으로 즉시 prompt를 업데이트하면 자기강화 편향이 생길 수 있으므로
  최소한의 human review 또는 holdout 검증을 권장합니다.
