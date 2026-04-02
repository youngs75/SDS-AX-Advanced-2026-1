# Day3 AgentOps 운영 문서 (실행 로그 체크시트 포함)

이 문서는 **실제 운영자가 순서대로 실행**하고, 같은 파일에 **실행 로그를 남길 수 있도록** 설계된 런북입니다.

---

## 1) 운영 목적

- Golden Dataset 품질을 주기적으로 검증한다.
- Langfuse 모니터링 실패 샘플을 추출한다.
- 실패 샘플 기반으로 Evaluation Prompt를 개선한다.
- 변경 이력/판단 근거를 실행 로그로 남긴다.

---

## 2) 표준 실행 순서

### A. 빠른 운영 루프 (권장)

1. Step 5: Golden 샘플링 평가  
2. Step 6: Langfuse 실패 샘플 추출  
3. Step 8: Prompt 개선  
4. 필요 시 `--opt-apply` 반영

실행 명령:

```bash
# Step 5
.venv/bin/python scripts/run_pipeline.py --step 5 \
  --categories rag custom \
  --eval-sample-ratio 0.3 \
  --eval-sample-size 80 \
  --eval-sample-seed 42 \
  --eval-stratify-by source_file

# Step 6
.venv/bin/python scripts/run_pipeline.py --step 6 \
  --lf-tags env:prod \
  --lf-hours 24 \
  --lf-limit 500 \
  --lf-sample-ratio 0.2 \
  --lf-sample-size 80 \
  --lf-sample-seed 42 \
  --lf-score-prefix eval \
  --lf-threshold 0.7

# Step 8
.venv/bin/python scripts/run_pipeline.py --step 8 \
  --opt-iters 2 \
  --opt-max-cases 6 \
  --opt-lf-failures data/eval_results/langfuse_failed_samples.json \
  --opt-lf-hints-max 6
```

### B. 데이터 재생성 포함 루프

1. Step 1~4로 Golden 재구성  
2. Step 5 → 6 → 8 실행

```bash
.venv/bin/python scripts/run_pipeline.py --from 1 --to 4 --skip-review --num-goldens 50
```

---

## 3) 실행 로그 체크시트 템플릿

아래 블록을 복사해서 매 실행마다 기록하세요.

```md
# Run Log - YYYY-MM-DD HH:mm

## 0. Run Header
- run_id:
- operator:
- environment: (local/staging/prod)
- objective: (정기 점검/이슈 대응/릴리즈 전 검증)
- openrouter_model:
- langfuse_filter_tags:
- notes:

## 1. Step Checklist
| Step | Command | Start | End | Status | Key Result | Artifact |
|---|---|---|---|---|---|---|
| 5 | run_pipeline --step 5 ... | | | ☐ | | data/eval_results/eval_results.json |
| 6 | run_pipeline --step 6 ... | | | ☐ | | data/eval_results/langfuse_failed_samples.json |
| 8 | run_pipeline --step 8 ... | | | ☐ | | data/prompt_optimization/report.json |

## 2. Step 5 Result (Golden Offline Eval)
- input_dataset: data/golden/golden_dataset.json
- sampled_count:
- pass_count:
- fail_count:
- pass_rate:
- top_failed_metrics:
- skipped_metrics_summary:
- decision: (Step 6 진행 / 데이터 정제 필요)

## 3. Step 6 Result (Langfuse Monitoring)
- trace_sampled:
- score_collected:
- traces_failed:
- failure_rate:
- top_failed_metrics:
- failed_samples_path: data/eval_results/langfuse_failed_samples.json
- decision: (Step 8 진행 / 임계값 재설정)

## 4. Step 8 Result (Prompt Improve)
- baseline_fit_by_metric:
- best_fit_by_metric:
- uplift_summary:
- apply_executed: (yes/no)
- prompts_updated: (yes/no)
- report_path: data/prompt_optimization/report.json
- decision: (배포 반영 / 추가 iteration)

## 5. Final Decision
- release_gate: (PASS/FAIL)
- reason:
- follow_up_owner:
- follow_up_due_date:
```

---

## 4) Step별 합격 기준(운영 게이트)

### Step 5 게이트
- `pass_rate`가 직전 기준선 대비 유지/상승
- 특정 메트릭 급락(예: faithfulness) 시 Step 8 전 원인 분석

### Step 6 게이트
- `failure_rate`가 설정 임계치 이내
- 실패 샘플이 특정 태그/버전/시나리오에 집중되면 분리 대응

### Step 8 게이트
- `best_fit`이 `baseline_fit`보다 개선
- 개선 없으면 프롬프트 자동 반영(`--opt-apply`) 금지

---

## 5) 운영 산출물 확인 경로

- Golden Dataset: `data/golden/golden_dataset.json`
- Step 5 결과: `data/eval_results/eval_results.json`
- Step 6 스냅샷: `data/eval_results/langfuse_monitoring_snapshot.json`
- Step 6 실패 샘플: `data/eval_results/langfuse_failed_samples.json`
- Step 8 리포트: `data/prompt_optimization/report.json`
- Step 8 최적 프롬프트: `data/prompt_optimization/best_prompts.json`

---

## 6) 주의사항

- Step 6은 기본적으로 Langfuse score 기반 모니터링이며 DeepEval 호출이 아닙니다.
- Step 8은 개선 제안 단계이므로, 반영 전 결과 JSON과 로그 체크시트를 함께 검토해야 합니다.
- 운영 로그는 실행 단위로 남겨야 다음 회차에서 추세 비교가 가능합니다.

---

## 7) Langfuse Docs MCP 학습 자료

- 질문 템플릿: `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/docs/langfuse_mcp_question_playbook.md`
- 세션 로그 템플릿: `/Users/jhj/Desktop/personal/fastcampus_agentops_class/Day3/project/docs/langfuse_mcp_study_log_template.md`
