# Langfuse Docs MCP 질문 템플릿 플레이북

이 문서는 **Langfuse 공식 문서 MCP**를 통해 빠르게 학습/검증하기 위한 실전 템플릿 모음입니다.  
목표는 “문서 검색”이 아니라 **운영 가능한 결론**을 얻는 것입니다.

---

## 1) 기본 원칙

1. 개요 확인: `getLangfuseOverview` (세션당 1회)
2. 후보 탐색: `searchLangfuseDocs`
3. 구현 확정: `getLangfuseDocsPage`
4. 산출물 고정: “결론 + 실행 단계 + 주의사항 + 참조 링크”

---

## 2) 공통 출력 포맷 (권장)

모든 질문에 대해 아래 포맷으로 답변을 받도록 고정하세요.

```md
## 결론
- 핵심 결정 1~3줄

## 실행 단계
1. ...
2. ...
3. ...

## 주의사항
- 운영 리스크/한계

## 참조 문서
- URL
```

---

## 3) 축별 질문 템플릿

## A. Observability 템플릿

### O-1. Trace 스키마 표준화
```text
Langfuse Observability에서 우리 프로젝트 표준 trace/span 스키마를 제안해줘.
필수 필드(예: user/session/tags/metadata/input/output)와 각 필드의 목적, 최소 예시 payload를 제공해줘.
답변은 결론/실행 단계/주의사항/참조 문서 형식으로.
```

### O-2. 운영 디버깅 뷰 설계
```text
운영 장애 대응 관점에서 Langfuse 대시보드/조회에 필요한 최소 필터 조합을 설계해줘.
(예: env, app_version, prompt_version, error type, latency)
우리 Day3 파이프라인에 바로 적용 가능한 규칙으로 답해줘.
```

### O-3. Sampling/Cost 추적 설계
```text
Langfuse에서 비용/토큰/샘플링 효율을 추적하기 위한 KPI를 정의해줘.
KPI 이름, 계산 방식, 경보 임계값 초안까지 제안해줘.
```

### O-4. MCP Tracing 적용 기준
```text
Langfuse MCP Tracing을 언제 켜고 언제 끄는지 운영 기준을 제안해줘.
개발/스테이징/프로덕션 환경별 권장 설정을 표로 정리해줘.
```

### O-5. Incident 재현 플레이북
```text
특정 시간대 실패(trace failed)가 급증했을 때 Langfuse에서 원인을 좁혀가는 단계별 조사 절차를 작성해줘.
실행 순서와 중단 조건(go/no-go)을 포함해줘.
```

## B. Prompt Management 템플릿

### P-1. 버전/라벨 전략
```text
Langfuse Prompt Management에서 version + label 전략을 설계해줘.
dev/staging/prod 라벨 체계와 롤백 절차를 포함해줘.
```

### P-2. Prompt 릴리즈 게이트
```text
새 프롬프트를 운영 반영하기 전 필수 게이트를 정의해줘.
DeepEval Step5/Step8 결과와 Langfuse Step6 모니터링 결과를 함께 반영하는 기준으로 제안해줘.
```

### P-3. Prompt 변경 감사로그
```text
Prompt 변경 이력을 감사 가능하게 남기기 위한 최소 메타데이터 스키마를 제안해줘.
(author, reason, linked incident, expected impact, rollback condition 포함)
```

### P-4. Prompt A/B 비교 설계
```text
Langfuse Prompt Management를 기준으로 프롬프트 A/B 테스트 설계를 제안해줘.
샘플 크기, 관찰 기간, 성공 판정 기준을 포함해줘.
```

### P-5. Prompt + Eval 연결
```text
프롬프트 버전별로 평가 점수(quality/safety/completeness)를 안정적으로 추적하는 연결 규칙을 제안해줘.
태그/metadata naming convention까지 포함해줘.
```

## C. Evaluation 템플릿

### E-1. 실패 정의 표준화
```text
Langfuse Evaluation에서 "실패"를 정의하는 방법을 제안해줘.
절대 임계값, 상대 하락률, 메트릭 가중치 방식 3가지를 비교해서 권장안을 제시해줘.
```

### E-2. Dataset/Experiment 운영
```text
Langfuse Dataset/Experiment를 우리 프로젝트에 도입할 때의 최소 운영 플로우를 제안해줘.
데이터셋 생성, 버전 관리, 실험 실행, 결과 해석, 반영까지 순서대로 설명해줘.
```

### E-3. LLM-as-a-Judge 설계
```text
Langfuse의 LLM-as-a-Judge를 쓸 때 평가 프롬프트 설계 규칙을 제안해줘.
judge bias 완화, calibration, human review 결합 전략까지 포함해줘.
```

### E-4. Offline vs Online 평가 분리
```text
우리 Day3 구조(DeepEval Step5 + Langfuse Step6)를 기준으로 offline/online 평가 책임 분리를 명확히 정의해줘.
중복 지표를 줄이면서도 운영 안정성을 유지하는 권장안을 제시해줘.
```

### E-5. Score Analytics 운영 지표
```text
Langfuse Score Analytics에서 주간 보고에 포함해야 할 핵심 지표를 제안해줘.
각 지표의 의미, 경고 신호, 후속 액션을 함께 정리해줘.
```

---

## 4) 교차(Observability↔Prompt↔Evaluation) 템플릿

### X-1. 폐쇄 루프 설계 검토
```text
Observability → Evaluation → Prompt Management 폐쇄 루프를 우리 Day3 파이프라인에 맞춰 재설계해줘.
각 단계의 입력/출력 아티팩트와 책임자를 명시해줘.
```

### X-2. 개선 우선순위 자동화
```text
실패 샘플이 많을 때 어떤 항목부터 개선할지 우선순위 모델을 제안해줘.
impact, frequency, risk를 반영한 점수식으로 설명해줘.
```

### X-3. 운영 리뷰 회의 템플릿
```text
주간 AgentOps 리뷰 회의 아젠다를 제안해줘.
Langfuse 관측 결과, 평가 점수, 프롬프트 변경, 다음 액션이 한 번에 보이게 구성해줘.
```

---

## 5) Day3 프로젝트용 권장 질문 순서

1. O-1 → O-2 (관측 표준)
2. E-1 → E-4 (실패 정의/평가 분리)
3. P-1 → P-2 (배포/롤백 기준)
4. X-1 (폐쇄 루프 정합성 점검)

---

## 6) 세션 완료 체크리스트

- [ ] 결정된 규칙이 Step5/6/8 현재 코드와 충돌하지 않는다.
- [ ] 문서 링크가 최신 공식 문서로 정리되었다.
- [ ] 실행 가능한 액션이 최소 3개 도출되었다.
- [ ] 결과가 운영 로그(`operations_runbook.md`)에 기록되었다.

---

## 7) 핵심 공식 문서 링크

- Observability: https://langfuse.com/docs/observability/overview
- Prompt Management: https://langfuse.com/docs/prompt-management/overview
- Evaluation: https://langfuse.com/docs/evaluation/overview
- Docs MCP: https://langfuse.com/docs/docs-mcp
- API/Data MCP: https://langfuse.com/docs/api-and-data-platform/features/mcp-server
