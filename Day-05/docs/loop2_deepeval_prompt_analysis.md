# Loop2 DeepEval 기본 프롬프트 분석

## 목적
- DeepEval 기본 메트릭의 내부 프롬프트가 실제로 어떤 multi-step pipeline으로 동작하는지 정리
- 커스텀 프롬프트 최적화 시 어떤 지점을 기준으로 맞춰야 하는지 근거 제공

## 공통 관찰
- 다수 메트릭은 단일 채점 호출이 아니라 `추출 → 판정 → 요약` 패턴으로 구성된다.
- 추출 단계 품질이 낮으면 이후 단계 정밀도가 함께 낮아진다.
- GEval은 `criteria`로부터 evaluation steps를 생성하는 사전 호출이 핵심이다.

## 메트릭별 분석

### 1) Answer Relevancy
- 파이프라인:
1. `actual_output`에서 statement 목록 추출
2. statement별 relevance 판정(`yes/no/idk`)
3. irrelevant/ambiguous 근거를 요약(reason)
- 입력 파라미터: `INPUT`, `ACTUAL_OUTPUT`
- 핵심 JSON:
1. `{"statements": [...]}`
2. `{"verdicts": [{"verdict": "...", "reason": "..."}]}`
3. `{"reason": "..."}`
- 점수 로직: `no`가 아닌 verdict 비율(= `yes + idk`) / 전체
- 실패 모드:
1. statement 과분절(짧은 단편 남발)
2. `idk` 과다로 인한 점수 과대평가
3. 질문과 무관한 배경정보가 relevance로 남는 현상

### 2) Faithfulness
- 파이프라인:
1. `actual_output`에서 claim 추출
2. `retrieval_context`에서 truth 추출
3. claim별 모순 판정(`yes/no/idk`)
4. contradiction 요약(reason)
- 입력 파라미터: `INPUT`, `ACTUAL_OUTPUT`, `RETRIEVAL_CONTEXT`
- 핵심 JSON:
1. `{"claims": [...]}`
2. `{"truths": [...]}`
3. `{"verdicts": [{"verdict": "...", "reason": "..."}]}`
4. `{"reason": "..."}`
- 점수 로직: 기본적으로 `no`만 페널티 (`idk`는 기본 설정에서 감점 아님)
- 실패 모드:
1. truth 추출이 너무 요약되어 사실 비교 근거가 약해짐
2. 비언급(`idk`)이 많아도 고득점이 나올 수 있음
3. retrieval_context 자체 품질이 낮으면 metric 신뢰도도 낮아짐

### 3) Contextual Precision
- 파이프라인:
1. retrieval node별 유용성 `yes/no` 판정
2. 노드 순위까지 반영한 reason 생성
- 입력 파라미터: `INPUT`, `EXPECTED_OUTPUT`, `RETRIEVAL_CONTEXT`
- 핵심 JSON: `{"verdicts": [{"verdict": "yes|no", "reason": "..."}]}`
- 점수 로직: relevance의 위치(랭크)를 반영한 가중 정밀도
- 실패 모드:
1. 기대답변이 너무 짧으면 유용성 판정 근거가 빈약
2. ranking 신호보다 surface overlap에 과민 반응

### 4) Contextual Recall
- 파이프라인:
1. expected output의 문장 단위 분해
2. 문장별 context 귀속 가능 여부(`yes/no`) + 이유
3. supportive/unsupportive 근거 요약(reason)
- 입력 파라미터: `EXPECTED_OUTPUT`, `RETRIEVAL_CONTEXT`
- 핵심 JSON: `{"verdicts": [{"verdict": "yes|no", "reason": "..."}]}`
- 점수 로직: `yes` 문장 비율
- 실패 모드:
1. expected output 문장 분해 기준 흔들림
2. 같은 의미의 표현 변형(paraphrase)에 과도한 불일치 판정

### 5) GEval
- 파이프라인:
1. `criteria` → 3~4개 evaluation steps 자동 생성
2. steps + test case 기반 점수(0~10) + reason 생성
- 입력 파라미터: 사용자 지정 (`LLMTestCaseParams` 조합)
- 핵심 JSON:
1. `{"steps": [...]}`
2. `{"score": 0~10, "reason": "..."}`
- 점수 로직: 0~10을 0~1로 정규화
- 실패 모드:
1. criteria가 모호하면 steps 품질이 먼저 무너짐
2. 매호출 steps 변동으로 일관성 저하
3. 추상적 reason이 많아 개선 액션으로 연결 어려움

### 6) Task Completion
- 파이프라인:
1. workflow에서 task/outcome 추출
2. task 대비 outcome align 점수 산정(0~1)
- 입력 파라미터: trajectory(trace 또는 input/output/tools)
- 핵심 JSON:
1. `{"task": "...", "outcome": "..."}`
2. `{"verdict": 0~1, "reason": "..."}`
- 실패 모드:
1. outcome 추출 시 주관적 단어 삽입
2. partial completion 구간 정의가 판정마다 흔들림

### 7) Tool Correctness
- 파이프라인:
1. user goal + available tools + tools called를 단일 패스로 평가
- 입력 파라미터: `INPUT`, `TOOLS_CALLED`, `EXPECTED_TOOLS`(환경별)
- 핵심 JSON: `{"score": 0.0~1.0, "reason": "..."}`
- 실패 모드:
1. tool 사용 맥락보다 도구 이름 키워드 매칭에 과적합
2. over-selection / under-selection 경계가 모호한 케이스

## 정리
- 기본 메트릭의 품질은 결국 "초기 추출 단계 안정성"과 "판정 스키마 일관성"에 좌우된다.
- GEval 커스텀에서는 criteria를 사실상 "step generator prompt"로 다뤄야 한다.
- 따라서 커스텀 최적화는 criteria/prompt에 정의, 경계조건, 루브릭을 명시해 step 흔들림을 줄이는 방향이 유효하다.
