# 핵심 주제: DeepAgents 실전 연동 및 서빙 최적화

## Day-08 강의 계획안

### 1. 강의 한 줄 정의

오늘은 Day-07에서 구축한 vLLM 서빙 인프라 위에 LangChain DeepAgents를 연동하고, 실전에서 발생하는 컨텍스트 및 도구 호출 장애를 해결하며 Langfuse로 전체 과정을 관측하는 실전 워크숍이다.

### 2. 강의 목표

1. vLLM OpenAI 호환 API와 DeepAgents를 안정적으로 연동하고 기본 에이전트 루프를 실행할 수 있다.
2. KV Cache와 Context Engineering이 에이전트의 응답 속도 및 서빙 효율에 미치는 영향을 이해하고, 요약(Summarization) 및 프롬프트 최적화를 통해 이를 개선할 수 있다.
3. self-hosted OSS 모델이 가진 Tool Calling의 제약 사항을 파악하고, 스키마 단순화와 설명을 통해 호출 성공률을 높이는 트러블슈팅 능력을 갖춘다.
4. Langfuse를 활용하여 메인 에이전트, 서브 에이전트, 도구 호출로 이어지는 복잡한 실행 흐름(Trace)을 끊김 없이 관측하고 성능 지표를 분석할 수 있다.

### 3. 대상 수강생

- Day-01 ~ Day-07을 수강하여 LLM 서빙(vLLM) 및 에이전트의 기본 개념을 이해하고 있는 수강생
- 직접 서빙하는 모델을 에이전트 애플리케이션에 연결할 때 발생하는 실전 문제(OOM, 속도 저하, 호출 실패)를 해결하고 싶은 개발자
- 에이전트 시스템의 운영 안정성을 위해 정밀한 모니터링 및 관측(Observability) 체계를 구축하고자 하는 엔지니어

### 4. 이번 회차에서 새로 배우는 것

#### 이미 알고 있다고 가정하는 것 (Day-07 복습)

- vLLM의 핵심 원리 (PagedAttention, Continuous Batching, KV Cache)
- Docker 기반 vLLM 서빙 및 OpenAI 호환 API 엔드포인트 개념
- LangChain의 기본 구조와 Chat Model 인터페이스 사용법

#### 이번 회차에서 새로 닫아줄 것 (Day-08 신규)

- **DeepAgents 연동 실무**: self-hosted vLLM 엔드포인트를 DeepAgents 런타임에 안정적으로 결합하는 패턴
- **Context Engineering**: 에이전트의 긴 대화 기록이 KV Cache 효율에 주는 영향과 Summarization Middleware를 통한 최적화
- **OSS Tool Calling 최적화**: 클라우드 모델 대비 부족한 OSS 모델의 도구 호출 능력을 보완하는 스키마 설계 및 프롬프트 기법
- **Langfuse Trace Continuity**: 분산된 에이전트 실행 로그를 하나의 Trace ID로 묶어 관측하는 연속성 확보 전략

### 5. 전체 운영 원칙

- 전체 강의 시간은 `6시간` 워크숍으로 설계한다.
- 실습 인터페이스는 `vLLM OpenAI 호환 API` 단일 기준으로 고정하여 복잡도를 낮춘다.
- 실습 환경은 수강생이 인프라를 처음부터 구축하는 대신, **사전 프로비저닝된 self-hosted vLLM 엔드포인트**를 활용하여 연동과 트러블슈팅에 집중한다.
- 강의는 `이론 브리지 -> Baseline 연동 -> 장애 재현 및 복구 -> 관측 검증`의 단계별 실습 완주형으로 운영한다.
- **가시적 산출물 세트**: 본 계획안(`plan.md`)을 포함하여 아래 4종의 문서를 가시적 결과물로 선언한다.
  - [theory_outline.md](./theory_outline.md): 이론 아웃라인 및 개념 경계
  - [lab_guide.md](./lab_guide.md): DeepAgents ↔ vLLM 실습 가이드
  - [troubleshooting_runbook.md](./troubleshooting_runbook.md): 장애 증상별 진단 및 해결 런북
  - [langfuse_observability_checklist.md](./langfuse_observability_checklist.md): 관측성 확보 및 검증 체크리스트

### 6. 6시간 타임테이블

| 구간 | 시간 | 목표 | 핵심 내용 | 산출물 |
| --- | --- | --- | --- | --- |
| 이론 브리지 | 60분 | Day-07 복습 및 신규 개념 정렬 | vLLM 최적화 복습, DeepAgents 구조, Context Engineering 필요성, Tool Calling 제약, Langfuse 관측 전략 | 학습 로드맵 |
| Lab 1 | 90분 | Baseline 연동 및 관측 활성화 | vLLM 엔드포인트 연결, DeepAgents 기본 실행, Langfuse Callback 연동, Trace 확인 | Baseline 실행 로그, Langfuse Trace |
| Lab 2 | 90분 | Context 최적화 실습 | 긴 컨텍스트로 인한 성능 저하 재현, Summarization 적용, Prefix Caching 효율 측정 | 최적화 전/후 비교 리포트 |
| Lab 3 | 90분 | Tool Calling 트러블슈팅 | 도구 호출 실패 재현, 스키마 단순화 및 설명 보강, 호출 성공률 복구 | 수정된 Tool Schema, 성공 로그 |
| 클로징 | 30분 | 완료 기준 체크 및 실무 연결 | 세션 요약, 장애 대응 패턴 정리, 실무 적용을 위한 체크리스트 검토 | 최종 완료 체크리스트 |

### 7. 세션 상세 구성

#### 7-1. 가시적 산출물 맵 (Artifact Map)

Day-08 패키지는 아래 문서들을 통해 수강생과 강사에게 일관된 가이드를 제공한다.

1.  **[theory_outline.md](./theory_outline.md)**: Day-07과 Day-08의 개념적 경계를 명확히 하고, 오늘 배울 핵심 기술(KV Cache 컨트롤, Context Engineering 등)이 왜 실무에서 중요한지 설명한다.
2.  **[lab_guide.md](./lab_guide.md)**: `base_url`, `model_provider` 설정을 포함하여 DeepAgents와 vLLM을 연결하는 표준 성공 경로(Baseline)를 단계별로 안내한다.
3.  **[troubleshooting_runbook.md](./troubleshooting_runbook.md)**: 실습 중 의도적으로 발생시키는 장애 상황(컨텍스트 비대화, 도구 호출 오류)에 대한 "증상 -> 진단 -> 해결 -> 재검증" 프로세스를 제공한다.
4.  **[langfuse_observability_checklist.md](./langfuse_observability_checklist.md)**: 에이전트 실행 기록이 Langfuse에서 파편화되지 않고 하나의 흐름으로 이어지는지 확인하는 기술적 체크포인트를 다룬다.

#### 7-2. 실습 시나리오 설계 (Baseline -> Failure -> Recovery)

단순히 "따라하기"식 실습이 아니라, 실제 운영 환경에서 겪을 수 있는 문제를 해결하는 흐름으로 구성한다.

-   **Baseline**: 가장 단순한 도구 1개를 가진 에이전트를 vLLM에 붙여 성공시킨다.
-   **Failure 1 (Context)**: 시스템 프롬프트를 의도적으로 길게 만들거나 대화 이력을 누적시켜 응답 지연 및 KV Cache 효율 저하를 유도한다.
-   **Recovery 1**: DeepAgents의 미들웨어를 활용해 요약 및 컨텍스트 격리 전략을 적용하여 성능을 복구한다.
-   **Failure 2 (Tool)**: 복잡한 중첩 스키마를 가진 도구를 제공하여 OSS 모델이 잘못된 인자를 생성하거나 호출에 실패하게 만든다.
-   **Recovery 2**: 스키마를 평평하게(Flatten) 만들고 인자 설명을 보강하여 호출 성공률을 높인다.

### 8. 수강생 최종 산출물

-   **Baseline 연동 증빙**: vLLM 엔드포인트 호출 성공 로그 및 Langfuse 초기 Trace 캡처
-   **최적화 결과물**: 컨텍스트 최적화(Summarization 등) 적용 전/후의 응답 속도 및 토큰 사용량 비교 데이터
-   **트러블슈팅 기록**: 실패하던 도구 호출 스키마를 수정한 내역과 최종 성공 실행 로그
-   **관측성 체크리스트**: Langfuse에서 메인/서브 에이전트 및 도구 호출이 하나의 Trace ID로 연결된 화면 증빙

### 9. 평가 및 완료 기준

#### 수강생 완료 기준

1.  **Baseline 통과**: 사전 제공된 vLLM 엔드포인트에 DeepAgents를 연결하여 첫 번째 응답을 성공적으로 수신했다.
2.  **장애 재현 및 해결 (Context)**: 컨텍스트 비대화로 인한 문제를 인지하고, 최적화 기법을 적용하여 응답 안정성을 확보했다.
3.  **장애 재현 및 해결 (Tool)**: 실패하는 도구 호출 상황을 진단하고, 스키마 수정을 통해 정상 호출을 이끌어냈다.
4.  **관측성 검증**: Langfuse에서 에이전트의 전체 실행 흐름이 파편화 없이(Trace Continuity) 기록됨을 확인했다.

#### 강사 검증 기준

-   제공된 vLLM 엔드포인트가 모든 수강생의 요청을 처리할 수 있는 상태인지 사전에 체크한다.
-   실습용 도구 스키마와 프롬프트가 의도한 장애를 정확히 유발하는지 확인한다.
-   수강생이 제출한 Langfuse Trace가 부모-자식 관계(Parent-Child relationship)를 올바르게 유지하고 있는지 검토한다.
-   장애 상황 발생 시 즉시 전환 가능한 **Stable Endpoint 및 Reference Trace**를 확보하여 실습 지연을 방지한다.

### 10. 오늘의 핵심 메시지

-   에이전트의 성능은 모델 자체의 능력뿐만 아니라, 이를 감싸는 애플리케이션의 **Context Engineering** 설계에 크게 의존한다.
-   self-hosted 환경에서는 클라우드 모델의 편의성에 기대기보다, **엔드포인트 레벨의 페이로드 진단**과 **스키마 최적화**가 필수적이다.
-   복잡한 에이전트 시스템일수록 "돌아간다"는 확인을 넘어, **Langfuse와 같은 도구로 실행 과정을 투명하게 관측**할 수 있어야 운영 가능하다.
