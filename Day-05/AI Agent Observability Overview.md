# AI Agent Observability 기반 평가와 개선 배경과 문제 정의

AI Agent(도구 호출, 메모리, 파일시스템, 서브에이전트 등으로 구성된 “시스템”)는 전통적 소프트웨어와 달리 **동일 입력·동일 코드에서도 실행 경로와 결과가 달라질 수 있는 비결정성**을 갖습니다. [\[1\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) 이런 특성 때문에 “왜 성공/실패했는지”의 원인을 **코드만으로** 재구성하기 어렵고, 실제 행동은 실행 과정에서 “드러난” 텍스트/도구 호출/상태 변화의 흔적에 남습니다. [\[2\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/)

이 관점에서 LangChain[\[3\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) 측은 “AI에서는 코드가 아니라 **트레이스(trace)가 앱의 행위를 문서화한다**”는 주장으로, 디버깅·테스팅·프로파일링·모니터링이 **코드 중심에서 트레이스 중심으로 이동**한다고 정리합니다. [\[2\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) 또한 에이전트는 **(1) 비결정성, (2) 프롬프트 민감성, (3) 무한(비정형) 입력 공간**이라는 이유로 사전 테스트만으로는 실제 사용/실패 패턴을 예측하기 어렵고, 운영 데이터 기반 반복 개선이 더 중요해진다고 설명합니다. [\[4\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/)

산업적으로는 Microsoft[\[5\]](https://arxiv.org/abs/2303.16634) Microsoft Azure[\[6\]](https://arize.com/docs/ax/prompts/prompt-optimization/prompt-learning-sdk?utm_source=chatgpt.com) 쪽이 “에이전트 관측성은 단순히 METRICS/LOGS/TRACES를 넘어 **EVALUATIONS(품질·안전 평가)** 및 **GOVERNANCE(정책·규정 준수)**를 포함해야 한다”고 정리합니다. [\[7\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) Arize AI[\[8\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) 역시 “복잡한 멀티에이전트/멀티모달/외부 프로토콜 연동 환경에서 ‘바이브로 디버깅’(prompt만 만지며 추측)하는 방식은 한계가 있으며, ‘유리 상자(glass box)’ 수준의 가시화가 필요하다”는 문제의식을 명확히 합니다. [\[9\]](https://arize.com/ai-agents/agent-observability/)

이 보고서는 위 흐름을 바탕으로, **Observability(측정) → Evaluation(평가) → Improvement(개선)**을 하나의 폐루프로 설계하는 방법을 다룹니다. 특히 (1) 어떤 데이터를 어떤 표준 스키마로 수집할지, (2) 무엇을 어떻게 평가할지(단위/통합/운영 평가), (3) 평가 결과를 어떻게 시스템적으로 개선(하네스/프롬프트/정책/런타임)으로 연결할지에 초점을 둡니다. [\[10\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/)

## Observability 설계의 핵심 개념: 트레이스·세션·행동 궤적

에이전트 관측성에서 기본 단위는 보통 “요청 1건(또는 세션)”이며, 해당 요청이 처리되는 동안의 **다단계 추론 및 도구 호출의 연쇄**를 트레이스로 기록합니다. [\[11\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) 트레이스는 “에이전트가 어떤 단계들을 밟았는지(trajectory), 어떤 도구를 어떤 인자로 호출했는지, 결과/시간/비용이 어땠는지”를 담아, 성공/실패의 원인 분석·전후 비교·회귀 탐지의 기반이 됩니다. [\[12\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/)

특히 멀티턴(대화형) 에이전트에서는 “단일 응답 품질”보다 **세션 전체의 일관성, 문맥 유지, 목표 달성**이 중요해지기 때문에, 세션 레벨 평가/관측이 별도의 1급 개념으로 다뤄집니다. [\[9\]](https://arize.com/ai-agents/agent-observability/) 또한 멀티에이전트 환경에서는 에이전트 간 라우팅/위임/핸드오프가 복잡해지며, 상호작용 흐름을 추적할 수 있어야 병목/루프/책임 경계를 파악할 수 있습니다. [\[13\]](https://arize.com/ai-agents/agent-observability/)

## 계측과 데이터 표준: OpenTelemetry GenAI Semantic Conventions 중심

관측성은 “데이터가 있어야” 시작됩니다. 최근 흐름의 핵심은 **OpenTelemetry(OTel)** 기반 표준화를 통해, 벤더/프레임워크가 달라도 동일한 형태로 트레이스·메트릭·로그를 수집·분석할 수 있게 만드는 것입니다. [\[14\]](https://opentelemetry.io/blog/2025/ai-agent-observability/)

### GenAI Semantic Conventions가 해결하는 것

OTel의 GenAI 스펙은 **프롬프트/응답, 토큰 사용량, 모델·프로바이더 메타데이터, 툴/에이전트 호출** 등을 일관된 키(예: gen\_ai.\*)로 기록하는 방식(Attributes/Spans/Events/Metrics)을 정의합니다. [\[15\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) 예를 들어 모델 호출(span)의 이름은 {gen\_ai.operation.name} {gen\_ai.request.model} 형태를 권고하고, span kind 및 오류 기록 방식도 가이드합니다. [\[16\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)

또한 “chat history 등 입력/출력 상세”는 비용/민감정보 이슈 때문에 기본적으로는 비활성(opt-in)로 두고, 필요 시 gen\_ai.client.inference.operation.details 같은 이벤트로 분리 저장할 수 있음을 명시합니다. [\[17\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) 이 분리는 **샘플링/익명화/보존 정책**과 강하게 결합됩니다.

### 프레임워크·벤더 독립성과 수집 파이프라인

OTel 기반 계측의 장점은, (1) 프레임워크가 내장 계측을 제공하거나, (2) 별도의 OTel instrumentation library로 분리 제공하는 두 접근이 모두 가능하다는 점입니다. OTel 측은 baked-in instrumentation의 장단점(도입 용이 vs. 프레임워크 비대화/버전 락인 등)과, 분리형 instrumentation의 장단점을 비교합니다. [\[18\]](https://opentelemetry.io/blog/2025/ai-agent-observability/)

실무에서는 OTel Collector(또는 동급 파이프라인)에서 **redaction, sampling, enrichment, routing**을 수행해 “데이터가 나가기 전에” 정책을 집행하는 설계가 자주 권장됩니다. [\[19\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) 특히 Datadog[\[20\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) 은 OTel GenAI 스펙(예: v1.37+) 기반의 span을 단일 계측으로 수집하고, Collector에서 거버넌스를 적용한 뒤 자사 LLM Observability로 분석하는 구성을 강조합니다. [\[21\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)

### 민감정보와 “중복 계측” 문제

민감 정보(프롬프트/응답/함수 인자·결과 등)는 관측성에 매우 유용하지만, 운영에서 그대로 저장하면 개인정보/기밀 유출 위험이 큽니다. [\[22\]](https://learn.microsoft.com/en-us/agent-framework/agents/observability) Microsoft Agent Framework 문서는 **민감 데이터 계측은 개발/테스트에서만 권장**하며, 클라이언트와 에이전트에 동시에 계측을 넣을 때 동일 컨텍스트가 중복 span에 기록될 수 있으니 범위를 조절하라고 명시합니다. [\[23\]](https://learn.microsoft.com/en-us/agent-framework/agents/observability)

### 계측 구현의 현실적인 패턴

Snowflake 문서는 Snowflake[\[24\]](https://arxiv.org/html/2507.21504v1) 환경에서 TruLens[\[25\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com) SDK의 @instrument() 데코레이터로 함수 입·출력과 지연 시간을 추적하고, RAG에서 RETRIEVAL/GENERATION 같은 span 유형을 지정해 가독성을 높이는 패턴을 예시로 제시합니다. [\[26\]](https://docs.snowflake.com/ko/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications)  
이 패턴의 실용적 해석은 다음과 같습니다: “에이전트 파이프라인의 논리적 경계(검색/재랭킹/생성/툴 호출/검증)를 span으로 쪼개고, 각 span에 최소한의 식별자·버전·비용·결과 상태를 붙여, 평가와 개선이 가능한 형태의 데이터셋을 만든다.” [\[27\]](https://docs.snowflake.com/ko/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications)

## Evaluation 설계: 무엇을 어떻게 “정량화”할 것인가

에이전트 평가는 “결과 텍스트”만으로 끝나지 않습니다. 최근 서베이(ACM KDD 2025)는 에이전트 평가를 **(A) 평가 목표(behavior/capabilities/reliability/safety)**와 **(B) 평가 프로세스(static vs online, 데이터셋, 메트릭 산출, 툴링, 환경)**의 2차원 분류로 정리합니다. [\[28\]](https://arxiv.org/html/2507.21504v1) 이는 Observability가 “데이터 수집”이라면, Evaluation은 그 데이터로 **목표 함수를 정의하는 과정**에 가깝다는 뜻입니다.

### 단위 평가에서 세션 평가까지: Deep Agents의 테스트 패턴

LangSmith[\[29\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) 팀은 Deep Agents 평가에서 “동일 앱 로직·동일 평가자”로 처리하는 전통적 LLM 평가가 깨지는 이유를 설명하며, 에이전트는 datapoint마다 성공 기준이 달라지고 **궤적(trajectory), 최종 응답, 부수 상태(파일/메모리/아티팩트)**를 각각 단언(assert)해야 한다고 강조합니다. [\[30\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)

또한 평가 실행 모드를 다음처럼 계층화합니다. (1) **Single-step**: 특정 시점에서 “다음 액션이 무엇인지”를 검증(단위 테스트 성격), (2) **Full turn**: 한 입력에 대해 도구 호출을 포함한 end-to-end 실행을 검증, (3) **Multiple turns**: 실제 사용자 상호작용을 모사하되, 분기 때문에 “레일(conditional logic)”을 두어 테스트가 무의미해지지 않게 만든다는 방식입니다. [\[31\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)

여기서 중요한 실무 결론은 두 가지입니다.  
첫째, “에이전트 평가”는 전통적 소프트웨어 테스트처럼 **단위 테스트(결정 지점) \+ 통합 테스트(세션 결과)**의 조합이 필요합니다. [\[32\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)  
둘째, 상태를 가진 에이전트는 평가 환경을 매 실행마다 리셋해야 재현성이 생기며(Docker/샌드박스/임시 디렉토리), 외부 API 의존은 replay/mocking으로 비용·변동성을 줄이는 것이 중요합니다. [\[33\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)

### Code evals vs LLM-as-a-Judge: 혼합 전략의 표준화

Arize는 에이전트 평가를 크게 **LLM-as-a-Judge**와 **Code evals(프로그램적 검증)**로 나누고, 특히 “다단계 궤적”을 LLM 평가로 채점해 루프/불필요 단계/골든 패스 이탈을 탐지하는 예시를 제공합니다. [\[9\]](https://arize.com/ai-agents/agent-observability/) 동시에 경로 수렴(convergence) 같은 지표는 코드로 계산할 수 있으며(최소 단계 대비 실제 단계 비율의 평균 등), 이런 객관적 체크는 빠르고 재현성이 좋아 운영 모니터링에 적합하다고 제시합니다. [\[9\]](https://arize.com/ai-agents/agent-observability/)

핵심은 “정답이 명확하면 코드 eval로, 정답이 애매하거나 다차원 품질이면 LLM/Human eval로”라는 이분법이 아니라, **동일 트레이스를 두 종류의 평가로 교차 검증**해 신뢰도를 올리는 혼합 전략입니다. [\[34\]](https://arize.com/ai-agents/agent-observability/)

### LLM-as-a-Judge의 신뢰성 리스크: 편향·불안정성·검증 프레임

LLM-as-a-Judge는 확장성이 뛰어나지만, “평가자 자체의 신뢰성”이 병목이 됩니다.  
대표적으로 (1) 자기 선호(self-preference) 편향: LLM이 자신에게 익숙한(낮은 perplexity) 출력에 더 높은 점수를 주는 경향이 보고되며, GPT-4에서 유의미한 자기 선호 편향이 관찰되었다고 합니다. [\[35\]](https://arxiv.org/abs/2410.21819)  
(2) LLM 기반 평가가 LLM 생성 텍스트에 bias를 가질 수 있다는 점도 지적됩니다. [\[36\]](https://arxiv.org/abs/2303.16634)  
(3) 더 최근에는 “LLM 판정의 신뢰성을 **Item Response Theory(IRT)** 기반으로 진단”하여, 프롬프트 변형에 대한 **내재적 일관성(intrinsic consistency)**과 인간 평가와의 **정렬(human alignment)**을 분리 측정하자는 프레임이 제안됩니다. [\[37\]](https://www.arxiv.org/abs/2602.00521)

따라서 실무에서 LLM-as-a-Judge를 채택할 때는, “점수 하나”로 운영 경보를 만들기 전에 최소한 다음의 검증이 권장됩니다: (a) 평가 프롬프트 변형에 대한 점수 안정성, (b) 샘플링된 human audit 세트와의 상관, (c) 모델/버전 변경 시 재검증. [\[38\]](https://www.arxiv.org/abs/2602.00521)

## Traces-to-Insights: 운영 규모에서의 분석과 회귀 탐지

운영에서 하루 수천\~수십만 트레이스가 쏟아지면 “가시화(visibility)”만으로는 부족하고, **분석(analysis)**이 병목이 됩니다. LangChain은 “100k+ 트레이스를 쌓아놓고도 아무 것도 못 하는” 문제를 언급하며, 기존 제품 분석 도구가 이벤트 기반이라 비정형 대화·행동 궤적을 다루기 어렵다고 지적합니다. [\[39\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/)

### 온라인 평가(known questions)와 탐색 분석(unknown questions)의 분리

트레이스에 평가자(온라인 eval)를 붙이면 “사용자 불만”, “특정 실패 기준”처럼 **미리 정의한 질문(known questions)**에는 답할 수 있습니다. [\[40\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) 그러나 “사용자들이 실제로 어떻게 쓰는지”, “어떤 실패 패턴이 존재하는지”처럼 **정의되지 않은 질문(unknown questions)**은 평가자만으로는 포착이 어렵다고 하며, 이를 위해 트레이스 클러스터링 기반 패턴 발굴(Insights Agent)을 제안합니다. [\[41\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/)

이 구분은 설계적으로 중요합니다. 운영 분석 체계를 만들 때는: \- 온라인 eval: KPI를 만드는 “센서(미리 아는 질문)”  
\- 클러스터링/요약: 원인 후보를 만드는 “탐색 엔진(모르는 질문)”  
로 역할을 분리해야, 평가 프롬프트가 과도하게 복잡해지거나(비용 폭증) 반대로 탐색이 공백이 되는 문제를 줄일 수 있습니다. [\[41\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/)

### 회귀 테스트의 재정의: 비결정성과 프롬프트 취약성 다루기

전통적 회귀 테스트는 “한 케이스가 깨지면 버그”에 가깝지만, LLM/에이전트는 (a) 개별 샘플의 플립이 흔하고, (b) 프롬프트 설계가 모델 업데이트에 따라 다르게 변하며, (c) 비결정성으로 테스트가 flaky해질 수 있습니다. [\[42\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)  
CMU의 회귀 테스트 논의는 이런 환경에서 회귀 테스트를 **데이터 슬라이스 단위(집계 지표)**로 정의하고, 프롬프트/모델 버전을 함께 추적하며, 비결정성을 전제로 샘플링/반복 실행을 설계해야 한다고 주장합니다. [\[43\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)

이 관점은 곧 “Observability 데이터 모델”로 환원됩니다. 즉, 트레이스에 **모델 버전, 프롬프트(템플릿) 버전, 하네스 버전, 툴/정책 버전**이 태깅되지 않으면 슬라이스 기반 회귀를 제대로 할 수 없습니다. [\[44\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)

## Improvement: 평가를 실제 성능 향상으로 바꾸는 방법

Observability와 Evaluation이 “지도”라면 Improvement는 “핸들”입니다. 최근의 중요한 흐름은 **모델을 바꾸지 않고도** 시스템(하네스/프롬프트/미들웨어/검증 루프)의 변경만으로 큰 성능 향상을 얻는 사례들이 축적되고 있다는 점입니다. [\[45\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

### Harness Engineering: “하네스만 바꿔도 성능이 오른다”

LangChain은 코딩 에이전트가 Terminal-Bench[\[46\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) 2.0에서 “Top 30 → Top 5”로 상승했는데, **모델은 고정하고 하네스만 변경**했다고 요약합니다. [\[47\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) 하네스의 목표는 “모델의 들쑥날쑥한(spiky) 지능을 과업 목표에 맞게 성형”하는 것이며, 시스템 프롬프트/도구/실행 흐름/미들웨어 훅이 주요 조절점이라고 설명합니다. [\[48\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

이 글에서 특히 실무적으로 재사용 가능한 개선 패턴은 다음과 같습니다. \- **Self-verification을 강제하는 훅**: 종료 직전에 체크리스트/테스트 실행을 강제해 “제출 전 검증”을 습관화. [\[48\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)  
\- **환경 컨텍스트 주입**: 디렉토리 구조/툴 존재 여부/제약(타임아웃)을 탐색 대신 주입해 오류 표면을 줄임. [\[48\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)  
\- **루프 감지 미들웨어**: 동일 파일 과도 편집 등 패턴을 추적해 “doom loop”를 완화. [\[48\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)  
\- **추론 예산 스케줄링**: 계획/검증 단계에 더 많은 추론을 배분하는 식의 “reasoning sandwich”로 시간제약을 다룸. [\[48\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

여기서 Observability의 역할은 단순 로그가 아니라, 하네스 변경이 실제로 어떤 실패 모드를 줄였는지 “트레이스 기반으로 재현 가능한 분석”을 만드는 것입니다. [\[49\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

참고로 Terminal-Bench 2.0 자체도 “컨테이너화된 환경 \+ 테스트 기반 검증”이라는 하네스 친화적 구조를 강조하며, 89개 과업 각각이 독립 환경/해법/테스트를 갖고 있고 outcome-driven 검증을 한다고 설명합니다. [\[50\]](https://arxiv.org/html/2601.11868v1)

### Prompt Learning Loop: 영어 피드백 기반의 지속 최적화

Arize는 프롬프트 최적화를 “수작업 프롬프트 튜닝”이 아니라, **데이터·평가·피드백을 통해 시스템 프롬프트(규칙)를 계속 업데이트하는 온라인 루프**로 정식화합니다. [\[51\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com) 핵심 차별점은 “스칼라 점수”가 아니라 **영어 설명(왜 틀렸는지, 어떤 규칙이 필요한지)**을 에러 신호로 쓰는 것입니다. [\[51\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com)

이 접근이 Observability/Evaluation과 만나는 지점은 명확합니다. \- 운영 트레이스에서 실패 사례를 수집(Observe)  
\- LLM-as-a-Judge나 인간 주석으로 “실패 원인 설명”을 생성(Evaluate)  
\- 그 설명을 메타프롬프트로 흡수해 규칙/지시문을 강화(Improve)  
라는 구조이며, Arize 문서는 Prompt Learning SDK가 이 과정을 오케스트레이션하는 구성요소(Optimizer/MetaPrompt/Annotator 등)를 갖는다고 설명합니다. [\[52\]](https://arize.com/docs/ax/prompts/prompt-optimization/prompt-learning-sdk?utm_source=chatgpt.com)

또한 Prompt Playground/Registry는 운영 환경에서 프롬프트 버전 관리·비교·배포를 지원해, 루프를 팀 단위 개발 프로세스로 “제품화”하는 역할을 한다고 설명합니다. [\[53\]](https://arize.com/blog-course/evaluating-prompt-playground/)

### DSPy: “프롬프트를 코드로” 다루는 최적화 컴파일러 접근

YouTube 강연 **“DSPy: The End of Prompt Engineering”**(Kevin Madura)이 주목받는 배경은, 프롬프트를 문자열로 다루는 대신 **시그니처/모듈/옵티마이저**로 구성된 프로그램을 “컴파일”해, 명시한 메트릭을 최대화하도록 프롬프트(또는 가중치)를 자동 최적화하는 접근이 확산되었기 때문입니다. [\[54\]](https://www.youtube.com/watch?v=-cKUW6n8hBU&utm_source=chatgpt.com) (강연 요약 성격의 2차 자료들도 같은 방향의 설명을 제공합니다.) [\[55\]](https://www.zenml.io/llmops-database/building-production-llm-applications-with-dspy-framework?utm_source=chatgpt.com)

DSPy 문서에 따르면 옵티마이저는 (1) DSPy 프로그램, (2) 메트릭, (3) 학습/평가용 데이터셋을 입력으로 받아, 지정한 목표(예: 정확도)를 최대화하도록 프로그램의 프롬프트/파라미터를 조정합니다. [\[56\]](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com) 그리고 DSPy의 원 논문은 하드코딩된 프롬프트 템플릿의 시행착오를 넘어, **선언적 모듈과 컴파일러로 파이프라인을 최적화**하는 모델을 제안합니다. [\[57\]](https://arxiv.org/abs/2310.03714?utm_source=chatgpt.com)

이 접근은 Agent Observability 관점에서 두 가지 함의를 갖습니다. 1\) “좋은 평가 메트릭이 없으면 자동 최적화도 없다” — 즉 Evaluation 설계가 곧 학습 신호가 됩니다. [\[58\]](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com)  
2\) 자동 최적화는 오히려 “측정 가능한 스키마(트레이스/데이터셋/버전)”가 없으면 재현/감사가 불가능해, Observability의 요구 수준을 끌어올립니다. [\[59\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)

## End-to-End 참조 아키텍처: Observe → Evaluate → Improve를 시스템으로 만드는 법

이 섹션은 “현업에서 구현 가능한” 형태로 관측성/평가/개선 루프를 아키텍처로 정리합니다. 설계의 목적은 단순 대시보드가 아니라, **개선 작업이 계속 누적되어도 품질·안전·비용이 관리되는 실행 시스템**입니다. [\[60\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/)

### 데이터 계층: 트레이스 스키마와 식별자

최소 요구 스키마는 다음과 같습니다. \- Trace/Span: 에이전트 세션(루트)과 하위 단계(LLM 호출/검색/툴 호출/검증)를 span으로 분할. [\[61\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)  
\- Attributes: gen\_ai.\*(모델/프로바이더/토큰), error.type, tool 호출 정보, 비용/지연/종료 이유 등. [\[62\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)  
\- Version tags: prompt template 버전/하네스 버전/툴 버전/정책 버전/데이터 리트리벌 버전. (슬라이스 회귀 테스트의 전제) [\[63\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)  
\- Privacy controls: input/output 상세 이벤트는 opt-in \+ redaction \+ 저장 기간 정책. [\[64\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/)

### 평가 계층: 오프라인·온라인·휴먼 루프의 결합

평가 파이프라인은 보통 3개 레일로 구성합니다. \- 오프라인(개발/PR/릴리스): 고정 데이터셋(“골든 세트”)에서 회귀 탐지. LangChain이 말하는 single-step/unit \+ full-turn/integration 조합이 여기 들어갑니다. [\[65\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)  
\- 온라인(운영): 샘플링된 트레이스에 judge/code-eval을 부착해 품질/안전 드리프트를 감지(알림). [\[66\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/)  
\- 휴먼(라벨링/감사): 온라인에서 “중요하지만 자동평가가 불안정한” 축(규정 준수/브랜드 리스크)을 사람에게 에스컬레이션. 이후 judge 검증 세트로 환류. [\[67\]](https://www.arxiv.org/abs/2602.00521)

MLflow[\[68\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) 문서는 에이전트 평가가 “트레이스와 스코어러를 사용해 중간 행동까지 평가”해야 하며, 평가 데이터셋도 기존 트레이스에서 추출해 만들 수 있다고 정리합니다. [\[69\]](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/) 이는 “운영 실패가 곧 테스트 케이스가 된다”는 LangChain의 주장과 구조적으로 일치합니다. [\[70\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/)

### 개선 계층: 변경의 단위와 안전장치

개선(Improvement)은 크게 네 레벨에서 일어납니다. 1\) **프롬프트/규칙(정책)**: Prompt Learning/DSPy류의 자동·반자동 업데이트, Prompt Playground에서 버전 관리. [\[71\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com)  
2\) **하네스/미들웨어**: self-verification 훅, 루프 감지, 컨텍스트 주입, 추론 예산 스케줄링. [\[72\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)  
3\) **툴링/환경**: 평가 샌드박스, 환경 리셋, API mocking/replay로 재현성 강화. [\[73\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)  
4\) **거버넌스/보안**: CI/CD gating, 레드팀, 모니터링 기반 정책 집행. [\[74\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/)

특히 Azure는 “(1) 벤치마크 기반 모델 선택, (2) 개발·운영에서의 지속 평가, (3) CI/CD 통합, (4) 레드팀, (5) 운영 모니터링”을 5대 관측성 베스트 프랙티스로 제시하며, 평가·거버넌스를 수명주기 전 단계에 내장할 것을 권고합니다. [\[7\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) OpenTelemetry[\[75\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) 측도 에이전트 관측성의 표준화된 semantic conventions 정립과, 프레임워크 간 상호운용이 중요하다고 강조합니다. [\[76\]](https://opentelemetry.io/blog/2025/ai-agent-observability/)

## 사례 중심 정리와 실무 체크포인트

### 코딩 에이전트: “평가 환경”이 곧 제품 품질

코딩 에이전트는 결과가 코드/파일/테스트 통과 여부로 명확해 code eval과 궁합이 좋습니다. SWE-bench는 GitHub 이슈를 기반으로 패치를 생성하고 Docker로 평가를 재현 가능하게 하는 harness를 제공한다고 명시합니다. [\[77\]](https://github.com/SWE-bench/SWE-bench) Terminal-Bench 2.0도 컨테이너 환경과 테스트 기반 검증, outcome-driven 평가를 강조합니다. [\[50\]](https://arxiv.org/html/2601.11868v1)

이 영역에서 Observability의 실무적 포인트는: \- “무슨 코드를 썼는가”가 아니라 “어떤 순서로 탐색/수정/검증했는가(trajectory)”가 비용·성공률을 좌우하므로, 툴 호출/파일 변경/테스트 실행을 span으로 남겨야 합니다. [\[78\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)  
\- 동일 벤치마크에서도 하네스가 성능을 크게 바꿀 수 있으므로, 모델 성능과 하네스 성능을 분리 측정해야 합니다. [\[79\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

### RAG/업무 에이전트: 환각 방지와 “모른다” 응답의 제품화

Arize의 Prompt Playground 예시는 “근거가 없으면 ‘I don’t know’라고 답하라”는 지시로 환각을 줄이는 패턴을 보여주며, 운영 단계에서 템플릿 비교/평가가 중요해진다고 설명합니다. [\[80\]](https://arize.com/blog-course/evaluating-prompt-playground/) 이런 패턴은 RAG 평가에서도 자주 등장하며, RAGAs는 정답 레이블이 없어도 RAG 파이프라인의 여러 차원을 평가하는 메트릭 세트를 제안합니다. [\[81\]](https://arxiv.org/abs/2309.15217?utm_source=chatgpt.com)

다만 “모른다” 응답은 사용자 경험/업무 KPI와 충돌할 수 있으므로, 온라인 평가에서는 (a) groundedness/faithfulness, (b) refusal 적정성, (c) 업무 성공률을 함께 모니터링해야 합니다. [\[82\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/)

### 멀티에이전트·시스템 레벨 관측성: 의미(semantic)와 행위(system)를 연결

애플리케이션 내부 트레이싱은 “에이전트의 의도/도구 호출”은 잘 보지만, 시스템 호출·프로세스 간 영향까지 완전히 잡기 어렵습니다. AgentSight 연구는 eBPF 기반의 “boundary tracing”으로 TLS 트래픽에서 의미 정보를 추출하고, 커널 이벤트와 인과적으로 연결해 프레임워크 무관·낮은 오버헤드의 관측성을 제안합니다. [\[83\]](https://arxiv.org/abs/2508.02736?utm_source=chatgpt.com) 이 계열은 특히 **보안(프롬프트 인젝션), 자원 낭비 루프, 멀티에이전트 병목** 같은 운영 문제를 “시스템 증거”로 포착한다는 점에서, 기존 LLM 트레이싱을 보완하는 방향으로 해석할 수 있습니다. [\[84\]](https://arxiv.org/abs/2508.02736?utm_source=chatgpt.com)

### 실무 체크리스트: 실패하기 쉬운 지점들

* **평가 프롬프트의 품질이 곧 최적화의 한계**입니다. DSPy/Prompt Learning 모두 “메트릭(평가)”이 학습 신호이므로, 평가가 부실하면 개선도 오작동합니다. [\[85\]](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com)

* **LLM-as-a-Judge 편향/불안정성**(자기 선호, 프롬프트 변형 민감도)을 전제로, 최소한의 신뢰성 진단(일관성/휴먼 정렬)을 수행해야 합니다. [\[86\]](https://arxiv.org/abs/2410.21819)

* **회귀 테스트는 슬라이스 기반**으로 설계해야 하며, 모델·프롬프트·하네스 버전 태깅이 없으면 “왜 변했는지”를 설명할 수 없습니다. [\[87\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)

* **민감정보**는 opt-in/마스킹/보존 정책이 시스템 레벨로 필요합니다(Collector에서의 redaction 권장). [\[88\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)

* **테스트 환경 리셋/재현성**이 없으면(특히 상태ful 에이전트) 평가 자체가 flaky해집니다. [\[73\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)

결론적으로, “에이전트 관측성”은 대시보드 기능이 아니라 **(표준화된 트레이스) \+ (신뢰 가능한 평가) \+ (개선이 배포로 이어지는 공학적 루프)**로 정의되는 운영 체계입니다. 이 체계가 갖춰질 때, LangChain이 말하는 “트레이스 중심 개발”(디버깅/테스트/최적화/모니터링의 중심 이동)이 현실의 엔지니어링 프로세스로 구현됩니다. [\[89\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/)

---

[\[1\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) [\[4\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) [\[39\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) [\[40\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) [\[41\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) [\[75\]](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/) From Traces to Insights: Understanding Agent Behavior at Scale

[https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/](https://blog.langchain.com/from-traces-to-insights-understanding-agent-behavior-at-scale/)

[\[2\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) [\[10\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) [\[11\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) [\[12\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) [\[46\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) [\[70\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) [\[89\]](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/) In software, the code documents the app. In AI, the traces do.

[https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/](https://blog.langchain.com/in-software-the-code-documents-the-app-in-ai-the-traces-do/)

[\[3\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[29\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[45\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[47\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[48\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[49\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[68\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[72\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[78\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) [\[79\]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) Improving Deep Agents with harness engineering

[https://blog.langchain.com/improving-deep-agents-with-harness-engineering/](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

[\[5\]](https://arxiv.org/abs/2303.16634) [\[36\]](https://arxiv.org/abs/2303.16634) \[2303.16634\] G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

[https://arxiv.org/abs/2303.16634](https://arxiv.org/abs/2303.16634)

[\[6\]](https://arize.com/docs/ax/prompts/prompt-optimization/prompt-learning-sdk?utm_source=chatgpt.com) [\[52\]](https://arize.com/docs/ax/prompts/prompt-optimization/prompt-learning-sdk?utm_source=chatgpt.com) Prompt Learning via SDK \- Arize AX Docs

[https://arize.com/docs/ax/prompts/prompt-optimization/prompt-learning-sdk?utm\_source=chatgpt.com](https://arize.com/docs/ax/prompts/prompt-optimization/prompt-learning-sdk?utm_source=chatgpt.com)

[\[7\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) [\[8\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) [\[20\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) [\[60\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) [\[66\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) [\[74\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) [\[82\]](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/) Agent Factory: Top 5 agent observability best practices for reliable AI | Microsoft Azure Blog

[https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/](https://azure.microsoft.com/en-us/blog/agent-factory-top-5-agent-observability-best-practices-for-reliable-ai/)

[\[9\]](https://arize.com/ai-agents/agent-observability/) [\[13\]](https://arize.com/ai-agents/agent-observability/) [\[34\]](https://arize.com/ai-agents/agent-observability/) Agent Observability and Tracing

[https://arize.com/ai-agents/agent-observability/](https://arize.com/ai-agents/agent-observability/)

[\[14\]](https://opentelemetry.io/blog/2025/ai-agent-observability/) [\[18\]](https://opentelemetry.io/blog/2025/ai-agent-observability/) [\[76\]](https://opentelemetry.io/blog/2025/ai-agent-observability/) AI Agent Observability \- Evolving Standards and Best Practices | OpenTelemetry

[https://opentelemetry.io/blog/2025/ai-agent-observability/](https://opentelemetry.io/blog/2025/ai-agent-observability/)

[\[15\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) [\[19\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) [\[21\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) [\[88\]](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) Datadog LLM Observability natively supports OpenTelemetry GenAI Semantic Conventions | Datadog

[https://www.datadoghq.com/blog/llm-otel-semantic-convention/](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)

[\[16\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) [\[61\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) [\[62\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) Semantic conventions for generative client AI spans | OpenTelemetry

[https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)

[\[17\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) [\[64\]](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) Semantic conventions for Generative AI events | OpenTelemetry

[https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/)

[\[22\]](https://learn.microsoft.com/en-us/agent-framework/agents/observability) [\[23\]](https://learn.microsoft.com/en-us/agent-framework/agents/observability) Observability | Microsoft Learn

[https://learn.microsoft.com/en-us/agent-framework/agents/observability](https://learn.microsoft.com/en-us/agent-framework/agents/observability)

[\[24\]](https://arxiv.org/html/2507.21504v1) [\[28\]](https://arxiv.org/html/2507.21504v1) Evaluation and Benchmarking of LLM Agents: A Survey

[https://arxiv.org/html/2507.21504v1](https://arxiv.org/html/2507.21504v1)

[\[25\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com) [\[51\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com) [\[71\]](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com) Prompt Learning: Using English Feedback to Optimize LLM ...

[https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm\_source=chatgpt.com](https://arize.com/blog/prompt-learning-using-english-feedback-to-optimize-llm-systems/?utm_source=chatgpt.com)

[\[26\]](https://docs.snowflake.com/ko/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications) [\[27\]](https://docs.snowflake.com/ko/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications) AI 애플리케이션 평가하기 | Snowflake Documentation

[https://docs.snowflake.com/ko/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications](https://docs.snowflake.com/ko/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications)

[\[30\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/) [\[31\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/) [\[32\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/) [\[33\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/) [\[65\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/) [\[73\]](https://blog.langchain.com/evaluating-deep-agents-our-learnings/) Evaluating Deep Agents: Our Learnings

[https://blog.langchain.com/evaluating-deep-agents-our-learnings/](https://blog.langchain.com/evaluating-deep-agents-our-learnings/)

[\[35\]](https://arxiv.org/abs/2410.21819) [\[86\]](https://arxiv.org/abs/2410.21819) \[2410.21819\] Self-Preference Bias in LLM-as-a-Judge

[https://arxiv.org/abs/2410.21819](https://arxiv.org/abs/2410.21819)

[\[37\]](https://www.arxiv.org/abs/2602.00521) [\[38\]](https://www.arxiv.org/abs/2602.00521) [\[67\]](https://www.arxiv.org/abs/2602.00521) \[2602.00521\] Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory

[https://www.arxiv.org/abs/2602.00521](https://www.arxiv.org/abs/2602.00521)

[\[42\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf) [\[43\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf) [\[44\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf) [\[59\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf) [\[63\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf) [\[87\]](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf) (Why) Is My Prompt Getting Worse? Rethinking Regression Testing for Evolving LLM APIs

[https://www.cs.cmu.edu/\~cyang3/papers/cain24.pdf](https://www.cs.cmu.edu/~cyang3/papers/cain24.pdf)

[\[50\]](https://arxiv.org/html/2601.11868v1) Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces

[https://arxiv.org/html/2601.11868v1](https://arxiv.org/html/2601.11868v1)

[\[53\]](https://arize.com/blog-course/evaluating-prompt-playground/) [\[80\]](https://arize.com/blog-course/evaluating-prompt-playground/) Evaluating Prompts: A Developer’s Guide \- Arize AI

[https://arize.com/blog-course/evaluating-prompt-playground/](https://arize.com/blog-course/evaluating-prompt-playground/)

[\[54\]](https://www.youtube.com/watch?v=-cKUW6n8hBU&utm_source=chatgpt.com) DSPy: The End of Prompt Engineering \- Kevin Madura ...

[https://www.youtube.com/watch?v=-cKUW6n8hBU\&utm\_source=chatgpt.com](https://www.youtube.com/watch?v=-cKUW6n8hBU&utm_source=chatgpt.com)

[\[55\]](https://www.zenml.io/llmops-database/building-production-llm-applications-with-dspy-framework?utm_source=chatgpt.com) AlixPartners: Building Production LLM Applications with DSPy ...

[https://www.zenml.io/llmops-database/building-production-llm-applications-with-dspy-framework?utm\_source=chatgpt.com](https://www.zenml.io/llmops-database/building-production-llm-applications-with-dspy-framework?utm_source=chatgpt.com)

[\[56\]](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com) [\[58\]](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com) [\[85\]](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com) Optimizers \- DSPy

[https://dspy.ai/learn/optimization/optimizers/?utm\_source=chatgpt.com](https://dspy.ai/learn/optimization/optimizers/?utm_source=chatgpt.com)

[\[57\]](https://arxiv.org/abs/2310.03714?utm_source=chatgpt.com) DSPy: Compiling Declarative Language Model Calls into ...

[https://arxiv.org/abs/2310.03714?utm\_source=chatgpt.com](https://arxiv.org/abs/2310.03714?utm_source=chatgpt.com)

[\[69\]](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/) Evaluating Agents | MLflow

[https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/agents/)

[\[77\]](https://github.com/SWE-bench/SWE-bench) GitHub \- SWE-bench/SWE-bench: SWE-bench: Can Language Models Resolve Real-world Github Issues?

[https://github.com/SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench)

[\[81\]](https://arxiv.org/abs/2309.15217?utm_source=chatgpt.com) Automated Evaluation of Retrieval Augmented Generation

[https://arxiv.org/abs/2309.15217?utm\_source=chatgpt.com](https://arxiv.org/abs/2309.15217?utm_source=chatgpt.com)

[\[83\]](https://arxiv.org/abs/2508.02736?utm_source=chatgpt.com) [\[84\]](https://arxiv.org/abs/2508.02736?utm_source=chatgpt.com) AgentSight: System-Level Observability for AI Agents Using eBPF

[https://arxiv.org/abs/2508.02736?utm\_source=chatgpt.com](https://arxiv.org/abs/2508.02736?utm_source=chatgpt.com)