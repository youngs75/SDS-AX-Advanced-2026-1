# Context Engineering for AI Agents: Fresh Lessons from Building Manus

## 기본 정보
- **저자**: @peakji (Manus AI)
- **발행일/출처**: LangChain Webinar 발표 자료, 날짜 미상 (2025년 추정)
- **페이지 수**: 24페이지 (프레젠테이션 슬라이드)
- **키워드**: Context Engineering, AI Agents, Context Reduction, Compaction, Summarization, Context Isolation, Context Offloading, Hierarchical Action Space, Manus AI

## 한줄 요약
> 이 발표는 Manus AI 에이전트를 구축하면서 얻은 컨텍스트 엔지니어링의 실전 교훈을 공유하며, 컨텍스트 축소(Reduction), 격리(Isolation), 오프로딩(Offloading)의 세 가지 핵심 전략과 "덜 만들고, 더 이해하라(Build less, understand more)"는 철학을 제시한다.

## 초록 (Abstract)
이 프레젠테이션은 Manus AI의 AI 에이전트 구축 경험에서 얻은 컨텍스트 엔지니어링에 대한 비합의적(non-consensus) 관점과 실전 교훈을 다룬다. 왜 컨텍스트 엔지니어링이 중요한지, 그리고 축소(Reduction), 격리(Isolation), 오프로딩(Offloading)의 세 가지 핵심 기법을 상세히 설명한다.

## 상세 내용

### 1. 왜 컨텍스트 엔지니어링인가? (Why Context Engineering?)

#### 첫 번째 함정: 자체 모델 학습의 유혹
- "왜 자체 모델을 학습하지 않는가?"라는 유혹이 있다.
- 파인 튜닝과 후처리(post-training)는 이전보다 쉬워졌다.
- **현실 점검**: 모델 반복이 제품 반복을 제한한다.
- **핵심 메시지**: PMF(Product-Market Fit) 달성 전에 전문화된 모델을 구축하지 말라!

#### 두 번째 함정: RL 파인 튜닝의 유혹
- 상황이 안정되면 RL로 파인 튜닝하려는 유혹이 온다.
- "행동 공간 + 보상 + 롤아웃을 정의하자..."
- "...하지만 MCP를 지원해야 한다!"
- **핵심 메시지**: 베이스 모델 회사가 이미 가지고 있는 것을 다시 만들지 말라.

#### 컨텍스트 엔지니어링의 위치
> "Context Engineering is the clearest and most practical boundary between application and model."

컨텍스트 엔지니어링은 애플리케이션과 모델 사이의 가장 명확하고 실용적인 경계이다.

### 2. 컨텍스트 축소 (Context Reduction)

#### 압축(Compaction) vs 요약(Summarization)
두 가지 서로 다른 축소 전략을 명확히 구분한다:

**압축(Compaction)**:
- 도구 호출의 인자와 결과에서 불필요한 정보를 제거하고 필요시 다시 검색할 수 있는 참조만 남긴다.
- 예: `file_write`의 전체 내용을 경로만으로 압축하고 "필요시 /home/ubuntu/foo.txt에서 검색"이라는 메모 추가
- 예: `browser_navigate`의 전체 페이지 내용을 "필요시 https://example.com을 다시 방문"으로 압축

**요약(Summarization)**:
- 도구 호출 결과를 파일 시스템에 저장하고, 전체 컨텍스트를 LLM이 생성한 요약으로 대체한다.
- 예: "사용자가 ...을 원한다. 다음 경로에 3개의 파일을 생성했다: ..."

#### "사전 부패 임계값" (Pre-rot threshold)
- 컨텍스트가 특정 크기를 넘으면 성능이 저하되기 시작한다.
- 128K 토큰 전에 압축을 시작해야 하며, 200K이나 1M까지 기다리면 안 된다.
- "그냥 무시해라..."는 불가능하다.

#### 압축 전략의 시각화
컨텍스트 길이 대비 턴 수 그래프에서, 여러 차례의 압축을 적용하여 컨텍스트 길이를 32K-96K 범위 내로 유지하면서 250턴 이상의 대화를 처리할 수 있음을 보여준다.

### 3. 컨텍스트 격리 (Context Isolation)

#### 멀티 에이전트의 동기화 오버헤드
- 다중 에이전트 설정은 무거운 동기화 오버헤드를 생성한다.
- 이는 새로운 문제가 아니다 -- 고전적인 동시성 문제와 같다.
- 프로그래밍 언어에서의 지혜를 빌려올 수 있다.

#### Go 언어의 철학 적용
> "Do not communicate by sharing memory; instead, share memory by communicating."

"메모리를 공유하여 통신하지 말고, 통신하여 메모리를 공유하라." (Go 언어 공식 블로그)

이를 에이전트 컨텍스트에 적용하면:

**통신을 통한 격리 (By communicating)**:
- 메인 에이전트가 시스템 프롬프트, 도구, 컨텍스트를 가지고 있다.
- 서브 에이전트에게 명령(Instruction)을 보내고 결과(Result)를 받는다.
- 서브 에이전트는 자체 시스템 프롬프트와 도구를 가지며, 메인 에이전트의 컨텍스트에 접근하지 않는다.

**컨텍스트 공유를 통한 격리 (By sharing context) - "Fork"**:
- 메인 에이전트의 컨텍스트를 "포크"하여 서브 에이전트에 전달한다.
- 서브 에이전트는 포크된 컨텍스트 위에서 작업하고 결과를 반환한다.

### 4. 컨텍스트 오프로딩 (Context Offloading)

#### 문제 인식
- 보통 작업 메모리를 파일에 저장하는 것을 의미한다.
- 하지만 **도구 자체도 컨텍스트를 어지럽힌다**.
- 너무 많은 도구 = 혼란, 잘못된 호출.
- "도구도 오프로드하면 어떨까?"

#### 계층적 행동 공간 (Hierarchical Action Space)
세 가지 추상화 수준:

**Level 1: Function Calling**
- 표준적이고 스키마 안전하지만:
  - 변경할 때마다 캐시가 깨진다.
  - 너무 많은 함수 = 컨텍스트 혼란.

**Level 2: Sandbox Utilities**
- 각 세션이 완전한 VM 샌드박스에서 실행된다.
- 모델이 쉘 유틸리티(CLI)를 호출할 수 있다.
- 모델 컨텍스트를 건드리지 않고 쉽게 확장 가능하다.
- 큰 출력 = 파일에 쓰기.
- "네, 제 에이전트는 sudo를 할 수 있습니다."

**Level 3: Packages & APIs**
- Manus가 사전 승인된 API를 호출하는 Python 스크립트를 작성한다.
- 데이터 중심적이거나 연쇄적인 태스크에 적합하다.
- 예: 도시 가져오기 -> ID 가져오기 -> 날씨 가져오기 -> 요약
- 모델 컨텍스트를 깨끗하게 유지하고, 메모리는 추론에만 사용한다.

#### 모든 수준의 통합
모든 레이어가 여전히 표준 함수 호출을 프록시로 사용한다. 깨끗한 인터페이스, 캐시 친화적, 직교 설계:
- Functions: message, shell, search, file, browser
- Sandbox: `$ manus-mcp-cli`, `$ manus-render-diagram`, `$ python ./analyze_data.py`
- Scripts: `create_3d_model.py`, `analyze_data.py`

### 5. 전체 통합 (Bringing It All Together)
- 오프로드 + 검색 = 축소를 가능하게 함
- 신뢰할 수 있는 검색 = 격리를 가능하게 함
- 격리 = 축소의 빈도를 줄임
- 모두 캐시 최적화 하에서

### 6. 과도한 컨텍스트 엔지니어링 경계 (Avoid Context Over-Engineering)
- 더 많은 컨텍스트 ≠ 더 많은 지능
- 단순화가 확장을 이긴다
- **가장 큰 성과는 추가가 아닌 제거에서 나왔다**
- 경계를 명확히 하고, 모델의 길에서 비켜라!

> "Build less, understand more."

## 핵심 키 포인트
1. PMF 달성 전에 자체 모델을 학습하지 말고, 베이스 모델 회사가 이미 가진 것을 다시 만들지 말라 -- 컨텍스트 엔지니어링이 애플리케이션과 모델의 경계이다.
2. 컨텍스트 압축(Compaction)과 요약(Summarization)은 별개의 전략이며, 둘 다 필요하다. 압축은 불필요한 세부사항을 제거하고 참조를 남기며, 요약은 전체를 축약한다.
3. "사전 부패 임계값"이 존재하여 컨텍스트가 너무 커지기 전에 축소를 시작해야 한다.
4. 다중 에이전트 간 컨텍스트 격리는 Go 언어의 동시성 철학("통신으로 메모리를 공유")을 빌려와 적용할 수 있다.
5. 도구 오프로딩을 위한 계층적 행동 공간(Function Calling / Sandbox / Packages)이 컨텍스트를 깨끗하게 유지한다.
6. 축소, 격리, 오프로딩은 상호 보완적이며 캐시 최적화 하에서 함께 작동한다.
7. 더 많은 컨텍스트가 더 많은 지능을 의미하지 않으며, 가장 큰 성과는 제거에서 나온다.

## 주요 인용 (Key Quotes)
> "Context Engineering is the clearest and most practical boundary between application and model." (Why Context Engineering 슬라이드)

> "Don't build specialized models before PMF!" (The First Trap 슬라이드)

> "Don't rebuild what base-model companies already have." (The Second Trap 슬라이드)

> "Do not communicate by sharing memory; instead, share memory by communicating." (Context Isolation 슬라이드, Go 언어 공식 블로그 인용)

> "Yes, my agent can sudo." (Level 2: Sandbox Utilities 슬라이드)

> "More context ≠ more intelligence" (Avoid Context Over-Engineering 슬라이드)

> "Our biggest gains came from removing, not adding." (Avoid Context Over-Engineering 슬라이드)

> "Build less, understand more." (마지막 슬라이드)

## 시사점 및 의의
이 프레젠테이션은 실제 프로덕션 AI 에이전트(Manus)를 구축한 경험에서 나온 실전적 교훈을 제공한다는 점에서 높은 가치가 있다:

1. **실용적 경계 설정**: 자체 모델 학습과 컨텍스트 엔지니어링 사이의 명확한 경계를 제시하여, 스타트업과 개발팀이 자원을 어디에 투자해야 하는지에 대한 실용적 가이드를 제공한다. PMF 이전에 모델 학습에 투자하지 말라는 조언은 특히 중요하다.

2. **동시성 프로그래밍 패턴의 차용**: Go 언어의 CSP(Communicating Sequential Processes) 철학을 다중 에이전트 컨텍스트 관리에 적용한 것은 소프트웨어 엔지니어링의 오랜 지혜를 AI 에이전트 설계에 접목하는 창의적인 접근이다.

3. **계층적 도구 설계**: Function Calling → Sandbox → Packages & APIs의 3계층 구조는 도구 사용의 확장성과 컨텍스트 효율성을 동시에 달성하는 실용적인 패턴을 제시한다.

4. **반직관적 통찰**: "더 많은 컨텍스트 ≠ 더 많은 지능"이라는 메시지와 "가장 큰 성과는 제거에서 나왔다"는 경험은 컨텍스트 윈도우를 최대한 채우려는 일반적 경향에 대한 중요한 경고이다.

5. **캐시 최적화 중심 설계**: 축소, 격리, 오프로딩의 세 전략이 모두 캐시 최적화라는 하나의 목표 아래 통합되어 있다는 점은 시스템 설계의 일관성과 효율성을 보여준다.

6. **에이전트 시스템 설계의 성숙**: 이 발표는 AI 에이전트 구축이 단순한 프롬프트 엔지니어링을 넘어 시스템 엔지니어링의 영역으로 진입했음을 보여주며, 운영체제와 분산 시스템의 개념(메모리 관리, 프로세스 격리, 계층적 추상화)이 에이전트 설계에 직접 적용됨을 시사한다.
