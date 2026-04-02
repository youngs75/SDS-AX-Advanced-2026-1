# A Large-Scale Study on the Development and Issues of Multi-Agent AI Systems

## 기본 정보
- **저자**: Daniel Liu, Krishna Upadhyay, Vinaik Chhetri (Louisiana State University), A.B. Siddique (University of Kentucky), Umar Farooq (Louisiana State University)
- **발행일/출처**: arXiv:2601.07136v1, 2026년 1월 12일
- **페이지 수**: 8페이지
- **키워드**: Multi-Agent Systems, Software Repositories, Software Mining, Software Maintenance, LangChain, CrewAI, AutoGen, LLM

## 한줄 요약
> 이 논문은 8개 주요 오픈소스 멀티 에이전트 AI 시스템에 대한 최초의 대규모 실증 연구를 수행하여, 42K+ 커밋과 4.7K+ 이슈를 분석하고 세 가지 개발 프로필(지속적, 안정적, 폭발적)을 식별하며, 기능 향상 중심의 개발과 버그/인프라/에이전트 조율 문제의 빈도를 밝혀 생태계의 추진력과 취약성을 동시에 보여준다.

## 초록 (Abstract)
LangChain, CrewAI, AutoGen을 포함한 멀티 에이전트 AI 시스템(MAS)의 급격한 출현은 대규모 언어 모델(LLM) 애플리케이션의 개발과 오케스트레이션 방식을 형성했다. 그러나 이러한 시스템이 실제로 어떻게 진화하고 유지되는지에 대해서는 알려진 바가 거의 없다. 이 논문은 오픈소스 MAS에 대한 최초의 대규모 실증 연구를 제시하며, 8개 주요 시스템에 걸쳐 42K 이상의 고유 커밋과 4.7K 이상의 해결된 이슈를 분석한다.

분석 결과 세 가지 뚜렷한 개발 프로필을 식별한다: 지속적(sustained), 안정적(steady), 폭발적(burst-driven). Perfective 커밋이 전체 변경의 40.8%를 차지하여 기능 향상이 수정적 유지보수(27.4%)와 적응적 업데이트(24.3%)보다 우선시됨을 시사한다. 이슈 데이터는 가장 빈번한 관심사가 버그(22%), 인프라(14%), 에이전트 조율 문제(10%)임을 보여준다. 중앙값 해결 시간은 1일 미만에서 약 2주까지 범위이며, 분포는 빠른 응답 쪽으로 치우쳐 있지만 소수의 이슈가 장기간의 주의를 필요로 한다.

## 상세 내용

### 1. 서론 (Introduction)
대규모 언어 모델(LLM)의 출현은 자율 에이전트 간 협업을 지원하는 멀티 에이전트 시스템(MAS)의 설계에 큰 영향을 미쳤다. AutoGen, CrewAI, LangChain 같은 프레임워크는 에이전트 생성, 통신, 조율을 위한 추상화를 제공한다. 이러한 시스템은 추론, 계획, 도구 사용을 통합하는 워크플로우 구축을 지원하며 현대 LLM 기반 소프트웨어 개발의 핵심 구성요소가 되었다.

그러나 채택이 증가하고 있음에도 불구하고, 이러한 시스템이 어떻게 진화하고 유지되는지에 대해서는 알려진 바가 거의 없다. 기존 MAS 연구는 주로 알고리즘적, 아키텍처적 발전에 초점을 맞추었으며, 소프트웨어 개발 관행에 대한 실증적 이해는 제한적이다. 이 격차는 이러한 시스템이 새로운 모델, API, 오케스트레이션 메커니즘을 지원하면서도 안정성과 장기적 지속 가능성을 유지하기 위해 빠르게 진화해야 하기 때문에 중요하다.

### 2. 배경 및 관련 연구 (Background and Related Work)

#### 2.1 멀티 에이전트 AI 시스템
단일 만능 모델에 의존하는 대신, 개발자들은 여러 전문 에이전트가 협력하여 복잡한 문제를 해결하는 멀티 에이전트 아키텍처로 전환하고 있다. 연구 대상 8개 프레임워크:

| MAS | 아키텍처 | GitHub Stars |
|---|---|---|
| AutoGen | 대화형 워크플로우 | 51.3K |
| CrewAI | 역할 기반 계층 | 40K |
| Haystack | 그래프 기반 파이프라인 | 23.2K |
| LangChain | 모듈형 체인 | 119K |
| Letta | 메모리 중심 에이전트 | 19K |
| LlamaIndex | 데이터 중심 파이프라인 | 45K |
| Semantic Kernel | 플러그인 기반 오케스트레이터 | 26.6K |
| SuperAGI | 확장 가능 에이전트 툴킷 | 16.8K |

설계 철학의 차이에도 불구하고, 이 프레임워크들은 에이전트 정의, 에이전트 간 통신, 태스크 분해, 실행 제어를 위한 공통 아키텍처 기반을 공유한다.

#### 2.2 관련 연구
소프트웨어 저장소 마이닝은 소프트웨어가 어떻게 구축되고 유지되는지 이해하는 데 오랫동안 중요한 방법이었다. 그러나 이러한 분석이 멀티 에이전트 AI 시스템에는 아직 적용되지 않았다. MAS에 대한 최근 연구는 주로 아키텍처와 알고리즘적 혁신에 초점을 맞추고 있으며(Guo et al., Xi et al.), AgentBench 같은 벤치마킹 노력이 에이전트 협업 평가를 위한 표준화된 태스크를 제공한다.

### 3. 방법론 (Methodology)

#### 3.1 연구 목표와 연구 질문
두 가지 핵심 연구 질문(RQ):

**RQ1. MAS 저장소 간 개발 패턴은 어떻게 다른가?**
- RQ1.1: 저장소별 뚜렷한 커밋 활동 패턴은 무엇인가?
- RQ1.2: MAS 저장소에서 커밋 유형의 분포는 어떠한가?

**RQ2. MAS 프레임워크의 이슈 보고 패턴과 해결 특성은 무엇인가?**
- RQ2.1: 프레임워크 전반에서 이슈 패턴은 시간에 따라 어떻게 진화하는가?
- RQ2.2: 프레임워크 전반에서 가장 빈번한 이슈 유형은 무엇인가?

#### 3.2 데이터셋 구성
- GitHub GraphQL API를 사용하여 10,813개의 닫힌 이슈를 추출
- PR이 연결된 이슈로 필터링하여 4,731개 이슈 확보
- 44,041개의 커밋에서 중복 제거 후 42,267개의 고유 커밋 확보
- 커밋 분류에 DistilBERT 모델(GitHub 커밋 메시지로 미세 조정) 사용
- 이슈 주제 분석에 BERTopic 사용

### 4. 결과 (Results)

#### 4.1 개발 및 기여 패턴 (RQ1)

**RQ1.1: 커밋 활동 패턴**
세 가지 뚜렷한 개발 프로필을 식별:

1. **지속적(Sustained)**: LangChain이 약 14,000 커밋으로 선두이며, 2023년 중반부터 급성장 후 2025년까지 안정화.
2. **안정적(Steady)**: Haystack이 2020년 이후 가장 일관된 궤적을 보이며 변동 계수(CV) 48.6%로 가장 낮음.
3. **폭발적(Burst-driven)**: SuperAGI가 2023년 중반 급격한 스파이크 후 최소한의 활동을 보이며 CV 456.1%.

대부분의 프로젝트가 2023년에 가속화되어, AI 에이전트에 대한 관심 증가와 일치하는 광범위한 변곡점을 나타낸다.

**코드 변동(Code Churn) 패턴**:
- SuperAGI: 2023년 초 거의 300만 줄 추가 후 최소한의 유지보수 (빠른 프로토타이핑)
- Haystack: 2023년과 2024년에 대규모 삭제 이벤트(~140K, ~290K 줄)로 의도적 리팩토링
- LangChain: 반복적인 변동 피크(300K-400K 줄)와 2025년 중반 대규모 삭제 스파이크(~400K 줄)
- 생태계 수준에서 2023년부터 대규모 삭제가 동등한 삽입과 균형을 이루어, 확장보다 코드 구조조정으로의 전환을 시사

**RQ1.2: 커밋 유형 분포**
파인 튜닝된 DistilBERT로 42,266개 커밋을 분류:

| 커밋 유형 | 비율 |
|---|---|
| Perfective (기능 향상) | 40.83% |
| Corrective (버그 수정) | 27.36% |
| Adaptive (적응적 업데이트) | 24.30% |
| 혼합 유형 | 7.51% |

Perfective 유지보수가 모든 프레임워크에서 지배적(34.2-51.5%). Semantic Kernel이 가장 높은 Perfective 비율(51.5%)과 가장 낮은 Corrective 비율(18.3%)을 보여 성숙한 코드베이스를 시사. SuperAGI와 Letta는 더 높은 Corrective 비율(32.8%, 33.5%)로 덜 안정적인 아키텍처를 시사.

#### 4.2 이슈 환경 (RQ2)

**RQ2.1: 이슈 패턴의 시간적 진화**
- Haystack과 Semantic Kernel이 2025년까지 약 4,000개의 누적 이슈로 지배적
- SuperAGI는 2023년 중반 극적인 스파이크 후 급격히 감소하는 집중적 폭발 패턴
- 중앙값 해결 시간은 LlamaIndex의 약 1일에서 Semantic Kernel/Haystack의 10일 이상까지 다양
- 모든 프레임워크에서 평균이 중앙값을 일관되게 초과하여, 소수의 이슈가 상당히 더 긴 해결 시간을 요구하는 우편향 분포를 시사

**RQ2.2: 가장 빈번한 이슈 유형**

| 이슈 유형 | 비율 | 건수 |
|---|---|---|
| Language/Framework | 31% | 3,301 |
| Project Workflow | 24% | 2,563 |
| Triage | 23% | 2,439 |
| Bug | 22% | 2,355 |
| Infrastructure | 14% | 1,467 |
| Data Processing | 11% | 1,161 |
| Agent Issues | 10% | 1,049 |
| Documentation | 7% | 734 |
| Feature | 7% | 727 |
| Community | 6% | 682 |
| UX | 2% | 195 |

버그 보고가 가장 빈번한 제품 특정 관심사(22%). 인프라 이슈(14%)는 빌드 시스템, 배포 구성, CI/CD 파이프라인, 테스트 프레임워크, 의존성 등을 포함. 에이전트 관련 이슈(10%)는 에이전트 행동, 다중 에이전트 조율, 채팅 히스토리 관리, 계획 메커니즘, 함수 호출, 도구 사용에 초점.

**BERTopic 기반 주제 분석**:
- Agent Systems & Intelligence: 58.42% (에이전트 프레임워크 개발 14.58%, 채팅 시스템 10.50%, 계획 & 순차 실행 9.79%, AI 서비스 제공자 통합 9.48% 등)
- Technical Implementation & Operations: 32.51% (평가 & 성능 메트릭 8.97%, 모델 학습 & 파인 튜닝 6.83%, 함수 호출 & 예외 처리 6.52% 등)

### 5. 타당성 위협 (Threats to Validity)
- **내적 타당성**: GitHub 메타데이터와 프로젝트 활동에 의존하여 주변적이거나 프로토타입 저장소를 포함할 수 있음
- **외적 타당성**: GitHub의 오픈소스 프로젝트에만 기반하여 독점 MAS에 완전히 일반화되지 않을 수 있음
- **구성 타당성**: 커밋과 이슈에서 파생된 메트릭이 코드 품질이나 설계 근거 같은 질적 측면을 간과할 수 있음
- **신뢰성**: GitHub API나 저장소 구조의 변경이 향후 복제에 영향을 미칠 수 있음

### 6. 결론 (Conclusions)
멀티 에이전트 AI 시스템의 개발 및 유지보수 관행에 대한 최초의 대규모 실증 분석을 제시한다. 8개의 가장 저명한 오픈소스 프로젝트를 조사하여 42K+ 고유 커밋과 4.7K+ 해결된 이슈를 다룬다. 급속한 성장 중이지만 아직 안정화 과정에 있는 생태계를 밝혀낸다. 개발은 주로 기능 향상에 의해 주도되며, 이슈 데이터는 버그, 인프라 문제, 에이전트 조율 문제가 가장 흔한 문제임을 나타낸다. 2023년 이후 커밋과 이슈 활동의 급격한 증가는 MAS에서의 LLM 사용 증가에 힘입은 채택 증가와 커뮤니티 참여를 시사한다.

## 핵심 키 포인트
1. MAS 생태계는 세 가지 뚜렷한 개발 프로필을 보인다: 지속적(LangChain), 안정적(Haystack), 폭발적(SuperAGI).
2. Perfective 커밋이 40.8%로 지배적이며, 이는 반응적 버그 수정보다 지속적 개선을 우선시하는 능동적 소프트웨어 진화를 나타낸다.
3. 2023년이 MAS 생태계의 중요한 성장 기점이며, 대부분의 프레임워크가 이 시기에 개발을 강화했다.
4. 버그(22%), 인프라(14%), 에이전트 조율(10%)이 가장 빈번한 이슈 유형으로, 기본적인 소프트웨어 품질과 운영 안정성 문제가 MAS 고유의 기능 관련 이슈만큼 또는 더 빈번하다.
5. 이슈 해결 시간이 프레임워크 간 크게 다르며(중앙값 1일~2주), 성숙도와 유지보수 역량의 차이를 반영한다.
6. 2023년부터 코드 삭제가 삽입과 균형을 이루어, 확장에서 아키텍처 정제로의 전환을 시사한다.
7. 에이전트 시스템 & 인텔리전스(58%)가 개발자 관심의 대부분을 차지하지만, 기술적 구현 & 운영(33%)도 상당한 엔지니어링 도전을 나타낸다.

## 주요 인용 (Key Quotes)
> "little is known about how these systems evolve and are maintained in practice." (Abstract, p.1)

> "Perfective commits constitute 40.8% of all changes, suggesting that feature enhancement is prioritized over corrective maintenance (27.4%) and adaptive updates (24.3%)." (Abstract, p.1)

> "Issue reporting also increased sharply across all frameworks starting in 2023." (Abstract, p.1)

> "These results highlight both the momentum and the fragility of the current ecosystem, emphasizing the need for improved testing infrastructure, documentation quality, and maintenance practices." (Abstract, p.1)

> "The high frequency of Bug (22%) and Infrastructure (14%) labels, combined with the relatively modest proportion of Agent-specific labels (10%), indicates that foundational software quality and operational stability challenges are reported as frequently or more frequently than issues related to the unique multi-agent capabilities." (Section IV.B.2, p.6)

> "Most repositories intensified their development starting in 2023, suggesting a critical growth period for the ecosystem, though development consistency varies dramatically across projects with CV ranging from 48.6% to 456.1%." (Finding 1.1, p.4)

> "The prevalence of perfective commits indicates that the development of MAS prioritizes continuous improvement over reactive bug fixing, indicating a proactive approach to software evolution." (Finding 1.2, p.5)

> "developers prioritize agent capabilities (58%) while contending with operational challenges (33%), reflecting a feature-driven community facing persistent deployment barriers." (Finding 2.2, p.7)

## 시사점 및 의의
이 논문은 멀티 에이전트 AI 시스템의 소프트웨어 엔지니어링 관행에 대한 최초의 체계적이고 정량적인 분석이라는 점에서 중요한 기여를 한다:

1. **생태계 성숙도 평가**: 세 가지 개발 프로필의 식별은 MAS 프레임워크 선택 시 장기적 유지보수 가능성과 커뮤니티 지원의 차이를 이해하는 데 실용적인 가이드를 제공한다. LangChain과 Haystack 같은 지속적/안정적 프로필의 프레임워크가 프로덕션 사용에 더 적합할 수 있다.

2. **기능 중심 개발의 양면성**: Perfective 커밋이 지배적이라는 발견은 빠른 혁신의 지표이지만, 동시에 테스트, 문서화, 유지보수에 대한 투자가 상대적으로 부족할 수 있음을 경고한다. 이는 장기적 안정성에 위험 요소가 될 수 있다.

3. **에이전트 조율의 도전**: 에이전트 관련 이슈가 10%를 차지한다는 것은 멀티 에이전트 조율이 여전히 해결되지 않은 핵심 기술적 도전임을 보여준다. 에이전트 행동, 계획, 함수 호출, 도구 사용 관련 문제가 프레임워크 전반에서 반복적으로 나타난다.

4. **인프라 복잡성**: 인프라 이슈(14%)의 높은 빈도는 MAS의 운영 복잡성이 단순한 LLM API 호출을 넘어 빌드 시스템, 배포, CI/CD, 다양한 플랫폼 호환성 등 광범위한 엔지니어링 도전을 수반함을 보여준다.

5. **2023년 변곡점**: 2023년이 MAS 생태계의 결정적 성장 기점이라는 발견은 ChatGPT/GPT-4의 출시 이후 LLM 기반 에이전트에 대한 관심 폭발과 일치하며, 이 분야의 급속한 성장과 함께 수반되는 유지보수 부담의 증가를 시사한다.

6. **연구와 실천의 갭**: 이 연구는 MAS 연구가 주로 아키텍처와 알고리즘에 초점을 맞추는 반면, 실제 개발과 유지보수 관행에 대한 이해가 부족하다는 중요한 갭을 식별하고 채우기 시작한다. 향후 테스트 인프라, 문서 품질, 유지보수 관행의 개선이 장기적 신뢰성과 지속 가능성을 위해 필수적이다.
