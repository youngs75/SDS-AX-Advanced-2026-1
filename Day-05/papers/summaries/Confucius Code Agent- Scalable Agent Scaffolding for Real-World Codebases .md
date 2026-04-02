# Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases

## 기본 정보
- **저자**: Sherman Wong, Zhenting Qi, Zhaodong Wang, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Wenlin Chen, Yilun Du, Minlan Yu, Ying Zhang (Meta, Harvard)
- **발행일/출처**: arXiv:2512.10398v5, 2025년 12월 20일
- **페이지 수**: 25페이지
- **키워드**: Coding Agent, Agent Scaffolding, Software Engineering, Context Management, Working Memory, Note-Taking, Meta-Agent, SWE-Bench

## 한줄 요약
> 이 논문은 대규모 코드베이스에서 작동할 수 있는 소프트웨어 엔지니어링 에이전트인 Confucius Code Agent(CCA)를 소개하며, Agent Experience(AX), User Experience(UX), Developer Experience(DX) 세 축의 설계 철학과 계층적 작업 메모리, 노트 테이킹, 메타 에이전트를 통해 SWE-Bench-Pro에서 54.3%의 Resolve@1 성능을 달성한다.

## 초록 (Abstract)
실제 소프트웨어 엔지니어링 태스크는 대규모 리포지토리에서 작동하고, 장기 세션을 유지하며, 복잡한 도구 체인을 신뢰성 있게 조율할 수 있는 코딩 에이전트를 요구한다. 기존의 연구 등급 코딩 에이전트는 투명성을 제공하지만 무거운 프로덕션 수준 작업으로 확장하기 어렵고, 프로덕션 등급 시스템은 강력한 실용적 성능을 달성하지만 확장성, 해석 가능성, 제어 가능성이 제한적이다. Confucius Code Agent(CCA)는 Confucius SDK 위에 구축된 소프트웨어 엔지니어링 에이전트로, AX, UX, DX의 세 가지 보완적 관점을 중심으로 구조화되어 있다. SDK는 장기 컨텍스트 추론을 위한 계층적 작업 메모리를 갖춘 통합 오케스트레이터, 교차 세션 지속적 학습을 위한 노트 테이킹 시스템, 신뢰성 있는 도구 사용을 위한 모듈형 확장 시스템을 통합한다. 또한 빌드-테스트-개선 루프를 통해 에이전트 구성의 합성, 평가, 개선을 자동화하는 메타 에이전트를 도입한다. SWE-Bench-Pro에서 CCA는 54.3%의 Resolve@1을 달성하여 기존 연구 기준선을 초과하고 상업적 결과와 유리하게 비교된다.

## 상세 내용

### 1. 서론 (Introduction)
소프트웨어 엔지니어링은 LLM의 최전선 응용 분야로 빠르게 부상했다. 모델 능력이 향상됨에 따라 단순 프로그램 합성에서 자동 코드 완성, 범용 코드 생성, 경쟁 수준 프로그래밍, 실제 이슈 해결로 발전해왔다. 그러나 실제 소프트웨어 엔지니어링에서의 성공은 기저 LLM뿐만 아니라 **에이전트 스캐폴드(agent scaffold)** -- 모델을 둘러싼 오케스트레이션, 메모리 구조, 도구 추상화 -- 에도 의존한다.

동일한 백본 모델을 사용해도 서로 다른 스캐폴딩 전략이 큰 성능 차이를 만들 수 있다. 기존 코딩 에이전트는 평탄한 대화 히스토리, 경험적 프롬프트 엔지니어링, 강하게 결합된 도구 파이프라인에 의존하며, 이는 장기적이고 다중 파일에 걸친 엔터프라이즈 수준의 소프트웨어 엔지니어링 워크플로우로 확장하기 어렵다.

두 가지 핵심 도전 과제를 식별한다:
- **C1: 장기 컨텍스트 추론** - 대규모 리포지토리 내에서 관련 코드를 효율적으로 찾고, 분산된 모듈 간 다중 홉 추론을 수행해야 한다.
- **C2: 장기 메모리** - 태스크와 세션 간에 재사용 가능한 패턴, 실패 모드, 불변량을 누적하는 지속적 지식을 축적해야 한다.

### 2. 방법론 (Method)

#### 2.1 설계 철학: AX, UX, DX
Confucius SDK는 세 축의 설계 철학을 채택한다:

- **Agent Experience (AX)**: 에이전트의 내부 인지 작업 공간. 어떤 정보를 받고, 어떻게 구조화되며, 추론과 도구 사용을 위한 어떤 어포던스를 가지는지 정의한다. 장황한 로그, 원시 diff, 메타데이터는 모델을 방해하므로, 증류된 작업 메모리와 적응형 요약을 강조한다.

- **User Experience (UX)**: 인간이 에이전트를 관찰하고 상호작용하는 방식. 읽기 쉬운 로그, 실행 트레이스, 아티팩트 미리보기를 통한 투명성과 해석 가능성을 우선시한다.

- **Developer Experience (DX)**: 에이전트를 구축, 검사, 개선하는 것. 재현성, 절삭(ablation), 디버깅, 빠른 반복을 위한 프롬프트, 도구, 메모리에 대한 모듈형 인터페이스가 필요하다.

핵심 차별점: 많은 프레임워크가 UX와 AX를 동일시하여 인간 지향 트레이스를 모델에 직접 전달하지만, CCA는 채널을 분리한다. 사용자는 풍부하고 계측된 트레이스를 보고, 에이전트는 압축되고 구조화된 메모리를 보며, 개발자는 둘 다 본다.

#### 2.2 오케스트레이터
CCA의 핵심은 Confucius Orchestrator로, LLM을 반복적으로 호출하고 출력을 해석하며 도구 사용을 조율하는 최소한이지만 확장 가능한 실행 루프이다. 네이티브 도구 사용 API를 가진 모델은 구조화된 JSON 도구 호출을 내보내고, 그렇지 않은 모델은 XML 스타일 태그를 파싱한다. 최대 반복 제한이 있지만 종료는 주로 에이전트 주도적이다.

#### 2.3 핵심 기능

**F1: 컨텍스트 관리 (C1; AX)**
계층적 작업 메모리와 적응형 컨텍스트 압축을 결합한다. 각 에이전트는 구성 가능한 가시성 범위(세션, 항목, 실행 가능)를 가진 계층적 작업 메모리로 지원된다. 프롬프트 길이가 임계값에 접근하면 Architect라는 플래너 에이전트가 별도의 LLM 호출에서 대화 히스토리를 분석하고 핵심 정보(태스크 목표, 결정, 오픈 TODO, 오류 트레이스)를 보존하는 구조화된 요약을 구성한다. 원본 히스토리를 이 압축된 요약으로 대체하면서 최근 메시지의 롤링 윈도우를 유지한다.

**F2: 노트 테이킹 에이전트 (C2; AX, UX)**
평탄한 채팅 로그 대신 구조화된 지속적 지식으로 변환한다. 전용 노트 테이킹 에이전트가 트래젝토리를 마크다운 파일의 파일 시스템과 유사한 트리 구조로 증류한다. **사후 분석 노트(hindsight notes)** 에 특별히 중점을 두어 컴파일 오류, 런타임 예외, 비생산적 전략과 그 해결책을 기록한다. 향후 세션에서 유사한 실패가 나타나면 해당 노트를 검색하여 알려진 수정 사항을 즉시 제공할 수 있다.

**F3: 확장 시스템 (C1; AX, DX)**
모듈형 컴포넌트인 확장(extensions)이 오케스트레이터에 부착되어 각 반복에 참여한다. 유형이 지정된 구성 객체가 콜백(on_input_messages, on_plain_text, on_tag, on_llm_output)을 등록한다. 확장은 인지(perception), 추론(reasoning), 행동(action)을 담당한다. CCA는 파일 편집, CLI, 코드 검색, 계획, 프롬프트 캐싱 등의 확장 번들로 인스턴스화된다.

**F4: 메타 에이전트 (DX)**
에이전트 행동을 자동으로 구축하고 개선하는 빌드-테스트-개선 루프를 자동화한다. 개발자가 자연어로 대상 에이전트를 설명하면, 메타 에이전트가 구성과 프롬프트를 합성하고, 선택된 확장을 연결하며, 후보 에이전트를 회귀 태스크에서 테스트하고, 실패 시 수정을 제안하여 반복적으로 개선한다. CCA 자체가 이 메타 에이전트의 빌드-개선-테스트 루프의 산물이다.

### 3. 실험 (Experiments)

#### 3.1 설정
- **모델**: Claude 4 Sonnet, Claude 4.5 Sonnet, Claude 4.5 Opus를 백본 LLM으로 사용
- **기준선**: SWE-Agent, Live-SWE-Agent
- **벤치마크**: SWE-Bench-Pro (731 태스크), SWE-Bench-Verified (500 태스크)

#### 3.2 SWE-Bench-Pro 주요 결과
| 백본 모델 | 스캐폴드 | Resolve Rate |
|---|---|---|
| Claude 4 Sonnet | SWE-Agent | 42.7 |
| Claude 4 Sonnet | CCA | 45.5 |
| Claude 4.5 Sonnet | SWE-Agent | 43.6 |
| Claude 4.5 Sonnet | Live-SWE-Agent | 45.8 |
| Claude 4.5 Sonnet | CCA | 52.7 |
| Claude 4.5 Opus | Anthropic System Card | 52.0 |
| Claude 4.5 Opus | CCA | 54.3 |

핵심 발견: 더 약한 모델 + 강한 스캐폴드(Claude 4.5 Sonnet + CCA: 52.7%)가 더 강한 모델(Claude 4.5 Opus + Anthropic 독점 스캐폴드: 52.0%)를 능가할 수 있다.

#### 3.3 메타 에이전트 학습 도구 사용
메타 에이전트가 학습한 도구 사용 기능을 비활성화하면 Resolve@1이 크게 하락한다(Claude 4.5 Sonnet: 51.6 → 44.0). 이는 도구 사용 관습이 성능의 주요 동인임을 확인한다.

#### 3.4 컨텍스트 관리 평가
고급 컨텍스트 관리가 Claude 4 Sonnet에서 Resolve@1을 42.0에서 48.6으로 개선(+6.6). 플래너 에이전트가 핵심 추론 체인을 누락시키지 않으면서 프롬프트 길이를 40% 이상 줄이는 것으로 관찰된다. 계획 반복 수도 증가(평균 2.7 vs 1.4)하여 깊은 다단계 추론을 촉진한다.

#### 3.5 장기 메모리 평가
노트 테이킹 모듈을 2회 연속 실행으로 평가:
- Run 1 → Run 2: 평균 턴 64→61(-3), 토큰 비용 104k→93k(-11k), Resolve Rate 53.0→54.4(+1.4)
- 노트가 실행 가능하고 재사용 가능한 지식을 포착하여 교차 세션 학습을 가능하게 함

#### 3.6 SWE-Bench-Verified 결과
Claude 4 Sonnet으로 CCA가 74.6%를 달성하여 OpenHands(72.8%)를 초과하고, 더 강력한 Claude 4.5 Sonnet을 사용하는 mini-SWE-Agent(70.6%)도 능가한다.

### 4. 관련 연구 (Related Work)
- **대규모 소프트웨어 엔지니어링**: Google의 모노레포 모델, ECO 같은 LLM 기반 코드 최적화기
- **소프트웨어 엔지니어링 에이전트**: SWE-Agent, Live-SWE-Agent, Satori-SWE, Agentless, OpenHands 등
- **소프트웨어 엔지니어링을 위한 LLM 학습**: SWE-Gym, SWE-Smith, SWE-RL

### 5. 향후 연구 (Future Work)
강화 학습(RL)이 SFT만으로 달성할 수 없는 수준으로 LLM 기반 소프트웨어 엔지니어링 에이전트를 향상시킬 수 있다. AX 프레임워크가 RL 학습에 적합한 트래젝토리 형식으로 추론 트레이스를 구조화하며, 메타 에이전트가 다양한 보상 함수로 변환 가능한 세밀한 피드백 신호를 생산한다. Confucius Orchestrator의 확장성은 RL에서 커리큘럼 설계를 위한 자연스러운 기반을 제공한다.

### 6. 결론 (Conclusion)
CCA는 AX, UX, DX를 명시적으로 분리하고 최적화하는 Confucius SDK 위에 인스턴스화된 코딩 에이전트로, 견고한 다단계 추론, 모듈형 도구 사용, 구조화된 메모리 관리, 해석 가능한 실행 트레이스를 가능하게 한다. 에이전트 스캐폴딩이 원시 모델 능력을 능가할 수 있음을 입증한다.

### 부록 주요 내용

#### A. 사고 예산 스케일링
Claude 4 Sonnet에서 사고 예산 8k→16k→32k 토큰으로 증가 시 Resolve Rate가 67.3→68.4→68.7로 증가하나 16k 이후 수확 체감.

#### B. 노트 테이킹 예시
프로젝트별(openlibrary)과 공유(shared) 노트로 잘 조직되며, 와일드카드 이스케이핑, 접두사 제거 엣지 케이스 등 구체적 인사이트를 기록한다.

#### C. Claude Code와의 비교 사례 연구
PyTorch 이슈에서 CCA와 Claude Code(CC)를 비교. CCA는 단일 에이전트로 직접 탐색하여 최소한의 수정을 선호하고, CC는 다중 에이전트로 위임하여 더 포괄적이지만 과설계된 솔루션을 생산하는 경향이 있다. PyTorch 팀의 최종 수정이 CCA의 접근 방식과 일치한 사례가 있어, CCA의 원칙적 엔지니어링 스타일을 검증한다.

## 핵심 키 포인트
1. 에이전트 스캐폴딩이 모델 능력 못지않게 성능의 주요 결정 요인이다 (약한 모델 + 강한 스캐폴드 > 강한 모델 + 약한 스캐폴드).
2. AX/UX/DX 삼축 설계 철학이 에이전트 시스템의 확장성, 해석 가능성, 개발 효율성을 동시에 달성한다.
3. 계층적 작업 메모리와 적응형 컨텍스트 압축이 장기 추론 안정성을 크게 향상시킨다.
4. 사후 분석 노트(hindsight notes)를 포함한 노트 테이킹 시스템이 교차 세션 학습을 가능하게 한다.
5. 메타 에이전트의 빌드-테스트-개선 루프가 수동 에이전트 설계를 능가하는 자동화된 에이전트 개발을 실현한다.
6. SWE-Bench-Pro에서 54.3% Resolve@1을 달성하여 기존 연구 기준선과 상업적 결과를 모두 초과한다.
7. 단일 에이전트 아키텍처가 잘 범위가 정해진 디버깅 태스크에서 다중 에이전트보다 더 정확한 솔루션을 생산할 수 있다.

## 주요 인용 (Key Quotes)
> "even when the same backbone model is used, different scaffolding strategies can lead to large performance disparities, suggesting that the design of the agent's cognitive and operational environment is a fundamental research dimension." (Section 1, p.2)

> "even a weaker model equipped with a strong agent scaffold (Claude 4.5 Sonnet + CCA at 52.7%) can outperform a stronger model (Claude 4.5 Opus + Anthropic's proprietary scaffold at 52.0%)." (Section 3.2, p.10)

> "Manual inspection further reveals that the planner agent frequently reduces prompt length by over 40% without omitting key reasoning chains." (Section 3.4.1, p.10)

> "the note-taking system provides CCA with a lightweight form of cross-session learning, enabling more efficient reasoning and more reliable patch generation in subsequent attempts." (Section 3.5, p.11)

> "The production Confucius Code Agent proposed in this paper is itself the outcome of the Meta-agent's build-improve-test loop." (Section 2.3.4, p.8)

> "subagents separate concerns and allow the main agent to focus on its main task. However, our analysis suggests that for well-scoped debugging tasks, the benefits of delegation may be outweighed by the risk of context loss and derailment via inter-agent misalignment." (Appendix C.3, p.24)

> "CCA favored minimal intervention, while CC pursued a more holistic solution. In this case, we note that the PyTorch team's eventual fix matched CCA's approach." (Appendix C.2, p.22)

> "AX must avoid noise. Verbose logs, raw diffs, and metadata that help humans often distract or bias the model." (Section 2.1, p.4)

## 시사점 및 의의
이 논문은 코딩 에이전트 연구에서 몇 가지 중요한 패러다임 전환을 제시한다:

1. **스캐폴딩의 중요성 재확인**: 모델 능력의 향상만으로는 충분하지 않으며, 에이전트를 둘러싼 인프라(오케스트레이션, 메모리, 도구 추상화)의 설계가 동등하거나 더 큰 성능 향상을 가져올 수 있음을 실증적으로 보여준다.

2. **에이전트 인지 작업 공간의 분리**: AX/UX/DX 프레임워크는 에이전트가 보는 것(AX), 사용자가 보는 것(UX), 개발자가 보는 것(DX)을 명시적으로 분리하여 각 축을 독립적으로 최적화할 수 있게 한다. 이는 많은 기존 프레임워크의 근본적 설계 결함을 해결한다.

3. **자동 에이전트 개발**: 메타 에이전트의 도입은 에이전트 설계를 수동적 엔지니어링에서 자동화된 반복적 최적화로 전환하는 것을 시사하며, 에이전트 개발의 확장성에 중요한 기여를 한다.

4. **교차 세션 학습**: 노트 테이킹 시스템은 에이전트가 경험에서 학습하여 점점 더 나아지는 형태의 경량 지속적 학습을 구현하며, 이는 실제 소프트웨어 엔지니어링에서 매우 가치 있는 능력이다.

5. **단일 vs 다중 에이전트**: Claude Code와의 비교를 통해 잘 범위가 정해진 태스크에서 단일 에이전트가 컨텍스트 유지 측면에서 유리할 수 있음을 보여주며, 다중 에이전트 아키텍처의 맹목적 채택에 대한 경고를 제공한다.
