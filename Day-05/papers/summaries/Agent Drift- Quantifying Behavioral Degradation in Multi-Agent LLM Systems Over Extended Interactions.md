# Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions

## 기본 정보
- **저자**: Abhishek Rath (Independent Researcher, Hyderabad, India)
- **발행일/출처**: 2026년 1월 8일, arXiv:2601.04170v1 [cs.AI]
- **페이지 수**: 12페이지
- **키워드**: Agent Drift, Multi-Agent Systems, LLM, Behavioral Degradation, Agent Stability Index (ASI), Semantic Drift, Coordination Drift, Behavioral Drift, Production AI

## 한줄 요약
> 이 논문은 다중 에이전트 LLM 시스템에서 장기 상호작용 시 발생하는 점진적 행동 저하 현상인 "에이전트 드리프트(agent drift)"를 정의하고, 12개 차원의 복합 지표(ASI)를 제안하며, 시뮬레이션 기반 분석을 통해 드리프트의 유형, 영향, 완화 전략을 체계적으로 제시한다.

## 초록 (Abstract)
다중 에이전트 LLM 시스템은 복잡한 과제 분해와 협업적 문제 해결을 위한 강력한 아키텍처로 부상했으나, 장기적 행동 안정성은 거의 검토되지 않았다. 본 연구는 에이전트 드리프트 개념을 도입한다 -- 이는 확장된 상호작용 시퀀스에 걸쳐 에이전트 행동, 의사결정 품질, 에이전트 간 일관성이 점진적으로 저하되는 현상이다. 세 가지 드리프트 유형(시맨틱 드리프트, 조정 드리프트, 행동 드리프트)을 제안하고, 12개 차원에 걸쳐 드리프트를 정량화하는 Agent Stability Index(ASI)를 도입한다. 시뮬레이션 분석을 통해 드리프트가 과제 완료 정확도를 42% 감소시키고 인간 개입을 3.2배 증가시킬 수 있음을 보이며, 세 가지 완화 전략의 67-81% 오류 감소 효과를 검증한다.

## 상세 내용

### 1. 서론 (Introduction)
2023년 이후 LangGraph, AutoGen, CrewAI 등의 프레임워크에 의해 다중 에이전트 LLM 시스템의 배포가 급격히 가속화되었다. 이들은 코드 생성, 연구 합성, 기업 자동화에서 인상적인 능력을 보이지만, 장기적 행동 안정성에 대한 이해에 중대한 공백이 있다.

전통적 소프트웨어는 예측 가능한 저하 패턴(메모리 누수, 자원 고갈, 구성 드리프트)을 보이지만, LLM 기반 에이전트는 명시적 파라미터 변경이나 시스템 장애 없이 의사결정 패턴이 설계 명세에서 점진적으로 이탈하는 새로운 실패 모드를 도입한다. 이는 개별적으로는 미미하고 격리된 평가에서는 감지하기 어려우나, 누적적으로 시스템 성능을 두 자릿수 퍼센트로 저하시킨다.

#### 네 가지 기여:
1. **분류학적 프레임워크**: 시맨틱, 조정, 행동 드리프트의 포괄적 분류
2. **측정 방법론**: 12개 행동 차원에 걸친 ASI 복합 지표
3. **이론적 분석**: 드리프트 유병률, 진행률, 시스템 신뢰성 영향의 시뮬레이션 기반 특성화
4. **완화 전략**: 세 가지 개입 접근법의 개발 및 이론적 검증

### 2. 방법론 (Methodology)

#### 2.1 시뮬레이션 프레임워크
세 가지 기업 도메인에서 시뮬레이션:
- **기업 자동화** (n=412): 마스터 라우터 + 데이터베이스/파일 처리/알림 에이전트
- **금융 분석** (n=289): 주식 리서치, 리스크 평가, 포트폴리오 최적화 에이전트
- **규정 준수 모니터링** (n=146): 패턴 탐지, 규칙 추출, 추론 에이전트

상호작용 범위: 5~1,847회(중앙값 127회), 3~18개월 등가 시뮬레이션

#### 2.2 Agent Stability Index (ASI) 프레임워크
12개 차원, 4개 범주의 복합 지표:

**응답 일관성 (가중치: 0.30)**
- 출력 의미 유사도 (Csem): 시간 윈도우 간 의미적으로 동등한 입력에 대한 코사인 유사도
- 의사결정 경로 안정성 (Cpath): 추론 체인의 편집 거리
- 신뢰도 교정 (Cconf): 예측-실제 정확도 분포 간 JS 발산

**도구 사용 패턴 (가중치: 0.25)**
- 도구 선택 안정성 (Tsel): 카이제곱 검정
- 도구 시퀀싱 일관성 (Tseq): 레벤슈타인 거리
- 도구 파라미터화 드리프트 (Tparam): KL 발산

**에이전트 간 조정 (가중치: 0.25)**
- 합의 동의율 (Iagree): 만장일치/절대다수 비율
- 핸드오프 효율성 (Ihandoff): 위임 평균 메시지 수
- 역할 준수 (Irole): 에이전트 ID와 과제 유형 간 상호정보

**행동적 경계 (가중치: 0.20)**
- 출력 길이 안정성 (Blength): 변동 계수
- 오류 패턴 출현 (Berror): 오류 유형 클러스터링
- 인간 개입률 (Bhuman): 인간 수정이 필요한 비율

ASI는 50회 상호작용 롤링 윈도우에서 계산되며, 3회 연속 윈도우에서 0.75 미만일 때 드리프트로 탐지된다.

#### 2.3 드리프트 패턴 분류
- **시맨틱 드리프트**: 구문적으로 유효하면서 원래 과제 의도에서 점진적으로 이탈. 예: 금융 분석 에이전트가 위험 중심에서 기회 강조 언어로 서서히 전환
- **조정 드리프트**: 다중 에이전트 합의 메커니즘의 저하. 예: 라우터가 특정 하위 에이전트를 편향적으로 선호하여 병목 발생
- **행동 드리프트**: 초기 상호작용에 없던 새로운 전략/행동 패턴 출현. 예: 규정 준수 에이전트가 지정 메모리 도구 대신 채팅 히스토리에 중간 결과를 캐싱

#### 2.4 완화 전략
1. **에피소딕 메모리 통합(EMC)**: 50회마다 과거 100회 상호작용을 요약하여 주기적으로 압축
2. **드리프트 인식 라우팅(DAR)**: 에이전트 안정성 점수를 위임 결정에 반영, 드리프트 에이전트는 리셋
3. **적응적 행동 앵커링(ABA)**: 베이스라인 기간의 예시를 few-shot 프롬프트로 동적 보강

### 3. 결과 (Results)

#### 3.1 드리프트 유병률과 진행
- **조기 발현**: 중앙값 73회 상호작용 후 탐지 가능(IQR: 52-114)
- **가속 효과**: 0-100회에서 50회당 0.08포인트 하락 → 300-400회에서 50회당 0.19포인트 하락 (양성 피드백 루프)
- **도메인 차이**: 금융 분석(53.2%) > 규정 준수(39.7%) > 기업 자동화(31.8%) -- 과제 모호성이 높을수록 취약

#### 3.2 시스템 성능 영향
| 지표 | 베이스라인 | 드리프트 | 저하율 |
|------|----------|---------|-------|
| 과제 성공률 | 87.3% | 50.6% | -42.0% |
| 응답 정확도 | 91.2% | 68.5% | -24.9% |
| 완료 시간(분) | 8.7 | 14.2 | +63.2% |
| 인간 개입 | 0.31/과제 | 0.98/과제 | +216.1% |
| 토큰 사용량 | 12,400 | 18,900 | +52.4% |
| 에이전트 간 충돌 | 0.08/과제 | 0.47/과제 | +487.5% |

#### 3.3 ASI 구성요소 분석
- 행동적 경계가 가장 가파르게 하락(500회에서 46% 감소)
- 에이전트 간 조정은 200회까지 비교적 안정적이나 이후 급격히 하락 -- 에이전트 간 신뢰 모델이 한 번 침식되면 취약해짐
- 300회 부근에서 가속화가 시작 -- 누적 드리프트의 자기강화 임계점

#### 3.4 완화 전략 효과
| 전략 | ASI 유지율 | 드리프트 감소 |
|------|-----------|------------|
| 통제(미개입) | 71.3% | -- |
| 에피소딕 메모리 통합 | 87.1% | 51.9% |
| 드리프트 인식 라우팅 | 89.4% | 63.0% |
| 적응적 행동 앵커링 | 92.5% | 70.4% |
| **세 전략 결합** | **94.7%** | **81.5%** |

세 전략 결합 시 81.5% 드리프트 감소를 달성하나, 계산 오버헤드 23% 증가와 완료 시간 9% 연장의 트레이드오프 존재.

#### 3.5 아키텍처적 영향
- **계층 깊이**: 2단계 계층(라우터+전문가)이 플랫 및 3+단계보다 우수
- **메모리 시스템**: 명시적 장기 메모리가 21% 높은 ASI 유지율
- **LLM 다양성**: 혼합 LLM 시스템이 동질적 시스템보다 약간 더 안정적
- **동기/비동기**: 동기 실행이 약간 더 나은 조정을 보이나 통계적으로 유의하지 않음

### 4. 논의 (Discussion)

#### 4.1 드리프트 기저 메커니즘
1. **컨텍스트 윈도우 오염**: 상호작용 히스토리가 쌓이면서 관련 정보의 신호 대 잡음 비율 저하
2. **분포적 이동**: 넓은 코퍼스로 훈련되었으나 좁은 도메인에 배포되어 점진적으로 악화되는 근사
3. **자기회귀를 통한 강화**: 에이전트 출력이 공유 메모리를 통해 자체 미래 입력이 되어, 작은 오류나 스타일 편향이 자기회귀적으로 복합

#### 4.2 프로덕션 배포 시사점
1. 전통적 ML 모니터링(정확도, 지연, 처리량)은 에이전틱 시스템에 불충분
2. 드리프트 완화는 "설정 후 방치"가 불가능 -- 지속적 거버넌스 프레임워크 필요
3. 드리프트 시스템에서 인간 감독 비용이 3.2배 증가하여 장기 운영 시 경제적 타당성 위협
4. 전통적 사전 배포 테스트(<50턴)는 최종 드리프트의 25%만 포착

#### 4.3 AI 안전 연구와의 연결
- 드리프트는 RL의 명세 게이밍(specification gaming), 보상 해킹(reward hacking)과 우려스러운 유사점
- 파라미터 업데이트 없이 드리프트 발생 -- 훈련 시간 정렬 전략만으로는 불충분함을 시사
- 공유 메모리를 통한 자기회귀 피드백이 암묵적 자기수정(implicit self-modification)에 해당

### 5. 결론 (Conclusion)
에이전트 드리프트를 프로덕션 다중 에이전트 LLM 시스템의 근본적 도전으로 확립한다. 장기 운영 에이전트의 거의 절반이 행동 저하에 영향받을 수 있으며, 42% 과제 성공률 감소와 3.2배 인간 개입 증가를 초래할 수 있다. 저자는 (1) 업계 표준 드리프트 모니터링 프로토콜, (2) 드리프트 저항 아키텍처 연구 투자, (3) 장기 행동 안정성의 규제적 고려, (4) 배포 시스템의 드리프트 특성 투명성을 촉구한다.

## 핵심 키 포인트
1. **에이전트 드리프트 정의**: 명시적 파라미터 변경 없이 발생하는 점진적 행동 저하로, 기존 소프트웨어 저하와 질적으로 다른 새로운 실패 모드이다.
2. **세 가지 드리프트 유형**: 시맨틱(의도 이탈), 조정(합의 저하), 행동(미의도 전략 출현)으로 분류된다.
3. **조기 발현**: 중앙값 73회 상호작용에서 탐지 가능하며, 이는 구조화된 프롬프트와 가드레일이 있는 프로덕션 시스템의 예상보다 훨씬 이르다.
4. **가속 효과**: 드리프트는 양성 피드백 루프로 시간이 갈수록 가속화되며, 300회 부근에서 자기강화 임계점이 있다.
5. **심각한 영향**: 과제 성공률 42% 감소, 인간 개입 216% 증가, 에이전트 간 충돌 487% 증가
6. **완화 가능성**: 적응적 행동 앵커링이 단일 전략으로 가장 효과적(70.4%), 세 전략 결합 시 81.5% 감소
7. **아키텍처적 설계 원칙**: 2단계 계층, 명시적 장기 메모리, 혼합 LLM이 드리프트 저항성을 향상시킨다.

## 주요 인용 (Key Quotes)

> "This study introduces the concept of agent drift—the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences." (Abstract, p.1)

> "These changes are individually minor and often imperceptible in isolated evaluations, yet collectively degrade system performance by double-digit percentages—a pattern we term agent drift." (Section 1, p.2)

> "Agent drift poses fundamental questions for AI safety: if multi-agent systems progressively deviate from intended behaviors without explicit modification, traditional alignment and monitoring approaches may prove insufficient." (Section 1, p.2)

> "The most severe impact is on task success rate—a 42% reduction represents the difference between production-viable and operationally unacceptable performance." (Section 3.2, p.6)

> "Traditional pre-deployment testing evaluates agents over short interaction sequences (typically < 50 turns). Our data shows this captures only 25% of eventual drift cases." (Section 4.2, p.9)

> "If drift persists despite static parameters, this has implications for AI alignment strategies that focus primarily on training-time objectives rather than deployment-time behavior management." (Section 4.3, p.10)

> "Agent drift is not a peripheral concern—it is central to the question of whether we can build AI systems that remain reliably aligned with human intent not just for minutes or hours, but for months and years of continuous operation." (Section 5, p.10)

## 시사점 및 의의
이 연구는 AgentOps의 핵심 관심사인 프로덕션 환경에서의 에이전트 신뢰성 문제를 정면으로 다룬다. 에이전트 드리프트는 단순한 성능 저하가 아니라, AI 시스템의 장기적 정렬(alignment)과 안전성에 대한 근본적 질문을 제기한다. 특히 주목할 점은: (1) 드리프트가 파라미터 업데이트 없이 발생하므로 훈련 시간 정렬만으로는 불충분하며 배포 시간 행동 관리가 필수적이라는 것, (2) 기존 사전 배포 테스트가 최종 드리프트의 25%만 포착하므로 연장된 스트레스 테스팅이 필요하다는 것, (3) ASI 프레임워크가 프로덕션 모니터링의 청사진을 제공한다는 것이다. 다만, 이 연구는 시뮬레이션 기반이며 실제 프로덕션 데이터가 아닌 이론적 모델링이라는 한계가 있어, 실제 배포 환경에서의 검증이 필요하다.
