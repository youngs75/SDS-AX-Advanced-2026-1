# Graph Retrieval-Augmented Generation: A Survey

## 기본 정보
- **제목**: Graph Retrieval-Augmented Generation: A Survey
- **저자**: Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, Siliang Tang
- **소속**: Peking University, Zhejiang University, Ant Group, Renmin University of China, Rutgers University
- **발표일**: 2025년 12월 (온라인 2025년 11월 19일)
- **학회/저널**: ACM Transactions on Information Systems (TOIS), Volume 44, Issue 2, Article 35
- **페이지 수**: 52페이지
- **DOI**: https://doi.org/10.1145/3777378
- **GitHub**: https://github.com/pengboci/GraphRAG-Survey

## 한줄 요약
그래프 구조 정보를 활용하여 LLM의 검색 증강 생성(RAG)을 개선하는 GraphRAG의 전체 워크플로우를 G-Indexing, G-Retrieval, G-Generation의 3단계로 체계화하고, 핵심 기술, 훈련 방법, 하위 과제, 산업 응용, 미래 방향을 포괄적으로 정리한 최초의 종합 서베이이다.

## 초록 (Abstract) 요약
RAG(Retrieval-Augmented Generation)는 LLM을 재훈련하지 않고도 외부 지식 기반을 참조하여 "환각(hallucination)", 도메인 지식 부족, 정보 비최신성 등의 문제를 효과적으로 완화하는 데 성공했다. 그러나 데이터베이스 내 엔티티 간의 복잡한 관계 구조는 기존 RAG 시스템에 도전을 제기한다. 이에 대응하여 GraphRAG는 엔티티 간 구조적 정보를 활용하여 더 정밀하고 포괄적인 검색을 가능하게 하고, 관계형 지식을 포착하여 더 정확하고 맥락 인식적인 응답을 생성한다.

이 논문은 GraphRAG 방법론에 대한 최초의 포괄적 개요를 제공하며, 워크플로우를 Graph-Based Indexing(G-Indexing), Graph-Guided Retrieval(G-Retrieval), Graph-Enhanced Generation(G-Generation)으로 공식화한다.

## 상세 내용

### 1. 서론 (Section 1)
GPT, Qwen, LLaMA, DeepSeek 등의 LLM은 자연어 처리에 혁명을 일으켰지만, 도메인 특화 지식 부족, 실시간 정보 미반영, 독점 지식 부재 등으로 인해 "환각" 문제가 발생한다. RAG는 생성 과정에 검색 컴포넌트를 통합하여 이를 완화하지만, 실세계에서 세 가지 한계에 직면한다:

1. **관계 무시(Neglecting Relationships)**: 텍스트 콘텐츠는 고립되어 있지 않고 상호 연결되어 있으나, 전통적 RAG는 의미적 유사성만으로 표현할 수 없는 구조화된 관계형 지식을 포착하지 못함
2. **중복 정보(Redundant Information)**: 텍스트 조각 형태의 콘텐츠를 프롬프트로 연결하면 컨텍스트가 과도하게 길어져 "lost in the middle" 딜레마 발생
3. **전역 정보 부족(Lacking Global Information)**: 문서의 하위 집합만 검색할 수 있어 Query-Focused Summarization(QFS) 등 전역적 이해가 필요한 과제에 어려움

GraphRAG는 사전 구축된 그래프 데이터베이스에서 쿼리와 관련된 관계형 지식을 포함하는 그래프 요소(노드, 트리플, 경로, 서브그래프)를 검색하여 응답을 생성함으로써 이러한 과제를 해결한다.

### 2. 관련 기술 및 서베이 비교 (Section 2)

#### 2.1 RAG와의 비교
GraphRAG는 넓은 관점에서 RAG의 한 분기이지만, 핵심적 차이가 있다:
- **인덱싱**: 텍스트 기반 RAG는 텍스트 청크를 직접 벡터화하지만, GraphRAG는 원시 텍스트를 먼저 그래프(예: KG)로 분해한 후 인덱스 구축
- **검색**: 텍스트 기반 RAG는 텍스트 검색 알고리즘을 사용하지만, GraphRAG는 그래프 검색 알고리즘에 의존
- **생성**: 텍스트 기반 RAG는 검색된 청크를 프롬프트에 직접 통합하지만, GraphRAG는 그래프 구조 데이터를 LM이 처리할 수 있는 형식으로 변환하는 추가 단계 필요

GraphRAG의 핵심 장점은 다중 홉 추론(multi-hop reasoning) 능력으로, 그래프의 여러 노드를 순회하여 복잡한 쿼리에 응답할 수 있다.

#### 2.2 LLMs on Graphs와의 구분
- LLMs on Graphs: LLM을 사용하여 그래프 중심 과제(노드 분류 등)의 성능 향상
- GraphRAG: 외부 그래프 구조를 검색하여 LLM의 자연어 과제 성능 향상 (역방향 접근)

#### 2.3 KBQA(Knowledge Base Question Answering)와의 관계
IR 기반 KBQA는 GraphRAG의 하위 집합. GraphRAG는 KG에 한정되지 않고 다양한 그래프 유형을 구성하며, 검색 정보도 트리플/경로뿐 아니라 엔티티 설명, 커뮤니티 요약, 원시 텍스트 청크까지 확장.

### 3. 사전 지식 (Section 3)

#### 3.1 TAGs (Text-Attributed Graphs)
GraphRAG에서 사용되는 그래프 데이터의 보편적 형식. 노드와 엣지가 텍스트 속성을 갖는다.
- 공식 정의: G = (V, E, A, {x_v}, {e_i,j}) - V(노드 집합), E(엣지 집합), A(인접 행렬), 텍스트 속성

#### 3.2 GNNs (Graph Neural Networks)
메시지 패싱 방식으로 노드 표현을 학습. GCN, GAT, GraphSAGE 등이 대표적. 이웃 노드의 정보를 집계하여 노드 표현 업데이트.

#### 3.3 LMs (Language Models)
판별적 모델(BERT, RoBERTa, SentenceBERT)과 생성적 모델(GPT-3, GPT-4)로 분류. 최근 LLM의 강력한 인컨텍스트 학습 능력으로 GraphRAG 연구가 가속화.

### 4. GraphRAG 개요 (Section 4)

GraphRAG의 목표는 외부 구조화된 KG를 활용하여 LM의 맥락 이해를 향상시키고 더 정보에 기반한 응답을 생성하는 것이다.

**공식 정의**: a* = arg max_a p(a|q, G)

이를 그래프 리트리버와 답변 생성기로 분해:
- p(a|q, G) = sum_G p_phi(a|q, G) * p_theta(G|q, G) 는 약 p_phi(a|q, G*) * p_theta(G*|q, G) 로 근사

여기서 G*는 최적 서브그래프로, 모든 가능한 서브그래프에 대한 합산이 계산적으로 다루기 힘들기 때문에 근사한다.

3단계 프로세스:
1. **G-Indexing**: 그래프 데이터베이스 식별/구축 및 인덱스 설정
2. **G-Retrieval**: 사용자 쿼리에 대해 관련 그래프 요소 추출
3. **G-Generation**: 검색된 그래프 데이터를 기반으로 응답 합성

### 5. G-Indexing (Section 5)

#### 5.1 그래프 데이터

**공개 KG(Open KGs):**
- **일반 KG**: 백과사전적 KG(Wikidata, Freebase, DBpedia, YAGO)와 상식 KG(ConceptNet, ATOMIC)
- **도메인 KG**: 생의학(CMeKG, CPubMed-KG), 영화(Wiki-Movies) 등 특정 분야

**자체 구축 그래프(Self-Constructed):**
- **트리 구조**: RAPTOR, SiReRAG - 텍스트 청크를 리프 노드, 의미 요약을 비리프 노드로 계층적 구성
- **문서 그래프**: ATLANTIC, KGP - 문서/패시지를 노드, 의미적/구조적 관계를 엣지로 모델링
- **지식 그래프**: DALK, HippoRAG, LightRAG, Microsoft-GraphRAG - NER + RE로 엔티티/관계 추출. 텍스트 KG는 노드/엣지에 추가 텍스트 정보 부착
- **이질적 그래프(Heterogeneous Graphs)**: NodeRAG, HippoRAG2 - 다중 유형의 노드와 엣지로 복잡한 도메인 지식 표현

#### 5.2 인덱싱 방법
- **그래프 인덱싱**: 전체 그래프 구조를 보존. BFS, DFS 등 그래프 이론 알고리즘으로 검색 지원
- **텍스트 인덱싱**: 그래프 데이터를 텍스트 설명으로 변환. 템플릿 기반 트리플 변환, 커뮤니티 요약 생성 등
- **벡터 인덱싱**: 그래프 데이터를 벡터 표현으로 변환. LM/GNN 기반 임베딩, 효율적 벡터 검색 알고리즘(LSH 등) 활용
- **하이브리드 접근**: NodeRAG는 그래프/텍스트/벡터 인덱싱을 동시 통합

### 6. G-Retrieval (Section 6)

두 가지 핵심 과제: (1) 폭발적 후보 서브그래프 수, (2) 불충분한 유사도 측정

#### 6.1 리트리버 유형
- **비모수적 리트리버(Non-Parametric)**: 휴리스틱 규칙 또는 전통적 그래프 검색 알고리즘 기반. k-hop 경로 검색, PCST 알고리즘, PageRank 변형 등
- **LM 기반 리트리버**: (1) LM을 인코더로 사용하여 쿼리 임베딩 생성 후 벡터 DB 검색, (2) LM으로 관계 경로/규칙 사전 생성, (3) LM을 검색기로 직접 사용하여 그래프 탐색
- **GNN 기반 리트리버**: GNN이 노드/엣지의 구조적 정보를 인코딩. 메시지 패싱으로 이웃 정보 집계하여 후보 노드 점수 산출

#### 6.2 검색 패러다임
- **1회 검색(Once Retrieval)**: 단일 쿼리로 1회 검색. 간단하지만 다중 홉 추론 어려움
- **반복적 검색(Iterative Retrieval)**: 여러 라운드의 검색/추론 반복. 복잡한 추론에 적합하나 오류 누적 위험
- **다단계 검색(Multi-stage)**: 초벌 검색 후 정밀 검색 등 단계별 검색. UniKGQA, GNN-RAG 등
- **적응적 검색(Adaptive)**: 쿼리 특성에 따라 검색 전략 동적 결정. Graph CoT 등

#### 6.3 검색 세분도
- **노드**: HippoRAG, HippoRAG2 등이 관련 노드 검색
- **트리플**: SubgraphRAG, KAPING 등이 (주어, 관계, 목적어) 트리플 검색
- **경로**: RoG, KELP 등이 엔티티 간 추론 경로 검색
- **서브그래프**: G-Retriever, GraphRAG 등이 관련 서브그래프 검색 (가장 정보가 풍부하나 노이즈 위험)

#### 6.4 검색 향상 기법
- **쿼리 향상**: 쿼리 분해(decomposition), 쿼리 확장(expansion) 등
- **지식 향상**: 병합(merging) - 다중 소스 정보 통합, 가지치기(pruning) - 무관한 정보 제거

### 7. G-Generation (Section 7)

검색된 그래프 데이터를 기반으로 의미 있는 출력을 합성하는 단계.

#### 7.1 생성기 유형
- **GNN 기반**: 그래프 구조를 직접 처리하여 노드/그래프 표현 생성
- **LM 기반**: 자연어 이해/생성 능력 활용
- **하이브리드**: GNN + LM 결합으로 구조/의미 정보 동시 활용

#### 7.2 그래프 형식 (그래프 -> LM 입력 변환)
- **임베딩(Embeddings)**: GNN/LM으로 그래프를 밀집 벡터로 인코딩
- **자연어(Languages)**: 그래프를 자연어 텍스트로 변환. 트리플 직렬화, 경로 서술, 서브그래프 설명 등

#### 7.3 생성 향상 기법
- **사전 향상(Pre-enhancement)**: 생성 전 검색 결과 정제
- **중간 향상(Mid-enhancement)**: 생성 중 그래프 정보 동적 통합
- **사후 향상(Post-enhancement)**: 생성 후 출력 교정/검증

### 8. 훈련 전략 (Section 8)
- 리트리버 훈련: 비감독/감독/강화 학습
- 생성기 훈련: SFT, RLHF 등

### 9. 하위 과제 및 응용 (Section 9)
- KGQA, 상식 추론, 사실 검증, 링크 예측, QFS 등 다양한 과제
- 의료, 금융, 과학, 교육, 법률 등 다양한 도메인 응용
- Microsoft GraphRAG, Neo4j 등 산업용 시스템

### 10. 미래 방향 (Section 10)
효율적 그래프 구축, 멀티모달 GraphRAG, 시간적 그래프, 대규모 확장성, GraphRAG 에이전트, 평가 방법론 등

## 핵심 키 포인트

1. **GraphRAG의 3단계 프레임워크**: G-Indexing -> G-Retrieval -> G-Generation의 체계적 워크플로우

2. **전통 RAG 대비 핵심 개선**: (1) 엔티티 간 관계형 지식 포착, (2) 정보 중복 완화, (3) 전역 정보 접근

3. **다양한 그래프 데이터 유형**: 공개 KG, 자체 구축(트리, 문서 그래프, KG, 이질적 그래프) 등

4. **검색 패러다임의 스펙트럼**: 1회/반복적/다단계/적응적 검색과 노드/트리플/경로/서브그래프 세분도

5. **그래프-LM 인터페이스 문제**: 그래프 구조 데이터를 LM이 처리 가능한 형식으로 변환이 핵심 과제

6. **하이브리드 인덱싱의 중요성**: 그래프/텍스트/벡터 인덱싱 결합이 실용적으로 가장 효과적

7. **다중 홉 추론 능력**: GraphRAG의 근본적 장점으로, 복잡한 쿼리에 대한 정확한 응답 생성

## 주요 인용 (Key Quotes)

1. **전통 RAG의 한계**:
   > "Traditional RAG fails to capture significant structured relational knowledge that cannot be represented through semantic similarity alone." (Section 1, p.2)

2. **GraphRAG의 핵심 가치**:
   > "GraphRAG leverages structural information across entities to enable more precise and comprehensive retrieval, capturing relational knowledge and facilitating more accurate, context-aware responses." (Abstract, p.1)

3. **다중 홉 추론의 장점**:
   > "The graph architecture of GraphRAG enables multi-hop reasoning, that is, the ability to comprehend and answer complex queries by traversing multiple nodes in the graph. This process does not rely solely on isolated chunks or simple semantic similarity but instead synthesizes information from multiple sources through edges (relationships) in the graph." (Section 2.1, p.5)

4. **정보 중복 완화**:
   > "Graph information primarily consists of entities, relationships, and concise textual elements, which to some extent mitigates the issue of information redundancy inherent in RAG approaches." (Section 2.1, p.4)

5. **그래프 구축의 근본적 과제**:
   > "Graph construction is inherently an information compression process, raising the fundamental question of how to maximize the retention of key semantic content while maintaining information conciseness." (Section 5.1.2, p.13)

6. **하이브리드 인덱싱의 필요성**:
   > "In practical applications, a hybrid approach combining these indexing methods is often preferred over relying solely on one... NodeRAG simultaneously integrates graph indexing, text indexing, and vector indexing." (Section 5.2, p.14-15)

7. **검색의 핵심 과제**:
   > "As the graph size increases, the number of candidate subgraphs grows exponentially, requiring heuristic search algorithms to efficiently explore and retrieve relevant subgraphs." (Section 6, p.15)

8. **GraphRAG와 RAG의 비교**:
   > "GraphRAG considers the interconnections between texts, enabling a more accurate and comprehensive retrieval of relational information. Additionally, graph data, such as Knowledge Graphs, offer abstraction and summarization of textual data, thereby significantly shortening the length of the input text." (Section 1, p.2-3)

## 시사점 및 의의

### 학술적 의의
1. **최초의 체계적 GraphRAG 서베이**: G-Indexing, G-Retrieval, G-Generation의 3단계 프레임워크를 공식 정의하고 기존 연구를 체계적으로 분류
2. **포괄적 방법론 비교**: Table 1에서 40+ 기존 방법의 그래프 데이터, 인덱싱, 리트리버, 검색 패러다임, 검색 세분도, 향상 기법, 생성기 등을 일목요연하게 비교
3. **기술적 분류 체계 확립**: 리트리버(비모수/LM/GNN), 검색 패러다임(1회/반복/다단계/적응), 생성 향상(사전/중간/사후) 등의 분류 기준 제시

### 산업적 시사점
1. **RAG 시스템의 진화 방향**: 단순 텍스트 검색에서 구조화된 그래프 기반 검색으로의 전환이 복잡한 추론 과제에서 필수적
2. **Microsoft GraphRAG의 성공**: 커뮤니티 기반 요약으로 QFS 과제를 효과적으로 해결하는 산업적 검증 사례
3. **그래프 구축 비용 문제**: LLM 기반 그래프 구축의 높은 계산 비용이 실용화의 주요 장벽이며, 오픈 KG 활용 + 동적 추출의 하이브리드 접근이 해결책

### 향후 연구 방향
1. **효율적 그래프 구축**: 비용 대비 효과를 최적화하는 그래프 구축 방법론
2. **멀티모달 GraphRAG**: 텍스트 외 이미지/오디오 등 다양한 모달리티 통합
3. **시간적 그래프**: 시간에 따른 지식 변화를 반영하는 동적 그래프
4. **GraphRAG 에이전트**: 자율적으로 그래프 기반 추론을 수행하는 에이전트 시스템
5. **대규모 확장성**: 수십억 노드 규모에서의 효율적 검색 및 생성
