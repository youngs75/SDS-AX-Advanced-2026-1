# 한국어 공개 데이터셋 카탈로그

> Day-06 LLM 파인튜닝 워크숍용 — HuggingFace 공개 데이터셋 종합 목록
> 조사일: 2026-04-05

---

## 사용법

모든 데이터셋은 HuggingFace `datasets` 라이브러리로 바로 다운로드할 수 있습니다.

```python
from datasets import load_dataset

# 기본 사용법
ds = load_dataset("데이터셋_ID")

# 특정 config가 있는 경우
ds = load_dataset("데이터셋_ID", "config_name")

# 특정 split만 로드
ds = load_dataset("데이터셋_ID", split="train")

# 일부만 샘플링
ds = load_dataset("데이터셋_ID", split="train[:1000]")
```

---

## Part 1: SFT (Supervised Fine-Tuning) 데이터셋

### Tier 1 — 대규모, 범용, 즉시 사용 가능

#### 1. maywell/korean_textbooks

```python
ds = load_dataset("maywell/korean_textbooks", "claude_evol")  # 239K rows
# 다른 config: tiny-textbooks (396K), normal_instructions (241K), code-alpaca (64K), ko_wikidata (128K)
```

| 항목 | 내용 |
|------|------|
| 크기 | **1.8M rows** (42개 config 합계) |
| 스키마 | `text` |
| 라이선스 | **Apache 2.0** |
| 품질 | Gemini Pro 합성, "Textbooks Are All You Need" 방식 |
| URL | https://huggingface.co/datasets/maywell/korean_textbooks |

---

#### 2. squarelike/OpenOrca-gugugo-ko

```python
ds = load_dataset("squarelike/OpenOrca-gugugo-ko")  # 1M+ rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **1M+ rows** (GPT-4 번역 640K + GPT-3.5 번역 1.59M) |
| 스키마 | system prompt + instruction + response (OpenOrca 형식) |
| 라이선스 | **MIT** |
| 품질 | OpenOrca 기계번역, 대규모 |
| URL | https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko |

---

#### 3. heegyu/open-korean-instructions

```python
ds = load_dataset("heegyu/open-korean-instructions")  # 997K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **997K rows** |
| 스키마 | `<sys>/<usr>/<bot>` 토큰 형식 |
| 라이선스 | **MIT** |
| 품질 | KoAlpaca v1.0+v1.1 + ShareGPT DeepL + OIG-small-chip2-ko + KorQuAD-Chat 통합 |
| URL | https://huggingface.co/datasets/heegyu/open-korean-instructions |

---

#### 4. maywell/koVast

```python
ds = load_dataset("maywell/koVast")  # 685K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **684,579 rows** |
| 스키마 | `conversations` (list of `{from: user/gpt, value: text}`), `split` |
| 라이선스 | **MIT** |
| 품질 | 멀티턴 대화, Sionic AI A100으로 생성, 14+ 모델 학습에 사용됨 |
| URL | https://huggingface.co/datasets/maywell/koVast |

---

#### 5. nlpai-lab/kullm-v2

```python
ds = load_dataset("nlpai-lab/kullm-v2", split="train")  # 153K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **152,630 rows** |
| 스키마 | `id`, `instruction`, `input`, `output` |
| 라이선스 | **Apache 2.0** |
| 품질 | DeepL 번역 (GPT4ALL + Dolly + Vicuna), 한국어 SFT 표준 데이터셋 |
| URL | https://huggingface.co/datasets/nlpai-lab/kullm-v2 |

---

### Tier 2 — 중규모, 범용

#### 6. heegyu/OIG-small-chip2-ko

```python
ds = load_dataset("heegyu/OIG-small-chip2-ko")  # 210K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **210,282 rows** |
| 스키마 | OIG 형식 |
| 라이선스 | **Apache 2.0** |
| 품질 | Google Translate 번역, 한국어+영어 이중언어 |
| URL | https://huggingface.co/datasets/heegyu/OIG-small-chip2-ko |

---

#### 7. heegyu/aulm-0809

```python
ds = load_dataset("heegyu/aulm-0809")  # 171K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **171,404 rows** |
| 스키마 | 통합 형식 |
| 라이선스 | - |
| 품질 | KoAlpaca + ShareGPT + KorQuAD-Chat + Evolve-Instruct 통합 |
| URL | https://huggingface.co/datasets/heegyu/aulm-0809 |

---

#### 8. heegyu/open-korean-instructions-v20231020

```python
ds = load_dataset("heegyu/open-korean-instructions-v20231020")  # 144K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **144,480 rows** |
| 스키마 | 통합 형식 |
| 라이선스 | - |
| 품질 | Evolve-Instruct + KoAlpaca + ShareGPT-74k-ko 포함, 2023년 10월 업데이트 |
| URL | https://huggingface.co/datasets/heegyu/open-korean-instructions-v20231020 |

---

#### 9. maywell/ko_wikidata_QA

```python
ds = load_dataset("maywell/ko_wikidata_QA")  # 138K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **137,505 rows** |
| 스키마 | `instruction`, `output` |
| 라이선스 | Non-commercial data (모델 학습은 허용) |
| 품질 | GPT + Synatra-7B로 Wikidata 기반 QA 생성, 중복 제거됨 |
| URL | https://huggingface.co/datasets/maywell/ko_wikidata_QA |

---

#### 10. jojo0217/korean_rlhf_dataset

```python
ds = load_dataset("jojo0217/korean_rlhf_dataset")  # 107K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **107,189 rows** |
| 스키마 | `instruction`, `input`, `output` |
| 라이선스 | **Apache 2.0** |
| 품질 | 5개 소스 통합 + GPT-3.5-Turbo 정제 + 중복 제거, 깨끗한 단일 파일 |
| URL | https://huggingface.co/datasets/jojo0217/korean_rlhf_dataset |

---

#### 11. devngho/korean-instruction-mix

```python
ds = load_dataset("devngho/korean-instruction-mix")  # 100K+ rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **100K~1M rows** |
| 스키마 | 혼합 |
| 라이선스 | **CC-BY-SA-4.0** |
| 품질 | 7개 소스 가중 샘플링, 글쓰기/대화/instruction/QA 혼합 |
| URL | https://huggingface.co/datasets/devngho/korean-instruction-mix |

---

### Tier 3 — 특화 / 소규모 고품질

#### 12. beomi/KoAlpaca-v1.1a

```python
ds = load_dataset("beomi/KoAlpaca-v1.1a")  # 21K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **21,155 rows** |
| 스키마 | `instruction`, `output`, `url` |
| 라이선스 | - (네이버 지식인 출처) |
| 품질 | **실제 인간 작성** QA (네이버 지식인), 한국어 SFT의 기초 데이터셋 |
| URL | https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a |

---

#### 13. kyujinpy/OpenOrca-KO

```python
ds = load_dataset("kyujinpy/OpenOrca-KO")  # 20K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **~20,000 rows** |
| 스키마 | OpenOrca 형식 |
| 라이선스 | **MIT** |
| 품질 | DeepL Pro 번역 + 수동 오류 수정, 깨끗하고 잘 관리됨 |
| URL | https://huggingface.co/datasets/kyujinpy/OpenOrca-KO |

---

#### 14. kyujinpy/KOpen-platypus

```python
ds = load_dataset("kyujinpy/KOpen-platypus")  # 25K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **~25,000 rows** |
| 스키마 | Open-Platypus 형식 |
| 라이선스 | **CC-BY-4.0** |
| 품질 | DeepL Pro 번역, STEM/추론 특화 |
| URL | https://huggingface.co/datasets/kyujinpy/KOpen-platypus |

---

#### 15. nlpai-lab/databricks-dolly-15k-ko

```python
ds = load_dataset("nlpai-lab/databricks-dolly-15k-ko")  # 15K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **15,011 rows** |
| 스키마 | `instruction`, `response`, `context`, `category` |
| 라이선스 | **CC-BY-SA 3.0** |
| 품질 | Databricks 직원 인간 작성 + DeepL 번역, 8가지 instruction 카테고리 |
| URL | https://huggingface.co/datasets/nlpai-lab/databricks-dolly-15k-ko |

---

#### 16. nlpai-lab/openassistant-guanaco-ko

```python
ds = load_dataset("nlpai-lab/openassistant-guanaco-ko")  # 10K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **10,364 rows** |
| 스키마 | `split`, `text`, `id` |
| 라이선스 | **Apache 2.0** |
| 품질 | OpenAssistant 최고 평점 경로 DeepL 번역 |
| URL | https://huggingface.co/datasets/nlpai-lab/openassistant-guanaco-ko |

---

#### 17. changpt/ko-lima-vicuna

```python
ds = load_dataset("changpt/ko-lima-vicuna")  # 1K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **1,030 rows** |
| 스키마 | `id`, `conversations` |
| 라이선스 | **CC BY 2.0 KR** |
| 품질 | GPT-4로 번역이 아닌 **재생성**, 소량이지만 최고 품질 |
| URL | https://huggingface.co/datasets/changpt/ko-lima-vicuna |

---

#### 18. Bingsu/ko_alpaca_data

```python
ds = load_dataset("Bingsu/ko_alpaca_data")  # 10K~100K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **10K~100K rows** |
| 스키마 | Alpaca 형식 |
| 라이선스 | CC-BY-NC-4.0 |
| 품질 | DeepL 번역 Stanford Alpaca + GPT-3.5-turbo 출력 |
| URL | https://huggingface.co/datasets/Bingsu/ko_alpaca_data |

---

#### 19. NLPBada/korean-persona-chat-dataset

```python
ds = load_dataset("NLPBada/korean-persona-chat-dataset")  # 10K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **10,328 rows** |
| 스키마 | 페르소나 채팅 |
| 라이선스 | **MIT** |
| 품질 | 채팅 페르소나, 존칭↔반말 변환 포함 |
| URL | https://huggingface.co/datasets/NLPBada/korean-persona-chat-dataset |

---

#### 20. heegyu/korquad-chat-v1

```python
ds = load_dataset("heegyu/korquad-chat-v1")  # 10K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **9,619 rows** |
| 스키마 | 멀티턴 대화 |
| 라이선스 | **MIT** |
| 품질 | KorQuAD 1.0 기반 ChatGPT 대화 생성, 지식 기반 |
| URL | https://huggingface.co/datasets/heegyu/korquad-chat-v1 |

---

#### 21. iknow-lab/ko-genstruct-v1

```python
ds = load_dataset("iknow-lab/ko-genstruct-v1")  # 5.7K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **5,726 rows** |
| 스키마 | `title`, `text`, `questions`, `source`, `generator` |
| 라이선스 | - |
| 품질 | 위키피디아 + Gemini 1.5 Flash QA, 다양한 난이도 |
| URL | https://huggingface.co/datasets/iknow-lab/ko-genstruct-v1 |

---

#### 22. didi0di/Chatbot_data_for_Korean_v1.0

```python
ds = load_dataset("didi0di/Chatbot_data_for_Korean_v1.0")  # 12K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **11,823 rows** |
| 스키마 | QA + 감성 라벨 |
| 라이선스 | - |
| 품질 | 인간 주석 챗봇 QA |
| URL | https://huggingface.co/datasets/didi0di/Chatbot_data_for_Korean_v1.0 |

---

### 도메인 특화 SFT

#### 23. 42MARU/korean-financial-sft (금융)

```python
ds = load_dataset("42MARU/korean-financial-sft")  # 2.9K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **2,862 rows** |
| 스키마 | TRL SFTTrainer 대화 형식 |
| 라이선스 | **CC-BY-4.0** |
| 품질 | 은행 예금 FAQ + 개인 신용 등급 QA |
| URL | https://huggingface.co/datasets/42MARU/korean-financial-sft |

---

#### 24. LuminaMotionAI/korean-legal-dataset (법률)

```python
ds = load_dataset("LuminaMotionAI/korean-legal-dataset")  # 80K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **80,000 rows** |
| 스키마 | QA + 요약 |
| 라이선스 | - |
| 품질 | 판결문, 법령, 판례 |
| URL | https://huggingface.co/datasets/LuminaMotionAI/korean-legal-dataset |

---

#### 25. JusWis/korean-legal-terminology (법률 용어)

```python
ds = load_dataset("JusWis/korean-legal-terminology")  # 17K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **17,484 rows** |
| 스키마 | instruction 형식 |
| 라이선스 | **CC-BY-4.0** |
| 품질 | 한국법제연구원 법률 용어 정의 |
| URL | https://huggingface.co/datasets/JusWis/korean-legal-terminology |

---

#### 26. squarelike/ko_medical_chat (의료)

```python
ds = load_dataset("squarelike/ko_medical_chat")  # 3K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **3,038 rows** |
| 스키마 | 환자/의사 대화 |
| 라이선스 | - |
| 품질 | MedText + ChatDoctor 변환, 의료 도메인 |
| URL | https://huggingface.co/datasets/squarelike/ko_medical_chat |

---

#### 27. kuotient/orca-math-word-problems-193k-korean (수학)

```python
ds = load_dataset("kuotient/orca-math-word-problems-193k-korean")  # 193K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **193,000 rows** |
| 스키마 | 수학 문제 + 풀이 |
| 라이선스 | **CC-BY-SA 4.0** |
| 품질 | Orca-Math 한국어 번역, 단계별 풀이 포함 |
| URL | https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean |

---

### DPO / RLHF 데이터셋

#### 28. maywell/ko_Ultrafeedback_binarized

```python
ds = load_dataset("maywell/ko_Ultrafeedback_binarized")  # 62K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **61,966 rows** |
| 스키마 | `prompt`, `chosen`, `rejected` |
| 라이선스 | Non-commercial |
| 품질 | Synatra-7B 번역, DPO/RLHF 학습용 |
| URL | https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized |

---

#### 29. heegyu/hh-rlhf-ko

```python
ds = load_dataset("heegyu/hh-rlhf-ko")  # 100K+ rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **100K+ rows** |
| 스키마 | RLHF 선호 쌍 |
| 라이선스 | **MIT** |
| 품질 | Anthropic hh-rlhf Synatra-7B 번역 |
| URL | https://huggingface.co/datasets/heegyu/hh-rlhf-ko |

---

#### 30. kuotient/orca-math-korean-dpo-pairs

```python
ds = load_dataset("kuotient/orca-math-korean-dpo-pairs")  # 100K+ rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **100K+ rows** |
| 스키마 | DPO 쌍 |
| 라이선스 | **CC-BY-SA 4.0** |
| 품질 | Claude Haiku + GPT-3.5 평가 수학 DPO |
| URL | https://huggingface.co/datasets/kuotient/orca-math-korean-dpo-pairs |

---

### 기타

#### 31. squarelike/sharegpt_deepl_ko_translation

```python
ds = load_dataset("squarelike/sharegpt_deepl_ko_translation")  # 487K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **487,000 rows** |
| 스키마 | `korean`, `english`, `num` |
| 라이선스 | - |
| 품질 | ShareGPT DeepL 번역, 한영 병렬 |
| URL | https://huggingface.co/datasets/squarelike/sharegpt_deepl_ko_translation |

---

#### 32. kyujinpy/KoCommercial-NoSSL

```python
ds = load_dataset("kyujinpy/KoCommercial-NoSSL")  # 175K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **~175,000 rows** |
| 스키마 | 혼합 |
| 라이선스 | CC-BY-NC-SA-4.0 |
| 품질 | 상업적 사용 가능하도록 필터링된 혼합 세트 |
| URL | https://huggingface.co/datasets/kyujinpy/KoCommercial-NoSSL |

---

#### 33. kyujinpy/KOR-OpenOrca-Platypus-v3

```python
ds = load_dataset("kyujinpy/KOR-OpenOrca-Platypus-v3")  # 10K~100K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **10K~100K rows** |
| 스키마 | OpenOrca + Platypus 통합 |
| 라이선스 | CC-BY-NC-4.0 |
| 품질 | OpenOrca-Ko + KOpen-platypus 통합 + 200+ 수동 수정 |
| URL | https://huggingface.co/datasets/kyujinpy/KOR-OpenOrca-Platypus-v3 |

---

#### 34. nmixx-fin/opensource_korean_finance_datasets (금융 코퍼스)

```python
ds = load_dataset("nmixx-fin/opensource_korean_finance_datasets")  # 100K~1M rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **100K~1M rows** |
| 스키마 | `source`, `text`, `category`, `token_count` |
| 라이선스 | - |
| 품질 | 한국어 금융 텍스트 통합 (instruction 형식 아님, 사전학습/지식 주입용) |
| URL | https://huggingface.co/datasets/nmixx-fin/opensource_korean_finance_datasets |

---

## Part 2: Embedding 학습 데이터셋

### Category A — 검색 트리플릿 (Query / Positive / Negative)

> 가장 직접적으로 contrastive learning에 사용 가능한 형태

#### E1. nlpai-lab/ko-triplet-v1.0 ★ 최우선 추천

```python
ds = load_dataset("nlpai-lab/ko-triplet-v1.0")  # 744K rows
# 바로 사용: anchor=query, positive=document, negative=hard_negative
```

| 항목 | 내용 |
|------|------|
| 크기 | **744,862 rows** |
| 스키마 | `query`, `document`, `hard_negative` |
| 라이선스 | - |
| 품질 | **한국어 임베딩 전용 목적 빌트**, KoE5/bge-m3-ko 학습에 사용, 고려대 NLP Lab |
| URL | https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0 |

---

#### E2. seongil-dn/korean_retrieval_583319

```python
ds = load_dataset("seongil-dn/korean_retrieval_583319")  # 583K rows
# anchor / positive / negative / subset(6종)
```

| 항목 | 내용 |
|------|------|
| 크기 | **583,319 rows** |
| 스키마 | `anchor`, `positive`, `negative`, `subset` (6개 카테고리) |
| 라이선스 | - |
| 품질 | 금융/법률 도메인 한국어 검색 데이터, 서브셋 라벨 포함 |
| URL | https://huggingface.co/datasets/seongil-dn/korean_retrieval_583319 |

---

#### E3. williamjeong2/msmarco-triplets-ko-v1

```python
ds = load_dataset("williamjeong2/msmarco-triplets-ko-v1")  # 499K rows
# pos, neg가 리스트이므로 첫 번째 요소 사용: row['pos'][0], row['neg'][0]
```

| 항목 | 내용 |
|------|------|
| 크기 | **499,000 rows** |
| 스키마 | `query`, `pos` (list), `neg` (list) |
| 라이선스 | MS MARCO 약관 |
| 품질 | MS MARCO 한국어 기계번역, 대규모 |
| URL | https://huggingface.co/datasets/williamjeong2/msmarco-triplets-ko-v1 |

---

#### E4. CocoRoF/msmacro_triplet_ko

```python
ds = load_dataset("CocoRoF/msmacro_triplet_ko")  # 503K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **503,000 rows** |
| 스키마 | `query`, `pos`, `neg` |
| 라이선스 | MS MARCO 약관 |
| 품질 | MS MARCO 한국어 다른 번역 버전 |
| URL | https://huggingface.co/datasets/CocoRoF/msmacro_triplet_ko |

---

#### E5. seongil-dn/korean_retrieval_dataset_v0

```python
ds = load_dataset("seongil-dn/korean_retrieval_dataset_v0")  # 483K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **482,743 rows** |
| 스키마 | `anchor`, `positive`, `negative` |
| 라이선스 | - |
| 품질 | 금융/법률 도메인, seongil-dn 시리즈 초기 버전 |
| URL | https://huggingface.co/datasets/seongil-dn/korean_retrieval_dataset_v0 |

---

#### E6. seongil-dn/korean_retrieval_subset

```python
ds = load_dataset("seongil-dn/korean_retrieval_subset")  # 287K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **287,413 rows** |
| 스키마 | `query`, `positive`, `hard_negative` |
| 라이선스 | - |
| 품질 | v0의 정제 서브셋, query/positive/hard_negative 컬럼명 |
| URL | https://huggingface.co/datasets/seongil-dn/korean_retrieval_subset |

---

#### E7. williamjeong2/msmarco-triplets-ko-v2

```python
ds = load_dataset("williamjeong2/msmarco-triplets-ko-v2")  # 332K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **332,000 rows** |
| 스키마 | `query`, `pos` (list), `neg` (list) |
| 라이선스 | MS MARCO 약관 |
| 품질 | v1의 개정/필터링 버전 |
| URL | https://huggingface.co/datasets/williamjeong2/msmarco-triplets-ko-v2 |

---

#### E8. crjoya/korean-proptech-retrieval (부동산 도메인)

```python
ds = load_dataset("crjoya/korean-proptech-retrieval")  # 1K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **1,000 rows** |
| 스키마 | `query`, `positive_document` (list, 3개), `negative_document` (list, 3개) |
| 라이선스 | - |
| 품질 | 부동산 도메인 특화, 소규모 |
| URL | https://huggingface.co/datasets/crjoya/korean-proptech-retrieval |

---

### Category B — NLI 데이터셋 (SimCSE/Contrastive 변환 가능)

> `entailment` → positive, `contradiction` → hard_negative로 변환하여 임베딩 학습에 사용

#### E9. kakaobrain/kor_nli ★ 핵심

```python
# 전체 로드
ds_multi = load_dataset("kakaobrain/kor_nli", "multi_nli")  # 392K rows
ds_snli = load_dataset("kakaobrain/kor_nli", "snli")         # 550K rows
ds_xnli = load_dataset("kakaobrain/kor_nli", "xnli_dev")     # 2.5K rows

# SimCSE 트리플릿 변환
positives = ds_multi['train'].filter(lambda x: x['label'] == 0)   # entailment
negatives = ds_multi['train'].filter(lambda x: x['label'] == 2)   # contradiction
```

| 항목 | 내용 |
|------|------|
| 크기 | **950,354 rows** 합계 |
| 스키마 | `premise`, `hypothesis`, `label` (0=entailment, 1=neutral, 2=contradiction) |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | KakaoBrain 공식, MultiNLI/SNLI/XNLI 번역, 한국어 임베딩 평가 표준 |
| 논문 | arXiv:2004.03289 (Ham et al., 2020) |
| URL | https://huggingface.co/datasets/kakaobrain/kor_nli |

---

#### E10. dkoterwa/kor_nli_simcse ★ 변환 완료된 트리플릿

```python
ds = load_dataset("dkoterwa/kor_nli_simcse")  # 487K rows
# 바로 사용: anchor=premise, positive=entailment, negative=contradiction
```

| 항목 | 내용 |
|------|------|
| 크기 | **486,868 rows** (train 414K, valid 49K, test 24K) |
| 스키마 | `premise`, `entailment`, `contradiction` — **이미 트리플릿 형태** |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | kor_nli에서 SimCSE 형태로 사전 변환 완료, 바로 사용 가능 |
| URL | https://huggingface.co/datasets/dkoterwa/kor_nli_simcse |

---

#### E11. klue/klue — NLI config

```python
ds = load_dataset("klue/klue", "nli")  # 28K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **28,000 rows** |
| 스키마 | `premise`, `hypothesis`, `label` |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | **네이티브 한국어 작성** (번역이 아님), 뉴스/위키/리뷰 출처, 공식 벤치마크 |
| 논문 | arXiv:2105.09680 (Park et al., 2021) |
| URL | https://huggingface.co/datasets/klue/klue |

---

#### E12. phnyxlab/klue-nli-simcse

```python
ds = load_dataset("phnyxlab/klue-nli-simcse")  # 9K rows
# 바로 사용: premise / entailment / contradiction
```

| 항목 | 내용 |
|------|------|
| 크기 | **9,047 rows** |
| 스키마 | `premise`, `entailment`, `contradiction` |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | KLUE NLI에서 SimCSE 트리플릿 변환 완료, 소규모 고품질 네이티브 |
| URL | https://huggingface.co/datasets/phnyxlab/klue-nli-simcse |

---

### Category C — STS 데이터셋 (유사도 점수)

> `CosineSimilarityLoss`로 직접 사용하거나, score >= 임계값으로 이진화하여 contrastive 학습

#### E13. mteb/KorSTS ★ 평가 표준

```python
ds = load_dataset("mteb/KorSTS")  # 8.5K rows
# score: 0~5, 정규화: score / 5.0
```

| 항목 | 내용 |
|------|------|
| 크기 | **8,532 rows** (train 5,690, test 1,376, valid 1,466) |
| 스키마 | `sentence1`, `sentence2`, `score` (0~5) |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | MTEB 공식 벤치마크, 인간 후편집 |
| URL | https://huggingface.co/datasets/mteb/KorSTS |

---

#### E14. klue/klue — STS config

```python
ds = load_dataset("klue/klue", "sts")  # 12K rows
# binary-label 포함: labels['binary-label'] (0 or 1)
```

| 항목 | 내용 |
|------|------|
| 크기 | **12,200 rows** |
| 스키마 | `sentence1`, `sentence2`, `labels` (label 0~5, binary-label 0/1) |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | 네이티브 한국어, 에어비앤비 리뷰/정책 문서/패러프레이즈 출처, 이진 라벨 포함 |
| URL | https://huggingface.co/datasets/klue/klue |

---

#### E15. CocoRoF/misc_sts_pairs_v2_kor

```python
ds = load_dataset("CocoRoF/misc_sts_pairs_v2_kor")  # 450K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **449,904 rows** |
| 스키마 | `sentence1`, `sentence2`, `score` (0.52~1.0) |
| 라이선스 | - |
| 품질 | 영어→한국어 기계번역, 대규모, score 범위 편향 주의 (음성 쌍 없음) |
| URL | https://huggingface.co/datasets/CocoRoF/misc_sts_pairs_v2_kor |

---

#### E16. 금융/법률 도메인 STS

```python
ds_fin_news = load_dataset("nmixx-fin/NMIXX_kor_fin_news_STS")     # 853 rows
ds_fin_law  = load_dataset("nmixx-fin/NMIXX_kor_fin_law_STS")      # 715 rows
ds_report   = load_dataset("nmixx-fin/NMIXX_kor_fin_report_STS")   # 421 rows
ds_law_sts  = load_dataset("tokkilab/kor-law-sts")                  # 2,794 rows
```

| 데이터셋 | 크기 | 도메인 |
|----------|------|--------|
| `nmixx-fin/NMIXX_kor_fin_news_STS` | 853 | 금융 뉴스 |
| `nmixx-fin/NMIXX_kor_fin_law_STS` | 715 | 금융 법률 |
| `nmixx-fin/NMIXX_kor_fin_report_STS` | 421 | 금융 리포트 |
| `tokkilab/kor-law-sts` | 2,794 | 건축/민법 |

---

### Category D — QA / MRC 데이터셋 (검색 쌍으로 변환 가능)

> (question, context) → positive pair, 다른 문서에서 negative 샘플링

#### E17. KorQuAD/squad_kor_v1 ★ 표준 벤치마크

```python
ds = load_dataset("KorQuAD/squad_kor_v1")  # 66K rows
# query=question, positive=context, negative=다른 문서의 context 샘플링
```

| 항목 | 내용 |
|------|------|
| 크기 | **66,181 rows** |
| 스키마 | `question`, `context`, `answers` |
| 라이선스 | CC BY-ND 4.0 (파생물 제한 주의) |
| 품질 | 한국어 위키피디아 인간 작성 QA, 표준 벤치마크 |
| URL | https://huggingface.co/datasets/KorQuAD/squad_kor_v1 |

---

#### E18. LGCNS/KorQuAD_2.0

```python
ds = load_dataset("LGCNS/KorQuAD_2.0")  # 94K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **93,651 rows** |
| 스키마 | `question`, `context` (전체 문서), `answer`, `url` |
| 라이선스 | **CC BY-SA 3.0** |
| 품질 | 전체 위키 문서 + HTML, 긴 컨텍스트 검색 학습에 적합 |
| URL | https://huggingface.co/datasets/LGCNS/KorQuAD_2.0 |

---

#### E19. lcw99/wikipedia-korean-20240501-1million-qna

```python
ds = load_dataset("lcw99/wikipedia-korean-20240501-1million-qna")  # 990K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **990,023 rows** |
| 스키마 | `question`, `answer`, `context` |
| 라이선스 | - |
| 품질 | 한국어 위키피디아 자동 생성 QA, 초대규모 |
| URL | https://huggingface.co/datasets/lcw99/wikipedia-korean-20240501-1million-qna |

---

#### E20. klue/klue — MRC config

```python
ds = load_dataset("klue/klue", "mrc")  # 23K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **23,400 rows** |
| 스키마 | `question`, `context`, `answers`, `is_impossible` |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | 뉴스 도메인 네이티브 QA, 답변 불가 문항 포함 |
| URL | https://huggingface.co/datasets/klue/klue |

---

#### E21. sungmineom/korquad_negative_samples

```python
ds = load_dataset("sungmineom/korquad_negative_samples")  # 147K rows
# 이미 4개의 hard negative가 포함됨!
```

| 항목 | 내용 |
|------|------|
| 크기 | **147,045 rows** |
| 스키마 | `question`, `context`, `answer_text`, `negative_samples` (list, 4개) |
| 라이선스 | - |
| 품질 | KorQuAD + BM25 기반 hard negative 4개 사전 포함, 바로 contrastive 학습 가능 |
| URL | https://huggingface.co/datasets/sungmineom/korquad_negative_samples |

---

#### E22. Doohae/klue-mrc-bm25

```python
ds = load_dataset("Doohae/klue-mrc-bm25")  # 17.6K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **17,600 rows** |
| 스키마 | `question`, `context`, `answers`, `negatives` (BM25 hard negative) |
| 라이선스 | **CC BY-SA 4.0** |
| 품질 | KLUE MRC + BM25 hard negative 추가, dense retrieval 학습용 |
| URL | https://huggingface.co/datasets/Doohae/klue-mrc-bm25 |

---

#### E23. daydrill/QG_korquad_aihub

```python
ds = load_dataset("daydrill/QG_korquad_aihub")  # 227K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **227,096 rows** |
| 스키마 | `question`, `paragraph`, `answer`, `sentence` |
| 라이선스 | - |
| 품질 | KorQuAD + AIHub 질문 생성, 문장/문단 레벨 다양한 컨텍스트 |
| URL | https://huggingface.co/datasets/daydrill/QG_korquad_aihub |

---

#### E24. iamjoon/klue-mrc-ko-rag-dataset

```python
ds = load_dataset("iamjoon/klue-mrc-ko-rag-dataset")  # 1.9K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **1,884 rows** |
| 스키마 | `question`, `search_result` (passage 리스트), `answer`, `type` |
| 라이선스 | - |
| 품질 | RAG 구조: query + 복수 검색 결과 + 근거 답변, RAG 검색 학습에 직접 적합 |
| URL | https://huggingface.co/datasets/iamjoon/klue-mrc-ko-rag-dataset |

---

### Category E — 다국어 병렬 (한국어 포함)

> 이중언어 임베딩 정렬에 사용

#### E25. sentence-transformers/parallel-sentences-ccmatrix (en-ko)

```python
ds = load_dataset("sentence-transformers/parallel-sentences-ccmatrix", "en-ko")  # 19.4M rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **19,400,000 rows** |
| 스키마 | `english`, `non_english` (한국어) |
| 라이선스 | CC BY-SA |
| 품질 | 웹 크롤링 정렬, 초대규모, 노이즈 필터링 필요 |
| URL | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix |

---

#### E26. sentence-transformers/parallel-sentences-talks (en-ko)

```python
ds = load_dataset("sentence-transformers/parallel-sentences-talks", "en-ko")  # 390K rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **390,000 rows** |
| 스키마 | `english`, `non_english` (한국어) |
| 라이선스 | TED Creative Commons |
| 품질 | TED Talk 인간 번역, 고품질 |
| URL | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks |

---

#### E27. bongsoo/news_talk_en_ko

```python
ds = load_dataset("bongsoo/news_talk_en_ko")  # 1.3M rows
```

| 항목 | 내용 |
|------|------|
| 크기 | **1,300,000 rows** |
| 스키마 | 영한 뉴스 병렬 |
| 라이선스 | - |
| 품질 | 뉴스 도메인 영한 병렬 |
| URL | https://huggingface.co/datasets/bongsoo/news_talk_en_ko |

---

### Category F — 한국어 코퍼스 (비지도 학습 / Hard Negative 마이닝용)

| 데이터셋 | 크기 | 내용 |
|----------|------|------|
| `lcw99/wikipedia-korean-20240501` | 515K | 한국어 위키피디아 2024.05 |
| `devngho/korean_wikipedia` | 1.02M | 한국어 위키피디아 대안 |
| `korean-corpus/namu_wiki_512_char_seg` | 6.23M | 나무위키 512자 분할 |
| `seanswyi/korean-wikipedia-namu-wiki` | 1.67M | 위키+나무위키 통합 |
| `bongsoo/kowiki20220620` | 4.39M | 한국어 위키 2022.06 |

```python
# 예시: 나무위키 코퍼스 로드
ds = load_dataset("korean-corpus/namu_wiki_512_char_seg")
```

---

## 워크숍 추천 조합

### Lab 1 (SFT) 추천 — 우선순위순

| 순위 | 데이터셋 | 크기 | 이유 |
|------|----------|------|------|
| 1 | Lab 0 도메인 합성 데이터 | 150~250 | AgentOps 도메인 최적 |
| 2 | `nlpai-lab/kullm-v2` | 153K | Apache 2.0, Alpaca 형식, 표준 |
| 3 | `jojo0217/korean_rlhf_dataset` | 107K | Apache 2.0, 정제 통합, 깨끗 |
| 4 | `beomi/KoAlpaca-v1.1a` | 21K | 실제 인간 QA, 높은 진정성 |
| 5 | `heegyu/open-korean-instructions` | 997K | MIT, 올인원, 대규모 |

### Lab 2 (Embedding) 추천 — 우선순위순

| 순위 | 데이터셋 | 크기 | 이유 |
|------|----------|------|------|
| 1 | Lab 0 도메인 합성 데이터 | 300~600 | AgentOps 도메인 최적 |
| 2 | `nlpai-lab/ko-triplet-v1.0` | 744K | 목적 빌트 트리플릿, 바로 사용 |
| 3 | `dkoterwa/kor_nli_simcse` | 487K | SimCSE 변환 완료, 바로 사용 |
| 4 | `sungmineom/korquad_negative_samples` | 147K | Hard negative 4개 포함 |
| 5 | `kakaobrain/kor_nli` (multi_nli) | 393K | 대규모 NLI, CC BY-SA |

---

## 빠른 다운로드 스크립트

```python
"""Day-06 워크숍용 데이터셋 일괄 다운로드 및 미리보기"""
from datasets import load_dataset
import pandas as pd

# ========== SFT 데이터셋 ==========
print("=" * 60)
print("SFT 데이터셋 다운로드")
print("=" * 60)

sft_datasets = {
    "kullm-v2":         ("nlpai-lab/kullm-v2", None, "train"),
    "korean-rlhf":      ("jojo0217/korean_rlhf_dataset", None, "train"),
    "koalpaca":         ("beomi/KoAlpaca-v1.1a", None, "train"),
    "dolly-15k-ko":     ("nlpai-lab/databricks-dolly-15k-ko", None, "train"),
    "openorca-ko":      ("kyujinpy/OpenOrca-KO", None, "train"),
    "ko-lima-vicuna":   ("changpt/ko-lima-vicuna", None, "train"),
}

for name, (dataset_id, config, split) in sft_datasets.items():
    try:
        ds = load_dataset(dataset_id, config, split=split)
        print(f"\n  {name}: {len(ds)} rows")
        print(f"    columns: {ds.column_names}")
        print(f"    load: load_dataset('{dataset_id}', split='{split}')")
    except Exception as e:
        print(f"\n  {name}: FAILED - {e}")

# ========== Embedding 데이터셋 ==========
print("\n" + "=" * 60)
print("Embedding 데이터셋 다운로드")
print("=" * 60)

emb_datasets = {
    "ko-triplet":       ("nlpai-lab/ko-triplet-v1.0", None, "train"),
    "kor-nli-simcse":   ("dkoterwa/kor_nli_simcse", None, "train"),
    "kor-nli-multi":    ("kakaobrain/kor_nli", "multi_nli", "train"),
    "klue-nli":         ("klue/klue", "nli", "train"),
    "korsts":           ("mteb/KorSTS", None, "test"),
    "korquad-neg":      ("sungmineom/korquad_negative_samples", None, "train"),
}

for name, (dataset_id, config, split) in emb_datasets.items():
    try:
        ds = load_dataset(dataset_id, config, split=split)
        print(f"\n  {name}: {len(ds)} rows")
        print(f"    columns: {ds.column_names}")
        print(f"    load: load_dataset('{dataset_id}'" + (f", '{config}'" if config else "") + f", split='{split}')")
    except Exception as e:
        print(f"\n  {name}: FAILED - {e}")

print("\n" + "=" * 60)
print("다운로드 완료!")
print("=" * 60)
```

---

## 참고

- 모든 데이터셋은 2026-04-05 기준으로 확인되었습니다.
- 라이선스가 `-`인 경우 명시되지 않은 것이며, 사용 전 확인이 필요합니다.
- 상업적 사용 시: Apache 2.0, MIT, CC-BY, CC-BY-SA 라이선스를 우선 선택하세요.
- `Non-commercial` 표시 데이터는 모델 가중치도 비상업적 사용으로 제한될 수 있습니다.
