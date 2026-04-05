# 데이터셋 확장 및 커스텀 전략

> Day-06 워크숍: 공개 한국어 데이터셋을 기반으로 도메인 특화 데이터를 구축하는 실전 전략

---

## 핵심 원칙

공개 데이터셋은 "폴백"이 아니라 **학습의 기반(base)**이다.

```
공개 데이터셋 (일반 한국어 능력)
    ↓ 필터링 + 도메인 혼합 + 합성 확장
도메인 특화 데이터셋 (AgentOps/Observability 전문성)
```

**왜 이 접근이 중요한가:**
- 일반 한국어 응답 능력이 없으면 도메인 답변도 부자연스러워짐
- 도메인 데이터만으로는 양이 부족하여 overfitting 위험
- 공개 데이터 + 도메인 데이터 혼합이 실무에서 가장 효과적인 패턴

---

## Part 1: SFT 데이터 확장 전략

### 전략 A — 3단계 혼합 (Curriculum Mixing)

```python
from datasets import load_dataset, concatenate_datasets

# ─── Stage 1: 일반 한국어 기반 (60%) ───
# 한국어 instruction-following 능력의 기초
base_ds = load_dataset("jojo0217/korean_rlhf_dataset", split="train")
# 107K rows, Apache 2.0, 5개 소스 정제 통합

# 또는 더 큰 규모가 필요하면:
# base_ds = load_dataset("nlpai-lab/kullm-v2", split="train")  # 153K

# ─── Stage 2: 도메인 인접 필터링 (20%) ───
# 기존 데이터에서 IT/운영/모니터링과 관련된 샘플만 추출
DOMAIN_KEYWORDS = [
    "모니터링", "서버", "장애", "로그", "알림", "지표", "메트릭",
    "API", "레이턴시", "응답 시간", "에러", "배포", "운영",
    "인시던트", "SLA", "가용성", "트래픽", "성능", "디버깅",
    "observability", "monitoring", "latency", "error rate",
]

def is_domain_adjacent(example):
    text = f"{example.get('instruction', '')} {example.get('output', '')}".lower()
    return any(kw.lower() in text for kw in DOMAIN_KEYWORDS)

domain_adjacent = base_ds.filter(is_domain_adjacent)
print(f"도메인 인접 샘플: {len(domain_adjacent)} / {len(base_ds)} ({len(domain_adjacent)/len(base_ds):.1%})")

# ─── Stage 3: 도메인 전용 합성 (20%) ───
# Lab 0에서 생성한 AgentOps/Observability 전문 데이터
domain_ds = load_dataset("json", data_files="sft_train.jsonl", split="train")

# ─── 최종 혼합 ───
import random

# 비율 조정: 일반 60% + 도메인 인접 20% + 도메인 전용 20%
general_sample = base_ds.shuffle(seed=42).select(range(min(3000, len(base_ds))))
adjacent_sample = domain_adjacent.shuffle(seed=42).select(range(min(1000, len(domain_adjacent))))
# domain_ds는 전부 사용 (150~250건)

final_train = concatenate_datasets([general_sample, adjacent_sample, domain_ds])
final_train = final_train.shuffle(seed=42)

print(f"최종 혼합: {len(final_train)} rows")
print(f"  일반: {len(general_sample)}, 인접: {len(adjacent_sample)}, 전용: {len(domain_ds)}")
```

### 전략 B — 고품질 소량 + 도메인 집중

```python
# 고품질 인간 작성 데이터 (소량이지만 진정성 높음)
koalpaca = load_dataset("beomi/KoAlpaca-v1.1a", split="train")     # 21K, 네이버 지식인
lima_ko = load_dataset("changpt/ko-lima-vicuna", split="train")     # 1K, GPT-4 재생성
dolly_ko = load_dataset("nlpai-lab/databricks-dolly-15k-ko", split="train")  # 15K, 인간 작성

# 이 세 데이터셋을 기반으로 도메인 데이터를 추가하면
# 적은 양으로도 높은 품질의 SFT가 가능

# 도메인 데이터 비율을 30~40%까지 올릴 수 있음
# (기반 데이터의 품질이 높으므로 일반 능력이 적은 양으로도 유지됨)
```

### 전략 C — 도메인 리라이트 (Domain Rewrite)

```python
from openai import OpenAI

client = OpenAI()

def rewrite_to_domain(example, source_chunk):
    """일반 한국어 instruction을 AgentOps 도메인으로 변환"""
    resp = client.responses.create(
        model="gpt-5.4-mini",
        input=[{
            "role": "user",
            "content": f"""아래 일반 한국어 instruction-output 쌍을 
AgentOps/Observability 도메인으로 재작성하세요.
원본의 질문 패턴과 답변 구조는 유지하되, 
내용을 아래 소스 문서에 근거하여 바꾸세요.

원본 instruction: {example['instruction']}
원본 output: {example['output']}

소스 문서:
{source_chunk}

재작성 규칙:
- 한국어로 작성
- source_doc 근거 유지
- 원본의 질문 유형(설명/비교/절차/제안) 보존
- AgentOps 용어를 자연스럽게 사용"""
        }],
    )
    return resp.output_text

# 도메인 인접 샘플을 실제 도메인 데이터로 업그레이드
# 이 방식은 질문 패턴의 다양성을 유지하면서 도메인 커버리지를 높임
```

### SFT 혼합 비율 가이드

| 시나리오 | 일반 | 도메인 인접 | 도메인 전용 | 총 규모 |
|----------|------|------------|------------|---------|
| 최소 실습 (시간 제한) | 2,000 | 500 | 150 | ~2,650 |
| 표준 워크숍 | 3,000 | 1,000 | 250 | ~4,250 |
| 충분한 시간 | 5,000 | 2,000 | 500 | ~7,500 |
| 프로덕션 급 | 10,000+ | 3,000+ | 1,000+ | ~14,000+ |

> **핵심:** 도메인 전용 데이터는 전체의 5~20%만 되어도 도메인 성능이 크게 개선됨.
> 나머지는 일반 한국어 능력을 유지하는 역할.

---

## Part 2: Embedding 데이터 확장 전략

### 전략 D — 공개 트리플릿 + 도메인 트리플릿 통합

```python
from datasets import load_dataset, Dataset
import pandas as pd

# ─── Base: 범용 한국어 검색 트리플릿 ───
ko_triplet = load_dataset("nlpai-lab/ko-triplet-v1.0", split="train")
# 744K rows: query / document / hard_negative
# 한국어 임베딩 학습의 사실상 표준 데이터셋

# 컬럼명 통일 (Lab 2 스키마에 맞춤)
ko_triplet_df = ko_triplet.to_pandas().rename(columns={
    "query": "query",
    "document": "positive",
    "hard_negative": "hard_negative",
})
# task_instruction 추가 (instruction-aware 모델용)
ko_triplet_df["task_instruction"] = "Retrieve relevant Korean passages"
ko_triplet_df["source_doc"] = "ko-triplet-v1.0"

# ─── 도메인 확장: AgentOps 트리플릿 ───
domain_emb = load_dataset("json", data_files="data/embedding_train.jsonl", split="train")
domain_df = domain_emb.to_pandas()

# ─── 혼합: 범용 서브샘플 + 도메인 전체 ───
general_sample = ko_triplet_df.sample(n=min(5000, len(ko_triplet_df)), random_state=42)
final_df = pd.concat([general_sample, domain_df], ignore_index=True)
final_ds = Dataset.from_pandas(final_df)

print(f"최종 embedding 학습 데이터: {len(final_ds)} rows")
print(f"  범용: {len(general_sample)}, 도메인: {len(domain_df)}")
```

### 전략 E — NLI 기반 사전 학습 → 도메인 미세조정 (2단계)

```python
# ─── 1단계: NLI 기반 contrastive 사전 학습 ───
# 대규모 NLI 데이터로 일반적인 의미 유사도 학습
nli_simcse = load_dataset("dkoterwa/kor_nli_simcse", split="train")
# 414K rows: premise / entailment / contradiction
# 이미 SimCSE 트리플릿 형태 → 바로 학습 가능

# Sentence Transformers 형식으로 변환
nli_train = nli_simcse.rename_columns({
    "premise": "anchor",
    "entailment": "positive", 
    "contradiction": "negative",
})

# 1단계 학습: 일반 의미 유사도 (3~5 epochs)
# → 모델이 "비슷한 의미 = 가까운 벡터" 기초를 학습

# ─── 2단계: 도메인 특화 미세조정 ───
# 1단계 모델 위에 도메인 트리플릿으로 추가 학습
# 도메인 데이터가 적어도(300~600건) 1단계 기반이 있으므로 효과적

domain_emb = load_dataset("json", data_files="data/embedding_train.jsonl", split="train")
# 2단계 학습: 도메인 검색 특화 (1~2 epochs)
# → AgentOps 용어/개념의 검색 품질 개선
```

### 전략 F — 기존 QA 데이터에서 Hard Negative 마이닝

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── KorQuAD에서 검색 트리플릿 자동 생성 ───
korquad = load_dataset("KorQuAD/squad_kor_v1", split="train")

# 1. 모든 context를 인코딩
model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base")
contexts = list(set(korquad["context"]))
context_embeddings = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)

# 2. 각 question에 대해 BM25 또는 dense search로 hard negative 마이닝
def mine_hard_negative(question, gold_context, all_contexts, all_embeddings, model, top_k=10):
    """정답이 아닌 가장 유사한 context를 hard negative로 선택"""
    q_emb = model.encode([question], normalize_embeddings=True)
    scores = np.matmul(q_emb, all_embeddings.T)[0]
    top_indices = np.argsort(-scores)[:top_k]
    
    for idx in top_indices:
        if all_contexts[idx] != gold_context:
            return all_contexts[idx]  # 정답이 아닌 가장 유사한 것
    return None

# 3. 트리플릿 생성
triplets = []
for row in korquad:
    hard_neg = mine_hard_negative(
        row["question"], row["context"], 
        contexts, context_embeddings, model
    )
    if hard_neg:
        triplets.append({
            "query": row["question"],
            "positive": row["context"],
            "hard_negative": hard_neg,
            "task_instruction": "Retrieve the passage that answers this Korean question",
            "source_doc": "korquad_mined",
        })

print(f"KorQuAD에서 마이닝한 트리플릿: {len(triplets)}")
# → 이 트리플릿을 도메인 데이터와 혼합하여 학습
```

### 전략 G — 기존 hard negative 포함 데이터 활용

```python
# 이미 hard negative가 포함된 데이터셋 활용 (마이닝 불필요)

# 옵션 1: KorQuAD + BM25 hard negative (바로 사용 가능)
korquad_neg = load_dataset("sungmineom/korquad_negative_samples", split="train")
# 147K rows, negative_samples에 4개의 hard negative 포함

triplets_from_korquad = []
for row in korquad_neg:
    if row["negative_samples"]:
        triplets_from_korquad.append({
            "anchor": row["question"],
            "positive": row["context"],
            "negative": row["negative_samples"][0],  # 가장 어려운 negative
        })

# 옵션 2: KLUE MRC + BM25 (CC BY-SA 4.0)
klue_neg = load_dataset("Doohae/klue-mrc-bm25", split="train")
# 17.6K rows, negatives 포함

# 옵션 3: MS MARCO 한국어 (대규모)
msmarco_ko = load_dataset("williamjeong2/msmarco-triplets-ko-v1", split="train")
# 499K rows, query/pos/neg
```

### Embedding 혼합 비율 가이드

| 시나리오 | 범용 트리플릿 | NLI/SimCSE | 도메인 전용 | 총 규모 |
|----------|-------------|-----------|------------|---------|
| 최소 실습 | 2,000 (ko-triplet) | - | 300 | ~2,300 |
| 표준 워크숍 | 5,000 (ko-triplet) | 3,000 (kor_nli_simcse) | 500 | ~8,500 |
| 2단계 학습 | - | 50,000 (1단계) | 500 (2단계) | 50,500 |
| 프로덕션 급 | 50,000+ | 100,000+ | 2,000+ | 150,000+ |

> **핵심:** Embedding 학습에서는 batch_size가 클수록 in-batch negative가 많아져 효과적.
> 따라서 데이터 규모를 키우는 것이 SFT보다 더 직접적인 성능 개선으로 이어짐.

---

## Part 3: 워크숍 실전 레시피

### 레시피 1: "빠른 도메인 적응" (Lab 시간 내 완료)

```python
"""
목표: 공개 데이터 + 소량 도메인 데이터로 빠르게 도메인 적응
시간: Lab 0 (60분) + Lab 1 (120분)
"""

# SFT 데이터
sft_base = load_dataset("jojo0217/korean_rlhf_dataset", split="train[:3000]")
sft_domain = load_dataset("json", data_files="sft_train.jsonl", split="train")
# → 3,000 + 150~250 = ~3,200건으로 학습

# Embedding 데이터  
emb_base = load_dataset("nlpai-lab/ko-triplet-v1.0", split="train[:5000]")
emb_domain = load_dataset("json", data_files="data/embedding_train.jsonl", split="train")
# → 5,000 + 300~600 = ~5,500건으로 학습
```

### 레시피 2: "깊은 도메인 적응" (충분한 시간)

```python
"""
목표: 2단계 학습으로 일반 능력 + 도메인 전문성 모두 확보
시간: 사전 준비 + Lab 1 확장 + Lab 2 확장
"""

# ─── SFT 2단계 ───
# 1단계: 일반 한국어 (max_steps=120)
sft_general = load_dataset("heegyu/open-korean-instructions", split="train[:10000]")

# 2단계: 도메인 집중 (max_steps=60)  
sft_domain = concatenate_datasets([
    load_dataset("json", data_files="sft_train.jsonl", split="train"),
    # + 도메인 리라이트된 일반 샘플
])

# ─── Embedding 2단계 ───
# 1단계: NLI contrastive (3 epochs)
nli_base = load_dataset("dkoterwa/kor_nli_simcse", split="train[:50000]")

# 2단계: 도메인 retrieval (1 epoch)
emb_domain = concatenate_datasets([
    load_dataset("nlpai-lab/ko-triplet-v1.0", split="train[:5000]"),
    load_dataset("json", data_files="data/embedding_train.jsonl", split="train"),
])
```

### 레시피 3: "도메인 리라이트 중심" (OpenAI API 활용)

```python
"""
목표: 기존 고품질 데이터의 질문 패턴을 보존하면서 도메인 전환
시간: Lab 0 확장 (90분)
"""

# 1. 고품질 기반 데이터에서 도메인 인접 샘플 추출
koalpaca = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
adjacent = koalpaca.filter(is_domain_adjacent)  # IT/운영 관련 필터
print(f"도메인 인접: {len(adjacent)}건 (네이버 지식인 실제 QA)")

# 2. 인접 샘플을 AgentOps 도메인으로 리라이트
# rewrite_to_domain() 함수로 변환
# → 질문의 자연스러움은 유지하면서 내용만 도메인 전환

# 3. 리라이트된 데이터 + 순수 합성 데이터 혼합
# 리라이트: 질문 패턴 다양성 ↑
# 합성: 도메인 정확도 ↑
```

---

## Part 4: 데이터 품질 검증 파이프라인

### 혼합 후 반드시 확인할 항목

```python
def validate_mixed_dataset(dataset, domain_keywords=DOMAIN_KEYWORDS):
    """혼합 데이터셋의 품질을 검증한다."""
    
    total = len(dataset)
    
    # 1. 도메인 커버리지
    domain_count = sum(1 for ex in dataset if is_domain_adjacent(ex))
    print(f"도메인 관련 비율: {domain_count}/{total} ({domain_count/total:.1%})")
    
    # 2. 한국어 비율
    def korean_ratio(text):
        total_chars = sum(1 for c in text if not c.isspace())
        if total_chars == 0: return 0
        return sum(1 for c in text if '가' <= c <= '힣') / total_chars
    
    kr_ratios = [korean_ratio(ex.get('instruction', '') + ex.get('output', '')) for ex in dataset]
    low_korean = sum(1 for r in kr_ratios if r < 0.3)
    print(f"한국어 비율 낮은 샘플: {low_korean}/{total} ({low_korean/total:.1%})")
    
    # 3. 중복 검사
    instructions = [ex.get('instruction', '').strip() for ex in dataset]
    unique = len(set(instructions))
    print(f"고유 instruction: {unique}/{total} (중복: {total - unique})")
    
    # 4. 길이 분포
    lengths = [len(ex.get('output', '')) for ex in dataset]
    print(f"output 길이: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
    
    # 5. 소스 분포
    if 'source_doc' in dataset.column_names:
        from collections import Counter
        sources = Counter(ex.get('source_doc', 'unknown') for ex in dataset)
        print(f"소스 분포: {dict(sources.most_common(10))}")

validate_mixed_dataset(final_train)
```

### 혼합 비율 자동 조정

```python
def auto_balance(general_ds, domain_ds, target_domain_ratio=0.2, max_total=5000):
    """도메인 비율을 목표에 맞게 자동 조정한다."""
    
    domain_count = len(domain_ds)
    
    if domain_count == 0:
        print("⚠ 도메인 데이터가 없습니다. 일반 데이터만 사용합니다.")
        return general_ds.shuffle(seed=42).select(range(min(max_total, len(general_ds))))
    
    # 도메인 비율에 맞는 일반 데이터 크기 계산
    general_needed = int(domain_count * (1 - target_domain_ratio) / target_domain_ratio)
    general_needed = min(general_needed, max_total - domain_count, len(general_ds))
    
    general_sample = general_ds.shuffle(seed=42).select(range(general_needed))
    
    final = concatenate_datasets([general_sample, domain_ds]).shuffle(seed=42)
    actual_ratio = domain_count / len(final)
    
    print(f"혼합 결과: {len(final)} rows (일반 {len(general_sample)} + 도메인 {domain_count})")
    print(f"도메인 비율: {actual_ratio:.1%} (목표: {target_domain_ratio:.1%})")
    
    return final
```

---

## 요약: 전략 선택 가이드

| 상황 | SFT 전략 | Embedding 전략 |
|------|----------|---------------|
| **시간 부족** (Lab 시간만) | A: 3단계 혼합 (kullm-v2 + 도메인 필터 + 합성) | D: ko-triplet 서브셋 + 도메인 트리플릿 |
| **OpenAI API 가용** | C: 도메인 리라이트 + 합성 혼합 | D + 합성 확장 |
| **품질 최우선** | B: 고품질 소량 (KoAlpaca + LIMA) + 도메인 집중 | E: NLI 사전학습 → 도메인 미세조정 |
| **규모 최우선** | A: open-korean-instructions 10K+ + 도메인 | F: KorQuAD hard negative 마이닝 + 도메인 |
| **API 없음** | A: kullm-v2 + 도메인 키워드 필터링만 | G: korquad_negative_samples 직접 활용 |
