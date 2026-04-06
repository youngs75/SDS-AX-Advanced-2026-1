# vLLM 이론과 활용 마스터 가이드

> **대상 독자**: Day-01~06을 수강한 백엔드 개발자
> **vLLM 기준 버전**: v0.19.0
> **강의 시간**: 8시간 (이론:실습 = 50:50)
> **관련 실습 프로젝트**: [04_project_benchmark.md](04_project_benchmark.md)

---

## 목차

- [Part 1: vLLM 서빙 기초](#part-1-vllm-서빙-기초) (3h — 이론 1.5h + 실습 1.5h)
- [Part 2: Agent 시스템 통합](#part-2-agent-시스템-통합) (1h — 이론 30m + 데모 30m)
- [Part 3: 성능 최적화](#part-3-성능-최적화) (2h — 이론 1h + 실습 1h)
- [Part 4: 프레임워크 비교](#part-4-프레임워크-비교) (참고 자료)
- [Part 5: KV Cache 심화 + LMCache](#part-5-kv-cache-심화--lmcache) (심화 보충 자료)
- [부록](#부록)

---

## Part 1: vLLM 서빙 기초

> **핵심 메시지**: vLLM은 OS의 가상 메모리 관리 원리를 GPU KV Cache에 적용하여 메모리 낭비와 배칭 비효율을 동시에 해결한다.

---

### 1.1 LLM 추론 서빙의 도전 과제

#### 1.1.1 왜 LLM 서빙은 특별한가

백엔드 서버를 운영할 때 일반적인 REST API 요청은 처리 시간이 예측 가능하고 메모리 사용량이 고정되어 있다. 그러나 LLM 추론은 근본적으로 다른 특성을 가진다.

**LLM 추론의 Auto-Regressive 특성:**

```
입력: "삼성 SDS의 주요 사업 분야는"
출력 토큰 1: "클라우드"     ← 생성 후 다음 토큰 계산에 사용
출력 토큰 2: "및"           ← 생성 후 다음 토큰 계산에 사용
출력 토큰 3: "IT"           ← 생성 후 다음 토큰 계산에 사용
...
출력 토큰 N: <EOS>          ← 언제 끝날지 모름
```

토큰을 한 개씩 순차적으로 생성하기 때문에 **출력 길이를 사전에 알 수 없다**. 이것이 모든 문제의 근원이다.

#### 1.1.2 호텔 방 배정 문제 (핵심 비유)

> **비유: 체크아웃 시간을 모르는 호텔**
>
> 투숙객이 예약할 때 "며칠 묵을지 모르겠어요"라고 말한다면?
> 전통적인 호텔(기존 LLM 서빙)은 **최대 숙박 기간(max_sequence_length) 만큼의 방을 미리 통째로 예약**해 둔다.
> 투숙객이 하루만 묵어도 그 방은 끝까지 비어있다.
>
> 결과: GPU 메모리의 **60~80%가 실제로 사용되지 않는 빈 공간**으로 낭비된다.

#### 1.1.3 전통적 서빙 방식의 세 가지 문제

| 문제 | 설명 | 백엔드 비유 |
|------|------|------------|
| **내부 단편화 (Internal Fragmentation)** | max_len만큼 메모리를 선점해 두지만 실제로는 일부만 사용 | DB 테이블에서 VARCHAR(1000) 컬럼에 실제로는 10글자만 저장하는 것 |
| **외부 단편화 (External Fragmentation)** | 짧은 요청들이 처리되며 남긴 메모리 조각들이 흩어져 연속된 큰 공간을 확보할 수 없음 | 힙 메모리의 메모리 단편화와 동일 |
| **배칭 비효율** | 서로 다른 길이의 요청들을 묶으면 짧은 것은 기다리고, 긴 것은 전체 배치를 지연시킴 | Thread Pool에서 하나의 오래 걸리는 작업이 전체를 블로킹하는 상황 |

#### 1.1.4 실측 데이터로 보는 문제의 심각성

기존 HuggingFace Transformers 기반 서빙:
- KV Cache 메모리 효율: **20~40%** (60~80% 낭비)
- GPU 실제 연산 활용률 (MFU): **30~50%** (나머지는 메모리 대기)
- 배칭 가능 요청 수: 메모리 제약으로 극히 제한

✅ **출처**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.

#### 1.1.5 핵심 메시지

> **vLLM이 해결하는 것**: KV Cache 메모리 낭비 + 배칭 비효율을 **동시에** 해결한다.
> 해법의 영감은 GPU 전문 지식이 아니라 **OS의 가상 메모리 관리**에서 왔다.

---

### 1.2 PagedAttention 심층 분석

#### 1.2.1 OS 가상 메모리와의 완벽한 대응

백엔드 개발자라면 OS의 가상 메모리(Virtual Memory) 개념을 알고 있다. PagedAttention은 이 개념을 KV Cache 관리에 그대로 적용한다.

| OS 가상 메모리 | PagedAttention | 설명 |
|---------------|----------------|------|
| **Virtual Page** | **Logical KV Block** | 프로세스가 보는 연속된 메모리 주소 / 시퀀스가 보는 연속된 KV 저장 공간 |
| **Physical Frame** | **Physical KV Block** | 실제 RAM의 물리적 페이지 / 실제 GPU HBM의 물리적 KV 저장 블록 |
| **Page Table** | **Block Table** | Virtual→Physical 주소 변환 테이블 / Logical→Physical KV 블록 매핑 테이블 |
| **Demand Paging** | **On-demand Block Allocation** | 실제 접근 시점에 물리 페이지 할당 / 실제 토큰 생성 시점에 KV 블록 할당 |
| **Copy-on-Write** | **CoW for Parallel Sampling** | fork() 후 쓰기 발생 시에만 복사 / Beam Search/Parallel Sampling 시 분기 시점에만 복사 |

#### 1.2.2 KV Block 크기와 구조

vLLM에서 KV Block의 크기는 **`block_size`** 파라미터로 제어된다.

```
유효한 block_size 값: [1, 8, 16, 32, 64, 128] 토큰/블록
CUDA 커널 최적화 기준 기본값: 16 (FlashAttention 기준)
```

✅ **출처**: vLLM 소스코드 `vllm/config.py` - `BlockSize` 검증 로직

**블록 구조 예시 (block_size=4 기준, 교육용 단순화):**

```
시퀀스: "삼성 SDS vLLM 강의 수강생 여러분 안녕하세요"
        [토큰1][토큰2][토큰3][토큰4] | [토큰5][토큰6][토큰7][토큰8] | ...
        ←── Logical Block 0 ──────→   ←── Logical Block 1 ──────→

Block Table (이 시퀀스의):
  Logical Block 0 → Physical Block #47  (GPU 메모리 어딘가)
  Logical Block 1 → Physical Block #12  (GPU 메모리 어딘가, 연속일 필요 없음)
  Logical Block 2 → Physical Block #89  (GPU 메모리 어딘가)
```

#### 1.2.3 Block Table 매핑 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU HBM (Physical)                      │
│                                                             │
│  Phys#0  [시퀀스B 블록0]   Phys#1  [시퀀스A 블록1]          │
│  Phys#2  [FREE          ]   Phys#3  [시퀀스C 블록0]          │
│  Phys#4  [시퀀스A 블록0]   Phys#5  [FREE          ]          │
│  Phys#6  [시퀀스B 블록1]   Phys#7  [시퀀스C 블록1]          │
│  Phys#8  [FREE          ]   Phys#9  [시퀀스A 블록2]          │
└─────────────────────────────────────────────────────────────┘

Block Tables:
  시퀀스A: [Logical 0→Phys#4] [Logical 1→Phys#1] [Logical 2→Phys#9]
  시퀀스B: [Logical 0→Phys#0] [Logical 1→Phys#6]
  시퀀스C: [Logical 0→Phys#3] [Logical 1→Phys#7]

핵심: 물리적으로 비연속(Non-contiguous)이어도 논리적으로 연속처럼 동작
```

#### 1.2.4 Fused CUDA Kernel

기존 Attention 연산은 세 단계가 분리되어 실행된다:
1. Q×K 행렬 곱셈
2. Softmax
3. Softmax 결과 × V 집계

PagedAttention은 이 세 단계를 **하나의 CUDA 커널**로 융합(Fused)하여:
- GPU 메모리 왕복 횟수 감소
- Block Table 조회를 커널 내에서 직접 처리
- 비연속 물리 블록 접근을 커널이 직접 해결

#### 1.2.5 성능 결과

✅ **출처**: Kwon et al., SOSP 2023 (실측 벤치마크)

| 비교 대상 | 처리량 (tokens/sec) | 상대 성능 |
|-----------|--------------------|---------:|
| HuggingFace Transformers | 기준 | 1x |
| HuggingFace TGI (당시) | 약 2x | 2x |
| **vLLM (PagedAttention)** | **최대 24x** | **24x** |

> 핵심 이유: 메모리 효율이 올라가면 더 많은 요청을 동시에 배칭할 수 있고, GPU는 쉬지 않고 계속 연산한다.

#### 1.2.6 핵심 메시지

> PagedAttention은 OS의 가상 메모리 관리 원리를 그대로 KV Cache에 적용한 것이다.
> **OS Page Table = vLLM Block Table**: 물리 메모리는 비연속이어도 논리적으로는 연속처럼 보인다.
> 덕분에 "호텔 방 통째 선점" 없이 **실제 사용하는 만큼만** GPU 메모리를 점유한다.

---

### 1.3 KV Cache 관리 및 양자화

#### 1.3.1 KV Cache란 무엇인가

LLM의 Self-Attention 연산에서 각 토큰은 이전 모든 토큰의 Key와 Value 벡터를 참조해야 한다.

> **비유: 회의록**
>
> 회의가 길어질수록 앞 발언을 다시 전체 낭독하는 대신, **회의록(KV Cache)**을 참조한다.
> 새 발언이 추가될 때마다 회의록에 기록하고, 다음 발언자는 회의록 전체를 읽는다.
>
> KV Cache가 없다면: 100번째 토큰을 생성하려면 앞 99개 토큰을 다시 전부 계산해야 한다.
> KV Cache가 있다면: 앞 99개 토큰의 K, V 텐서를 메모리에서 읽기만 하면 된다.

**메모리 사용량 공식 (단순화):**

```
KV Cache 크기 = 2 × num_layers × num_heads × head_dim × seq_len × precision_bytes

예: Qwen3-7B (32 layers, 32 heads, head_dim=128, seq_len=8192, FP16)
  = 2 × 32 × 32 × 128 × 8192 × 2 bytes
  = 약 8.6 GB (모델 가중치 별도)
```

#### 1.3.2 정밀도 조합별 유효성 (검증된 7가지 케이스)

✅ **출처**: CaseStudy_검증결과.md 강의 내용 + vLLM v0.19.0 소스코드 검증

| # | 모델 가중치 | KV Cache | 유효성 | 설명 |
|---|------------|---------|--------|------|
| 1 | **FP16** | **FP16** | ✅ 유효 (기본값) | 모든 연산이 FP16. 기준 케이스 |
| 2 | **FP16** | **FP8** | ✅ 유효 (권장 최적화) | KV 저장 시 FP8로 압축, 읽을 때 FP16 복원. 메모리 절약, 미세한 정확도 손실 |
| 3 | **FP8** | **FP16** | ❌ 의미 없음 | 모델보다 KV가 더 정밀 → 낭비. 이 케이스는 피해야 함 |
| 4 | **FP8** | **FP8** | ✅ 유효 | 모델과 KV 모두 FP8. 대규모 배포에 효율적 |
| 5 | **FP4** | **FP16** | ✅ 유효 | FP4 양자화 모델 + FP16 KV. KV 정밀도 유지 |
| 6 | **FP4** | **FP8** | ✅ 유효 | FP4 양자화 모델 + FP8 KV. 균형잡힌 절충 |
| 7 | **FP4** | **FP4** | ❌ **vLLM 미지원** | v0.19.0 기준 WIP 상태. `KV_CACHE_QUANT_ALGOS = ["FP8"]`만 존재 |

**CaseStudy_검증결과.md 검증 결과:**

> "KV 캐시 4비트는 vLLM 지원 없고" → ✅ **v0.19.0에서 정확한 사실**
>
> vLLM 소스코드 `vllm/attention/backends/utils.py` 및 `vllm/_custom_ops.py`에서
> `KV_CACHE_QUANT_ALGOS`는 현재 `["FP8"]`만 포함. FP4 KV Cache는 WIP(진행 중).

#### 1.3.3 FP16/FP8 조합 상세 동작 (케이스 2)

```
[추론 흐름 - 모델 FP16, KV Cache FP8]

① Attention 연산 → 16비트 배열 생성
② 16비트 → 8비트 변환 (quantize, 연산량 적음)
③ 8비트 KV Cache에 저장
④ Cache Hit 발생 (다음 토큰 생성 시)
⑤ 8비트 → 16비트 복원 (dequantize, 오차 발생)
⑥ 복원된 16비트로 Attention 계산 → 출력 토큰

[오차 완화 방법]
- top_k 감소: 선택 가능 토큰 수를 줄여 오차 영향 최소화
- top_p 조정: 누적 확률 범위 조정
- temperature 낮춤: 확률 분포를 더 뾰족하게
- auto k_scale/v_scale: vLLM이 자동으로 스케일 팩터 계산 (권장)
```

#### 1.3.4 주요 KV Cache 관련 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--gpu-memory-utilization` | `0.9` | GPU 메모리 중 KV Cache에 할당할 비율 (0~1). 높을수록 더 많은 병렬 처리 가능 |
| `--kv-cache-dtype` | `auto` | `auto` (모델과 동일), `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| `--cpu-offload-gb` | `0` | KV Cache를 CPU RAM으로 오프로드할 GB. VRAM 부족 시 사용 |
| `--kv-cache-memory-bytes` | `None` | KV Cache에 사용할 바이트 수를 직접 지정 (v0.19.0+) |
| `--block-size` | `16` | KV Block 크기 (토큰 수). 유효값: 1, 8, 16, 32, 64, 128 |

> 💡 심화 내용은 [Part 5: KV Cache 심화 + LMCache](#part-5-kv-cache-심화--lmcache) 참조

#### 1.3.5 핵심 메시지

> KV Cache는 "회의록"이다. 메모리를 아끼려면 회의록 용지를 더 작게(FP8) 쓰면 된다.
> 단, 작은 글씨(저정밀도)는 오독(오차)을 유발하므로 top_k/top_p/temperature로 보완한다.
> **FP4 KV Cache는 v0.19.0에서 아직 지원하지 않는다.** 실무에서 혼동하지 말 것.

---

### 1.4 Continuous Batching

#### 1.4.1 세 가지 배칭 전략 비교

> **배경**: GPU는 병렬 연산에 최적화되어 있어 요청을 한 번에 묶어서 처리(배칭)하면 처리량이 크게 오른다.
> 문제는 **각 요청의 출력 길이가 다르다**는 것이다.

| 전략 | 백엔드 비유 | 동작 방식 | 단점 |
|------|-----------|-----------|------|
| **Static Batching** | 단체 식사 (모두 앉아야 출발) | 배치 크기 N을 고정. N개 모두 완료될 때까지 대기 | 빠른 요청이 느린 요청을 기다림. GPU 낭비 심각 |
| **Dynamic Batching** | 시간대별 단체 입장 (10분마다 모아서 처리) | 일정 시간 또는 일정 수를 모아서 배치 구성 | 여전히 배치 내 최장 요청까지 대기 |
| **Continuous Batching** | **이벤트 루프** (완료 즉시 새 요청 투입) | 토큰 생성 반복(iteration)마다 스케줄링 결정 | 구현 복잡도 높음 (vLLM이 해결) |

#### 1.4.2 Continuous Batching 상세 동작

> **비유: Node.js 이벤트 루프**
>
> `async/await`로 비동기 처리하면 하나의 함수가 완료되기를 기다리는 동안
> 다른 함수가 실행될 수 있다. Continuous Batching도 동일하다.
>
> 토큰 하나가 생성될 때마다 스케줄러가 개입해서:
> "완료된 시퀀스가 있나? → 빼내고 대기 중인 새 요청 넣자"

```
Iteration 1: [요청A(tok3)] [요청B(tok1)] [요청C(tok5)]
Iteration 2: [요청A(tok4)] [요청B(tok2)] [요청C(tok6→완료!)]
                                         ↓ C 제거, D 투입
Iteration 3: [요청A(tok5)] [요청B(tok3)] [요청D(tok1←새로운요청)]
Iteration 4: [요청A(tok6→완료!)] [요청B(tok4)] [요청D(tok2)]
              ↓ A 제거, E 투입
Iteration 5: [요청E(tok1←새로운요청)] [요청B(tok5)] [요청D(tok3)]
```

✅ **출처**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022. (Iteration-level scheduling 개념 최초 제안)

#### 1.4.3 스케줄링 정책

| 파라미터 | 옵션 | 설명 |
|---------|------|------|
| `--scheduling-policy` | `fcfs` (기본) | First-Come First-Served. 먼저 온 요청 먼저 처리 |
| `--scheduling-policy` | `priority` | 요청별 우선순위 지정 가능 (OpenAI API `priority` 필드 활용) |

**선점(Preemption):** KV Cache가 가득 찰 경우 LRU(Least Recently Used) 방식으로 시퀀스를 evict. evict된 시퀀스는 CPU로 swap하거나 재계산(recompute).

#### 1.4.4 Chunked Prefill (v0.19.0 V1 기본 활성화)

**문제**: 매우 긴 프롬프트(Prefill) 처리 중에는 Decode 요청이 완전히 블로킹된다.

```
기존 (Prefill/Decode 분리):
  Prefill [────────────────────────────────]  ← 10K 토큰 프롬프트 처리 중
  Decode  [                                ]  ← 완전 대기 (TTFT 폭발)

Chunked Prefill (V1 기본):
  Chunk 1 [────────]  Decode [▪▪▪]  ← 프리필 청크 + 디코드 인터리빙
  Chunk 2 [────────]  Decode [▪▪▪]  ← 다음 청크 + 디코드
  Chunk 3 [────────]  Decode [▪▪▪]  ← ...
```

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--max-num-batched-tokens` | 모델별 상이 | 한 iteration에서 처리할 최대 토큰 수. Chunked Prefill의 청크 크기 상한 |
| `--max-num-seqs` | `256` | 동시 처리 시퀀스 최대 수 |
| `--max-num-partial-prefills` | `1` | 동시에 청크 처리할 수 있는 프리필 수 |

#### 1.4.5 핵심 메시지

> Continuous Batching은 Node.js 이벤트 루프와 같다.
> 요청 완료를 기다리지 않고, **매 토큰 생성 시점마다** 스케줄러가 새 요청을 투입한다.
> GPU는 거의 쉬지 않고 연산한다. Chunked Prefill은 여기에 더해 긴 프롬프트도 디코드와 인터리빙한다.

---

### 1.5 OpenAI API 호환 서버

#### 1.5.1 서버 기동 명령

**기본 기동 (Docker):**

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8080:8080 \
  vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-7B \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

**분산 서빙 (Tensor Parallelism, Ray 활용):**

```bash
# master 노드에서
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.95
```

#### 1.5.2 v0.19.0 전체 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/v1/chat/completions` | POST | OpenAI 호환 채팅 완성 (주력 엔드포인트) |
| `/v1/completions` | POST | OpenAI 호환 텍스트 완성 (레거시) |
| `/v1/embeddings` | POST | 텍스트 임베딩 생성 |
| `/v1/responses` | POST | **v0.19.0 신규** - OpenAI Responses API 호환 |
| `/v1/messages` | POST | **v0.19.0 신규** - Anthropic API 호환 (`--served-model-name` 활용) |
| `/v1/models` | GET | 로드된 모델 목록 조회 |
| `/health` | GET | 서버 헬스체크 (ready 상태 확인) |
| `/metrics` | GET | Prometheus 형식 메트릭 (처리량, 지연시간, 큐 길이 등) |
| `/tokenize` | POST | 텍스트를 토큰으로 변환 (토큰 수 사전 확인용) |
| `/v1/load_lora_adapter` | POST | 런타임 LoRA 어댑터 로드 |
| `/v1/unload_lora_adapter` | POST | 런타임 LoRA 어댑터 언로드 |
| `/reset_prefix_cache` | POST | Prefix Cache 초기화 |
| `/v1/score` | POST | 교차 인코더 스코어링 (Reranker 용도) |
| `/pooling` | POST | 커스텀 풀링 모드 |

**gRPC 서버 (v0.19.0 신규):**

```bash
vllm serve Qwen/Qwen3-7B --grpc --grpc-port 8081
```

#### 1.5.3 Python 클라이언트 예제 (OpenAI SDK)

```python
from openai import OpenAI

# vLLM 서버를 OpenAI 클라이언트로 호출
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # vLLM은 기본적으로 API 키 불필요
)

# 일반 요청
response = client.chat.completions.create(
    model="Qwen/Qwen3-7B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "vLLM의 PagedAttention을 한 문장으로 설명해줘."},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)

# 스트리밍 요청
stream = client.chat.completions.create(
    model="Qwen/Qwen3-7B",
    messages=[{"role": "user", "content": "Continuous Batching을 설명해줘."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### 1.5.4 핵심 메시지

> vLLM 서버는 **OpenAI API와 완전 호환**된다. 기존 OpenAI SDK 코드에서 `base_url`만 바꾸면 된다.
> v0.19.0에서는 Anthropic API와 gRPC도 추가 지원. `/metrics` 엔드포인트로 Prometheus 연동 가능.

---

### 1.6 vLLM 최신 기능 (v0.19.0)

#### 1.6.1 Speculative Decoding (추측적 디코딩)

**문제**: LLM 디코딩은 본질적으로 순차적이다. 한 번에 토큰 하나씩만 생성한다.

> **비유: 초안 작성자 + 편집장**
>
> 편집장(메인 모델, 7B)이 혼자 글을 쓰는 대신:
> 1. 초안 작성자(드래프트 모델, 0.5B)가 5개 토큰을 빠르게 작성
> 2. 편집장이 5개를 한 번에 검토 (병렬 처리로 매우 빠름)
> 3. 동의하면 5개 모두 채택, 틀린 것부터는 버리고 편집장이 직접 작성
>
> 결과: 초안이 맞으면 5배 빠름. 틀려도 손해 없음 (출력 품질 동일 보장).

**지원 방법 (v0.19.0 기준):**

| 방법 | 설명 | 권장 케이스 |
|------|------|------------|
| `ngram` | 입력 텍스트 내 n-gram 패턴으로 드래프트 | 문서 요약, 코드 완성 |
| `eagle` | EAGLE 드래프트 모델 활용 | 범용 (EAGLE-LLaMA 등) |
| `eagle3` | EAGLE3 (개선된 버전, v0.19.0 신규 지원) | 더 높은 수락률 |
| `medusa` | Medusa 헤드 활용 (다중 후보 동시 예측) | 범용 |
| `mtp` | Multi-Token Prediction | DeepSeek V3 계열 |

**설정 예시:**

```bash
vllm serve Qwen/Qwen3-7B \
  --speculative-model Qwen/Qwen3-0.5B \
  --num-speculative-tokens 5 \
  --speculative-disable-by-batch-size 4
```

**v0.19.0 신규**: Zero-Bubble Async Scheduling으로 드래프트/검증 단계 오버랩. 지연 시간 추가 감소.

#### 1.6.2 Structured Output (구조화된 출력)

LLM 출력을 JSON Schema 또는 정규식으로 강제. 파싱 오류 없는 API 응답 구현에 필수.

**지원 백엔드:**

| 백엔드 | 특징 | 기본값 여부 |
|--------|------|-----------|
| `xgrammar` | 빠른 GPU 가속 문법 처리 | ✅ 기본값 (v0.19.0) |
| `guidance` | Microsoft Guidance 라이브러리 | 선택적 |
| `outlines` | Outlines 라이브러리 | 선택적 |
| `lm-format-enforcer` | LM-Format-Enforcer | 선택적 |

**Python SDK 예시:**

```python
from openai import OpenAI
from pydantic import BaseModel

class ServerInfo(BaseModel):
    model: str
    throughput_tokens_per_sec: int
    gpu_memory_util: float

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="Qwen/Qwen3-7B",
    messages=[{"role": "user", "content": "vLLM 서버 상태를 JSON으로 알려줘."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ServerInfo",
            "schema": ServerInfo.model_json_schema(),
        },
    },
)
import json
data = json.loads(response.choices[0].message.content)
server = ServerInfo(**data)
```

#### 1.6.3 Multi-LoRA Serving (다중 LoRA 어댑터)

하나의 베이스 모델 위에 여러 LoRA 어댑터를 **동시에** 로드하고 요청별로 선택적 적용.

> **비유**: 하나의 서버 인스턴스에서 여러 고객사(tenant)별 커스터마이징을 처리하는 멀티테넌트 구조.

**설정:**

```bash
vllm serve Qwen/Qwen3-7B \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --lora-modules \
    customer-a=/adapters/customer_a \
    customer-b=/adapters/customer_b
```

**런타임 로드/언로드 (v0.19.0):**

```python
import requests

# 런타임에 어댑터 로드
requests.post("http://localhost:8080/v1/load_lora_adapter", json={
    "lora_name": "customer-c",
    "lora_path": "/adapters/customer_c",
})

# 런타임에 어댑터 언로드
requests.post("http://localhost:8080/v1/unload_lora_adapter", json={
    "lora_name": "customer-b",
})
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--max-loras` | `1` | 동시 GPU에 로드할 최대 LoRA 수 |
| `--max-lora-rank` | `16` | LoRA rank 최대값. 훈련된 rank보다 크거나 같아야 함 (유효값: 8~512) |
| `--max-cpu-loras` | `None` | CPU에 캐시할 LoRA 수 (GPU 용량 초과 시 swap) |

#### 1.6.4 Automatic Prefix Caching (자동 접두사 캐싱)

**v0.19.0에서 기본 활성화 (V1 엔진)**

동일한 시스템 프롬프트나 Context를 가진 요청들이 반복될 때 KV Cache를 재사용.

**동작 원리:**

```
첫 번째 요청 (10,000 토큰 시스템 프롬프트 + 질문):
  시스템 프롬프트 10,000 토큰 → KV 계산 (TTFT: 4.3초)
  결과를 해시(sha256) 키로 캐시에 저장

두 번째 요청 (동일 시스템 프롬프트 + 다른 질문):
  시스템 프롬프트 10,000 토큰 → 해시 계산 → Cache HIT!
  KV 재계산 없이 즉시 사용 (TTFT: 0.6초)
```

✅ **실측 효과**: 10,000 토큰 프롬프트 기준 TTFT 4.3초 → 0.6초 (약 **7배 단축**)

**해싱 방식:**

| 방식 | 설명 |
|------|------|
| `sha256` | 기본값. SHA-256 해시로 블록 식별 |
| `sha256_cbor` | CBOR 인코딩 + SHA-256. 약간 더 안전한 직렬화 |

**주의사항 - 로드 밸런서 설정:**

```
문제: Round-Robin 방식 LB는 Prefix Cache를 파괴한다!

요청1 → 서버A (캐시 생성)
요청2 → 서버B (캐시 없음, 재계산)
요청3 → 서버C (캐시 없음, 재계산)

해결책: 동일한 시스템 프롬프트를 가진 요청은
        항상 동일한 vLLM 인스턴스로 라우팅 (Sticky Routing)
        → L7 LB에서 세션 어피니티 또는 해시 기반 라우팅 구성 필요
```

**파라미터:**

```bash
# Prefix Cache 비활성화 (필요한 경우)
vllm serve Qwen/Qwen3-7B --no-enable-prefix-caching

# Prefix Cache 초기화 (API 호출)
curl -X POST http://localhost:8080/reset_prefix_cache
```

> 💡 심화 내용은 [Part 5: KV Cache 심화 + LMCache](#part-5-kv-cache-심화--lmcache) 참조

#### 1.6.5 Tool Calling (도구 호출)

20개 이상의 모델별 파서 내장. 모델이 JSON 형식으로 함수 호출을 출력하면 파싱해서 전달.

**지원 모델 파서 (선택):**

| 파서 | 지원 모델 |
|------|----------|
| `hermes` | Nous-Hermes, OpenHermes 계열 |
| `llama3_json` | LLaMA 3.x 계열 |
| `mistral` | Mistral, Mixtral 계열 |
| `qwen3` | Qwen3 계열 (포함: `reasoning-parser`) |
| `deepseek_v3` | DeepSeek V3, R1 계열 |
| `internlm` | InternLM2 계열 |

```bash
vllm serve Qwen/Qwen3-7B \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3
```

#### 1.6.6 Performance Mode (성능 프로파일)

v0.17+부터 지원. 단일 파라미터로 워크로드별 최적화 프리셋 적용.

| 모드 | 목적 | 내부 조정 |
|------|------|---------|
| `balanced` | 기본값. 지연시간과 처리량 균형 | 기본 설정 유지 |
| `interactivity` | 대화형 서비스. TTFT 최소화 | Chunked Prefill 청크 크기 축소, 큐 대기시간 축소 |
| `throughput` | 배치 처리. 최대 처리량 | Chunked Prefill 비활성화, 배치 크기 최대화 |

```bash
vllm serve Qwen/Qwen3-7B --performance-mode throughput
```

#### 1.6.7 V1 엔진 아키텍처

| 변경 사항 | 버전 | 내용 |
|---------|------|------|
| V0 엔진 완전 제거 | **v0.11** | V1 엔진만 남음. `VLLM_USE_V1=0` 환경변수로 전환 불가 |
| FlashAttention 4 기본 적용 | v0.17 | NVIDIA Hopper(H100) 이상에서 자동 활성화 |
| Async Scheduling 기본 적용 | v0.14 | 스케줄링과 모델 실행 비동기화. 처리량 향상 |
| Chunked Prefill 기본 활성화 | v0.19 | V1에서 기본으로 활성화 |

#### 1.6.8 핵심 메시지

> v0.19.0의 주요 신규 기능: **Responses API**, **Anthropic API 호환**, **gRPC**, **EAGLE3**, **Zero-Bubble Async Scheduling**.
> **Automatic Prefix Caching은 이미 기본 활성화**. RAG 시스템이나 시스템 프롬프트가 긴 서비스에서 즉각적인 효과.
> Multi-LoRA는 멀티테넌트 LLM 서비스의 핵심 패턴. 어댑터를 런타임에 동적으로 교체 가능.

---

### 1.7 v0.11 → v0.19 버전 변경 요약

#### 1.7.1 버전별 주요 변경 사항

| 버전 | 주요 변경 | 백엔드 영향 |
|------|---------|------------|
| **v0.11** | V0 엔진 완전 제거. V1 엔진 단일화. Chunked Prefill 안정화 | `VLLM_USE_V1=0` 환경변수 무효화 |
| **v0.12** | Multi-Modal 입력 개선 (이미지/비디오). `--limit-mm-per-prompt` 파라미터 추가 | 멀티모달 서비스에서 주의 |
| **v0.13** | LoRA 런타임 로드/언로드 API 안정화 | `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` 공식화 |
| **v0.14** | Async Scheduling 기본 활성화. CPU 오프로드 개선 | 스케줄링 지연 감소, `--cpu-offload-gb` 안정화 |
| **v0.15** | Prefix Caching V2. sha256_cbor 해싱 추가 | 캐시 히트율 향상 |
| **v0.16** | DeepSeek V3/R1 공식 지원. MTP Speculative Decoding | `deepseek_v3` 파서, `mtp` spec decode |
| **v0.17** | FlashAttention 4 (H100+). Performance Mode (`--performance-mode`) 추가 | H100 환경에서 자동 활성화 |
| **v0.18** | EAGLE3 Speculative Decoding. Tool Calling 파서 20+ 확장 | Qwen3, DeepSeek 파서 포함 |
| **v0.19** | **Responses API** (`/v1/responses`). **Anthropic API** (`/v1/messages`). **gRPC** (`--grpc`). Zero-Bubble Async Scheduling. `--kv-cache-memory-bytes` 파라미터 | API 호환성 확장. gRPC 클라이언트 연동 가능 |

#### 1.7.2 주요 Breaking Changes (v0.11~v0.19)

| 버전 | Breaking Change | 마이그레이션 |
|------|----------------|------------|
| v0.11 | V0 엔진 제거 | `VLLM_USE_V1=0` 제거, V1 기반으로 전환 |
| v0.14 | `--disable-async-output-proc` 기본값 변경 | 동기 처리가 필요하면 명시적으로 설정 |
| v0.17 | `--performance-mode` 기본값 `balanced` | 기존 throughput 설정 확인 필요 |
| v0.19 | Prefix Caching 기본 활성화 | 결정론적 출력이 필요하면 `--no-enable-prefix-caching` |

#### 1.7.3 핵심 메시지

> 이 강의에서 사용하는 Docker 이미지는 `vllm/vllm-openai:v0.19.0`.
> 강의 자료의 모든 파라미터와 API 엔드포인트는 이 버전 기준.
> 팀 내 기존 vLLM 서버가 v0.11 이하라면 V0 제거 Breaking Change 먼저 확인.

---

### 1.8 Part 1 실습 가이드

#### 실습 1A: vLLM 기본 서빙 (45분)

**목표**: Docker로 vLLM 서버를 기동하고 OpenAI API 호환 방식으로 호출한다.

**단계별 진행:**

**Step 1: Docker 이미지 확인 및 서버 기동 (10분)**

```bash
# GPU 확인
nvidia-smi

# vLLM 서버 기동 (Qwen3-0.6B - 빠른 실습용)
docker run --gpus all \
  --name vllm-lab \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8080:8080 \
  vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# 헬스체크 (서버 준비 완료까지 대기)
curl http://localhost:8080/health
```

**Step 2: API 호출 - curl (5분)**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "PagedAttention을 한 문장으로 설명해줘."}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**Step 3: Python OpenAI SDK 호출 (10분)**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

# 일반 호출
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "당신은 vLLM 전문가입니다."},
        {"role": "user", "content": "Continuous Batching이 왜 중요한지 설명해줘."},
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n사용 토큰: {response.usage}")
```

**Step 4: 스트리밍 호출 (10분)**

```python
# 스트리밍 - 토큰이 생성되는 즉시 출력
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "vLLM의 장점 5가지를 설명해줘."}],
    stream=True,
    max_tokens=500,
)

print("스트리밍 출력:")
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()  # 줄바꿈
```

**Step 5: 메트릭 확인 (10분)**

```bash
# Prometheus 메트릭 확인
curl http://localhost:8080/metrics | grep vllm

# 주요 메트릭:
# vllm:num_requests_running      ← 현재 실행 중인 요청 수
# vllm:num_requests_waiting      ← 대기 중인 요청 수
# vllm:kv_cache_usage_perc       ← KV Cache 사용률
# vllm:num_preemptions_total     ← KV Cache 선점 횟수 (0이 이상적)
# vllm:request_success_total     ← 성공한 요청 수
```

---

#### 실습 1B: 오프라인 추론 (45분)

**목표**: vLLM의 `LLM` 클래스를 직접 사용하여 배치 추론을 수행한다.

> **오프라인 추론 vs 온라인 서빙**:
> - 온라인 서빙: HTTP 서버로 실시간 요청 처리 (`vllm serve`)
> - 오프라인 추론: Python 코드에서 직접 `LLM` 객체 생성, 배치 처리 (서버 불필요)

**Step 1: 기본 LLM 클래스 사용 (15분)**

```python
from vllm import LLM, SamplingParams
from loguru import logger

# 모델 로드 (GPU 직접 사용)
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=4096,
    gpu_memory_utilization=0.8,  # 다른 프로세스와 공유할 경우 낮춤
)

# SamplingParams 설정
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    stop=["</s>", "<|endoftext|>"],
)

# 단일 요청
outputs = llm.generate(
    ["vLLM의 PagedAttention이 무엇인지 설명해줘."],
    sampling_params,
)

for output in outputs:
    logger.info(f"입력: {output.prompt}")
    logger.info(f"출력: {output.outputs[0].text}")
    logger.info(f"토큰 수: {len(output.outputs[0].token_ids)}")
```

**Step 2: 배치 추론 (15분)**

```python
# 여러 요청을 한 번에 처리 (Continuous Batching 자동 적용)
prompts = [
    "PagedAttention이란 무엇인가?",
    "Continuous Batching의 장점을 설명해줘.",
    "vLLM v0.19.0의 주요 신기능은?",
    "KV Cache 양자화의 장단점은?",
    "Speculative Decoding이 왜 빠른가?",
]

sampling_params = SamplingParams(temperature=0.3, max_tokens=200)

import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
logger.info(f"총 {len(prompts)}개 요청 처리 완료")
logger.info(f"처리 시간: {elapsed:.2f}초")
logger.info(f"총 생성 토큰: {total_tokens}")
logger.info(f"처리량: {total_tokens/elapsed:.1f} tokens/sec")

for i, output in enumerate(outputs):
    print(f"\n[Q{i+1}] {output.prompt}")
    print(f"[A{i+1}] {output.outputs[0].text}")
```

**Step 3: Chat 형식 (15분)**

```python
# llm.chat() - 메시지 형식 직접 지원
messages_batch = [
    [
        {"role": "system", "content": "당신은 vLLM 전문가입니다. 간결하게 답변하세요."},
        {"role": "user", "content": "PagedAttention과 OS 가상 메모리의 공통점은?"},
    ],
    [
        {"role": "user", "content": "KV Cache FP8 양자화의 장단점은?"},
    ],
]

sampling_params = SamplingParams(temperature=0.5, max_tokens=300)
outputs = llm.chat(messages_batch, sampling_params)

for i, output in enumerate(outputs):
    print(f"\n[응답 {i+1}]")
    print(output.outputs[0].text)
```

---

#### 실습 완료 후 확인 사항

수강생이 스스로 체크해야 할 항목:

| 항목 | 확인 방법 |
|------|---------|
| ✅ vLLM 서버가 `docker ps`에서 실행 중 | `docker ps | grep vllm` |
| ✅ `/health` 엔드포인트가 200 응답 | `curl http://localhost:8080/health` |
| ✅ `/v1/chat/completions` 호출 성공 | Python 또는 curl로 응답 수신 |
| ✅ 스트리밍 출력이 토큰 단위로 나옴 | 터미널에서 글자가 하나씩 출력 |
| ✅ `/metrics`에서 `vllm:kv_cache_usage_perc` 확인 | `curl http://localhost:8080/metrics` |
| ✅ 오프라인 추론으로 5개 배치 처리 | 처리량(tokens/sec) 수치 확인 |

---

### 1.9 참고 자료

| 자료 | 링크 |
|------|------|
| vLLM 공식 GitHub | https://github.com/vllm-project/vllm |
| vLLM 공식 문서 | https://docs.vllm.ai |
| PagedAttention 논문 | Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 |
| Orca 논문 (Continuous Batching 원형) | Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022 |
| EAGLE3 논문 | Li et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test," 2025 |
| vLLM v0.19.0 Release Notes | https://github.com/vllm-project/vllm/releases/tag/v0.19.0 |
| vLLM v0.19.0 Docker Image | `docker pull vllm/vllm-openai:v0.19.0` |

---

## Part 2: Agent 시스템 통합

### 2.1 왜 vLLM + Agent인가?

#### 학습 목표

- vLLM과 기존 에이전트 시스템을 연동하는 3가지 패턴을 이해한다.
- LangGraph에서 vLLM을 LLM 백엔드로 교체하는 방법을 알 수 있다.
- LLM Gateway(Proxy)의 역할과 필요성을 이해한다.

#### Day-01~04 복습 연결

Day-01~04에서 구축한 LangGraph 에이전트, RAG 파이프라인, MCP 도구들은 모두 **외부 LLM API**(OpenAI, Anthropic 등)를 호출합니다.

프로덕션 환경에서 자체 vLLM 서버를 사용하면:

| 항목 | 외부 API | 자체 vLLM 서버 |
|------|---------|---------------|
| **비용** | 토큰당 과금 | GPU 인프라 비용 (고정) |
| **데이터 통제** | 외부 서버로 데이터 전송 | 사내 네트워크 내 처리 |
| **레이턴시** | 네트워크 왕복 포함 | 내부 네트워크, 최소 지연 |
| **가용성** | 외부 서비스 장애 의존 | 자체 관리 가능 |
| **커스터마이징** | 제한적 (프롬프트만) | 파인튜닝 모델 직접 서빙 |

#### 핵심 메시지

> "Day-06에서 파인튜닝한 모델을 Day-07에서 vLLM으로 서빙하면, 외부 API 의존 없이 **비용 절감 + 데이터 통제 + 커스텀 모델 서비스**가 가능합니다."

---

### 2.2 통합 패턴 3가지

#### 패턴 1: 직접 연동 (가장 간단)

vLLM은 OpenAI 호환 API를 제공하므로, 기존 코드의 `base_url`만 변경하면 됩니다.

```python
from langchain_openai import ChatOpenAI

# 기존: OpenAI API 사용
# llm = ChatOpenAI(model="gpt-4o")

# 변경: 자체 vLLM 서버 사용
llm = ChatOpenAI(
    base_url="http://vllm-server:8000/v1",
    api_key="not-needed",  # vLLM은 기본적으로 API 키 불필요
    model="Qwen/Qwen3-0.6B",  # vLLM에 로드된 모델명
    temperature=0.7,
)

# 이후 코드는 동일하게 사용
response = llm.invoke("안녕하세요, 무엇을 도와드릴까요?")
```

**장점**: 코드 변경 최소, 기존 LangChain/LangGraph 코드 100% 재사용
**단점**: 단일 서버 의존, 장애 시 폴백 없음

#### 패턴 2: LLM Gateway(Proxy) 경유

여러 vLLM 인스턴스나 모델을 관리할 때 게이트웨이를 사용합니다.

```
[Agent] → [LLM Gateway] → [vLLM 서버 1: Qwen3-0.6B]
                         → [vLLM 서버 2: Qwen3-8B]
                         → [OpenAI API (폴백)]
```

**LLM Gateway의 역할**:
- **로드밸런싱**: 여러 vLLM 인스턴스에 요청 분산
- **폴백**: vLLM 서버 장애 시 외부 API로 자동 전환
- **API 키 관리**: 중앙 집중식 인증/인가
- **사용량 추적**: 팀/프로젝트별 토큰 사용량 모니터링
- **요청 라우팅**: 모델/태스크별 적합한 서버로 라우팅

**대표 도구**:
- **LiteLLM**: 100+ LLM 프로바이더를 단일 인터페이스로 통합
- **AI Gateway**: 캐싱, 레이트리밋, 로깅 기능 포함

```python
# LiteLLM Proxy 사용 예시
llm = ChatOpenAI(
    base_url="http://litellm-proxy:4000/v1",
    api_key="sk-litellm-key",
    model="vllm/Qwen3-0.6B",  # LiteLLM이 vLLM 서버로 라우팅
)
```

#### 패턴 3: MCP Tool 래핑

vLLM 서버를 MCP (Model Context Protocol) Tool로 래핑하여, 에이전트가 도구로 사용할 수 있습니다.

```python
# MCP Tool로 vLLM 호출을 래핑
from langchain_core.tools import tool

@tool
def generate_with_vllm(prompt: str, max_tokens: int = 256) -> str:
    """vLLM 서버를 사용하여 텍스트를 생성합니다."""
    from openai import OpenAI

    client = OpenAI(
        base_url="http://vllm-server:8000/v1",
        api_key="not-needed",
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

**활용 시나리오**: 메인 에이전트(GPT-4o)가 특정 도메인 작업을 위해 파인튜닝된 vLLM 모델을 도구로 호출

---

### 2.3 통합 데모

#### 데모: LangGraph + vLLM 연동

Day-01에서 만든 기본 에이전트의 LLM 백엔드를 vLLM으로 교체합니다.

```python
"""LangGraph + vLLM 통합 데모"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


# 1. 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. vLLM 기반 LLM 설정
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="Qwen/Qwen3-0.6B",
    temperature=0.7,
    streaming=True,
)


# 3. 챗봇 노드
def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. 그래프 구성
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

# 5. 실행
if __name__ == "__main__":
    result = app.invoke({
        "messages": [{"role": "user", "content": "vLLM이 무엇인지 설명해주세요."}]
    })
    print(result["messages"][-1].content)
```

#### 데모 실행 순서

1. vLLM 서버 기동 확인 (`/health` 엔드포인트)
2. 위 코드 실행하여 vLLM 기반 챗봇 동작 확인
3. `streaming=True`로 스트리밍 응답 확인
4. 기존 OpenAI API 대비 응답 품질/속도 비교

#### Prefix Caching 효과 체험

동일한 시스템 프롬프트로 여러 요청을 보내면 Prefix Caching 효과를 체험할 수 있습니다:

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
system_prompt = "당신은 Samsung SDS의 AI 전문 컨설턴트입니다. " * 50  # 긴 시스템 프롬프트

# 첫 번째 요청 (캐시 없음)
start = time.time()
r1 = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "vLLM의 장점은?"},
    ],
    max_tokens=100,
)
print(f"첫 번째 요청: {time.time() - start:.3f}초")

# 두 번째 요청 (동일 시스템 프롬프트 → Prefix Cache 적중)
start = time.time()
r2 = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "SGLang과 비교하면?"},
    ],
    max_tokens=100,
)
print(f"두 번째 요청: {time.time() - start:.3f}초 (Prefix Cache 적중)")
```

---

### 2.4 핵심 메시지

> 1. vLLM은 OpenAI 호환 API를 제공하므로, `base_url`만 변경하면 기존 에이전트 코드를 그대로 사용할 수 있다.
> 2. 프로덕션에서는 LLM Gateway를 통해 로드밸런싱, 폴백, 사용량 추적을 관리한다.
> 3. 파인튜닝 모델을 MCP Tool로 래핑하면, 범용 LLM 에이전트가 도메인 전문 모델을 도구로 활용할 수 있다.

---

## Part 3: 성능 최적화

> **학습 목표**
> - Continuous Batching, Streaming, Prefix Caching 등 핵심 성능 파라미터의 원리와 조정 방법을 이해한다.
> - GPU 모니터링 도구를 활용하여 시스템 상태를 실시간으로 파악한다.
> - 분산 서빙의 병렬화 전략을 이해하고 적절한 구성을 선택할 수 있다.

---

### 3.1 Continuous Batching 최적화

#### Continuous Batching 복습

> Part 1에서 다뤘듯이 (섹션 1.3 참조), Continuous Batching은 요청이 완료되는 즉시 새로운 요청을 배치에 편입하는 방식이다. 기존 정적 배치(Static Batching) 대비 GPU 활용률을 크게 향상시키는 vLLM의 핵심 메커니즘이다.

이 동작을 제어하는 핵심 파라미터가 있으며, **목적에 따라 다른 방향으로 튜닝**해야 한다.

백엔드 개발자 비유: Nginx의 `worker_connections`처럼, 동시 처리 용량(max_num_seqs)과 한 번에 처리하는 작업 단위(max_num_batched_tokens)를 조정하는 것과 동일하다.

#### 핵심 파라미터 튜닝 테이블

| 목적 | 파라미터 | 권장 방향 | 비고 |
|------|----------|-----------|------|
| **처리량(Throughput) 최대화** | `--max-num-seqs` | 크게 ↑ (예: 256) | 동시 처리 요청 수 상한 |
| **처리량(Throughput) 최대화** | `--max-num-batched-tokens` | 크게 ↑ (예: 65536) | 배치당 최대 토큰 수 |
| **지연(Latency) 최소화** | `--max-num-seqs` | 작게 ↓ (예: 32~64) | 큐 대기 줄임 |
| **지연(Latency) 최소화** | Chunked Prefill | 활성화 (`--enable-chunked-prefill`) | TTFT 분산 효과 |
| **메모리 효율** | `--gpu-memory-utilization` | 크게 ↑ (최대 0.95) | VRAM 활용률 |
| **메모리 효율** | KV Cache FP8 | `--kv-cache-dtype fp8` | 캐시 메모리 50% 절감 |

#### A100 80GB 기준 권장 베이스라인

```bash
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-8B \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
```

> **A100 80GB 벤치마크 참고치**
> - `--max-num-seqs=256`, `--max-num-batched-tokens=32768` → 처리량 최대화 세팅
> - `--max-num-seqs=32`, `--enable-chunked-prefill` → 저지연 세팅
> - 메모리를 더 확보하려면 `--gpu-memory-utilization 0.95`까지 상향 가능 (OOM 위험 증가)

#### Chunked Prefill이란?

일반적으로 긴 프롬프트(Prefill 단계)가 들어오면 해당 배치가 모든 토큰을 처리할 때까지 다른 요청의 Decode 단계가 블로킹된다. Chunked Prefill은 **긴 Prefill을 청크 단위로 쪼개서** Decode 단계와 인터리빙(interleaving)한다.

- TTFT(Time To First Token) 변동성을 **20~40% 감소**
- 짧은 요청의 응답성 향상
- `--enable-chunked-prefill` 플래그로 활성화 (v0.19.0 기본값: 모델 크기에 따라 자동 결정)

```bash
# Chunked Prefill 명시 활성화
vllm serve Qwen/Qwen3-8B \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192   # 청크 크기 조정
```

---

### 3.2 Streaming 활성화

#### 왜 Streaming인가?

LLM은 토큰을 **순차적으로 생성**한다. Streaming 없이는 모든 토큰이 생성된 후에야 응답을 받으므로, 1000 토큰짜리 답변의 경우 사용자가 수십 초를 기다려야 한다.

Streaming을 사용하면 **첫 번째 토큰부터 즉시 전달**되므로 체감 응답 시간이 극적으로 줄어든다.

#### 핵심 지표 구분

| 지표 | 정의 | Streaming 관련성 |
|------|------|-----------------|
| **TTFT** (Time To First Token) | 요청 후 첫 토큰 도착까지의 시간 | Streaming에서 가장 중요 |
| **TPOT** (Time Per Output Token) | 토큰 하나 생성당 평균 시간 | 생성 속도의 기준 |
| **E2E Latency** | 요청 시작~마지막 토큰까지의 전체 시간 | Non-streaming에서 중요 |

백엔드 개발자 비유:
- TTFT = HTTP 응답의 첫 바이트 도착 시간 (TTFB, Time To First Byte)
- TPOT = 청크 단위 스트리밍 전송 속도

#### SSE 기반 Token Streaming

vLLM의 OpenAI 호환 서버는 **Server-Sent Events(SSE)** 로 스트리밍을 구현한다. `stream=True` 파라미터 하나로 활성화된다.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",  # vLLM은 임의 값 허용
)

# Non-streaming: 전체 응답을 한 번에 수신
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "대한민국 수도는?"}],
    stream=False,
)
print(response.choices[0].message.content)

# Streaming: 토큰이 생성되는 즉시 수신
import time

start = time.time()
first_token_time = None
full_text = ""

stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Python의 GIL에 대해 500자로 설명해줘."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        if first_token_time is None:
            first_token_time = time.time()
            print(f"TTFT: {first_token_time - start:.3f}s")
        full_text += delta
        print(delta, end="", flush=True)

end = time.time()
total_tokens = len(full_text.split())  # 근사치
print(f"\nTPOT: {(end - first_token_time) / total_tokens * 1000:.1f}ms/token")
```

---

### 3.3 Prefix Caching & KV Caching 최적화

#### Automatic Prefix Caching (APC)

> Part 1에서 다뤘듯이 (섹션 1.3 참조), vLLM의 Automatic Prefix Caching은 PagedAttention을 기반으로 동일한 프롬프트 접두사(prefix)를 가진 요청들의 KV Cache를 재사용하는 핵심 최적화 기능이다.

접두사의 해시값을 키로 하여 KV 블록을 캐싱하므로, 두 번째 요청부터는 Prefill 단계를 거의 생략한다.

```
요청 1: [시스템 프롬프트 1000토큰] + [질문 A]
         → KV Cache 계산 후 저장 (hash: abc123)

요청 2: [시스템 프롬프트 1000토큰] + [질문 B]
         → hash: abc123 캐시 HIT → Prefill 생략!
```

**실측 효과 (10K 토큰 프롬프트 기준)**

| 조건 | TTFT |
|------|------|
| Prefix Caching OFF (첫 요청) | 4.3s |
| Prefix Caching ON (캐시 HIT) | **0.6s** |
| 개선율 | **약 7배** |

**활성화 방법** (v0.19.0 기본값: 자동 활성화)

```bash
vllm serve Qwen/Qwen3-8B \
  --enable-prefix-caching    # 명시적 활성화 (기본값이지만 명시 권장)
```

> 💡 심화 운영 가이드는 [Part 5](#part-5-kv-cache-심화--lmcache) 참조

**활용 케이스**

| 케이스 | 이유 |
|--------|------|
| RAG 파이프라인 | 동일한 시스템 프롬프트 + 검색 결과 반복 사용 |
| 멀티턴 챗봇 | 이전 대화 이력이 매 요청마다 prefix로 포함 |
| Agent 루프 | 동일한 tool description이 반복 포함 |
| Few-shot 프롬프트 | 동일한 예시들이 매 요청마다 prefix로 포함 |

**분산 환경 주의사항**: 로드 밸런서가 Round-Robin 방식이면 동일 prefix 요청이 다른 서버로 분산되어 캐시가 무효화된다. **Sticky Routing**(동일 session_id → 동일 서버)이 필요하다.

```nginx
# Nginx sticky session 예시
upstream vllm_backend {
    ip_hash;  # 클라이언트 IP 기반 고정
    server vllm-node1:8000;
    server vllm-node2:8000;
}
```

#### KV Cache 최적화 전략 테이블

> 💡 심화 운영 가이드는 [Part 5](#part-5-kv-cache-심화--lmcache) 참조

| 전략 | 파라미터 | 효과 | 고려사항 |
|------|----------|------|----------|
| **FP8 KV Cache** | `--kv-cache-dtype fp8` | 메모리 **50% 절감** | 미세한 정밀도 손실 (Part 1 섹션 1.2 KV Cache 양자화 참조) |
| **CPU Offload** | `--cpu-offload-gb N` | GPU VRAM 부족 시 용량 확장 | CPU ↔ GPU 전송 레이턴시 발생 |
| **FlexKV** (v0.18+) | 자동 적용 | LRU 기반 스마트 CPU Offload | 자주 쓰이는 블록은 GPU 유지 |

**FP8 KV Cache와 모델 정밀도의 관계** (Part 1 섹션 1.2 KV Cache 양자화 검증 결과 참조):

```
모델 FP16 + KV FP8 조합:
  어텐션 연산(FP16) → KV 저장 시 FP8 다운캐스팅 → 캐시 히트 시 FP8→FP16 업캐스팅
  → 오차 발생 → top_k/top_p 조정으로 완화 가능
```

```bash
# FP8 KV Cache 적용
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-8B \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90
```

---

### 3.4 GPU 모니터링

#### nvidia-smi 핵심 지표

```bash
# 현재 GPU 상태 한 번 조회
nvidia-smi

# 출력 예시:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0    |
# +-----------------------------------------------------------------------------+
# | GPU  Name         Temp  Perf  Pwr:Usage/Cap  |    Memory-Usage  |  Compute M |
# |=============================================================================|
# |   0  A100-SXM4-80GB  42C   P0   250W / 400W  | 45231MiB / 81920MiB|      Default |
# +-----------------------------------------------------------------------------+
```

| 지표 | 의미 | 정상 범위 (추론 중) |
|------|------|-------------------|
| **GPU Utilization** | GPU 컴퓨팅 사용률 | 70~95% (낮으면 배치 크기 부족) |
| **Memory Usage** | VRAM 사용량 | `gpu_memory_utilization` 설정 이하 |
| **Temperature** | GPU 온도 | A100: 80°C 이하 권장 |
| **Power Draw** | 전력 소비 | 400W 근처가 최대 성능 |

```bash
# 연속 모니터링: 1초 간격으로 핵심 지표 출력
nvidia-smi dmon -s u -d 1

# 출력 형식: gpu  sm  mem  enc  dec  jpg  ofa
# sm  = SM(Streaming Multiprocessor) 활용률 (%)
# mem = 메모리 컨트롤러 활용률 (%)

# 더 상세한 연속 모니터링
nvidia-smi dmon -s pucvmet -d 2
# p=power, u=utilization, c=proc-clocks, v=voltage, m=memory, e=ecc, t=temperature
```

#### vLLM Prometheus 메트릭

vLLM 서버는 `/metrics` 엔드포인트로 Prometheus 형식의 메트릭을 제공한다.

```bash
# 메트릭 수집
curl http://localhost:8000/metrics

# 주요 메트릭 필터링
curl -s http://localhost:8000/metrics | grep -E "vllm:(num_requests|kv_cache|throughput|latency)"
```

> 💡 심화 운영 가이드는 [Part 5](#part-5-kv-cache-심화--lmcache) 참조

**핵심 Prometheus 메트릭**

| 메트릭 | 설명 | 활용 |
|--------|------|------|
| `vllm:num_requests_running` | 현재 처리 중인 요청 수 | 서버 부하 파악 |
| `vllm:num_requests_waiting` | 큐에서 대기 중인 요청 수 | 병목 감지 |
| `vllm:kv_cache_usage_perc` | KV Cache 사용률 (0.0~1.0) | OOM 위험 조기 감지 |
| `vllm:avg_generation_throughput_toks_per_s` | 초당 평균 생성 토큰 수 | 처리량 모니터링 |
| `vllm:e2e_request_latency_seconds` | End-to-End 요청 레이턴시 분포 | SLA 모니터링 |

#### 모니터링 스택: Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['host.docker.internal:8000']  # vLLM 서버 주소
    metrics_path: '/metrics'
```

**Grafana 대시보드 주요 패널**:
- 요청 처리 현황: `num_requests_running` + `num_requests_waiting`
- KV Cache 압박: `vllm:kv_cache_usage_perc` (0.9 초과 시 알림)
- 처리량 추이: `avg_generation_throughput_toks_per_s`
- P50/P95 레이턴시: `e2e_request_latency_seconds` histogram

---

### 3.5 분산 서빙

#### vLLM의 병렬화 전략 (v0.19.0)

단일 GPU에 모델이 올라가지 않거나, 처리량을 선형으로 확장해야 할 때 분산 서빙이 필요하다.

| 병렬화 방식 | 파라미터 | 동작 방식 | 사용 케이스 |
|------------|----------|-----------|------------|
| **Tensor Parallel (TP)** | `--tensor-parallel-size N` | 레이어의 가중치 행렬을 N개 GPU에 분할 | 단일 노드 다중 GPU |
| **Pipeline Parallel (PP)** | `--pipeline-parallel-size N` | 레이어를 순서대로 N개 GPU에 배분 | 다중 노드 |
| **Data Parallel (DP)** | `--data-parallel-size N` | 동일 모델 N개 복제본 운영 | 처리량 수평 확장 |
| **Expert Parallel (EP)** | `--enable-expert-parallel` | MoE 모델의 Expert를 GPU에 분산 | Mixtral, Qwen3-MoE 등 |

**world_size 계산**:

```
world_size = Tensor Parallel × Pipeline Parallel
예: TP=4, PP=2 → 8개 GPU 필요
```

#### Ray 의존성 (v0.18.0 이후 변경)

- **v0.18.0 이전**: 다중 GPU 사용 시 Ray 필수
- **v0.18.0 이후**: Ray는 **선택적 의존성**
  - 단일 노드 다중 GPU: Ray 없이 동작
  - **다중 노드**: `--distributed-executor-backend ray` 필요

```bash
# 단일 노드 4GPU (Ray 불필요)
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 4

# 다중 노드 (Ray 필요)
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2
```

#### 분산 서빙 성능 개선 이력

| 버전 | 개선 사항 | 효과 |
|------|-----------|------|
| v0.16.0 | Async Scheduling + Pipeline Parallel 조합 | 처리량 **+30.8%** |
| v0.18.0 | Ray 선택적 의존성, FlexKV 도입 | 운영 복잡도 감소 |
| v0.19.0 | Expert Parallel (EP) 정식 지원 | MoE 모델 효율 향상 |

#### 실제 운영 예시: Qwen3-235B-A22B

기존 수업에서 사용한 `vllm_serve_distributed.sh` 참조:

```bash
# 16개 GPU (8×2 구성) 기반 Qwen3-235B-A22B 서빙
vllm serve Qwen/Qwen3-235B-A22B \
  --distributed-executor-backend ray \
  --host=0.0.0.0 --port=8080 \
  --tensor-parallel-size=8 \    # 8 GPU로 텐서 분산
  --pipeline-parallel-size=2 \  # 2 노드로 파이프라인 분산
  --gpu-memory-utilization=0.95 \
  --reasoning-parser qwen3
# world_size = 8 × 2 = 16개 GPU 필요
```

---

### 3.6 Part 3 실습 가이드

#### 실습 3A: 성능 파라미터 튜닝 (30분)

**목표**: 파라미터 변경 전후 처리량과 레이턴시 변화를 직접 측정한다.

##### Step 1: Baseline 서버 구동

```bash
# Baseline: 기본 설정
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-baseline \
  vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 64
```

##### Step 2: 간단한 부하 테스트 스크립트

```python
# load_test.py
import time
import concurrent.futures
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")

TEST_PROMPT = "Python에서 리스트 컴프리헨션의 장점을 설명하고 예시를 보여줘."

def single_request():
    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": TEST_PROMPT}],
        max_tokens=200,
        stream=False,
    )
    elapsed = time.time() - start
    tokens = response.usage.completion_tokens
    return elapsed, tokens

def run_concurrent(n_concurrent=10, n_total=30):
    results = []
    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(single_request) for _ in range(n_total)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start_all
    latencies = [r[0] for r in results]
    tokens = [r[1] for r in results]
    print(f"동시 요청: {n_concurrent} | 총 요청: {n_total}")
    print(f"평균 레이턴시: {sum(latencies)/len(latencies):.2f}s")
    print(f"P95 레이턴시: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
    print(f"총 처리량: {sum(tokens)/total_time:.1f} tokens/s")

if __name__ == "__main__":
    run_concurrent(n_concurrent=1)
    run_concurrent(n_concurrent=5)
    run_concurrent(n_concurrent=10)
```

##### Step 3: gpu_memory_utilization 조정 비교

```bash
# 설정 1: 보수적 (0.80)
docker rm -f vllm-test
docker run --runtime nvidia --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-test vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.80
# → python load_test.py 실행 후 수치 기록

# 설정 2: 공격적 (0.93)
docker rm -f vllm-test
docker run --runtime nvidia --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-test vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.93
# → python load_test.py 실행 후 수치 기록
```

##### Step 4: KV Cache FP8 전환 비교

```bash
# FP16 KV Cache (기본)
docker rm -f vllm-test
docker run --runtime nvidia --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-test vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B

# 메모리 사용량 기록: nvidia-smi | grep MiB

# FP8 KV Cache
docker rm -f vllm-test
docker run --runtime nvidia --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-test vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B \
  --kv-cache-dtype fp8

# 메모리 사용량 비교: FP8에서 KV Cache 공간이 ~50% 줄어드는지 확인
```

##### Step 5: Prefix Caching 효과 측정

```python
# prefix_cache_test.py
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")

# 긴 시스템 프롬프트 (prefix로 캐싱될 부분)
LONG_SYSTEM = """당신은 Python 전문가입니다. """ + ("Python은 인터프리터 언어입니다. " * 200)

def ask_with_prefix(question: str):
    start = time.time()
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": question},
        ],
        max_tokens=100,
        stream=True,
    )
    first_token_time = None
    for chunk in stream:
        if chunk.choices[0].delta.content and first_token_time is None:
            first_token_time = time.time()
            break
    return first_token_time - start

questions = ["리스트란?", "딕셔너리란?", "클래스란?", "함수란?", "모듈이란?"]

print("=== Prefix Caching 효과 측정 ===")
for i, q in enumerate(questions):
    ttft = ask_with_prefix(q)
    label = "(첫 요청 - 캐시 MISS)" if i == 0 else f"(캐시 HIT 예상)"
    print(f"요청 {i+1} {label}: TTFT = {ttft:.3f}s")
```

**예상 결과**:
```
요청 1 (첫 요청 - 캐시 MISS): TTFT = 2.1s
요청 2 (캐시 HIT 예상):       TTFT = 0.3s
요청 3 (캐시 HIT 예상):       TTFT = 0.3s
```

---

#### 실습 3B: GPU 모니터링 + Streaming (30분)

##### Step 1: nvidia-smi dmon 모니터링

```bash
# 터미널 1: 서버 구동
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3-0.6B

# 터미널 2: GPU 지속 모니터링 (1초 간격)
nvidia-smi dmon -s u -d 1

# 터미널 3: 부하 생성
python load_test.py
```

**관찰 포인트**:
- 요청 시작 시 SM 활용률(`sm`) 급등
- 유휴 상태와 부하 상태의 메모리 사용률(`mem`) 차이
- 처리 중 전력 소비 변화

##### Step 2: /metrics 엔드포인트 Prometheus 메트릭 확인

```bash
# 서버 구동 중 메트릭 조회
curl -s http://localhost:8000/metrics | grep vllm

# 부하 테스트 중 실시간 모니터링 (2초 간격)
watch -n 2 'curl -s http://localhost:8000/metrics | \
  grep -E "vllm:(num_requests|kv_cache|throughput)" | \
  grep -v "^#"'
```

**주요 확인 지표**:
```
vllm:num_requests_running{...} 8         # 현재 처리 중 8개 요청
vllm:num_requests_waiting{...} 12        # 12개 대기 (배치 용량 초과 신호)
vllm:kv_cache_usage_perc{...} 0.72       # KV Cache 72% 사용
vllm:avg_generation_throughput_toks_per_s{...} 342.5  # 342.5 tokens/s
```

##### Step 3: Streaming vs Non-Streaming 응답 시간 비교

```python
# streaming_comparison.py
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")

PROMPT = "대한민국의 역사를 500자 이내로 설명해줘."
MODEL = "Qwen/Qwen3-0.6B"

# Non-streaming
start = time.time()
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=300,
    stream=False,
)
non_stream_total = time.time() - start
print(f"[Non-Streaming] 전체 응답 대기: {non_stream_total:.3f}s")
print(f"  생성 토큰 수: {response.usage.completion_tokens}")

print()

# Streaming
start = time.time()
first_token_time = None
token_count = 0

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=300,
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - start
        token_count += 1

stream_total = time.time() - start
tpot = (stream_total - ttft) / token_count * 1000 if token_count > 1 else 0

print(f"[Streaming]")
print(f"  TTFT (첫 토큰까지): {ttft:.3f}s")
print(f"  TPOT (토큰당 평균): {tpot:.1f}ms")
print(f"  전체 시간: {stream_total:.3f}s")
print(f"  사용자 체감: Non-Streaming 대비 {non_stream_total/ttft:.1f}배 빠른 첫 응답")
```

##### Step 4: 결과 기록 템플릿

| 설정 | 처리량(tokens/s) | 평균 레이턴시 | P95 레이턴시 | VRAM 사용 |
|------|-----------------|--------------|--------------|-----------|
| Baseline (gpu_util=0.80) | | | | |
| gpu_util=0.93 | | | | |
| KV Cache FP8 | | | | |
| Prefix Caching ON | | | | |
| Streaming (TTFT 기준) | | | | |

---

### Part 3 핵심 정리

| 최적화 | 효과 | 트레이드오프 |
|--------|------|-------------|
| `max-num-seqs` ↑ | 처리량 ↑ | 지연 ↑, 메모리 ↑ |
| Chunked Prefill | TTFT 변동성 20~40% 감소 | 구현 복잡도 소폭 ↑ |
| Streaming | 사용자 체감 응답 속도 ↑↑ | 서버 측 연결 지속 필요 |
| Prefix Caching | 반복 prefix TTFT 7배 개선 | 분산 환경에서 Sticky Routing 필요 |
| KV Cache FP8 | VRAM 50% 절감 | 미세 정밀도 손실 |
| Tensor Parallel | 대형 모델 적재 가능 | GPU 수 증가, 네트워크 대역폭 필요 |

---

## Part 4: 프레임워크 비교

> **학습 목표**
> - Production LLM Serving 프레임워크(vLLM, TGI, SGLang, TEI)의 아키텍처 차이를 이해한다.
> - 워크로드 특성에 따라 적절한 프레임워크를 선택할 수 있다.
> - 각 프레임워크의 핵심 최적화 전략과 트레이드오프를 파악한다.

---

### 4.1 왜 비교하는가?

Production LLM Serving을 구축할 때 프레임워크 선택은 처음부터 끝까지 모든 설계 결정에 영향을 미친다. 세 프레임워크 모두 Apache 2.0 라이선스로 상업적 사용이 자유롭지만, 내부 아키텍처와 최적화 전략이 근본적으로 다르다.

| 항목 | vLLM | TGI | SGLang |
|------|------|-----|--------|
| 개발 주체 | Anyscale / UC Berkeley | HuggingFace | LMSYS Org / UC Berkeley |
| 논문 | SOSP 2023 | — | OSDI 2024 |
| 라이선스 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| 핵심 혁신 | PagedAttention | Continuous Batching (Rust) | RadixAttention |

> vLLM의 PagedAttention 원리와 KV Cache 관리 방식은 [Part 1](#part-1-vllm-핵심-아키텍처)에서 자세히 다뤘다.

선택 기준을 한 문장으로 정리하면:
- **일반 API 서빙 → vLLM**
- **HF 생태계 / 인코더-디코더 혼용 → TGI**
- **Agent / RAG / 구조화 출력 → SGLang**
- **임베딩 / 리랭커 전용 서빙 → TEI**

> **중요한 구분**: TEI(Text Embeddings Inference)는 vLLM/TGI/SGLang처럼 생성 모델을 서빙하는 엔진이라기보다, **임베딩·리랭커·시퀀스 분류 전용 고속 서버**에 가깝다. 따라서 "생성 프레임워크 4강 비교"가 아니라 **RAG 파이프라인의 별도 축**으로 보는 것이 정확하다.

---

### 4.2 TGI (Text Generation Inference) — HuggingFace

#### 4.2.1 아키텍처

TGI는 **Rust HTTP/gRPC 라우터 + Python 모델 백엔드** 이중 레이어 구조다.

```
클라이언트
    |
[Rust HTTP/gRPC Router]   <- 연결 관리, 배치 스케줄링, 부하 분산
    |
[Python Model Backend]    <- 실제 forward pass, GPU 연산
    |
[Flash Attention 2 / CUDA Kernels]
```

Rust 라우터가 요청 수신과 Continuous Batching 스케줄링을 담당하기 때문에 Python GIL의 영향을 받지 않는다. 이는 낮은 연결 오버헤드가 필요한 엣지 배포에서 유리하다.

#### 4.2.2 주요 기능

**Continuous Batching**
Rust 사이드 스케줄러가 in-flight 요청을 동적으로 배치에 추가/제거한다. 기존 정적 배치 대비 GPU 활용률을 크게 향상시킨다.

**Flash Attention 2**
메모리 효율적인 어텐션 연산. TGI v2.0 이후 Flash Attention 2를 통해 PagedAttention과 유사한 KV 캐시 효율을 간접적으로 구현했다. 단, vLLM의 custom CUDA 커널 기반 PagedAttention과는 구현 방식이 다르다.

**Safetensors-first Loading**
HuggingFace safetensors 포맷을 우선 지원한다. 기존 `.bin` 파일(pt 직렬화 포맷) 대비 안전하며, 로딩 속도도 빠르다.

**양자화 지원 (역사적으로 vLLM보다 선행)**
- GPTQ, AWQ, EETQ, bitsandbytes, FP8

#### 4.2.3 KV Cache 관리

TGI v2.0 이후 Flash Attention 2를 통해 PagedAttention 유사 기능을 채택했다. 그러나 vLLM의 custom 커널과 달리 Flash Attention 2의 청크 처리 방식에 의존한다.

**결정적 차이: Prefix Caching 없음**

TGI는 vLLM의 Automatic Prefix Caching(Part 1 섹션 1.3 참조)과 SGLang의 RadixAttention에 해당하는 기능이 없다. 반복되는 시스템 프롬프트나 few-shot 예시가 있는 워크로드에서 TTFT(Time-To-First-Token)가 최대 7배까지 느려질 수 있다.

#### 4.2.4 vLLM 대비 장점

| 장점 | 설명 |
|------|------|
| HF 생태계 통합 | HuggingFace Hub 모델을 가장 자연스럽게 사용 |
| Rust 라우터 낮은 오버헤드 | Python GIL 우회, 연결 처리 효율적 |
| 인코더 모델 지원 | BERT, T5 등 encoder-only/encoder-decoder 지원 |
| 임베딩 엔드포인트 | `/embed` 엔드포인트로 임베딩 벡터 직접 서빙 |
| Dynamic LoRA Loading | 런타임 중 LoRA 어댑터 핫스왑 지원 (vLLM보다 성숙) |

#### 4.2.5 vLLM 대비 단점

| 단점 | 설명 |
|------|------|
| Prefix Caching 없음 | 반복 프롬프트에서 TTFT 최대 7배 페널티 |
| PagedAttention 구현 | Flash Attention 2 의존, vLLM custom 커널 대비 효율 낮음 |
| 고동시성 처리량 | vLLM 대비 0.5-0.7x 처리량 |
| 분산 서빙 성숙도 | Multi-GPU 분산 처리에서 vLLM에 비해 덜 성숙 |

---

### 4.3 SGLang — LMSYS Org / UC Berkeley

#### 4.3.1 아키텍처

SGLang은 3계층 아키텍처로 구성된다.

```
[SGLang DSL Frontend]      <- LLM 프로그램 표현 언어
        |
[Runtime Backend]          <- 요청 스케줄링, 배치 관리
        |
[RadixAttention Engine]    <- KV 캐시 공유 핵심 엔진
        |
[FlashInfer Kernels]       <- GPU 연산
```

논문: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", OSDI 2024

#### 4.3.2 RadixAttention vs PagedAttention — 핵심 차이

이것이 SGLang과 vLLM의 가장 근본적인 아키텍처 차이다.

> vLLM PagedAttention의 원리는 [Part 1](#part-1-vllm-핵심-아키텍처) 섹션 1.2에서 상세히 다뤘다.

**vLLM PagedAttention**
- 토큰 시퀀스를 고정 크기 페이지(블록)로 분할
- 요청 단위 prefix 재사용: 동일한 prefix 해시가 있으면 블록을 공유
- 요청 간(cross-request) KV 공유는 가능하지만, 동일 요청의 여러 호출(cross-call) 간 공유는 불가

**SGLang RadixAttention**
- 토큰 시퀀스를 Radix Tree(트라이) 자료구조로 관리
- 트리 탐색으로 캐시 히트를 찾기 때문에 prefix 길이에 무관하게 O(prefix_len) 탐색
- **cross-request** AND **cross-call** KV 공유 모두 가능
- fork/join 분기 구조를 네이티브로 지원

**구체적 예시 — Few-shot 프롬프트 + 3개 분기**

```
공통 prefix: [시스템 프롬프트 500토큰] + [Few-shot 예시 200토큰]
분기 A: [질문 A 50토큰]
분기 B: [질문 B 50토큰]
분기 C: [질문 C 50토큰]
```

| 프레임워크 | KV 캐시 처리 |
|-----------|------------|
| vLLM | 공통 prefix에 대해 best-effort 해시 매칭, 분기마다 별도 페이지 할당 |
| SGLang | Radix Tree에서 공통 700토큰을 단일 트리 경로로 구조적 공유, 분기 노드에서 leaf 분기 |

결과: few-shot + 다중 분기 패턴에서 SGLang의 TTFT가 2-8배 낮다.

#### 4.3.3 구조화 출력 (Structured Output)

**vLLM 방식**
- JSON Schema를 FSM(Finite State Machine)으로 컴파일
- 각 토큰 생성 시 FSM 전이 마스킹으로 유효하지 않은 토큰 제거
- 강제 토큰(forced tokens)도 여전히 forward pass를 실행

**SGLang 방식**
- **Compressed FSM**: 동일 전이를 묶어 상태 수 대폭 감소
- **Jump-Forward Decoding**: 강제 토큰 구간을 forward pass 없이 건너뜀
- 결과: 동일 JSON 스키마 기준 2-3x 낮은 오버헤드

```python
# SGLang DSL 예시 — 구조화 출력
@function
def classify(s, review):
    s += system("You are a sentiment classifier.")
    s += user(review)
    s += assistant(gen("result", max_tokens=10,
                       regex=r"positive|negative|neutral"))
```

#### 4.3.4 Programming DSL

SGLang은 복잡한 LLM 프로그램을 파이썬 함수처럼 표현할 수 있는 DSL을 제공한다.

```python
# 병렬 생성 + 분기
@function
def multi_branch(s, question):
    s += user(question)
    forks = s.fork(3)           # 3개 병렬 분기
    for f in forks:
        f += assistant(gen("answer", max_tokens=100))
    s.join(forks)               # 결과 병합
```

이 구조 자체가 RadixAttention의 cross-call KV 공유와 맞물려 최적화된다.

#### 4.3.5 vLLM 대비 장점

| 장점 | 설명 |
|------|------|
| RadixAttention | multi-call 프로그램에서 TTFT 2-8배 감소 |
| 구조화 출력 속도 | Jump-Forward Decoding으로 2-3배 빠른 JSON 생성 |
| Programming DSL | 복잡한 LLM 파이프라인을 선언적으로 표현 |
| FlashInfer 통합 | 최신 attention 커널, 특히 decode phase 최적화 |

#### 4.3.6 vLLM 대비 단점

| 단점 | 설명 |
|------|------|
| 모델 지원 폭 | vLLM보다 지원 모델 수 적음 |
| 분산 서빙 | Multi-node 설정이 vLLM보다 복잡 |
| 커뮤니티 규모 | GitHub star 수, 플러그인 생태계 vLLM 대비 작음 |
| 인코더 모델 | encoder-only 모델 지원 없음 |

---

### 4.4 Feature Comparison Matrix

| Feature | vLLM v0.19.0 | TGI | SGLang |
|---------|:---:|:---:|:---:|
| **Continuous Batching** | O | O | O |
| **KV Cache Management** | PagedAttention (custom kernel) | FlashAttn2 (post v2.0) | RadixAttention |
| **Automatic Prefix Caching** | O | X | O (RadixAttention) |
| **Cross-call KV Sharing** | X | X | O |
| **Quantization — GPTQ** | O | O | O |
| **Quantization — AWQ** | O | O | O |
| **Quantization — FP8** | O | O | O |
| **Quantization — FP4 (NVFP4)** | O (Blackwell, experimental) | X | 부분 지원 |
| **bitsandbytes** | O | O | X |
| **OpenAI-compatible API** | O | O | O |
| **Structured Output (JSON Schema)** | O | O | O (Jump-Forward) |
| **LoRA (Static)** | O | O | O |
| **LoRA (Dynamic hot-swap)** | 제한적 | O | O |
| **Encoder-only Model (BERT 등)** | X | O | X |
| **Embedding Endpoint** | O | O | O |
| **Tensor Parallelism (TP)** | O | O | O |
| **Pipeline Parallelism (PP)** | O | 제한적 | O |
| **Data Parallelism (DP)** | O | O | O |
| **Expert Parallelism (EP, MoE)** | O | X | O |
| **Speculative Decoding** | O | O | O |
| **Tool Calling** | O | O | O |
| **Programming DSL** | X | X | O |
| **구현 언어 (라우터)** | Python | Rust + Python | Python |

---

### 4.5 성능 비교

> 아래 수치는 공개 벤치마크 및 논문 기반 상대적 비교다. 실제 수치는 모델, 하드웨어, 워크로드에 따라 달라진다.

#### 4.5.1 단일 GPU 처리량 (Single-turn, 동시성 높음)

```
SGLang ≈ vLLM  >>  TGI (0.5–0.7x)
```

TGI는 Rust 라우터 오버헤드가 낮지만, 고동시성 배치에서 Python 백엔드 병목이 발생한다.

#### 4.5.2 Prefix 반복 워크로드 TTFT

```
SGLang (2–8x 빠름)  >  vLLM  >>  TGI
```

동일한 시스템 프롬프트나 few-shot 예시를 반복 사용하는 RAG, Agent 시나리오에서 SGLang의 RadixAttention이 가장 효과적이다. TGI는 Prefix Caching이 없어 매 요청마다 전체 prefix를 재연산한다.

#### 4.5.3 JSON Constrained Generation (Throughput)

```
SGLang (2–3x 빠름)  >  vLLM ≈ TGI
```

Jump-Forward Decoding이 강제 토큰 구간의 forward pass를 건너뛰기 때문에 JSON 스키마가 복잡할수록 차이가 커진다.

#### 4.5.4 Multi-GPU 분산 서빙

```
vLLM  >  SGLang  >  TGI
```

vLLM은 Tensor Parallelism, Pipeline Parallelism, Expert Parallelism을 안정적으로 지원하며 multi-node 설정이 가장 성숙하다. (분산 서빙 상세 파라미터는 Part 3 섹션 3.5 참조)

---

### 4.6 배포 비교

#### 4.6.1 Docker 이미지

| 항목 | vLLM | TGI | SGLang |
|------|------|-----|--------|
| 공식 이미지 | `vllm/vllm-openai:latest` | `ghcr.io/huggingface/text-generation-inference:latest` | `lmsysorg/sglang:latest` |
| 이미지 크기 | ~8GB | ~10GB | ~7GB |
| CUDA 버전 기반 태그 | O | O | O |

**vLLM 예시**
```bash
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:v0.19.0 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.9
```

**TGI 예시**
```bash
docker run --gpus all \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8B-Instruct
```

**SGLang 예시**
```bash
docker run --gpus all \
  -p 30000:30000 \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000
```

#### 4.6.2 HuggingFace Hub 통합

| 항목 | vLLM | TGI | SGLang |
|------|------|-----|--------|
| HF Hub 직접 로딩 | O | O (최우선 지원) | O |
| safetensors 지원 | O | O (기본값) | O |
| PEFT/LoRA Hub 로딩 | O | O | O |

#### 4.6.3 클라우드 지원

| 클라우드 | vLLM | TGI | SGLang |
|---------|------|-----|--------|
| AWS SageMaker | O (공식 컨테이너) | O (공식 DLC) | 수동 설정 |
| GCP Vertex AI | O | O | O |
| Azure ML | O | O | 제한적 |
| On-premise Docker | O | O | O |

#### 4.6.4 설정 복잡도 및 학습 곡선

| 항목 | vLLM | TGI | SGLang |
|------|------|-----|--------|
| 기본 설정 난이도 | 낮음 | 낮음 | 중간 |
| 분산 서빙 설정 | 중간 | 높음 | 중간 |
| 모니터링 통합 | Prometheus 기본 제공 | Prometheus 기본 제공 | 수동 설정 |
| 공식 문서 품질 | 우수 | 우수 | 중간 |

---

### 4.7 커뮤니티 & 생태계

| 항목 | vLLM | TGI | SGLang |
|------|------|-----|--------|
| GitHub Stars (약) | ~45K | ~10K | ~8K |
| 주요 기여 조직 | Anyscale, UC Berkeley, Google | HuggingFace | LMSYS, UC Berkeley |
| 논문 | SOSP 2023 | 없음 | OSDI 2024 |
| 기여자 수 | 가장 많음 | 중간 | 적음 |
| 릴리즈 주기 | 2-4주 | 1-3주 | 2-4주 |
| 엔터프라이즈 지원 | Anyscale (상업적 지원) | HuggingFace Enterprise | 없음 (학술 주도) |

**vLLM SOSP 2023 논문**: "Efficient Memory Management for Large Language Model Serving with PagedAttention", Kwon et al.

**SGLang OSDI 2024 논문**: "SGLang: Efficient Execution of Structured Language Model Programs", Zheng et al.

---

### 4.8 선택 가이드 (Decision Framework)

#### 4.8.1 워크로드 기반 선택

```
단순 Chat API 서빙
  └── 단일 GPU → vLLM 또는 SGLang (벤치마크 후 선택)
  └── Multi-GPU (70B+) → vLLM

반복 시스템 프롬프트 / RAG / Agent
  └── SGLang (RadixAttention으로 TTFT 2-8x 감소)

JSON / 구조화 출력이 핵심
  └── SGLang (Jump-Forward Decoding으로 2-3x 빠름)

HuggingFace 모델 생태계 최우선
  └── TGI

BERT / T5 등 인코더 모델 함께 서빙
  └── TGI (유일하게 encoder-only 지원)

임베딩 / 리랭커 전용 마이크로서비스
  └── TEI (Text Embeddings Inference)

LoRA 어댑터 런타임 핫스왑
  └── TGI (가장 성숙한 dynamic LoRA)

70B+ 모델 분산 서빙
  └── vLLM (가장 성숙한 분산 서빙)
```

#### 4.8.2 팀/운영 기반 선택

| 상황 | 추천 |
|------|------|
| 팀이 Python 친숙, HF 스택 사용 중 | TGI |
| 범용 OpenAI-compatible API 필요 | vLLM |
| 연구/실험 환경, 고급 최적화 필요 | SGLang |
| 임베딩 모델 / 리랭커를 별도 서비스로 분리 | TEI |
| 엔터프라이즈 SLA 필요 | vLLM (Anyscale 지원 가능) |

#### 4.8.3 TEI (Text Embeddings Inference)는 언제 쓰는가?

TEI는 HuggingFace가 제공하는 **임베딩·리랭커·시퀀스 분류 전용 서버**다. 생성 모델을 위한 범용 Chat API 서버가 아니라, RAG 파이프라인에서 다음과 같은 역할에 특화되어 있다.

- **문서 인덱싱용 임베딩 생성**
- **온라인 질의 임베딩 생성**
- **리랭커(cross-encoder) 서비스**
- **시퀀스 분류 / 감성 분류**

공식 README 기준 핵심 특징:

- **Token-based dynamic batching**
- **Flash Attention / Candle / cuBLASLt 기반 최적화**
- **Safetensors / ONNX 로딩**
- **Prometheus metrics + OpenTelemetry tracing**
- REST 엔드포인트:
  - `/embed`
  - `/rerank`
  - `/predict`
  - `/embed_sparse`
- **gRPC 지원**
- CPU / GPU / Apple Silicon / ARM64 / ROCm 등 다양한 배포 경로

실무 판단 기준은 간단하다:

- **생성 모델 서빙**이 목적이면 vLLM/TGI/SGLang 중에서 고른다.
- **임베딩 / 리랭커만 빠르게 안정적으로 서빙**하려면 TEI가 가장 직접적인 선택이다.
- **생성 + 임베딩이 모두 필요한 RAG 시스템**이라면:
  - 생성: vLLM 또는 SGLang
  - 임베딩 / 리랭커: TEI
  - 처럼 **사이드카/분리 마이크로서비스**로 구성하는 것이 자연스럽다.

---

### Part 4 핵심 요약

| 프레임워크 | 한 줄 요약 |
|-----------|-----------|
| **vLLM** | PagedAttention 기반 범용 고성능 서빙 — 가장 넓은 모델 지원과 성숙한 분산 서빙 |
| **TGI** | HuggingFace 공식 서버 — 인코더 모델 · Dynamic LoRA · HF 생태계 네이티브 통합 |
| **SGLang** | RadixAttention + Jump-Forward로 Agent · RAG · 구조화 출력에 최적화된 차세대 엔진 |
| **TEI** | 임베딩·리랭커·분류 전용 고속 서버 — RAG 파이프라인의 retrieval 축에 가장 잘 맞는 HuggingFace 옵션 |

---

## Part 5: KV Cache 심화 + LMCache

> **관련 내용**: vLLM KV Cache 기초 → Part 1 섹션 1.2 참조. LMCache 통합 실습 → Part 3 참조.

---

### 5.1 vLLM KV Cache 관리 방법 (공식 문서 기반)

#### 5.1.1 Automatic Prefix Caching (APC)

##### 개념과 동작 원리

Automatic Prefix Caching(APC)은 동일한 prefix를 공유하는 여러 요청이 KV Cache를 재사용할 수 있게 하는 vLLM의 핵심 최적화 기능이다. 동일한 system prompt나 few-shot 예제를 반복 사용하는 경우, prefill 연산을 생략하고 캐시에서 직접 KV 텐서를 가져와 TTFT(Time To First Token)를 크게 줄인다.

> **참조**: PagedAttention 기반 블록 구조에 대한 자세한 설명은 Part 1 섹션 1.2 참조.

**Hash-based block caching 메커니즘**

vLLM은 KV Cache를 고정 크기의 블록(기본값: 16 토큰)으로 나눈다. 각 블록은 다음 방식으로 해시 키를 생성한다.

```
block_hash = hash(parent_block_hash || token_ids_in_block)
```

이 체이닝(chaining) 방식 덕분에 prefix의 어느 지점에서 분기가 발생해도 정확히 그 지점까지만 캐시를 재사용한다.

```
[System Prompt Block 0] → hash_0
[System Prompt Block 1] → hash(hash_0 + tokens_1) = hash_1
[User Query Block]      → hash(hash_1 + tokens_2) = hash_2 (새로 계산)
```

**마지막 블록(Partial Trailing Block) 처리**

중요한 제약사항: **완전히 채워진 블록만 캐시된다.** 시퀀스의 마지막 블록이 `block_size`보다 적은 토큰을 가지면 캐시 대상이 아니다. 이는 불완전한 블록에서 해시 충돌이 발생하는 것을 방지한다.

##### Hash 알고리즘 선택

| 알고리즘 | 특성 | 추천 상황 |
|---------|------|---------|
| `sha256` (기본값) | Python 직렬화 기반, 암호학적 보안, 비교적 느림 | 단일 테넌트, 보안 중요 환경 |
| `sha256_cbor` | CBOR 인코딩 + sha256, 결정론적 직렬화 | 멀티 인스턴스 일관성 필요 시 |
| `xxhash` | 매우 빠름, 비암호학적 | 단일 테넌트 고속 처리 |
| `xxhash_cbor` | xxhash + CBOR | 빠른 단일 테넌트 환경 |

> **주의**: `xxhash`는 멀티 테넌트 환경에서 해시 충돌 위험이 있다. 서로 다른 테넌트의 다른 내용이 같은 해시를 가질 수 있다. `sha256`은 이 위험이 사실상 없다.

내부 구현 참고: `sha256` 알고리즘은 Python 객체 직렬화를 통해 토큰 ID를 바이트로 변환한 뒤 sha256을 적용한다. `sha256_cbor`는 CBOR(Concise Binary Object Representation) 인코딩으로 더 결정론적인 직렬화를 보장한다.

##### LRU Eviction

APC는 LRU(Least Recently Used) 방식으로 캐시 블록을 제거한다.

- 내부적으로 doubly linked list로 구현
- 새 블록은 리스트의 앞(most recent)이 아닌 **뒤(least recent) 방향으로 삽입**된다 (reverse order insertion). 이는 APC에서 새로 추가된 블록이 완성 전까지 캐시 우선순위가 낮게 유지되도록 의도된 설계다.
- 블록이 실제로 prefill 재사용될 때 MRU 위치로 이동

##### 멀티 테넌트 격리: cache_salt

동일 서버에서 여러 테넌트를 서비스할 때 KV Cache 분리가 필요하다. `cache_salt`를 테넌트별로 다르게 설정하면 해시 계산에 salt가 추가되어 테넌트 간 캐시 간섭이 없어진다.

```bash
# 테넌트 A용 인스턴스
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --cache-salt "tenant-a-secret"

# 테넌트 B용 인스턴스
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --cache-salt "tenant-b-secret"
```

##### 캐시 초기화

서버 재시작 없이 prefix cache를 비워야 할 때:

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_prefix_caching=True)
# ... 서비스 중 ...
llm.llm_engine.reset_prefix_cache()  # 모든 캐시된 블록 제거
```

##### 활성화 방법

```bash
# V1 엔진에서는 기본 활성화됨 (vllm >= 0.8.0)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

---

#### 5.1.2 KV Cache Quantization (FP8)

##### 개요

KV Cache를 FP32/BF16 대신 FP8로 저장하면 **동일한 GPU 메모리로 2배의 KV 블록**을 수용할 수 있다. Hopper(H100) 및 Ada(L40S, RTX 4090) 아키텍처에서 하드웨어 가속도 지원한다.

##### CacheDType 옵션

| CacheDType | 설명 | 요구사항 |
|-----------|------|---------|
| `auto` | 모델 dtype 그대로 사용 | - |
| `fp8` | FP8 (하드웨어 자동 선택) | Hopper/Ada GPU |
| `fp8_e4m3` | 4비트 지수, 3비트 가수 | Hopper GPU (SM90) |
| `fp8_e5m2` | 5비트 지수, 2비트 가수 | Ada GPU (SM89) |
| `fp8_inc` | Intel Neural Compressor 기반 FP8 | Intel GPU |
| `int8_per_token_head` | Attention head 단위 INT8 | Flash Attention 3 |
| `fp8_per_token_head` | Attention head 단위 FP8 | Flash Attention 3 |

##### 두 가지 양자화 전략

**전략 1: Per-tensor quantization (기본)**

전체 KV Cache 텐서에 단일 scale factor를 적용한다. 구현이 단순하고 대부분의 경우 충분한 품질을 제공한다.

**전략 2: Per-attention-head quantization (Flash Attention 3 전용)**

각 attention head별로 독립적인 scale factor를 사용한다. head마다 activation 분포가 다를 수 있어 더 정밀한 양자화가 가능하다. `int8_per_token_head` 또는 `fp8_per_token_head` 사용 시 활성화된다. llm-compressor로 calibration이 필요하다.

##### Scale Calibration 방법

**옵션 1: No calibration (기본값)**

Scale factor를 1.0으로 설정. 성능은 약간 낮지만 즉시 사용 가능.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8
```

**옵션 2: On-the-fly random calibration**

서버 시작 시 랜덤 입력으로 scale을 추정. 실제 데이터와 다를 수 있어 품질 손실 가능.

**옵션 3: Dataset calibration (권장)**

llm-compressor를 사용해 실제 데이터셋으로 calibration scale을 사전 계산하고 모델 파일에 저장.

```python
# llm-compressor를 이용한 KV Cache scale 계산 (오프라인)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    kv_cache_scheme={"type": "fp8"},
)

oneshot(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dataset="ultrachat-200k",
    recipe=recipe,
    output_dir="./llama-3.1-8b-fp8-kv",
    num_calibration_samples=512,
)
```

> **Note (v0.19+)**: `--calculate-kv-scales` 플래그는 deprecated. vLLM이 자동으로 scale을 처리한다.

##### 특정 레이어 양자화 제외

일부 레이어(예: 첫 번째와 마지막 레이어)는 양자화에 민감할 수 있다:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 31  # 레이어 0, 31 제외
```

##### Flash Attention 3 (FA3) 특이사항

FA3를 사용할 때는 key/value뿐만 아니라 **query도 FP8로 양자화**된다. 이로 인해 attention score 계산 전체가 FP8 precision으로 진행된다.

---

#### 5.1.3 GPU Memory Management

##### gpu_memory_utilization

vLLM이 GPU 메모리에서 모델 가중치 + KV Cache를 위해 사용할 비율이다.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.90  # 기본값
```

**내부 메모리 프로파일링 과정**:

1. **Dummy forward pass**: 최대 `max_model_len` 길이의 더미 입력으로 forward를 실행해 활성화 메모리 피크를 측정
2. **Peak measurement**: CUDA memory profiler로 실제 사용량 기록
3. **Available KV 계산**: `available_kv = (total_gpu_memory × gpu_memory_utilization) - model_weights - peak_activations - cudagraph_memory`
4. **Block count**: `num_blocks = available_kv / (block_size × num_layers × num_heads × head_dim × dtype_size × 2)`

##### 세밀한 메모리 제어 방법

**방법 1: `--kv-cache-memory-bytes`**

KV Cache에 할당할 메모리를 바이트 단위로 직접 지정. `gpu_memory_utilization`을 무시한다.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-memory-bytes 10737418240  # 10GB
```

**방법 2: `--num-gpu-blocks-override`**

KV 블록 수를 강제로 설정. 프로파일링을 건너뛰므로 테스트와 디버깅에 유용하다.

```bash
# 블록 수를 1000으로 강제 (프로덕션에서는 비권장)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --num-gpu-blocks-override 1000
```

##### Block Size 설정

블록 크기는 내부 단편화와 병렬성 트레이드오프에 영향을 준다.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --block-size 32  # 기본값: 16, 옵션: 8, 16, 32
```

- **작은 block_size (8)**: 단편화 감소, 블록 수 증가로 관리 오버헤드 증가
- **큰 block_size (32)**: 관리 효율, 짧은 시퀀스에서 내부 단편화 증가

---

#### 5.1.4 KV Cache Offloading (v0.19.0)

vLLM v0.19.0부터 KV Cache를 GPU 외부 저장소로 오프로드하는 기능이 공식 지원된다.

##### UVA Backend (기존): CPU 오프로드

모델 가중치를 CPU 핀드 메모리(pinned memory)로 이동시켜 GPU에 더 많은 KV Cache 공간을 확보한다.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --cpu-offload-gb 20  # 20GB를 CPU로 오프로드
```

> **Note**: 이 플래그는 모델 가중치 오프로드이며 KV Cache 자체를 CPU로 옮기는 것이 아니다.

##### Prefetch Backend (신규): 레이어 단위 오프로드

여러 레이어를 그룹으로 묶어 async prefetch와 함께 CPU 메모리에 오프로드한다.

##### KV Cache 직접 오프로드 (핵심 기능)

KV Cache 자체를 CPU/Disk/Remote로 오프로드:

```bash
# CPU로 KV Cache 오프로드
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-offloading-size 40 \
  --kv-offloading-backend native

# LMCache 백엔드 사용 (더 많은 기능)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-offloading-size 40 \
  --kv-offloading-backend lmcache
```

---

#### 5.1.5 Preemption Behavior (V1 Engine)

##### V1에서의 변경사항

vLLM V1 엔진에서는 V0의 **CPU swap 기능이 제거**되었다. V0에서는 `swap_space` 설정으로 실행 중인 요청의 KV Cache를 CPU로 swap out했지만, V1에서는 이 메커니즘이 없다.

**V1 preemption 동작**:

1. GPU KV Cache가 부족해 새 요청을 스케줄링할 수 없음
2. **가장 낮은 우선순위 요청** 선택 (주로 FCFS에서는 가장 나중에 들어온 요청)
3. 해당 요청의 KV Cache 블록을 즉시 **해제**
4. 요청 상태를 `PREEMPTED`로 변경
5. 요청을 **대기 큐(waiting queue)로 되돌림**
6. 나중에 다시 스케줄링되면 **처음부터 recompute** (swap 없음)

```python
# preemption 횟수 모니터링
stats = llm.llm_engine.get_stats()
print(f"Preemptions: {stats.num_preemptions}")
```

##### Preemption의 성능 영향

Preemption이 발생하면 해당 요청의 모든 prefill 연산을 다시 해야 한다. 긴 컨텍스트 요청일수록 비용이 크다. `num_preemptions` 메트릭이 높다면 KV Cache 용량 부족 신호다.

---

#### 5.1.6 KV Transfer Connectors

##### Disaggregated Prefill/Decode (P/D 분리)

Prefill(first token 생성)과 Decode(이후 token 생성)를 서로 다른 서버에서 처리하는 아키텍처다. Prefill은 compute-intensive, Decode는 memory-bandwidth-intensive하므로 서로 다른 하드웨어에 최적화할 수 있다.

> **참조**: P/D 분리 아키텍처의 배경 및 시스템 구성 개요는 Part 3 참조.

```
[Prefill 서버]          [Decode 서버]
KV Producer    --KV--> KV Consumer
(A100 x 2)             (L4 x 4)
```

##### 지원 Connector

| Connector | 전송 방식 | 특징 |
|----------|---------|-----|
| `NixlConnector` | RDMA/UCX | 저지연, InfiniBand 활용 |
| `MooncakeConnector` | Mooncake Store | ByteDance 개발, 대규모 클러스터 |
| `P2pNcclConnector` | NCCL P2P | NVIDIA GPU 간 직접 전송 |

##### 설정 예시

```bash
# Prefill 서버 (KV Producer)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_port":14579}' \
  --port 8100

# Decode 서버 (KV Consumer)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1,"kv_ip":"192.168.1.10","kv_port":14579}' \
  --port 8200
```

##### kv_load_failure_policy

KV 전송 실패 시 동작 정책:

| 정책 | 동작 | 적합한 상황 |
|-----|------|-----------|
| `fail` (기본값) | 에러 반환 | SLA가 엄격한 프로덕션 |
| `recompute` | 로컬에서 재계산 | 가용성 우선 환경 |

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_load_failure_policy":"recompute"}'
```

---

### 5.2 KV Cache 문제 증상과 해결 방법

#### 5.2.1 KV Cache 부족 증상

| 증상 | 근본 원인 | 해결 방법 |
|-----|---------|---------|
| `num_preemptions` 지속 증가 | KV 블록 부족으로 실행 중 요청 강제 중단 | `gpu_memory_utilization` 높이기, FP8 적용, `max_model_len` 줄이기 |
| `num_requests_waiting` 지속 증가 | 새 요청을 위한 블록 부족으로 스케줄 불가 | 동시 요청 수 제한 (`max_num_seqs`), 스케일 아웃 |
| 높은 TTFT (p95 > 수초) | 요청이 대기 큐에서 장시간 대기 | Prefix Caching 적용, Chunked Prefill 활성화 |
| CUDA OOM / 서버 크래시 | `gpu_memory_utilization`이 너무 높아 예기치 않은 사용량 초과 | 0.90 이하로 설정, 안전 마진 유지 |
| Prefix cache hit rate 급락 | 캐시 교체(eviction)가 너무 빈번 | KV 메모리 증가, `cache_salt` 오설정 점검 |
| Decode 속도 저하 | GPU memory bandwidth 포화 | FP8 KV Cache 적용, batch size 조정 |

#### 5.2.2 흔한 설정 실수

##### 실수 1: `gpu_memory_utilization` 과도하게 높음

```bash
# 위험: 예기치 않은 activation spike로 OOM 가능
--gpu-memory-utilization 0.98

# 권장
--gpu-memory-utilization 0.90  # 10% 안전 마진 유지
```

CUDA context, PyTorch caching allocator, NCCL buffer 등 vLLM 외부에서도 GPU 메모리를 사용하므로 항상 마진을 둬야 한다.

##### 실수 2: `gpu_memory_utilization` 불필요하게 낮음

```bash
# 낭비: KV 공간을 절반밖에 못 씀
--gpu-memory-utilization 0.50

# 합리적 범위
--gpu-memory-utilization 0.85  # 안전하면서도 효율적
```

##### 실수 3: `max_model_len` 과도하게 높음

```bash
# 비효율: 대부분의 요청이 4K이지만 128K를 설정
--max-model-len 131072

# 결과: 프로파일링 시 128K 기준으로 메모리를 예약하여 실제 가용 블록 수 급감
# 권장: 실제 p99 요청 길이 * 1.2 정도로 설정
--max-model-len 8192
```

##### 실수 4: Hopper/Ada에서 FP8 미사용

```bash
# H100, A100 (Hopper), L40S, RTX 4090 (Ada)에서 FP8 미사용
# BF16 대비 2배 KV 메모리 낭비

# FP8 적용 (단순 설정)
--kv-cache-dtype fp8

# 더 나은 품질 (calibration 포함 모델 사용 시)
--kv-cache-dtype fp8_e4m3  # Hopper
--kv-cache-dtype fp8_e5m2  # Ada
```

##### 실수 5: `cache_salt` 미설정 (멀티 테넌트)

멀티 테넌트 환경에서 `cache_salt` 없이 APC를 사용하면 서로 다른 테넌트가 서로의 캐시를 "오염"시킬 수 있다. (동일한 system prompt로 시작하는 경우 의도치 않게 공유)

#### 5.2.3 단편화 이슈

##### 외부 단편화 (External Fragmentation)

비연속적인 메모리 블록을 사용할 수 없는 현상. vLLM의 Paged Attention이 이 문제를 **완전히 제거**했다. 비연속 블록도 논리적 KV Cache로 매핑하므로 외부 단편화가 없다.

##### 내부 단편화 (Internal Fragmentation)

각 블록 내에서 발생하는 낭비. 시퀀스 길이가 block_size의 배수가 아닐 때 마지막 블록에 빈 슬롯이 생긴다.

```
block_size = 16, 시퀀스 = 25 토큰
  블록 1: 16/16 사용 (낭비 0)
  블록 2:  9/16 사용 (낭비 7 = block_size - 1 최대)
```

최대 낭비: 시퀀스당 `block_size - 1` 토큰 = 15 토큰(block_size=16). 대규모 배치에서는 평균적으로 `(block_size - 1) / 2 = 7.5` 토큰 낭비.

##### APC 중복 블록 (Duplicate Blocks)

동일한 prefix에 대해 여러 요청이 동시에 prefill을 진행할 때, 캐시 히트가 발생하기 전에 중복 블록이 임시로 생성될 수 있다. 이 중복 블록은 요청이 완료되거나 취소될 때 정리된다. APC 구현의 한계이며 특히 트래픽 급증 시 일시적으로 발생한다.

---

### 5.3 KV Cache 모니터링

#### 5.3.1 Prometheus 메트릭 (`/metrics`)

vLLM은 Prometheus 형식의 메트릭을 `/metrics` 엔드포인트로 노출한다.

> **참조**: Prometheus + Grafana 대시보드 설정 방법은 Part 3 참조.

##### 핵심 KV Cache 메트릭

| 메트릭 이름 | 타입 | 설명 | 경보 임계값 |
|-----------|-----|------|-----------|
| `vllm:kv_cache_usage_perc` | Gauge | 현재 KV Cache 사용률 (0.0~1.0) | > 0.90 |
| `vllm:num_requests_running` | Gauge | 현재 GPU에서 실행 중인 요청 수 | 모델별 상이 |
| `vllm:num_requests_waiting` | Gauge | 대기 큐의 요청 수 | > 50 (3분 지속) |
| `vllm:num_preemptions` | Counter | 누적 preemption 횟수 | rate > 1/초 |
| `vllm:prefix_cache_queries` | Counter | 로컬 Prefix Cache 조회 횟수 | - |
| `vllm:prefix_cache_hits` | Counter | 로컬 Prefix Cache 히트 횟수 | hit rate < 10% |
| `vllm:external_prefix_cache_queries` | Counter | 외부 KV Cache 조회 (LMCache 등) | - |
| `vllm:external_prefix_cache_hits` | Counter | 외부 KV Cache 히트 횟수 | - |
| `vllm:prompt_tokens_cached` | Counter | 캐시에서 재사용된 프롬프트 토큰 수 | - |
| `vllm:prompt_tokens_recomputed` | Counter | Preemption 후 재계산된 토큰 수 | - |
| `vllm:prompt_tokens_by_source` | Counter (labeled) | KV 소스별 토큰 수 | - |

##### `vllm:prompt_tokens_by_source` 레이블

이 메트릭은 `source` 레이블로 세분화된다:

| source 값 | 의미 |
|----------|-----|
| `local_cache_hit` | 로컬 GPU APC에서 히트 |
| `external_kv_transfer` | 외부 KV 스토어(LMCache/NixlConnector 등)에서 로드 |
| `computed` | 새로 계산됨 (캐시 미스) |

```promql
# Prometheus 쿼리 예시: Prefix cache hit rate
rate(vllm:prefix_cache_hits[5m])
  / rate(vllm:prefix_cache_queries[5m])

# Preemption 발생률
rate(vllm:num_preemptions[1m])
```

> **v0 -> v1 마이그레이션 주의**: V0에서 `vllm:gpu_cache_usage_perc`로 사용하던 메트릭이 V1에서 `vllm:kv_cache_usage_perc`로 이름이 변경되었다. 기존 Grafana 대시보드의 메트릭 이름을 업데이트해야 한다.

#### 5.3.2 Prometheus Alert Rules

```yaml
# vllm-alerts.yaml
groups:
  - name: vllm_kv_cache_alerts
    rules:
      # Alert 1: KV Cache 포화
      - alert: VllmKvCacheHigh
        expr: vllm:kv_cache_usage_perc > 0.90
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "vLLM KV Cache usage is high ({{ $value | humanizePercentage }})"
          description: |
            Instance {{ $labels.instance }} KV cache usage has exceeded 90% for 2 minutes.
            Consider: increasing gpu_memory_utilization, enabling FP8, or scaling out.

      # Alert 2: Preemption 빈발
      - alert: VllmHighPreemptionRate
        expr: rate(vllm:num_preemptions[1m]) > 1
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "vLLM experiencing high preemption rate ({{ $value | humanize }}/sec)"
          description: |
            Instance {{ $labels.instance }} preemption rate exceeds 1/sec.
            Requests are being interrupted and recomputed, severely degrading throughput.

      # Alert 3: 대기 큐 증가
      - alert: VllmWaitingQueueHigh
        expr: vllm:num_requests_waiting > 50
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "vLLM waiting queue is large ({{ $value }} requests)"
          description: |
            Instance {{ $labels.instance }} has {{ $value }} requests waiting for 3+ minutes.
            New requests cannot be scheduled. Consider load balancing or capacity expansion.

      # Alert 4: Prefix Cache hit rate 저하
      - alert: VllmLowPrefixCacheHitRate
        expr: |
          (
            rate(vllm:prefix_cache_hits[5m])
            / rate(vllm:prefix_cache_queries[5m])
          ) < 0.10
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "vLLM prefix cache hit rate is low ({{ $value | humanizePercentage }})"
          description: |
            Instance {{ $labels.instance }} prefix cache hit rate is below 10% for 5 minutes.
            Check if workload has repetitive prefixes. Consider cache_salt configuration.
```

#### 5.3.3 nvidia-smi + vLLM 로그

##### nvidia-smi로 GPU 메모리 실시간 모니터링

```bash
# 1초 간격으로 GPU 메모리/사용률 모니터링
nvidia-smi dmon -s mu -d 1

# 특정 GPU만 모니터링 (GPU 0)
nvidia-smi dmon -s mu -d 1 -i 0

# 지속 기록 (파일 저장)
nvidia-smi dmon -s mu -d 5 -f gpu_monitor.log

# 현재 프로세스별 GPU 메모리 사용량
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

##### vLLM 시작 로그에서 KV Cache 정보 확인

서버 시작 시 vLLM은 프로파일링 결과를 로그에 출력한다:

```
INFO 01-01 00:00:00 gpu_executor.py:123] # GPU blocks: 6144, # CPU blocks: 512
INFO 01-01 00:00:00 gpu_executor.py:124] Maximum concurrency for 4096 tokens per request: 24.00x
```

- `# GPU blocks: 6144` → 사용 가능한 KV 블록 수. block_size=16이면 6144 x 16 = 98,304 토큰의 KV Cache
- `Maximum concurrency` → 최대 동시 처리 가능한 요청 수 (max_model_len 기준)

##### 런타임 로그 (INFO 레벨)

```
INFO 01-01 00:00:05 metrics.py:456] Avg prompt throughput: 1234.5 tokens/s, Avg generation throughput: 89.0 tokens/s, Running: 8, Swapped: 0, Pending: 0, GPU KV cache usage: 87.4%, Prefix cache hit rate: 72.1%
```

이 로그에서:
- `GPU KV cache usage: 87.4%` → 현재 KV 블록 사용률
- `Prefix cache hit rate: 72.1%` → APC hit rate
- `Pending: 0` → 대기 중인 요청 없음 (정상)

---

### 5.4 LMCache: vLLM KV Cache의 확장

#### 5.4.1 LMCache란?

LMCache는 vLLM의 Automatic Prefix Caching을 3차원으로 확장하는 오픈소스 KV Cache 관리 시스템이다.

**확장의 3가지 차원**:

1. **Beyond GPU (저장소 확장)**: GPU HBM에서 벗어나 CPU DRAM → NVMe SSD → 원격 스토리지(Redis, AWS S3, Mooncake) 계층적 저장
2. **Beyond single instance (인스턴스 간 공유)**: P2P KV 공유(NIXL 기반), Multi-Process(MP) 모드로 여러 vLLM 인스턴스 간 캐시 공유
3. **Beyond prefix (비-프리픽스 재사용)**: CacheBlend 기술로 RAG 등 비연속적 패턴에서도 KV Cache 재사용

> 📝 **논문 원문과의 프레이밍 차이**: 위 "3가지 확장 차원"은 교육적 재해석이다. 논문 원문은 시스템 관점에서 3가지 과제(I/O 비효율, 추론 엔진 호환성, 관리 API 부재)와 이에 대응하는 3가지 기여(성능 최적화, KV Connector 표준 인터페이스, Controller API)를 제시한다.

**개발 배경**:
- 개발: Tensormesh
- 학술 기반: University of Chicago 연구 그룹
- 라이선스: Apache 2.0
- 논문: arXiv:2510.09665 (EuroSys 2025)

#### 5.4.2 핵심 기능

##### Multi-tier Storage Hierarchy

```
+-----------------------------------------------------+
|                    LLM Request                      |
+---------------------+-------------------------------+
                      |
                      v
+-----------------------------------------------------+
|              vLLM GPU KV Cache (HBM)                |
|                   ~40-80 GB                         |
|              Hit rate: 100% (if present)            |
+---------------------+-------------------------------+
                      | miss
                      v
+-----------------------------------------------------+
|              LMCache CPU DRAM Tier                  |
|                  100-500 GB                         |
|           Bandwidth: ~50-100 GB/s (PCIe)            |
+---------------------+-------------------------------+
                      | miss
                      v
+-----------------------------------------------------+
|              LMCache Disk Tier (NVMe)               |
|                   1-10 TB                           |
|            Bandwidth: ~5-10 GB/s                    |
+---------------------+-------------------------------+
                      | miss
                      v
+-----------------------------------------------------+
|      LMCache Remote Tier (Redis/S3/Mooncake)        |
|                   Unlimited                         |
|              Bandwidth: network-bound               |
+-----------------------------------------------------+
```

각 티어는 독립적으로 구성하거나 조합해서 사용할 수 있다.

##### CacheBlend: 비-프리픽스 KV 재사용

기존 APC는 정확한 prefix match만 캐시를 재사용한다. RAG 파이프라인에서는 여러 문서를 동적으로 조합하므로 prefix가 매번 달라진다.

CacheBlend는 separator token을 활용해 각 문서 청크의 KV Cache를 독립적으로 저장하고, 요청 시 조합한다. 약 15%만 재계산하면서 나머지 85%는 캐시에서 재사용한다.

```
전통 RAG (APC):
[System][Doc1][Doc2][Doc3][Query]  모두 재계산 (prefix mismatch)

CacheBlend:
[System]  캐시 히트
[Doc1]    캐시 히트 (독립적으로 저장됨)
[Doc2]    캐시 히트
[Doc3]    캐시 히트
[Query]   신규 계산 (~15%)
```

##### CacheGen: KV Cache 압축 전송

SIGCOMM 2024에 발표된 CacheGen 기술을 활용해 네트워크를 통한 KV Cache 전송을 압축한다. 원시 FP16 전송 대비 4~8배 압축률.

##### Layerwise Pipelined Transfer

KV Cache 로딩 시 별도의 CUDA stream을 활용해 파이프라이닝한다:

```
Stream 1: Load Layer 0-5   -----[====]------------------
Stream 2: Load Layer 6-11  ----------[====]-------------
Stream 3: Compute          ---------------[compute]-----
```

이를 통해 KV 로딩과 연산이 오버랩되어 실질적인 레이턴시 증가가 최소화된다.

##### Disaggregated Prefill (NIXL 기반)

LMCache는 NIXL(NVIDIA Inference Xfer Library)을 통한 Prefill/Decode 분리를 지원한다. NixlConnector를 내장해 P/D 분리 아키텍처를 vLLM 기본 설정보다 쉽게 구성할 수 있다.

##### Cache Eviction 정책

| 정책 | 설명 | 적합한 워크로드 |
|-----|------|--------------|
| `lru` (기본값) | Least Recently Used | 일반 범용 |
| `mru` | Most Recently Used | 최신 데이터 우선 필요 시 |
| `lfu` | Least Frequently Used | 인기 프롬프트 캐시 유지 |
| `fifo` | First In First Out | 예측 가능한 순서 필요 시 |

##### Async Loading

KV Cache를 비동기로 로드해 메인 추론 스트림을 차단하지 않는다. `async_loading: true` (기본값)로 활성화된다.

---

#### 5.4.3 Controller Interface (관리 API)

> ⚠️ **논문 핵심 기여**: Controller Interface는 LMCache 논문(arXiv:2510.09665)의 3대 기여 중 하나로, 프로덕션 multi-instance 운영의 핵심이다.

**아키텍처**: Centralized controller manager + Per-instance workers
- Controller Manager: 독립 프로세스로 글로벌 조정 수행
- Per-instance Workers: 각 LMCache 인스턴스와 공존, 로컬 연산 + 매니저 통신

**API 목록**:

| API | 유형 | 기능 |
|-----|------|------|
| `lookup(tokens)` | External | 글로벌 KV cache 위치 조회 |
| `move(src, dst, tokens)` | External | 인스턴스 간 KV cache 마이그레이션 |
| `clear(tokens, inst_id, device)` | External | 특정 위치의 KV cache 삭제 |
| `pin/unpin(tokens, instance, device)` | External | KV cache 고정/해제 (eviction 방지) |
| `compress/decompress(tokens, instance, device, method)` | External | KV cache 압축/해제 |
| `batched_admit/batched_evict` | Internal | KV 입퇴장 이벤트 보고 |
| `batched_p2p_lookup` | Internal | P2P KV cache 존재 확인 |

**4가지 핵심 기능**:
1. **KV cache-aware routing**: `lookup()` API로 캐시 적중률 최고 인스턴스에 쿼리 라우팅
2. **KV cache migration**: `move()` API로 스케일 다운/로드밸런싱 시 KV cache 이동
3. **P2P KV cache sharing**: `batched_p2p_lookup`으로 피어 인스턴스에서 캐시 직접 로드
4. **KV cache clearance**: `clear()` API로 모델 전환/메모리 회수 시 캐시 정리

---

#### 5.4.4 vLLM 연동 방법

> **참조**: LMCache 통합 실습 환경 구성 및 Docker Compose 설정은 Part 3 참조.

##### 설치

```bash
# pip/uv로 설치
uv pip install lmcache vllm

# Docker 사용 (권장)
docker pull lmcache/vllm-openai:latest

# 특정 버전 (호환성 확인 필수)
docker pull lmcache/vllm-openai:0.4.2-v0.8.3
```

##### 연동 방법 1: KV Transfer Config JSON

vLLM의 KV Transfer Connector API를 통해 LMCache를 연동한다.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV2",
    "kv_role": "kv_both",
    "kv_connector_config": {
      "lmcache_config_file": "/path/to/lmcache_config.yaml"
    }
  }'
```

##### 연동 방법 2: KV Offloading Backend

더 간단한 설정으로 CPU offload를 활성화:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 40
```

##### LMCache YAML 설정 파일 전체 예시

```yaml
# lmcache_config.yaml

# 저장 백엔드 계층 설정
storage_backend:
  # 1티어: CPU DRAM
  - type: local_cpu
    size: 40GB

  # 2티어: 로컬 NVMe SSD (선택)
  - type: local_disk
    path: /mnt/nvme/lmcache
    size: 200GB

  # 3티어: 원격 Redis (선택)
  # - type: redis
  #   host: redis-server.internal
  #   port: 6379
  #   size: unlimited

# Eviction 정책
eviction_policy: lru

# 비동기 로딩 활성화
async_loading: true

# 전송 압축 (CacheGen, 네트워크 전송 시)
compression:
  enabled: true
  algorithm: cachegen

# CacheBlend 설정 (RAG 워크로드용)
cacheblend:
  enabled: false
  recompute_ratio: 0.15

# Layerwise 파이프라인 설정
pipeline:
  num_cuda_streams: 3
  prefetch_layers: 4

# 토큰 단위 chunk 크기 (vLLM block_size와 일치 권장)
chunk_size: 256

# 로깅
logging:
  level: INFO
  metrics_port: 9090
```

##### kv_role 옵션

| kv_role | 설명 | 사용 상황 |
|---------|------|---------|
| `kv_producer` | KV를 생성하고 외부로 전송 | Prefill 전용 서버 |
| `kv_consumer` | 외부에서 KV를 수신하여 사용 | Decode 전용 서버 |
| `kv_both` | 생성과 소비 모두 수행 | 단일 서버, 일반적 용도 |

##### 호환성 매트릭스

| LMCache 버전 | 지원 vLLM 버전 | 비고 |
|------------|-------------|-----|
| 0.4.2 | 0.17, 0.18 | 안정 버전 |
| 0.4.x | 0.8.x (V1) | V1 엔진 지원 |
| main | 최신 vLLM | 실험적 |

> **PYTHONHASHSEED 고정 필수**: 멀티 인스턴스 환경에서 Python의 해시 랜덤화로 인해 같은 토큰 시퀀스가 다른 해시를 생성할 수 있다. LMCache를 멀티 인스턴스로 사용할 때는 반드시:

```bash
export PYTHONHASHSEED=42
vllm serve ...
```

---

#### 5.4.5 벤치마크 성능

##### 공개 벤치마크 결과

| 시나리오 | 개선 효과 | 측정 환경 |
|--------|---------|---------|
| CPU Offload (단일 인스턴스) | TTFT **44.5배 감소** | NVIDIA L4, Llama-3.1-8B, 15K 토큰 입력 | ⚠️ LMCache 공식 벤치마크 출처 (논문 외부). 논문 기준 CPU Offload 최대치는 **8.1배** (H100, Figure 8) |
| Long Document QA | TTFT **75% 감소 (4배)** | Qwen3-8B, 46문서 x 10K 토큰, 반복 질의 |
| P2P Instance 공유 | TTFT **54.7% 감소 (2.2배)** | Llama-3.1-8B, 50문서, 2 인스턴스 |
| 전체 시스템 (논문 기준) | TTFT **약 2~8배 감소**, 처리량 최대 **15배 향상** | arXiv:2510.09665 |

##### CPU Offload 효과 메커니즘

vLLM APC만 사용 시, GPU KV Cache가 가득 차면 eviction 이후 재계산이 필요하다. LMCache의 CPU 오프로드를 사용하면:

1. GPU에서 evict된 KV Block -> CPU DRAM으로 이동 (수십 GB 확장)
2. 동일 prefix 재요청 -> CPU에서 GPU로 빠르게 로드 (재계산 없음)
3. 특히 긴 컨텍스트(15K+ 토큰) 요청에서 효과가 극대화

---

#### 5.4.6 Vanilla vLLM vs LMCache 비교

| Feature | Vanilla vLLM APC | LMCache + vLLM |
|---------|-----------------|----------------|
| KV 저장소 | GPU HBM만 사용 | GPU->CPU->Disk->Remote (계층적) |
| GPU eviction 후 KV 유지 | No (재계산 필요) | Yes (CPU/Disk 티어로 이동) |
| 서버 재시작 후 KV 유지 | No (모두 소실) | Yes (Disk/Remote 백엔드 사용 시) |
| 인스턴스 간 KV 공유 | No | Yes (P2P NIXL, MP 모드) |
| 비-프리픽스 KV 재사용 | No | Yes (CacheBlend, RAG 최적화) |
| 네트워크 효율 KV 전송 | N/A | Yes (CacheGen 압축, 4~8배) |
| Disaggregated Prefill | 제한적 (내장 Connector만) | Yes (NIXL 네이티브 지원) |
| Controller 기반 글로벌 캐시 관리 | ❌ | ✅ (routing, migration, clearance) |

---

#### 5.4.7 실전 적용 시나리오

##### 시나리오 1: Multi-turn Conversations (대화형 AI)

**문제**: 사용자가 긴 대화를 이어갈 때마다 전체 히스토리를 재계산.

**해결**: LMCache CPU 오프로드로 대화 컨텍스트를 캐시 유지.

```yaml
storage_backend:
  - type: local_cpu
    size: 80GB
eviction_policy: lru
chunk_size: 256
```

**효과**: 10턴 이상 대화에서 TTFT가 선형으로 증가하지 않고 일정 수준으로 유지됨.

##### 시나리오 2: RAG Pipelines (CacheBlend)

**문제**: 검색된 문서 조합이 매번 달라 APC hit rate가 낮음.

**해결**: CacheBlend로 각 문서 청크를 독립적으로 캐시하고 조합.

```yaml
cacheblend:
  enabled: true
  recompute_ratio: 0.15
```

```bash
export PYTHONHASHSEED=42
vllm serve Qwen/Qwen3-8B \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV2",
    "kv_role": "kv_both",
    "kv_connector_config": {
      "lmcache_config_file": "/config/lmcache_rag.yaml"
    }
  }'
```

**효과**: 동일 문서 세트 기반 질의에서 TTFT 4배 감소 (Qwen3-8B 측정).

##### 시나리오 3: Shared System Prompts across Pods

**문제**: Kubernetes에서 여러 vLLM 파드가 동일한 긴 system prompt를 각자 캐시해 GPU 메모리 낭비.

**해결**: 공유 Redis에 system prompt KV Cache 저장.

```yaml
storage_backend:
  - type: local_cpu
    size: 20GB
  - type: redis
    host: shared-redis.default.svc.cluster.local
    port: 6379
```

**효과**: 각 파드가 공유 캐시에서 system prompt KV를 로드, GPU 메모리를 실제 사용자 KV에 더 많이 할당.

##### 시나리오 4: Long Document Analysis

**문제**: 100K 토큰 PDF 분석 -> 여러 질문 -> 매번 전체 문서 재계산.

**해결**: 첫 번째 질문에서 생성된 KV를 Disk 백엔드에 저장, 이후 질문은 캐시에서 로드.

```yaml
storage_backend:
  - type: local_cpu
    size: 100GB
  - type: local_disk
    path: /mnt/nvme/kvcache
    size: 500GB
async_loading: true
```

**효과**: 두 번째 질문부터 TTFT가 수십 초에서 1~2초로 단축.

##### 시나리오 5: 비용 절감 (Cloud)

GPU 시간이 비싼 클라우드 환경에서:
- LMCache로 KV 재계산 감소 -> GPU 연산 절감
- CPU/Disk 오프로드 -> 더 적은 GPU 인스턴스로 동일 처리량
- 아키텍처 보고서 기준: 동일 SLA에서 GPU 비용 30~60% 절감 가능

---

## 부록

### A. v0.11 → v0.19 버전 변경 요약

#### A.1 버전별 주요 변경 사항

| 버전 | 주요 변경 | 백엔드 영향 |
|------|---------|------------|
| **v0.11** | V0 엔진 완전 제거. V1 엔진 단일화. Chunked Prefill 안정화 | `VLLM_USE_V1=0` 환경변수 무효화 |
| **v0.12** | Multi-Modal 입력 개선 (이미지/비디오). `--limit-mm-per-prompt` 파라미터 추가 | 멀티모달 서비스에서 주의 |
| **v0.13** | LoRA 런타임 로드/언로드 API 안정화 | `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` 공식화 |
| **v0.14** | Async Scheduling 기본 활성화. CPU 오프로드 개선 | 스케줄링 지연 감소, `--cpu-offload-gb` 안정화 |
| **v0.15** | Prefix Caching V2. sha256_cbor 해싱 추가 | 캐시 히트율 향상 |
| **v0.16** | DeepSeek V3/R1 공식 지원. MTP Speculative Decoding | `deepseek_v3` 파서, `mtp` spec decode |
| **v0.17** | FlashAttention 4 (H100+). Performance Mode (`--performance-mode`) 추가 | H100 환경에서 자동 활성화 |
| **v0.18** | EAGLE3 Speculative Decoding. Tool Calling 파서 20+ 확장 | Qwen3, DeepSeek 파서 포함 |
| **v0.19** | **Responses API** (`/v1/responses`). **Anthropic API** (`/v1/messages`). **gRPC** (`--grpc`). Zero-Bubble Async Scheduling. `--kv-cache-memory-bytes` 파라미터 | API 호환성 확장. gRPC 클라이언트 연동 가능 |

#### A.2 주요 Breaking Changes (v0.11~v0.19)

| 버전 | Breaking Change | 마이그레이션 |
|------|----------------|------------|
| v0.11 | V0 엔진 제거 | `VLLM_USE_V1=0` 제거, V1 기반으로 전환 |
| v0.14 | `--disable-async-output-proc` 기본값 변경 | 동기 처리가 필요하면 명시적으로 설정 |
| v0.17 | `--performance-mode` 기본값 `balanced` | 기존 throughput 설정 확인 필요 |
| v0.19 | Prefix Caching 기본 활성화 | 결정론적 출력이 필요하면 `--no-enable-prefix-caching` |

#### A.3 핵심 메시지

> 이 강의에서 사용하는 Docker 이미지는 `vllm/vllm-openai:v0.19.0`.
> 강의 자료의 모든 파라미터와 API 엔드포인트는 이 버전 기준.
> 팀 내 기존 vLLM 서버가 v0.11 이하라면 V0 제거 Breaking Change 먼저 확인.

---

### B. 핵심 CLI 플래그 레퍼런스

아래 표는 KV Cache와 관련된 모든 주요 vLLM CLI 플래그를 정리한 것이다.

| 플래그 | 타입 | 기본값 | 설명 |
|------|-----|-------|------|
| `--enable-prefix-caching` | bool | V1: True | Automatic Prefix Caching 활성화 |
| `--no-enable-prefix-caching` | bool | - | APC 비활성화 (V1에서 명시적 비활성화) |
| `--prefix-caching-hash-algo` | str | `sha256` | APC 해시 알고리즘 선택 |
| `--cache-salt` | str | None | 멀티 테넌트 해시 격리용 salt |
| `--gpu-memory-utilization` | float | `0.90` | GPU 메모리 중 모델+KV에 사용할 비율 |
| `--kv-cache-memory-bytes` | int | None | KV Cache 메모리 바이트 직접 지정 (utilization 무시) |
| `--num-gpu-blocks-override` | int | None | KV 블록 수 강제 설정 (테스트/디버그용) |
| `--block-size` | int | `16` | KV Cache 블록당 토큰 수 (8/16/32) |
| `--kv-cache-dtype` | str | `auto` | KV Cache 저장 데이터 타입 |
| `--kv-cache-dtype-skip-layers` | int... | None | FP8 양자화에서 제외할 레이어 인덱스 |
| `--max-model-len` | int | 모델 기본값 | 최대 시퀀스 길이 (컨텍스트 윈도우) |
| `--max-num-seqs` | int | `256` | 동시 처리 최대 시퀀스 수 |
| `--cpu-offload-gb` | float | `0` | CPU로 오프로드할 모델 가중치 크기 (GB) |
| `--kv-offloading-size` | int | None | KV Cache 오프로드 크기 (GB) |
| `--kv-offloading-backend` | str | `native` | KV 오프로드 백엔드 (`native` 또는 `lmcache`) |
| `--kv-transfer-config` | JSON str | None | KV Transfer Connector 설정 (JSON) |

#### `--kv-transfer-config` 전체 필드 참조

```json
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_rank": 0,
  "kv_parallel_size": 1,
  "kv_port": 14579,
  "kv_ip": "127.0.0.1",
  "kv_load_failure_policy": "recompute",
  "kv_connector_config": {
    "lmcache_config_file": "/path/to/config.yaml"
  }
}
```

#### FP8 설정 빠른 참조

```bash
# Hopper GPU (H100, H200)
--kv-cache-dtype fp8_e4m3

# Ada GPU (L40S, RTX 4090)
--kv-cache-dtype fp8_e5m2

# 하드웨어 자동 감지
--kv-cache-dtype fp8

# 특정 레이어 제외 (0번, 마지막 레이어)
--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers 0 31
```

---

### C. 참고 자료

#### vLLM 공식 문서 및 GitHub

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- [vLLM KV Cache Quantization](https://docs.vllm.ai/en/latest/features/kv_cache_quantization.html)
- [vLLM KV Cache Offloading](https://docs.vllm.ai/en/latest/features/kv_cache_offloading.html)
- [vLLM Disaggregated Prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/production_metrics.html)
- [vLLM GitHub: block_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py)
- [vLLM 공식 GitHub](https://github.com/vllm-project/vllm)
- [vLLM 공식 문서](https://docs.vllm.ai)
- [vLLM v0.19.0 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.19.0)
- [vLLM v0.19.0 Docker Image](https://hub.docker.com/r/vllm/vllm-openai) — `docker pull vllm/vllm-openai:v0.19.0`

#### LMCache

- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [LMCache 공식 문서](https://docs.lmcache.ai)
- [LMCache Docker Hub](https://hub.docker.com/r/lmcache/vllm-openai)

#### 학술 논문

- **PagedAttention (vLLM 원논문)**: Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. arXiv:2309.06180
- **Orca (Continuous Batching 원형)**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022
- **CacheGen**: Yuhan Liu et al., "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving", SIGCOMM 2024. arXiv:2310.07240
- **LMCache (CacheBlend, Full System)**: Yuhan Liu et al., "LMCache: Accelerating Large Language Model Serving with Large-Scale KV Cache Management", EuroSys 2025. arXiv:2510.09665
- **EAGLE3**: Li et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test", 2025
- **Disaggregated Serving**: Cunchen Hu et al., "Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads", arXiv:2401.11181

---

### D. 양자화 조합 검증 결과

> 검증 기준: vLLM v0.19.0 소스 코드 및 공식 문서

#### D.1 모델 정밀도 × KV Cache 정밀도 조합

| # | 모델 | KV Cache | 판정 | 결론 | 비고 |
|---|------|---------|------|------|------|
| 1 | FP16 | FP16 | ✅ 정확 | 기준 케이스. `--kv-cache-dtype auto` 기본 동작 | BF16 모델(Llama-3 등)은 auto → BF16 KV |
| 2 | FP16 | FP8 | ✅ 정확 | 메모리 50% 절감. FP8↔FP16 양자화/역양자화 시 오차 발생 | `fp8_e4m3`(NVIDIA) / `fp8_e5m2`(AMD) |
| 3 | FP8 | FP16 | ✅ 정확 | **비권장** — FP8 모델로 절약한 메모리를 KV FP16이 상쇄 | 최적화 관점 비효율 |
| 4 | FP8 | FP8 | ✅ 정확 | **권장 조합**. Hopper+(H100) FP8 하드웨어 가속 | 퍼플렉시티 약 0.1-0.5% 증가 |
| 5 | FP4 | FP16 | ⚠️ 보충 | 가능하나 비효율. **조합 6이 실무 권장** | — |
| 6 | FP4 | FP8 | ✅ 정확 | **실무 권장 조합**. AWQ/GPTQ 4비트 + FP8 KV 완전 지원 | 최적 균형점 |
| 7 | FP4 | FP4 | ⚠️ 정확 | **v0.19.0 미지원**. PR #37192(WIP), Blackwell GPU 필요 | 곧 지원 예정 |

#### D.2 양자화 오차 완화 전략

| 전략 | 방법 | 효과 |
|------|------|------|
| **top_k 조정** | 확률 상위 N개 토큰만 허용 (값 ↓ = 더 결정적) | 오차 영향 범위 축소 |
| **top_p 조정** | 누적 확률 P 이내 토큰만 허용 (값 ↓ = 더 결정적) | 저확률 토큰 오차 차단 |
| **temperature 조정** | 확률 분포 집중도 (값 ↓ = 더 greedy) | 미세 확률 변화 영향 감소 |
| **Per-head KV scale** | v0.17.0+ 자동 지원. head별 독립 scale | FP8 outlier 정밀도 향상 |

```python
# 양자화 오차 완화 설정 예시
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", kv_cache_dtype="fp8")
params = SamplingParams(temperature=0.3, top_k=40, top_p=0.9)
```

#### D.3 실제 사용 사례: 144개 분류 (Qwen2.5)

| 모델 | 방법 | 정확도 | 문제점 |
|------|------|--------|--------|
| Qwen2.5-3B | 파인튜닝 | 낮음 | 비용 과다 + 중국어 혼입 |
| Qwen2.5-7B | 8비트 양자화 | 85~90% | 성능 애매 |
| **Qwen2.5-14B** | **4비트 양자화 (AWQ)** | **~90%** | 중국어 일부 혼입 (Qwen 특성) |

> **중국어 혼입 완화**: 시스템 프롬프트에 `"반드시 한국어로 답하라"` + `temperature` 낮추기

#### D.4 FP4 KV Cache 개발 현황

| 항목 | 상태 |
|------|------|
| FP8 KV Cache | ✅ 안정 (v0.3.0+) |
| FP4 KV Cache (NVFP4) | WIP — PR #37192 |
| 요구 하드웨어 | NVIDIA Blackwell (SM120+, B100/B200/RTX 5090) |
| 알려진 이슈 | FP4 scale tensor NaN (PR #38148) |

---

### E. 외국어 차단 및 RAG LogitsProcessor

#### E.1 외국어 차단 관련 GitHub 리포지토리

1. **LLM_Foreign_Block**: https://github.com/workdd/LLM_Foreign_Block
2. **Llama4-Token-Editor**: https://github.com/sionic-ai/Llama4-Token-Editor

#### E.2 SelectiveCiteProcessor — RAG 할루시네이션 완화

> GitHub: https://github.com/suhan1433/SelectiveCiteProcessor

경량 LLM(0.5B)에서 RAG 수행 시, 특정 정보(전화번호, 고유명사 등)의 할루시네이션/누락 문제를 LogitsProcessor로 개선하는 방법.

**원리**: 모델의 다음 토큰 예측 단계에서, 참조 Chunk 내 토큰들의 logit 값을 인위적으로 증폭하여 해당 토큰이 선택될 확률을 높인다.

```python
class SelectiveCiteProcessor(LogitsProcessor):
    def __init__(self, tokenizer, chunk_token_ids, boost_factor=1.0):
        self.tokenizer = tokenizer
        self.chunk_token_ids = chunk_token_ids
        self.boost_factor = boost_factor

    def __call__(self, input_ids, logits):
        vocab_size = logits.shape[1]
        for i in range(logits.shape[0]):
            chunk_tokens = set(self.chunk_token_ids[i].tolist())
            chunk_tokens.add(self.tokenizer.eos_token_id)
            chunk_tokens = [t for t in chunk_tokens if t < vocab_size]
            logits[i, chunk_tokens] += self.boost_factor
        return logits
```

**적용 결과** (Qwen2-0.5B, boost_factor=2.5, temperature=0.8):

| 구분 | 응답 품질 |
|------|----------|
| 적용 전 | 할루시네이션 — "인공지능의 모델로 인식되지 않으며..." (무관한 답변) |
| 적용 후 | 정확한 인용 — "연차 신청은 그룹웨어 시스템을 통해..." (Chunk 기반 답변) |

> FAQ와 같이 정확한 답변이 요구되는 상황에서, LLM-Judge 및 정성적 평가를 통해 기존 대비 성능 향상 확인 (30% → 90%+)
