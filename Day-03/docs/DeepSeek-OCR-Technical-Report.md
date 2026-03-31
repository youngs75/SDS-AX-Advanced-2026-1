# DeepSeek-OCR: Contexts Optical Compression — 기술 분석 보고서

## 1. 핵심 주장 요약

이 논문의 핵심 주장은 한 문장으로 압축됩니다:

> **텍스트를 이미지로 렌더링한 뒤 vision token으로 압축하면, 원래 text token 대비 10배 이상의 압축률을 달성하면서도 97%의 디코딩 정확도를 유지할 수 있다.**

이것은 "a picture is worth a thousand words"를 문자 그대로 실현하려는 시도입니다. LLM의 quadratic attention 비용 문제를 우회하기 위해, 텍스트를 이미지로 변환 → vision encoder로 압축 → 소수의 vision token으로 표현하는 **2D optical mapping** 패러다임을 제안합니다. 기존에는 vision token이 "이미지 이해"를 위한 것이었다면, 여기서는 **"텍스트 압축 매체"**로 재정의됩니다.

---

## 2. 문제 정의: 왜 이 연구가 필요한가

### 2.1 LLM의 장문 처리 병목

현재 LLM은 시퀀스 길이에 대해 **O(n²)** 의 attention 비용을 가집니다. 100K 토큰 컨텍스트를 처리하려면 막대한 메모리와 연산이 필요합니다. 기존 해결 방법들(sparse attention, sliding window 등)은 정보 손실이 불가피합니다.

### 2.2 기존 Vision Encoder의 한계

논문은 현존하는 VLM의 vision encoder가 3가지 유형으로 나뉘며, 모두 고해상도 문서 처리에 심각한 한계가 있음을 지적합니다:

| 유형 | 대표 모델 | 핵심 문제 |
|------|----------|----------|
| **Dual-tower** | Vary, DeepSeekVL | 두 번의 전처리 필요, 초고해상도 미지원, 배포 어려움 |
| **Tile-based** | InternVL2.0 | 네이티브 해상도가 낮아(512x512 미만) 이미지가 과도하게 분할됨, vision token 수 폭증 |
| **Adaptive resolution** | Qwen2-VL (NaViT) | 대형 이미지에서 activation memory 폭발, GPU OOM 위험, prefill/generation 속도 저하 |

### 2.3 핵심 연구 질문

> *"1000단어를 담은 문서를 디코딩하는 데 최소 몇 개의 vision token이 필요한가?"*

이 질문은 vision-text 압축의 이론적 한계와 실용적 경계를 탐구하는 것으로, 이전 연구에서 체계적으로 다뤄지지 않았습니다.

---

## 3. 제안 아키텍처: DeepSeek-OCR

### 3.1 전체 구조

DeepSeek-OCR은 **인코더(DeepEncoder, ~380M params)** + **디코더(DeepSeek3B-MoE, ~570M activated params)** 의 2-컴포넌트 구조입니다.

```
Input Image → [SAM-base (80M)] → [Conv 16x 다운샘플] → [CLIP-large (300M)] → Vision Tokens
                  ↓ window attention          ↓ 압축기              ↓ global attention
                  (local perception)                               (semantic knowledge)
                                                                        ↓
                                                            [DeepSeek3B-MoE] → Text Output
```

### 3.2 DeepEncoder의 설계 철학

DeepEncoder가 충족해야 하는 5가지 요구사항:

1. **고해상도 처리 가능** — 문서 이미지는 보통 1024px 이상
2. **고해상도에서도 낮은 activation memory** — GPU OOM 방지
3. **적은 수의 vision token 출력** — 압축의 핵심
4. **다중 해상도 입력 지원** — 압축률 실험을 위해
5. **적정 파라미터 수** — 실용적 배포를 위해

DeepEncoder의 핵심 혁신은 **SAM(window attention) → 16x Conv 압축 → CLIP(global attention)** 의 직렬 연결입니다. Window attention이 먼저 고해상도 이미지를 처리하되 local 범위로 제한하여 메모리를 절약하고, 16x 합성곱 압축기가 토큰 수를 대폭 줄인 뒤에야 비로소 global attention(CLIP)에 진입합니다. 이는 **"비싼 연산은 적은 토큰에만 적용"** 이라는 효율적 설계 원칙을 따릅니다.

> 예: 1024x1024 입력 → SAM이 n×16×16 패치로 분할 → 16x 압축 → 4096/16 = **256 vision tokens** 만 CLIP에 진입

### 3.3 다중 해상도 모드

| 모드 | 해상도 | Vision Token 수 | 처리 방식 |
|------|--------|----------------|----------|
| Tiny | 512 | 64 | resize |
| Small | 640 | 100 | resize |
| Base | 1024 | 256 (유효 182) | padding |
| Large | 1280 | 400 (유효 285) | padding |
| Gundam | 640+1024 | n×100+256 | 타일링+resize+padding |
| Gundam-M | 1024+1280 | n×256+400 | 타일링+resize+padding |

Dynamic resolution 모드(Gundam)는 InternVL2.0의 타일링 방식을 차용하되, 네이티브 해상도가 높기 때문에(640~1024) 과도한 분할 문제가 발생하지 않습니다. 타일 수는 2~9개로 제한됩니다.

유효 vision token 수 계산 공식:

```
N_valid = ⌈N_actual × [1 − ((max(w,h) − min(w,h)) / max(w,h))]⌉
```

여기서 w, h는 원본 입력 이미지의 너비와 높이입니다.

### 3.4 MoE 디코더

DeepSeek3B-MoE를 디코더로 사용합니다. 추론 시 64개 라우팅 전문가 중 6개 + 공유 전문가 2개가 활성화되어 약 **570M 활성 파라미터**로 동작합니다. 이는 3B 모델의 표현력을 가지면서 500M급의 추론 효율을 달성합니다.

디코더의 역할은 수학적으로:

```
f_dec : R^{n × d_latent} → R^{N × d_text}   (단, n ≤ N)
```

n개의 압축된 vision token으로부터 N개의 text token을 복원하는 비선형 매핑입니다.

---

## 4. 데이터 엔진

### 4.1 학습 데이터 구성 (총 비율)

| 데이터 유형 | 비율 | 설명 |
|-----------|------|------|
| **OCR 데이터** | 70% | OCR 1.0 + OCR 2.0 |
| **일반 비전 데이터** | 20% | caption, detection, grounding |
| **텍스트 전용 데이터** | 10% | 언어 능력 유지용 |

### 4.2 OCR 1.0 데이터 (문서 OCR)

- **30M 페이지**의 다양한 PDF 데이터 (약 100개 언어)
- 중국어/영어 25M + 기타 언어 5M
- **Coarse annotation**: fitz로 직접 추출 (소수 언어 포함)
- **Fine annotation**: 레이아웃 모델(PP-DocLayout) + OCR 모델(MineurU, GOT-OCR2.0)로 레이아웃+인식 interleaved 데이터 구축
  - 중국어/영어 각 2M 페이지
  - 소수 언어: 레이아웃 모델의 일반화 능력 활용 + fitz 패치 데이터로 GOT-OCR2.0 학습 → 60만 개 샘플 생성 (model flywheel)
- **3M Word 데이터**: 레이아웃 없이 직접 텍스트 추출 (수식, HTML 테이블에 유리)
- **자연 장면 OCR**: LAION + Wukong 출처, PaddleOCR로 라벨링, 중국어/영어 각 10M

### 4.3 OCR 2.0 데이터 (구조화된 이미지 파싱)

| 유형 | 데이터 규모 | 생성 방법 |
|------|----------|----------|
| 차트 | 10M 이미지 | pyecharts/matplotlib로 렌더링, HTML 테이블 형식으로 변환 |
| 화학식 | 5M 이미지-텍스트 쌍 | PubChem SMILES → RDKit 렌더링 |
| 평면 기하학 | 1M | Slow Perception 방식, 기하학적 이동 불변 증강 |

데이터 엔진에서 주목할 점은 **model flywheel** 전략입니다. 소수 언어 데이터의 경우, 기존 모델로 작은 패치 데이터를 라벨링 → 이를 학습한 모델로 더 큰 데이터를 라벨링하는 자기 강화 루프를 구축합니다. 또한 화학식의 경우 SMILES→이미지→SMILES 왕복 변환 데이터를 대규모로 생성하는데, 이는 합성 데이터의 대표적 활용 사례입니다.

---

## 5. 학습 파이프라인

### 5.1 Stage 1: DeepEncoder 학습

- OCR 1.0/2.0 + LAION 100M 일반 데이터
- Batch size 1280, 2 epoch
- AdamW + cosine annealing, lr=5e-5
- 학습 시퀀스 길이: 4096
- 소형 언어 모델(Opt-IML)로 next token prediction

### 5.2 Stage 2: DeepSeek-OCR 학습

- HAI-LLM 플랫폼에서 Pipeline Parallelism (4 파트)
  - PP0: SAM + 압축기 (frozen)
  - PP1: CLIP (unfrozen)
  - PP2-3: DeepSeek3B-MoE 각 6 레이어
- 20 노드 (각 8× A100-40G), DP=40, global batch=640
- AdamW + step-based scheduler, lr=3e-5
- 학습 속도: 텍스트 전용 90B tokens/day, 멀티모달 70B tokens/day

---

## 6. 실험 결과 및 핵심 주장에 대한 분석

### 6.1 Vision-Text 압축 연구 (핵심 주장)

Fox 벤치마크에서 영어 문서 100페이지(600-1300 text tokens)를 대상으로 테스트:

| Text Tokens | Vision=64 정확도 | 압축률 | Vision=100 정확도 | 압축률 |
|-------------|---------------|--------|----------------|--------|
| 600-700 | 96.5% | 10.5x | 98.5% | 6.7x |
| 700-800 | 93.8% | 11.8x | 97.3% | 7.5x |
| 800-900 | 83.8% | 13.2x | 96.8% | 8.5x |
| 900-1000 | 85.9% | 15.1x | 96.8% | 9.7x |
| 1000-1100 | 79.3% | 16.5x | 91.5% | 10.6x |
| 1100-1200 | 76.4% | 17.7x | 89.8% | 11.3x |
| 1200-1300 | 59.1% | 19.7x | 87.1% | 12.6x |

**핵심 발견**:

- **10x 이내 압축률에서 ~97% 정확도** 달성 (주장 뒷받침됨)
- **20x 압축에서도 ~60% 정확도** 유지 (주장 뒷받침됨)
- 압축률 10x 초과 시 성능이 급격히 저하되는 **cliff edge** 존재

이 결과의 의미는 심오합니다. 100개의 vision token(640x640 이미지 1장)으로 1000개의 text token을 97% 정확도로 복원할 수 있다는 것은, **vision modality가 text modality보다 정보 밀도가 훨씬 높다**는 것을 정량적으로 증명합니다. 다만, 10x 이후의 급격한 성능 저하는 현재 해상도(512~640)에서의 물리적 한계(글자가 흐려짐)와 복잡한 레이아웃의 정보 손실이 결합된 결과로, 더 높은 해상도에서는 이 한계가 완화될 수 있음을 시사합니다.

### 6.2 OmniDocBench 실용 성능

OmniDocBench에서의 edit distance 기반 비교 (값이 작을수록 우수):

| 모델 | 평균 Vision Tokens | 영어 Overall | 중국어 Overall |
|------|-------------------|-------------|-------------|
| GOT-OCR2.0 | 256 | 0.287 | 0.528 |
| MinerU2.0 | ~6790 | 0.133 | 0.506 |
| dots.ocr†200dpi | 5545 | 0.125 | 0.416 |
| **DeepSeek-OCR (Small)** | **100** | 0.221 | 0.530 |
| **DeepSeek-OCR (Base)** | **256(182)** | 0.137 | 0.474 |
| **DeepSeek-OCR (Gundam)** | **795** | 0.127 | 0.432 |
| **DeepSeek-OCR (Gundam-M†200dpi)** | **1853** | **0.123** | **0.377** |

**핵심 발견**:

- **100 vision tokens**만으로 GOT-OCR2.0(256 tokens)을 능가
- **800 tokens 미만**(Gundam)으로 MinerU2.0(~7000 tokens)을 능가
- 가장 적은 vision token으로 end-to-end 모델 중 SOTA 달성

### 6.3 문서 유형별 성능

| 문서 유형 | 최적 모드 | 특이사항 |
|---------|---------|---------|
| 슬라이드 | Tiny (64 tokens) | 텍스트가 적어 64 토큰으로도 충분 |
| 책/보고서 | Small (100 tokens) | 텍스트 1000 이내로 10x 압축 범위 |
| 교과서/시험지 | Base (256 tokens) | 수식/도표 포함 시 더 많은 토큰 필요 |
| 신문 | Gundam-M (1853 tokens) | 텍스트 4-5K로 10x 압축 한계 초과 |

### 6.4 Deep Parsing 기능

DeepSeek-OCR은 문서 내 이미지를 2차 모델 호출로 추가 파싱하는 "deep parsing" 기능을 갖추고 있습니다:

- **차트** → HTML 테이블 + 재렌더링 가능한 구조화 데이터
- **화학식** → SMILES 형식 변환
- **평면 기하학** → 선분/좌표/유형 딕셔너리
- **자연 이미지** → 밀집 캡션(dense caption) 자동 생성

### 6.5 다국어 인식

약 100개 언어의 PDF 문서를 처리할 수 있으며, 아랍어/싱할라어 등 소수 언어도 레이아웃/비레이아웃 모두 지원합니다.

---

## 7. Contexts Optical Compression: 망각 메커니즘

논문에서 가장 사변적이면서도 흥미로운 주장은 **"망각 메커니즘(forgetting mechanism)"** 개념입니다:

```
시간 →     [최근 대화]          [1일 전]        [1주 전]        [1년 전]
해상도 →   Gundam(고해상도)    Large          Base           Tiny(저해상도)
토큰 수 →  많음 (상세함)       중간            적음            극소 (흐릿함)
```

**제안 패러다임**:

1. 멀티턴 대화에서 과거 k 라운드 이후의 텍스트를 이미지로 렌더링
2. 최근 컨텍스트는 고해상도로, 오래된 컨텍스트는 저해상도로 점진적 축소
3. 인간 기억의 망각 곡선(Ebbinghaus)과 시각적 거리에 따른 인지 감쇠를 모방

이 "광학적 망각(optical forgetting)" 개념은 현재 LLM의 context window 문제에 대한 매우 독창적인 접근입니다. 기존의 KV-cache 압축, sliding window, RAG 등과는 완전히 다른 차원의 해결책을 제시합니다. 텍스트를 이미지로 렌더링하여 해상도를 낮추면, **물리적으로 정보가 점진적으로 손실**되면서 자연스러운 "기억 감쇠"가 구현됩니다. 다만 이는 아직 proof-of-concept 수준이며, 실제 multi-turn 대화에서의 검증은 향후 과제로 남겨두고 있습니다.

---

## 8. 생산 환경 실용성

| 항목 | 수치 |
|------|------|
| 일일 처리량 | **3300만 페이지/일** (20노드, 각 8× A100-40G) |
| 단일 노드 | **200K+ 페이지/일** (A100-40G 1대) |
| 모델 크기 | 인코더 ~380M + 디코더 ~570M activated |
| 시퀀스 길이 | 8192 tokens |

이는 LLM/VLM 사전학습 데이터 생성용 도구로서의 실용성을 강조한 것입니다.

---

## 9. 한계점 및 비판적 분석

### 9.1 논문이 인정하는 한계

1. **OCR만으로는 contexts optical compression의 완전한 검증에 불충분** — 디지털-광학 텍스트 interleaved pretraining, needle-in-a-haystack 테스트 등 추가 평가 필요
2. **SFT 미수행** — 챗봇이 아니라 completion 모델, 일부 기능은 프롬프트로 활성화 필요
3. **평면 기하학 파싱**은 선분 간 복잡한 상호의존성으로 인해 아직 초기 단계

### 9.2 추가 비판적 고찰

| 관점 | 비판 |
|------|------|
| **평가 범위** | Fox 벤치마크 100페이지, OmniDocBench 중심의 제한된 평가. 다양한 실제 문서(계약서, 법률문서, 코드 등)에서의 검증 부재 |
| **압축 측정 방식** | "text tokens in ground truth / vision tokens"로 정의된 압축률은 information-theoretic 압축률과 다름. 실제 정보 보존률을 더 엄밀히 측정할 필요 |
| **해상도 의존성** | 10x 이후 성능 저하가 해상도 한계(512-640에서 글자 흐려짐)인지, 모델 능력의 근본적 한계인지 분리되지 않음 |
| **비교 공정성** | Pipeline 모델(Marker, Mathpix 등)과 end-to-end 모델의 직접 비교는 paradigm이 다르므로 주의 필요 |
| **Forgetting mechanism** | 매력적인 아이디어이나, 실험적 검증이 전무한 순수 사변적 제안 |
| **언어 편향** | 학습 데이터의 약 83%가 중국어/영어이며, 소수 언어에서의 성능은 체계적으로 평가되지 않음 |
| **Downstream task 미검증** | OCR 정확도는 측정했으나, 압축된 vision token을 실제 LLM에 입력했을 때의 downstream task 성능(QA, summarization 등)은 미검증 |

---

## 10. 종합 평가

### 10.1 기여도

| 기여 | 평가 |
|------|------|
| Vision-text 압축의 정량적 분석 | **높음** — 최초의 체계적 연구 |
| DeepEncoder 아키텍처 | **높음** — 실용적이며 기존 한계를 잘 해결 |
| 대규모 학습 데이터 파이프라인 | **높음** — 100개 언어, 다양한 문서 유형 |
| Contexts optical compression 비전 | **중간** — 영감을 주지만 검증 부재 |
| 오픈소스 공개 | **높음** — 코드 + 모델 가중치 공개 |

### 10.2 의의

이 논문은 **VLM을 "이미지 이해" 도구가 아닌 "텍스트 압축 매체"로 재정의**한 최초의 체계적 시도입니다. 10x 압축에서 97% 정확도라는 결과는, 향후 LLM의 장문 컨텍스트 처리, KV-cache 최적화, 에이전트 시스템의 메모리 관리 등 다양한 영역에 영향을 미칠 수 있는 중요한 발견입니다.

다만, 이것이 실제 LLM 추론 파이프라인에서 text token 대비 연산 절감으로 이어지려면, **vision encoder의 인코딩 비용 + 디코딩 없이 압축된 vision token을 직접 LLM에 입력하는 방법**에 대한 추가 연구가 필요합니다. 현재는 "압축 → 디코딩(텍스트 복원)"의 왕복 과정이 필요하므로, 순수한 연산 절감보다는 **저장/전송 효율** 관점에서 즉시 실용적 가치가 있습니다.

이 연구의 가장 깊은 함의는 **modality 간 정보 밀도의 비대칭성**을 활용한 압축입니다. 텍스트는 1D sequential이지만, 이미지는 2D spatial입니다. 같은 정보량을 2D로 배치하면 1D보다 훨씬 적은 토큰으로 표현할 수 있습니다. 이는 Shannon의 정보이론에서 차원 확장을 통한 효율적 부호화와 맥을 같이 합니다. 향후 LLM이 vision encoder를 "내장 압축기"로 활용하는 하이브리드 아키텍처가 등장할 가능성을 열어놓은 논문입니다.

---

## 참고 문헌 (원문에서 인용된 주요 문헌)

- [5] Bai et al. Qwen2.5-VL technical report. arXiv:2502.13923, 2025.
- [6] Blecher et al. Nougat: Neural optical understanding for academic documents. arXiv:2308.13418, 2023.
- [8] Chen et al. How far are we to GPT-4V? (InternVL2.0). arXiv:2404.16821, 2024.
- [17] Kirillov et al. Segment Anything (SAM). arXiv:2304.02643, 2023.
- [19] Liu et al. DeepSeek-V2. arXiv:2405.04434, 2024.
- [20] Liu et al. DeepSeek-V3. arXiv:2412.19437, 2024.
- [21] Liu et al. Fox: Focus anywhere for fine-grained multi-page document understanding. arXiv:2405.14295, 2024.
- [27] Ouyang et al. OmniDocBench. CVPR, 2025.
- [29] Radford et al. CLIP: Learning transferable visual models from natural language supervision. ICML, 2021.
- [34] Wang et al. MinerU: An open-source solution for precise document content extraction. arXiv:2409.18839, 2024.
- [36] Wei et al. Vary: Scaling up the vision vocabulary for large VLM. ECCV, 2024.
- [38] Wei et al. GOT-OCR2.0: General OCR theory. arXiv:2409.01704, 2024.
