# runpodctl 사용 가이드

> Day-06 LLM 파인튜닝 워크숍을 위한 RunPod CLI 가이드
> runpodctl v2.1.9 기준 (2026-04-05 확인)

---

## 1. runpodctl이란?

RunPod의 공식 CLI 도구(Go 바이너리)로, 터미널에서 GPU pod 생성/관리/삭제, SSH 접속, 파일 전송, 비용 확인까지 모든 작업을 수행할 수 있습니다.

**핵심 특징:**
- 모든 RunPod pod에 **사전 설치**되어 있음 (pod 내부에서도 사용 가능)
- 명령 패턴: `runpodctl <리소스> <액션>` (예: `runpodctl pod create`)
- 출력 형식: JSON (기본), `--output=table` (사람용), `--output=yaml`
- 파일 전송: croc 프로토콜 기반 `send`/`receive` 내장

---

## 2. 설치

### macOS (Homebrew — 권장)

```bash
brew install runpod/runpodctl/runpodctl
```

### Linux / WSL

```bash
wget -qO- cli.runpod.net | sudo bash
```

### Windows (PowerShell)

```powershell
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe -O runpodctl.exe
```

### Google Colab / Jupyter

```bash
!wget -qO- cli.runpod.net | sudo bash
```

### 설치 확인

```bash
runpodctl version
# 출력 예: runpodctl v2.1.9
```

설정 파일 위치: `~/.runpod/config.toml`

---

## 3. 인증 및 API 키 발급

Day-06 워크숍에서 사용하는 API 키는 3종입니다. `.env` 파일에 한 번만 설정하면 `setup_runpod.sh`가 자동으로 읽습니다.

```bash
cp .env.example .env   # 템플릿 복사
vi .env                # 아래 키 입력
```

### 3-1. RunPod API Key (필수)

| 항목 | 내용 |
|------|------|
| **발급 URL** | https://www.runpod.io/console/user/settings |
| **회원가입** | https://www.runpod.io/console/signup |
| **크레딧 충전** | https://www.runpod.io/console/user/billing |
| **키 형태** | 영숫자 문자열 (예: `rpa_XXXXXXXXXXXXXXXXXXXX`) |

**발급 절차:**
1. https://www.runpod.io/console/user/settings 접속 → 좌측 메뉴 **"API Keys"**
2. **"Create API Key"** 버튼 클릭
3. 키 이름 입력 (예: `day06-workshop`) → **"Create"** 클릭
4. 생성된 키 복사 → `.env` 파일의 `RUNPOD_API_KEY=` 뒤에 붙여넣기
5. **주의:** 키는 생성 시 한 번만 표시됩니다. 반드시 즉시 저장하세요.

> **비용 안내:** 최소 $10 충전 권장. 24GB GPU 기준 시간당 $0.4~$1.0. 워크숍 7시간 기준 약 $3~$7 예상.

### 3-2. HuggingFace Access Token (권장)

| 항목 | 내용 |
|------|------|
| **발급 URL** | https://huggingface.co/settings/tokens |
| **회원가입** | https://huggingface.co/join |
| **키 형태** | `hf_` 접두사 (예: `hf_AbCdEfGhIjKlMnOpQrStUv`) |

**발급 절차:**
1. https://huggingface.co/settings/tokens 접속 → HuggingFace 로그인
2. **"Create new token"** 클릭
3. Token name: `day06-workshop`, Type: **"Read"** 선택
4. **"Create token"** 클릭 → `hf_...` 토큰 복사 → `.env`의 `HF_TOKEN=` 뒤에 붙여넣기

> **없으면:** 공개 모델(Qwen3.5, Qwen3-Embedding)은 다운로드 가능하지만, rate limit에 걸릴 수 있습니다.

### 3-3. OpenAI API Key (선택 — Lab 0 합성 데이터 생성용)

| 항목 | 내용 |
|------|------|
| **발급 URL** | https://platform.openai.com/api-keys |
| **회원가입** | https://platform.openai.com/signup |
| **크레딧 확인** | https://platform.openai.com/settings/organization/billing/overview |
| **키 형태** | `sk-` 접두사 (예: `sk-proj-AbCdEfGhIj...`) |

**발급 절차:**
1. https://platform.openai.com/api-keys 접속 → OpenAI 로그인
2. **"+ Create new secret key"** 클릭
3. 키 이름: `day06-workshop` → **"Create secret key"** 클릭
4. `sk-...` 키 복사 → `.env`의 `OPENAI_API_KEY=` 뒤에 붙여넣기
5. **주의:** 키는 생성 시 한 번만 표시됩니다.

> **없으면:** Lab 0을 건너뛰고 폴백 데이터셋(`data/sft_train_fallback.jsonl` 등)으로 Lab 1/2 진행 가능. 비용 약 $0.5~$2.0 예상.

### CLI에 RunPod 키 등록

`.env` 사용 시 `setup_runpod.sh`가 자동 처리합니다. 수동으로 등록하려면:

```bash
runpodctl config --apiKey YOUR_API_KEY
```

> **참고:** 모든 RunPod pod 내부에는 pod 범위의 API 키가 이미 설정되어 있어, pod 안에서는 별도 인증 없이 `runpodctl` 명령을 사용할 수 있습니다.

---

## 4. GPU 조회

### 사용 가능한 GPU 목록

```bash
runpodctl gpu list --output=table
```

### 재고 없는 GPU 포함

```bash
runpodctl gpu list --include-unavailable --output=table
```

### Day-06 권장 GPU (24GB VRAM)

| GPU ID (`--gpu-id`에 사용) | VRAM | 특징 |
|---|---|---|
| `NVIDIA GeForce RTX 3090` | 24 GB | 가장 저렴, Community cloud |
| `NVIDIA GeForce RTX 4090` | 24 GB | 가장 빠름, Ada Lovelace |
| `NVIDIA RTX A5000` | 24 GB | Secure + Community |
| `NVIDIA L4` | 24 GB | 데이터센터급, 가장 안정적 |

---

## 5. Pod 관리

### 5-1. Pod 생성

**템플릿 기반 (권장):**

```bash
runpodctl pod create \
  --template-id runpod-torch-v21 \
  --gpu-id "NVIDIA GeForce RTX 4090"
```

**이미지 직접 지정:**

```bash
runpodctl pod create \
  --name my-finetune \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 1 \
  --volume-in-gb 50 \
  --container-disk-in-gb 30 \
  --volume-mount-path /workspace \
  --ports "8888/http,22/tcp" \
  --cloud-type SECURE \
  --ssh
```

**환경변수 주입:**

```bash
runpodctl pod create \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --gpu-id "NVIDIA L4" \
  --env '{"HF_TOKEN":"hf_xxx","OPENAI_API_KEY":"sk-xxx"}'
```

**네트워크 볼륨 연결:**

```bash
runpodctl pod create \
  --image runpod/pytorch:latest \
  --gpu-id "NVIDIA RTX A5000" \
  --network-volume-id <volume-id>
```

### 주요 플래그 레퍼런스

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--name` | string | "" | Pod 이름 |
| `--template-id` | string | "" | 템플릿 ID (`runpodctl template search`로 검색) |
| `--image` | string | "" | Docker 이미지 (템플릿 없으면 필수) |
| `--gpu-id` | string | "" | GPU 타입 (`runpodctl gpu list`에서 확인) |
| `--gpu-count` | int | 1 | GPU 수 |
| `--volume-in-gb` | int | 0 | 영속 볼륨 크기 (GB) |
| `--container-disk-in-gb` | int | 20 | 컨테이너 디스크 크기 (GB) |
| `--volume-mount-path` | string | /workspace | 볼륨 마운트 경로 |
| `--ports` | string | "" | 포트 노출 (예: `8888/http,22/tcp`) |
| `--env` | string | "" | 환경변수 JSON (예: `'{"KEY":"value"}'`) |
| `--cloud-type` | string | SECURE | `SECURE` 또는 `COMMUNITY` |
| `--ssh` | bool | true | SSH 활성화 |
| `--network-volume-id` | string | "" | 네트워크 볼륨 ID |
| `--data-center-ids` | string | "" | 데이터센터 지정 (예: `US-TX-3,US-GA-1`) |

> `--template-id` 또는 `--image` 중 하나는 반드시 지정해야 합니다.

### 5-2. Pod 목록 조회

```bash
# 실행 중인 pod만 (v2.1.0부터 기본값)
runpodctl pod list

# 모든 pod (중지/종료 포함)
runpodctl pod list --all

# 필터링
runpodctl pod list --status RUNNING
runpodctl pod list --name my-finetune
runpodctl pod list --since 1h          # 최근 1시간 내 생성

# 테이블 형식
runpodctl pod list --output=table
```

### 5-3. Pod 상세 정보

```bash
runpodctl pod get <pod-id>
```

### 5-4. Pod 시작 / 중지 / 삭제

```bash
runpodctl pod start <pod-id>     # 중지된 pod 시작
runpodctl pod stop <pod-id>      # pod 중지 (볼륨 유지)
runpodctl pod restart <pod-id>   # 재시작
runpodctl pod delete <pod-id>    # 영구 삭제
```

> **주의:** 네트워크 볼륨이 연결된 pod는 stop이 불가하며 delete만 가능합니다.

---

## 6. SSH 접속

### SSH 키 관리

```bash
# 키 목록
runpodctl ssh list-keys

# 새 키 생성 (자동으로 ~/.runpod/ssh/에 저장)
runpodctl ssh add-key

# 기존 키 등록
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
```

### SSH 접속 문자열 확인

```bash
runpodctl ssh connect              # 모든 pod
runpodctl ssh connect <pod-id>     # 특정 pod
```

출력 예:
```
ssh root@194.26.196.6 -p 43201
```

### 실제 접속

```bash
ssh root@194.26.196.6 -p 43201 -i ~/.runpod/ssh/RunPod-Key
```

### Jupyter 터널링

```bash
# pod 생성 시 8888 포트 노출 필수: --ports "8888/http,22/tcp"
ssh -L 8888:localhost:8888 root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 -N
# 브라우저에서 http://localhost:8888 접속
```

---

## 7. 파일 전송

### 방법 1: runpodctl send/receive (소용량)

```bash
# 보내는 쪽
runpodctl send data.tar.gz
# 출력: code is: 8338-galileo-collect-fidel

# 받는 쪽
runpodctl receive 8338-galileo-collect-fidel
```

> **주의:** 대용량 파일(수백 MB 이상)에서 90% 지점 stall 현상 보고됨. 대용량은 rsync 사용 권장.

### 방법 2: SCP (중간 크기)

```bash
# 업로드
scp -P <PORT> -i ~/.ssh/id_ed25519 ./local_file.py root@<IP>:/workspace/

# 다운로드
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/output.pt ./
```

### 방법 3: rsync (대용량 — 권장)

```bash
# 업로드 (디렉토리)
rsync -avz -e "ssh -p <PORT>" ./my_project/ root@<IP>:/workspace/my_project/

# 다운로드 (체크포인트)
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/checkpoints/ ./checkpoints/
```

### 전송 방법 비교

| 방법 | 명령 | 장점 | 단점 |
|------|------|------|------|
| `runpodctl send` | `runpodctl send file.tar.gz` | API 키 불필요, 간편 | 대용량 stall, 단일 파일만 |
| SCP | `scp -P port file root@ip:/path` | 표준적, 안정 | 대용량 timeout 가능 |
| rsync | `rsync -avz -e "ssh -p port" ...` | 재개 가능, 대용량 안정 | Linux/WSL 필요 |

---

## 8. 템플릿 관리

### 템플릿 검색

```bash
runpodctl template search pytorch              # 이름으로 검색
runpodctl template search unsloth              # Unsloth 템플릿 검색
runpodctl template search pytorch --type official  # 공식 템플릿만
```

### 템플릿 목록

```bash
runpodctl template list                    # 공식 + 커뮤니티
runpodctl template list --type user        # 내가 만든 것만
runpodctl template list --all              # 전부
```

### 템플릿 상세

```bash
runpodctl template get <template-id>
```

---

## 9. 네트워크 볼륨

### 볼륨 생성

```bash
runpodctl network-volume create \
  --name model-weights \
  --size 100 \
  --data-center-id US-TX-3
```

> `--name`, `--size`, `--data-center-id` 세 가지 모두 필수.

### 볼륨 목록 / 상세 / 삭제

```bash
runpodctl network-volume list --output=table
runpodctl network-volume get <volume-id>
runpodctl network-volume delete <volume-id>
```

### 가격

- 1TB 미만: **$0.07/GB/월**
- 1TB 이상: **$0.05/GB/월**

---

## 10. 비용 확인

```bash
# Pod 비용 (일별)
runpodctl billing pods --bucket-size day --output=table

# 특정 기간
runpodctl billing pods \
  --start-time 2026-04-01T00:00:00Z \
  --end-time 2026-04-06T00:00:00Z

# GPU별 그룹핑
runpodctl billing pods --grouping gpuId

# 네트워크 볼륨 비용
runpodctl billing network-volume --bucket-size month
```

---

## 11. Serverless 엔드포인트

```bash
# 목록
runpodctl serverless list

# 생성
runpodctl serverless create \
  --template-id <template-id> \
  --name my-endpoint \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --workers-min 0 \
  --workers-max 3

# 삭제
runpodctl serverless delete <endpoint-id>
```

---

## 12. 웹 UI vs CLI 비교

| 기능 | runpodctl CLI | 웹 UI |
|------|--------------|-------|
| Pod 생성 | `pod create` (스크립트화 가능) | GUI 클릭 |
| 대량 Pod 생성 | `create pods --podCount N` | 불가 |
| Pod 필터링 | `--status`, `--since`, `--name` | 제한적 |
| SSH 키 관리 | `ssh add-key` | 가능 |
| 파일 전송 (croc) | `send`/`receive` | 불가 |
| 비용 분석 (기간별) | `billing --bucket-size` | 제한적 |
| 자동화/스크립팅 | JSON 출력, CI/CD 연동 | 불가 |
| 실시간 터미널 | 불가 (SSH 필요) | 웹 터미널 |
| GPU 가격 시각화 | 불가 | 마켓플레이스 |

---

## 13. runpodctl vs runpod Python SDK

| 비교 | runpodctl (Go CLI) | runpod Python SDK |
|------|-------------------|-------------------|
| 언어 | Go 바이너리 (런타임 불필요) | Python (pip install) |
| 주요 용도 | Shell 스크립트, CI/CD, 빠른 작업 | Python 코드 내 오케스트레이션 |
| Pod 관리 | 전체 지원 | 전체 지원 |
| 파일 전송 | `send`/`receive` 내장 | 미지원 (SSH 별도) |
| SSH 키 관리 | 지원 | 미지원 |
| Pod 사전 설치 | **예** (별도 설치 불필요) | 아니오 (pip install 필요) |
| Serverless | 기본 관리 | 주력 기능 (워커 개발) |

**선택 기준:**
- Pod 외부에서 인프라 관리 → `runpodctl`
- Python 코드 안에서 프로그래밍적 제어 → `runpod` SDK
- 워크숍 환경 셋업 자동화 → `runpodctl` (스크립트화 용이)

---

## 14. 알려진 제약사항

1. `runpodctl send` — 대용량 파일(수백 MB+)에서 90% stall 발생 가능 → rsync 사용
2. 로그 스트리밍 — `logs`/`exec` 명령 없음 → SSH 접속 후 확인
3. Spot 인스턴스 — CLI에서 미지원 → 웹 UI 또는 Python SDK 사용
4. 설치 — sudo 필요 (`/usr/bin/`에 설치)
5. SSH 프록시 — `ssh.runpod.io` 경유 시 SCP/rsync 불가 → 퍼블릭 IP 사용
6. 네트워크 볼륨 Pod — stop 불가, delete만 가능

---

## 15. 참고 자료

- [runpodctl GitHub](https://github.com/runpod/runpodctl) — 소스코드, 릴리즈
- [RunPod CLI 문서](https://docs.runpod.io/runpodctl/overview) — 공식 설치/설정
- [RunPod 파일 전송](https://docs.runpod.io/runpodctl/transfer-files) — SCP/rsync/croc 비교
- [RunPod SSH 설정](https://docs.runpod.io/pods/configuration/use-ssh) — SSH 키 등록
- [RunPod GPU 타입](https://docs.runpod.io/references/gpu-types) — 전체 GPU 목록
- [RunPod 가격](https://docs.runpod.io/pods/pricing) — 과금 모델
