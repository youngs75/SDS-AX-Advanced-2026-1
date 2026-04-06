#!/usr/bin/env bash
# =============================================================================
# Day-06 RunPod 환경 원클릭 셋업 스크립트
# =============================================================================
#
# 사용법:
#   1. .env.example 을 복사하여 .env 를 만드세요:
#      cp .env.example .env
#   2. .env 파일에 API 키를 입력하세요.
#   3. 스크립트를 실행하세요:
#      chmod +x setup_runpod.sh
#      ./setup_runpod.sh
#
# API 키 발급처:
#   - RunPod:       https://www.runpod.io/console/user/settings  (API Keys)
#   - HuggingFace:  https://huggingface.co/settings/tokens       (Access Tokens)
#   - OpenAI:       https://platform.openai.com/api-keys         (Secret Keys)
#
# 이 스크립트가 수행하는 작업:
#   1. .env 파일 로드
#   2. runpodctl 설치 확인
#   3. API 키 설정
#   4. SSH 키 확인/생성
#   5. GPU 가용성 확인
#   6. Pod 생성 (H100 80GB GPU + PyTorch + 영속 볼륨)
#   7. Pod 준비 대기
#   8. SSH 접속 정보 출력
#
# =============================================================================

set -euo pipefail

# ---------- 색상 정의 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
hint()  { echo -e "${CYAN}  →${NC} $*"; }

# ---------- .env 파일 로드 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

echo ""
echo "============================================"
echo " Day-06 RunPod 환경 원클릭 셋업"
echo "============================================"
echo ""

if [ -f "$ENV_FILE" ]; then
  info ".env 파일을 로드합니다: $ENV_FILE"
  # .env 파일에서 빈 줄과 주석(#)을 제외하고 export
  while IFS='=' read -r key value; do
    # 빈 줄, 주석, 값이 없는 줄 건너뛰기
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
    # 키에서 앞뒤 공백 제거
    key=$(echo "$key" | xargs)
    # 값에서 앞뒤 공백과 따옴표 제거
    value=$(echo "$value" | xargs | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    # 값이 비어있으면 건너뛰기
    [ -z "$value" ] && continue
    export "$key=$value"
  done < "$ENV_FILE"
  ok ".env 로드 완료"
else
  warn ".env 파일이 없습니다. 환경변수에서 키를 읽습니다."
  echo ""
  if [ ! -f "${SCRIPT_DIR}/.env.example" ]; then
    warn ".env.example 파일도 없습니다."
  else
    echo "  .env 파일 생성 방법:"
    hint "cp .env.example .env"
    hint "vi .env  # 또는 선호하는 편집기로 API 키 입력"
    echo ""
  fi
fi

# ---------- 기본 설정 (.env에서 오버라이드 가능) ----------
POD_NAME="day06-finetune-$(date +%m%d-%H%M)"
DOCKER_IMAGE="${DOCKER_IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
VOLUME_GB="${VOLUME_GB:-50}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
VOLUME_MOUNT="/workspace"
PORTS="8888/http,22/tcp"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"

# GPU 우선순위 (.env에서 GPU_ID를 지정하면 그것만 사용)
# H100 80GB를 기본으로, A100 80GB/40GB를 폴백으로 사용
GPU_PRIORITY=(
  "NVIDIA H100 80GB HBM3"
  "NVIDIA H100 SXM"
  "NVIDIA A100 80GB PCIe"
  "NVIDIA A100-SXM4-80GB"
  "NVIDIA A100-SXM4-40GB"
  "NVIDIA L40S"
  "NVIDIA GeForce RTX 4090"
  "NVIDIA L4"
  "NVIDIA RTX A5000"
  "NVIDIA GeForce RTX 3090"
)

# ---------- Step 0: API 키 검증 ----------
info "API 키 상태 확인..."
echo ""

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  err "RUNPOD_API_KEY가 설정되지 않았습니다."
  echo ""
  echo "  [필수] RunPod API Key 발급:"
  hint "1. https://www.runpod.io/console/user/settings 접속"
  hint "2. 좌측 메뉴 'API Keys' → 'Create API Key' 클릭"
  hint "3. 생성된 키를 .env 파일의 RUNPOD_API_KEY= 뒤에 붙여넣기"
  echo ""
  hint "계정이 없으면: https://www.runpod.io/console/signup"
  hint "크레딧 충전:   https://www.runpod.io/console/user/billing (최소 \$10 권장)"
  echo ""
  exit 1
fi
echo -e "  RUNPOD_API_KEY  ${GREEN}✓ 설정됨${NC}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo -e "  HF_TOKEN        ${YELLOW}✗ 미설정${NC} (모델 다운로드 제한 가능)"
  echo ""
  echo "  [권장] HuggingFace Token 발급:"
  hint "1. https://huggingface.co/settings/tokens 접속"
  hint "2. 'Create new token' → Type: Read → 생성"
  hint "3. 'hf_...' 토큰을 .env 파일의 HF_TOKEN= 뒤에 붙여넣기"
  echo ""
else
  echo -e "  HF_TOKEN        ${GREEN}✓ 설정됨${NC}"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo -e "  OPENAI_API_KEY  ${YELLOW}✗ 미설정${NC} (Lab 0 건너뜀 → 폴백 데이터 사용)"
  echo ""
  echo "  [선택] OpenAI API Key 발급 (Lab 0 합성 데이터 생성에 필요):"
  hint "1. https://platform.openai.com/api-keys 접속"
  hint "2. '+ Create new secret key' 클릭"
  hint "3. 'sk-...' 키를 .env 파일의 OPENAI_API_KEY= 뒤에 붙여넣기"
  hint "크레딧 확인: https://platform.openai.com/settings/organization/billing/overview"
  echo ""
else
  echo -e "  OPENAI_API_KEY  ${GREEN}✓ 설정됨${NC}"
fi

echo ""

# ---------- Step 1: runpodctl 설치 확인 ----------
info "Step 1/7: runpodctl 설치 확인..."

if command -v runpodctl &>/dev/null; then
  INSTALLED_VERSION=$(runpodctl version 2>/dev/null || echo "unknown")
  ok "runpodctl 설치됨: $INSTALLED_VERSION"
else
  warn "runpodctl이 설치되어 있지 않습니다. 설치를 시작합니다..."

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &>/dev/null; then
      info "Homebrew로 설치 중..."
      brew install runpod/runpodctl/runpodctl
    else
      info "wget으로 설치 중 (sudo 필요)..."
      wget -qO- cli.runpod.net | sudo bash
    fi
  else
    # Linux / WSL
    info "설치 중 (sudo 필요)..."
    wget -qO- cli.runpod.net | sudo bash
  fi

  if command -v runpodctl &>/dev/null; then
    ok "runpodctl 설치 완료: $(runpodctl version 2>/dev/null)"
  else
    err "runpodctl 설치 실패. 수동 설치를 시도하세요:"
    echo "  https://github.com/runpod/runpodctl/releases"
    exit 1
  fi
fi

# ---------- Step 2: API 키 설정 ----------
info "Step 2/7: API 키 설정..."
runpodctl config --apiKey "$RUNPOD_API_KEY"
ok "API 키 설정 완료"

# ---------- Step 3: SSH 키 확인 ----------
info "Step 3/7: SSH 키 확인..."

SSH_KEYS=$(runpodctl ssh list-keys 2>/dev/null || echo "")
if echo "$SSH_KEYS" | grep -q "fingerprint\|key_id\|name"; then
  ok "SSH 키가 이미 등록되어 있습니다."
else
  info "SSH 키를 생성합니다..."
  runpodctl ssh add-key 2>/dev/null || true
  ok "SSH 키 생성 완료 (저장 위치: ~/.runpod/ssh/)"
fi

# ---------- Step 4: GPU 가용성 확인 ----------
info "Step 4/7: GPU 가용성 확인 (H100 80GB 우선)..."

echo ""
echo "  현재 사용 가능한 GPU (H100 > A100 > 24GB 순):"
echo "  ─────────────────────────────────"

GPU_LIST=$(runpodctl gpu list 2>/dev/null || echo "")
SELECTED_GPU=""

for gpu in "${GPU_PRIORITY[@]}"; do
  if echo "$GPU_LIST" | grep -q "$gpu"; then
    echo -e "  ${GREEN}✓${NC} $gpu — 사용 가능"
    if [ -z "$SELECTED_GPU" ]; then
      SELECTED_GPU="$gpu"
    fi
  else
    echo -e "  ${RED}✗${NC} $gpu — 미확인/재고 없음"
  fi
done

echo ""

if [ -z "$SELECTED_GPU" ]; then
  warn "우선순위 GPU를 찾지 못했습니다. 전체 목록을 확인합니다..."
  runpodctl gpu list --output=table 2>/dev/null || true
  echo ""
  read -rp "사용할 GPU ID를 직접 입력하세요: " SELECTED_GPU
  if [ -z "$SELECTED_GPU" ]; then
    err "GPU를 선택하지 않았습니다."
    exit 1
  fi
fi

ok "선택된 GPU: $SELECTED_GPU"

# ---------- Step 5: Pod 생성 ----------
info "Step 5/7: Pod 생성 중..."
echo ""
echo "  Pod 이름:    $POD_NAME"
echo "  GPU:         $SELECTED_GPU"
echo "  이미지:      $DOCKER_IMAGE"
echo "  볼륨:        ${VOLUME_GB}GB (마운트: $VOLUME_MOUNT)"
echo "  디스크:      ${CONTAINER_DISK_GB}GB"
echo "  Cloud:       $CLOUD_TYPE"
echo ""

# 환경변수 JSON 생성
ENV_JSON='{"HF_HOME":"/workspace/hf_cache"'
if [ -n "${HF_TOKEN:-}" ]; then
  ENV_JSON="${ENV_JSON},\"HF_TOKEN\":\"${HF_TOKEN}\""
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
  ENV_JSON="${ENV_JSON},\"OPENAI_API_KEY\":\"${OPENAI_API_KEY}\""
fi
ENV_JSON="${ENV_JSON}}"

POD_RESULT=$(runpodctl pod create \
  --name "$POD_NAME" \
  --image "$DOCKER_IMAGE" \
  --gpu-id "$SELECTED_GPU" \
  --gpu-count 1 \
  --volume-in-gb "$VOLUME_GB" \
  --container-disk-in-gb "$CONTAINER_DISK_GB" \
  --volume-mount-path "$VOLUME_MOUNT" \
  --ports "$PORTS" \
  --env "$ENV_JSON" \
  --cloud-type "$CLOUD_TYPE" \
  --ssh 2>&1) || {
    err "Pod 생성 실패:"
    echo "$POD_RESULT"
    exit 1
  }

# Pod ID 추출
POD_ID=$(echo "$POD_RESULT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('id', ''))
except:
    # JSON 파싱 실패 시 텍스트에서 ID 추출 시도
    import re
    text = sys.stdin.read() if not isinstance(data, str) else str(data)
    match = re.search(r'[a-z0-9]{20,}', text)
    print(match.group(0) if match else '')
" 2>/dev/null || echo "")

if [ -z "$POD_ID" ]; then
  warn "Pod ID를 자동 추출하지 못했습니다. 생성 결과:"
  echo "$POD_RESULT"
  echo ""
  read -rp "Pod ID를 직접 입력하세요: " POD_ID
fi

ok "Pod 생성됨: $POD_ID"

# ---------- Step 6: Pod 준비 대기 ----------
info "Step 6/7: Pod 준비 대기 중..."

MAX_WAIT=60  # 최대 5분 (10초 × 30)
for i in $(seq 1 $MAX_WAIT); do
  STATUS=$(runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('desiredStatus', data.get('status', 'UNKNOWN')))
except:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")

  if [ "$STATUS" = "RUNNING" ]; then
    ok "Pod 준비 완료! (${i}0초 소요)"
    break
  fi

  printf "\r  [%d/%d] 상태: %s ... 대기 중" "$i" "$MAX_WAIT" "$STATUS"
  sleep 10
done

echo ""

if [ "$STATUS" != "RUNNING" ]; then
  warn "Pod이 아직 RUNNING 상태가 아닙니다 (현재: $STATUS)."
  warn "잠시 후 다시 확인하세요: runpodctl pod get $POD_ID"
fi

# ---------- Step 7: SSH 접속 정보 ----------
info "Step 7/7: SSH 접속 정보..."
echo ""

SSH_INFO=$(runpodctl ssh connect "$POD_ID" 2>/dev/null || echo "")

if [ -n "$SSH_INFO" ]; then
  echo "  ┌─────────────────────────────────────────────┐"
  echo "  │ SSH 접속 명령:                               │"
  echo "  │                                              │"
  echo "  │  $SSH_INFO"
  echo "  │                                              │"
  echo "  └─────────────────────────────────────────────┘"
else
  warn "SSH 접속 정보를 가져오지 못했습니다."
  echo "  수동 확인: runpodctl ssh connect $POD_ID"
fi

# ---------- 완료 요약 ----------
echo ""
echo "============================================"
echo " 셋업 완료!"
echo "============================================"
echo ""
echo "  Pod ID:      $POD_ID"
echo "  Pod 이름:    $POD_NAME"
echo "  GPU:         $SELECTED_GPU"
echo "  볼륨:        ${VOLUME_GB}GB at $VOLUME_MOUNT"
echo ""
echo "  다음 단계:"
echo "  ─────────────────────────────────"
echo "  1. SSH로 접속하세요"
echo "  2. Pod 안에서 Unsloth 설치:"
echo "     pip install unsloth"
echo "  3. Jupyter 시작:"
echo "     jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"
echo "  4. 브라우저에서 Jupyter 접속"
echo ""
echo "  유용한 명령어:"
echo "  ─────────────────────────────────"
echo "  runpodctl pod list                  # Pod 상태 확인"
echo "  runpodctl pod stop $POD_ID     # Pod 중지 (볼륨 유지)"
echo "  runpodctl pod delete $POD_ID   # Pod 삭제"
echo "  runpodctl billing pods --bucket-size day  # 비용 확인"
echo ""

# Pod 정보를 파일로 저장
POD_INFO_FILE="pod_info_${POD_NAME}.txt"
cat > "$POD_INFO_FILE" <<EOF
# Day-06 RunPod Pod 정보
# 생성 시각: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
POD_ID=$POD_ID
POD_NAME=$POD_NAME
GPU=$SELECTED_GPU
SSH_CMD=$SSH_INFO

# Pod 관리 명령:
# runpodctl pod get $POD_ID
# runpodctl pod stop $POD_ID
# runpodctl pod delete $POD_ID
EOF

ok "Pod 정보가 $POD_INFO_FILE 에 저장되었습니다."
echo ""
