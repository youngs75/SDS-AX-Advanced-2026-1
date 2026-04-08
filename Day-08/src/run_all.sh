#!/bin/bash
# ============================================================
# 비동기 서브에이전트 데모 — 3개 프로세스 일괄 실행 스크립트
#
# 사용법:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# 종료:
#   Ctrl+C (오케스트레이터 종료 후 서버들도 자동 종료)
# ============================================================

cd "$(dirname "$0")"

cleanup() {
    echo ""
    echo "[run_all] 서버를 종료합니다..."
    kill $PID_COMPLEX $PID_SIMPLE 2>/dev/null
    wait $PID_COMPLEX $PID_SIMPLE 2>/dev/null
    echo "[run_all] 모든 프로세스가 종료되었습니다."
}
trap cleanup EXIT

# 1) 복잡한 작업 서버 (GLM, port 2024)
echo "[run_all] complex 서버 시작 (port 2024)..."
python task_server.py --name complex --port 2024 &
PID_COMPLEX=$!

# 2) 간단한 작업 서버 (Qwen, port 2025)
echo "[run_all] simple 서버 시작 (port 2025)..."
python task_server.py --name simple --port 2025 &
PID_SIMPLE=$!

# 서버 준비 대기 (느린 환경에서는 값을 늘려주세요)
echo "[run_all] 서버 준비 대기 (3초)..."
sleep 3

# 3) 오케스트레이터 (포그라운드)
echo "[run_all] 오케스트레이터 시작..."
echo ""
python _async_agent.py
