"""Qwen3.5 9B - H100 최대 max_tokens OOM 테스트"""

import time
from openai import OpenAI

BASE_URL = "https://skvc5zkikltaag-8000.proxy.runpod.net/v1"
client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

model_id = client.models.list().data[0].id
print(f"Model: {model_id}")

# 서버의 max_model_len 확인 시도 (큰 값으로 요청해서 에러 메시지에서 확인)
# H100 80GB, Qwen3.5-9B, gpu_memory_utilization=0.95
# 서버가 --max-model-len 120000으로 설정되어 있으므로, 실제 생성 가능한 max_tokens를 테스트

# 긴 출력을 유도하는 프롬프트
PROMPT = "Write a very long, detailed essay about the complete history of computer science from 1940 to 2025. Include every major milestone, invention, and breakthrough. Be as detailed as possible."

results = []

# 테스트할 max_tokens 값들 (점진적으로 증가)
test_values = [4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304, 110000, 120000]

for max_tok in test_values:
    print(f"\n{'='*50}")
    print(f"Testing max_tokens={max_tok}...")
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tok,
            temperature=0.7,
        )
        elapsed = time.time() - start
        usage = response.usage
        finish = response.choices[0].finish_reason
        actual_tokens = usage.completion_tokens
        total = usage.prompt_tokens + usage.completion_tokens
        tps = actual_tokens / elapsed if elapsed > 0 else 0

        status = "SUCCESS"
        print(f"  Status: {status}")
        print(f"  Finish reason: {finish}")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {actual_tokens}")
        print(f"  Total tokens: {total}")
        print(f"  Time: {elapsed:.1f}s | Speed: {tps:.1f} tok/s")
        results.append((max_tok, status, actual_tokens, total, finish, f"{elapsed:.1f}s", f"{tps:.1f}"))

    except Exception as e:
        elapsed = time.time() - start
        err_msg = str(e)[:200]
        status = "FAILED"
        print(f"  Status: {status}")
        print(f"  Error: {err_msg}")
        results.append((max_tok, status, 0, 0, "error", f"{elapsed:.1f}s", err_msg))

        # OOM이면 더 큰 값 테스트 중단
        if "OOM" in str(e).upper() or "out of memory" in str(e).lower() or "CUDA" in str(e):
            print("  >>> OOM detected, stopping further tests.")
            break

# 결과 요약
print(f"\n{'='*70}")
print("결과 요약")
print(f"{'='*70}")
print(f"{'max_tokens':>12} | {'Status':>8} | {'Actual':>8} | {'Total':>8} | {'Finish':>12} | {'Time':>8} | {'tok/s':>8}")
print("-" * 70)
for r in results:
    if r[1] == "SUCCESS":
        print(f"{r[0]:>12} | {r[1]:>8} | {r[2]:>8} | {r[3]:>8} | {r[4]:>12} | {r[5]:>8} | {r[6]:>8}")
    else:
        print(f"{r[0]:>12} | {r[1]:>8} | {'N/A':>8} | {'N/A':>8} | {'error':>12} | {r[5]:>8} | {r[6][:30]}")

# 파일 저장
output_path = "/home/sds/projects/SDS-AX-Advanced-2026-1/Day-08/test_max_tokens_result.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Qwen3.5 9B - H100 최대 max_tokens OOM 테스트 결과\n")
    f.write(f"서버: {BASE_URL}\n")
    f.write(f"{'='*70}\n")
    f.write(f"{'max_tokens':>12} | {'Status':>8} | {'Actual':>8} | {'Total':>8} | {'Finish':>12} | {'Time':>8} | {'tok/s':>8}\n")
    f.write("-" * 70 + "\n")
    for r in results:
        if r[1] == "SUCCESS":
            f.write(f"{r[0]:>12} | {r[1]:>8} | {r[2]:>8} | {r[3]:>8} | {r[4]:>12} | {r[5]:>8} | {r[6]:>8}\n")
        else:
            f.write(f"{r[0]:>12} | {r[1]:>8} | {'N/A':>8} | {'N/A':>8} | {'error':>12} | {r[5]:>8} | {r[6][:60]}\n")
    f.write(f"\n최대 성공 max_tokens: {max(r[0] for r in results if r[1]=='SUCCESS') if any(r[1]=='SUCCESS' for r in results) else 'N/A'}\n")

print(f"\n결과 저장: {output_path}")
