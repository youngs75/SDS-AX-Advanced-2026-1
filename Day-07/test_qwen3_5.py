"""Qwen3.5 9B 모델 테스트 - OpenAI Compatible Server"""

import json
import sys
from datetime import datetime
from openai import OpenAI

BASE_URL = "https://qdlc9sc32po6bg-8000.proxy.runpod.net/v1"
client = OpenAI(base_url=BASE_URL, api_key="sk-qdlc9sc32po6bg")

output_lines = []

def log(msg=""):
    print(msg)
    output_lines.append(msg)

def separator(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}\n")


def test_chat():
    """1. Chat Completion 테스트"""
    separator("1. Chat Completion 테스트")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer in Korean."},
        {"role": "user", "content": "인공지능의 미래에 대해 3문장으로 설명해주세요."}
    ]

    log(f"[Request] messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
    log("")

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,  # 서버에서 사용 가능한 첫 번째 모델
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )

    log(f"[Response]")
    log(f"  Model: {response.model}")
    log(f"  Usage: prompt_tokens={response.usage.prompt_tokens}, completion_tokens={response.usage.completion_tokens}")
    log(f"  Content:\n{response.choices[0].message.content}")
    return response


def test_tool_calling():
    """2. Tool Calling 테스트"""
    separator("2. Tool Calling 테스트")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name (e.g. Seoul, Tokyo)"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a math calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "서울의 현재 날씨를 알려주고, 123 * 456을 계산해줘."}
    ]

    log(f"[Request] messages: {json.dumps(messages, ensure_ascii=False)}")
    log(f"[Tools] {json.dumps([t['function']['name'] for t in tools])}")
    log("")

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
    )

    msg = response.choices[0].message
    log(f"[Response]")
    log(f"  Finish Reason: {response.choices[0].finish_reason}")

    if msg.tool_calls:
        log(f"  Tool Calls ({len(msg.tool_calls)}):")
        for tc in msg.tool_calls:
            log(f"    - Function: {tc.function.name}")
            log(f"      Arguments: {tc.function.arguments}")

        # Tool call 결과를 넣고 후속 응답 받기
        messages.append(msg)
        for tc in msg.tool_calls:
            if tc.function.name == "get_weather":
                result = json.dumps({"city": "Seoul", "temperature": 15, "condition": "맑음", "unit": "celsius"})
            elif tc.function.name == "calculate":
                result = json.dumps({"result": 56088})
            else:
                result = json.dumps({"error": "unknown function"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        log(f"\n  [Follow-up with tool results]")
        response2 = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=messages,
            tools=tools,
            temperature=0.0,
        )
        log(f"  Final Response:\n{response2.choices[0].message.content}")
    else:
        log(f"  Content (no tool calls):\n{msg.content}")

    return response


def test_reasoning():
    """3. Reasoning (Thinking) 테스트"""
    separator("3. Reasoning (Thinking) 테스트")

    # Qwen3 모델은 /think 태그로 reasoning을 지원
    messages = [
        {"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think step by step."}
    ]

    log(f"[Request] messages: {json.dumps(messages, ensure_ascii=False)}")
    log("")

    # 방법 1: 일반 호출 (모델이 자체적으로 thinking 수행)
    log("[방법 1] 일반 호출 - step by step 프롬프팅")
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
    )
    log(f"  Content:\n{response.choices[0].message.content}")

    # 방법 2: chat_template_kwargs로 enable_thinking 시도
    log(f"\n[방법 2] enable_thinking 파라미터 사용 시도")
    try:
        messages2 = [
            {"role": "user", "content": "What is 28 * 37? Show your reasoning."}
        ]
        response2 = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=messages2,
            temperature=0.6,  # Qwen3 thinking requires temp > 0
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        msg = response2.choices[0].message
        log(f"  Content:\n{msg.content}")

        # reasoning_content 필드 확인
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            log(f"\n  Reasoning Content:\n{msg.reasoning_content}")
        else:
            log(f"\n  (reasoning_content 필드 없음 - 응답 본문에 thinking이 포함될 수 있음)")
    except Exception as e:
        log(f"  Error: {e}")

    return response


if __name__ == "__main__":
    log(f"Qwen3.5 9B 모델 테스트 결과")
    log(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"서버: {BASE_URL}")

    # 모델 정보 확인
    separator("0. 모델 정보 확인")
    try:
        models = client.models.list()
        for m in models.data:
            log(f"  Available model: {m.id}")
    except Exception as e:
        log(f"  모델 목록 조회 실패: {e}")
        sys.exit(1)

    # 테스트 실행
    try:
        test_chat()
    except Exception as e:
        log(f"  Chat 테스트 실패: {e}")

    try:
        test_tool_calling()
    except Exception as e:
        log(f"  Tool Calling 테스트 실패: {e}")

    try:
        test_reasoning()
    except Exception as e:
        log(f"  Reasoning 테스트 실패: {e}")

    separator("테스트 완료")

    # 결과를 파일로 저장
    output_path = "/home/sds/projects/SDS-AX-Advanced-2026-1/Day-08/test_qwen3_5_result.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log(f"\n결과가 {output_path} 에 저장되었습니다.")
