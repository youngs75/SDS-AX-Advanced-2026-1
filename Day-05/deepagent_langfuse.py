"""Deep Agents + Langfuse 통합 예제.

create_deep_agent로 간단한 에이전트를 만들고,
Langfuse 콜백 핸들러를 통해 로컬 Langfuse에서 트레이스를 확인한다.

실행:
    cd Day-05
    export $(grep -v '^#' .env | xargs)
    python deepagent_langfuse.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr


# ── 도구 정의 ────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 3 * 4'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def get_current_date() -> str:
    """현재 날짜와 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── LLM 설정 ─────────────────────────────────────────

llm = ChatOpenAI(
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    model="gpt-4.1-mini",
    temperature=0.1,
)

# ── 에이전트 생성 ─────────────────────────────────────

agent = create_deep_agent(
    model=llm,
    tools=[calculate, get_current_date],
    system_prompt="당신은 계산과 날짜 확인을 도와주는 어시스턴트입니다. 한국어로 답변하세요.",
)


# ── 실행 ──────────────────────────────────────────────

if __name__ == "__main__":
    # Langfuse 콜백 핸들러 설정
    # v4: 환경변수 LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL에서 자동 로드
    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": "youngs75",
            "langfuse_session_id": "day05-demo",
            "langfuse_tags": ["demo", "deep-agents", "day05"],
        },
    }

    # 질문 목록
    questions = [
        "오늘 날짜가 뭐야?",
        "123 * 456 + 789 계산해줘",
        "2의 10승은 얼마야?",
    ]

    for q in questions:
        print(f"\n🧑 {q}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config=config,
        )
        ai_message = response["messages"][-1]
        print(f"🤖 {ai_message.content}")

    # Langfuse로 트레이스 전송 완료 대기
    lf = get_client()
    lf.flush()
    lf.shutdown()

    print(f"\n✅ 완료! Langfuse에서 트레이스를 확인하세요:")
    print(f"   → {os.environ.get('LANGFUSE_BASE_URL', 'http://localhost:3100')}")
