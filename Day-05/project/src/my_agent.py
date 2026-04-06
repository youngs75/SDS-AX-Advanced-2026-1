"""프로젝트 기본 평가 대상 agent 모듈.

이 파일은 참가자들이 앞으로 따라야 할 최소 계약(contract) 예시입니다.

중요 규칙:
1. 모든 agent 모듈은 전역 `agent` 변수를 export 해야 합니다.
2. `agent`는 `create_deep_agent(...)`로 생성된 객체여야 합니다.
3. 평가기는 `importlib`로 모듈을 import한 뒤, 이 전역 `agent`를 찾아 Step 5에서 사용합니다.
4. 따라서 파일을 import하는 순간 agent가 정상적으로 생성될 수 있어야 합니다.

이 파일의 역할:
- 직접 질문을 던져 보는 데모 실행도 가능
- `run_pipeline.py --step 5`의 기본 평가 대상 모듈
- 다른 참가자가 새 agent를 만들 때 따라야 하는 기준 템플릿
"""

from __future__ import annotations

from deepagents import create_deep_agent
from langchain.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.llm.openrouter import get_langchain_chat_model
from src.observability.langfuse import build_langchain_config
from src.settings import get_settings

settings = get_settings()

SYSTEM_PROMPT = """\
You are the default project evaluation agent.

Rules:
1. For factual, comparison, or information-seeking questions, call `search_web` before answering.
2. Use the search result as supporting evidence, then provide the final answer.
3. If the question is purely conversational and does not require factual lookup, you may answer directly.
4. Keep the final answer concise and task-focused.
"""


@tool
def search_web(query: str) -> str:
    """웹 검색 도구 예시.

    실제 서비스에서는 진짜 검색 도구로 교체하면 됩니다.
    지금은 평가 파이프라인 연결을 보여 주기 위한 간단한 예시 도구입니다.
    """
    return f"검색 결과: '{query}'에 대한 정보입니다."


agent_tools = [search_web]


def build_agent():
    """전역 `agent` 생성을 담당하는 작은 팩토리 함수.

    전역 `agent` 변수는 평가기가 직접 사용합니다.
    팩토리 함수를 따로 두는 이유는:
    - 생성 로직을 읽기 쉽게 분리하고
    - 나중에 다른 모듈에서도 재사용하기 쉽도록 하기 위해서입니다.
    """
    llm = get_langchain_chat_model(settings=settings)
    return create_deep_agent(
        model=llm,
        tools=agent_tools,
        system_prompt=SYSTEM_PROMPT,
        name="project_default_agent",
    )


# 평가기는 이 전역 변수만 찾습니다.
agent = build_agent()


if __name__ == "__main__":
    # 이 블록은 개발자가 agent를 단독으로 실행해 보는 데모용입니다.
    # 평가 파이프라인은 이 블록을 사용하지 않고, 위의 전역 `agent`만 import 해서 씁니다.
    user_prompt = "What is your task?"

    langfuse_handler = CallbackHandler()
    config = build_langchain_config(
        user_id="demo-user",
        session_id="demo-session",
        callbacks=[langfuse_handler],
        tags=["demo", "my-agent"],
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config=config,
    )

    get_client().flush()

    ai_message = response["messages"][-1]
    print(ai_message.content)
