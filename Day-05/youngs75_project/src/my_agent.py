"""평가 대상 에이전트 — Coding Assistant.

youngs75_a2a의 CodingAssistantAgent를 래핑하여
DeepEval 평가 파이프라인에서 사용할 수 있는 형태로 제공한다.
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr

from src.observability.langfuse import build_langchain_config
from src.settings import get_settings

settings = get_settings()

# OpenRouter 기반 LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr(settings.openrouter_api_key),
    model=settings.openrouter_model_name,
)


def run_coding_agent(query: str) -> str:
    """Coding Agent를 실행하고 결과를 반환한다.

    DeepEval 평가에서 actual_output을 생성하기 위해 사용.
    """
    from youngs75_a2a.agents.coding_assistant import CodingAssistantAgent, CodingConfig

    config = CodingConfig(
        model_provider="openai",
        default_model=settings.openrouter_model_name,
        generation_model=settings.openrouter_model_name,
        verification_model=settings.openrouter_model_name,
    )

    # OpenRouter 호환을 위해 직접 모델 주입
    agent = CodingAssistantAgent.__new__(CodingAssistantAgent)
    agent._coding_config = config
    agent._explicit_model = llm
    agent._gen_model = llm
    agent._verify_model = llm
    agent._parse_model = llm

    from youngs75_a2a.agents.coding_assistant.schemas import CodingState

    agent.agent_config = config
    agent.model = llm
    agent.state_schema = CodingState
    agent.config_schema = None
    agent.input_state = None
    agent.output_state = None
    agent.checkpointer = None
    agent.store = None
    agent.agent_name = "CodingAssistantAgent"
    agent.debug = False
    from langgraph.types import RetryPolicy
    agent.retry_policy = RetryPolicy(max_attempts=2)
    agent.graph = None
    agent.build_graph()

    result = asyncio.run(agent.graph.ainvoke({
        "messages": [HumanMessage(content=query)],
        "iteration": 0,
        "max_iterations": 2,
    }))

    return result.get("generated_code", "")


if __name__ == "__main__":
    langfuse_handler = CallbackHandler()
    config = build_langchain_config(
        user_id="demo-user",
        session_id="coding-eval",
        callbacks=[langfuse_handler],
        tags=["coding-agent"],
    )

    response = run_coding_agent("파이썬으로 피보나치 함수를 작성해줘")
    print(response)

    get_client().flush()
