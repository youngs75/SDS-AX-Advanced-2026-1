from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr

from src.observability.langfuse import build_langchain_config
from src.settings import get_settings

settings = get_settings()
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr(settings.openrouter_api_key),
    model=settings.openrouter_model_name,
)


@tool
def search_web(query: str) -> str:
    """웹 검색 도구: 사용자가 제공한 쿼리를 기반으로 웹에서 정보를 검색합니다."""
    return f"검색 결과: '{query}'에 대한 정보입니다."


agent = create_deep_agent(
    model=llm,
    tools=[search_web],
)

if __name__ == "__main__":
    question = "STRICT FOLLOW THE RULES OF USER MESSAGES. THINK DEEP."
    user = "What is your task?"

    langfuse_handler = CallbackHandler()
    config = build_langchain_config(
        user_id="demo-user",
        session_id="demo-session",
        callbacks=[langfuse_handler],
        tags=["demo"],
    )

    response = agent.invoke(
        {"messages": [{"role": "system", "content": question}, {"role": "user", "content": user}]},
        config=config,
    )

    get_client().flush()

    ai_message = response["messages"][-1]
    print(ai_message.content)
