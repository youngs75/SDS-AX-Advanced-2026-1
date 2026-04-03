import os

from langfuse import get_client
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")
langfuse = get_client()
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
)

agent = create_deep_agent(
    model=llm,
    tools=[],
    name="Day05_basic",
    debug=True,
)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "이름이 뭐에여?"}]},
        config={"callbacks": [langfuse_handler]},
    )
    print(response)
