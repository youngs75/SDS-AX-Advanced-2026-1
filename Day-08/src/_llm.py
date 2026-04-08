from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Orchestator 모델
# GPT-5.4
# API_KEY, BASE_URl 은 OpenAI
orchestator_model = ChatOpenAI(model="gpt-5.4")

# 2. 작업용 모델
# z-ai/GLM-5.1 & qwen/qwen3.5-35b-a3b
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
task_glm = ChatOpenAI(
    model="z-ai/glm-5.1",
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)
task_qwen = ChatOpenAI(
    model="qwen/qwen3.5-35b-a3b",
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# if __name__ == "__main__":
#     input_prompt = "니 이름은 뭐니?"

#     response_orchestator = orchestator_model.invoke(input_prompt)
#     print(f"Orchestator 모델 응답:{response_orchestator.content}")

#     response_glm = task_glm.invoke(input_prompt)
#     print(f"Task GLM 모델 응답:{response_glm.content}")

#     response_qwen = task_qwen.invoke(input_prompt)
#     print(f"Task Qwen 모델 응답:{response_qwen.content}")
