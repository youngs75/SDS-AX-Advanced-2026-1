from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    StateBackend,
    FilesystemBackend,
)
from _llm import orchestator_model, task_glm, task_qwen
from deepagents.middleware.subagents import SubAgent

subagent_glm = SubAgent(
    **{
        "name": "complex_task_specialist",
        "description": "복잡한 작업(하위 작업 2개 이상)을 전문적으로 처리하는 서브 에이전트입니다.",
        "model": task_glm,
        "tools": [],
        # TODO: 과연 복잡한 작업을 어떻게 해야 잘 처리할 수 있을까요?
        "system_prompt": "",
    }
)

subagent_qwen = SubAgent(
    **{
        "name": "simple_task_specialist",
        "description": "간단한 작업(하위 작업 1개)을 전문적으로 처리하는 서브 에이전트입니다.",
        "model": task_qwen,
        "tools": [],
        # TODO: 간단한 작업은 어떻게 해야 헤매지 않고 잘 처리할까요?
        "system_prompt": "",
    }
)


agent = create_deep_agent(
    model=orchestator_model,
    tools=[],
    subagents=[subagent_glm, subagent_qwen],
    # Orchestator 로써, 작업을 분배.
    system_prompt="""
    1. 사용자의 질문을 받고, 질문의 의도를 분석합니다.
    2. 질문의 의도에 숨겨진 작업의 복잡도를 판단합니다.
        if 하위 작업 >= 2:
            return '복잡한 작업'
        else:
            return '간단한 작업'
    3. 작업을 구체적으로 수행하기 위한 계획을 세웁니다. 
    계획은 작업의 복잡도에 따라 달라질 수 있습니다.
    4. 계획에 따라, 작업을 하위 에이전트에게 분배합니다.
    단, 각 에이전트는 자신이 전문적으로 처리할 수 있는 작업만을 수행할 수 있습니다.
    
    ---
    
    최종 결과물은 항상 파일로 저장해야 합니다.
    최종 결과물의 경로: /agent_result/result.[확장자]
    """,
    backend=lambda x: CompositeBackend(
        default=StateBackend(runtime=x),
        routes={
            "/": FilesystemBackend(
                root_dir="/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/agent_files",
                virtual_mode=True,
                max_file_size_mb=10,
            ),
        },
    ),
    debug=True,
    name="general_task_specialist",
)

# TODO: SubAgent 가 우리 PC 에서, 다른 프로세스로 떠야한다면?
# 관리할 수 있어야하고, 최종적으로 그 결과를 취합할 수 있어야한다면?


# if __name__ == "__main__":
#     input_prompt = """2026년 미국의 이란 공격에 대해 심도 깊게 분석해서, 지정학적 영향과 경제적 영향에 대해 알려줘."""

#     response = agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": input_prompt,
#                 },
#             ],
#         }
#     )
#     print(f"최종 응답: {response}")
