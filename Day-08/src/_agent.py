import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any

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

PROCESS_SUBAGENTS = {
    subagent_glm["name"]: subagent_glm,
    subagent_qwen["name"]: subagent_qwen,
}


def _build_backend(runtime: Any) -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(runtime=runtime),
        routes={
            "/": FilesystemBackend(
                root_dir="/Users/jhj/Desktop/2026_1_sds_ax_advanced/Day-08/agent_files",
                virtual_mode=True,
                max_file_size_mb=10,
            ),
        },
    )


def _build_single_subagent(spec_name: str):
    spec = PROCESS_SUBAGENTS[spec_name]
    return create_deep_agent(
        model=spec["model"],
        tools=list(spec.get("tools", [])),
        system_prompt=spec["system_prompt"],
        backend=_build_backend,
        debug=True,
        name=spec["name"],
    )


def _normalize_agent_output(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", str(block))
                    if isinstance(block, dict)
                    else str(block)
                    for block in content
                )
        return str(response)
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(response)


def _run_subagent_in_process(spec_name: str, prompt: str) -> dict[str, Any]:
    worker_agent = _build_single_subagent(spec_name)
    response = worker_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
    )
    return {
        "subagent": spec_name,
        "output": _normalize_agent_output(response),
        "raw_response": response,
    }


class ProcessSubAgentManager:
    def __init__(self, max_workers: int | None = None):
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers or len(PROCESS_SUBAGENTS),
            mp_context=mp.get_context("spawn"),
        )
        self._futures: dict[Future[dict[str, Any]], str] = {}

    def submit(self, subagent_name: str, prompt: str) -> Future[dict[str, Any]]:
        if subagent_name not in PROCESS_SUBAGENTS:
            raise ValueError(f"Unknown subagent: {subagent_name}")
        future = self._executor.submit(_run_subagent_in_process, subagent_name, prompt)
        self._futures[future] = subagent_name
        return future

    def gather(
        self, futures: list[Future[dict[str, Any]]] | None = None
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        target_futures = futures or list(self._futures)
        for future in as_completed(target_futures):
            subagent_name = self._futures.pop(future)
            try:
                results[subagent_name] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[subagent_name] = {
                    "subagent": subagent_name,
                    "error": str(exc),
                }
        return results

    def run_many(self, tasks: dict[str, str]) -> dict[str, dict[str, Any]]:
        futures = [
            self.submit(subagent_name=subagent_name, prompt=prompt)
            for subagent_name, prompt in tasks.items()
        ]
        return self.gather(futures)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


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
