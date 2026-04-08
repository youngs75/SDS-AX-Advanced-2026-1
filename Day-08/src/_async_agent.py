"""비동기 서브에이전트 오케스트레이터.

별도 프로세스로 실행 중인 Agent Protocol 서버들과 HTTP로 통신하여
작업을 비동기적으로 위임하고, 결과를 취합합니다.

이 파일은 _agent.py(동기 버전)의 비동기 버전입니다.
동기 버전과의 핵심 차이:
  - SubAgent → AsyncSubAgent (별도 프로세스, HTTP 통신)
  - task 도구 → start/check/update/cancel/list_async_task 5개 도구
  - 블로킹 실행 → 논블로킹 (즉시 task_id 반환, 나중에 결과 조회)

실행 방법:
    # 먼저 서버 2개를 각각 다른 터미널에서 실행
    python task_server.py --name complex --port 2024
    python task_server.py --name simple --port 2025

    # 그 다음 오케스트레이터 실행
    python _async_agent.py
"""

from __future__ import annotations

import asyncio
import uuid

from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from _llm import orchestator_model

# ── 비동기 서브에이전트 스펙 ──────────────────────────────────────────────────
# graph_id 필드가 있으면 create_deep_agent가 자동으로 AsyncSubAgentMiddleware를 적용합니다.
# graph_id는 서버의 assistant_id로 전달됩니다 (서버당 에이전트 1개이므로 라우팅에는 미사용).

async_subagents: list[AsyncSubAgent] = [
    {
        "name": "complex_task_specialist",
        "description": "복잡한 작업(하위 작업 2개 이상)을 전문적으로 처리하는 원격 에이전트입니다. 별도 프로세스(port 2024)에서 실행됩니다.",
        "graph_id": "complex_task_specialist",
        "url": "http://localhost:2024",
    },
    {
        "name": "simple_task_specialist",
        "description": "간단한 작업(하위 작업 1개)을 빠르게 처리하는 원격 에이전트입니다. 별도 프로세스(port 2025)에서 실행됩니다.",
        "graph_id": "simple_task_specialist",
        "url": "http://localhost:2025",
    },
]

# ── 오케스트레이터 생성 ───────────────────────────────────────────────────────

checkpointer = MemorySaver()
thread_id = str(uuid.uuid4())

agent = create_deep_agent(
    model=orchestator_model,
    tools=[],
    subagents=async_subagents,
    checkpointer=checkpointer,
    system_prompt="""당신은 비동기 작업 오케스트레이터입니다.
사용자의 질문을 받고, 작업의 복잡도를 판단하여 적절한 원격 에이전트에게 위임합니다.

## 작업 분배 기준

1. 사용자의 질문을 분석합니다.
2. 질문에 숨겨진 작업의 복잡도를 판단합니다:
    if 하위 작업 >= 2:
        → complex_task_specialist에게 위임 (start_async_task)
    else:
        → simple_task_specialist에게 위임 (start_async_task)

## 비동기 작업 흐름

1. **위임**: start_async_task로 원격 에이전트에게 작업을 보냅니다.
   - task_id를 사용자에게 알려주고, 작업이 백그라운드에서 진행 중임을 안내합니다.
2. **확인**: 사용자가 결과를 물으면 check_async_task로 상태를 확인합니다.
   - 아직 실행 중이면 "아직 진행 중입니다"라고 안내합니다.
   - 완료되었으면 결과를 정리하여 사용자에게 전달합니다.
3. **수정**: 사용자가 방향을 바꾸고 싶으면 update_async_task로 새 지시를 보냅니다.
4. **취소**: 더 이상 필요 없으면 cancel_async_task로 취소합니다.
5. **목록**: 여러 작업의 상태를 한번에 보려면 list_async_tasks를 사용합니다.

## 주의사항

- start_async_task 후에는 즉시 상태를 확인하지 마세요. task_id만 알려주고 멈추세요.
- check_async_task를 반복 호출하지 마세요. 한 번 확인하고, 아직이면 사용자에게 알려주세요.
- 간단한 인사나 대화에는 에이전트를 사용하지 말고 직접 답변하세요.
""",
    debug=True,
    name="async_orchestrator",
)


# ── 대화형 REPL ──────────────────────────────────────────────────────────────


async def chat(user_input: str) -> None:
    """오케스트레이터에게 메시지를 보내고 응답을 출력합니다."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    last = result["messages"][-1]
    content = last.content
    print(
        "\n"
        + (
            content
            if isinstance(content, str)
            else __import__("json").dumps(content, indent=2, ensure_ascii=False)
        )
        + "\n"
    )


async def main() -> None:
    """대화형 REPL을 실행합니다."""
    print("=" * 60)
    print("비동기 서브에이전트 오케스트레이터")
    print("=" * 60)
    print("complex_task_specialist → http://localhost:2024")
    print("simple_task_specialist  → http://localhost:2025")
    print()
    print("메시지를 입력하세요. 종료: Ctrl+C 또는 Ctrl+D")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not user_input:
            continue
        try:
            await chat(user_input)
        except Exception as exc:
            print(f"오류 발생: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
