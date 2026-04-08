"""매달린(dangling) 도구 호출을 패치하는 미들웨어 모듈.

이 모듈은 메시지 히스토리에서 발생할 수 있는 "매달린 도구 호출" 문제를 해결합니다.

## 매달린 도구 호출이란?

LLM이 도구 호출(tool_call)을 요청했지만, 그에 대응하는 ToolMessage(도구 실행 결과)가
메시지 히스토리에 존재하지 않는 경우를 말합니다. 이는 다음과 같은 상황에서 발생합니다:

1. 사용자가 도구 실행 도중 대화를 중단(interrupt)한 경우
2. 도구 실행이 타임아웃이나 오류로 실패한 경우
3. 컨텍스트 압축(summarization) 과정에서 ToolMessage가 누락된 경우

## 왜 패치가 필요한가?

대부분의 LLM API(특히 OpenAI, Anthropic)는 AIMessage에 tool_calls가 있으면
반드시 그에 대응하는 ToolMessage가 뒤따라야 한다는 제약을 가지고 있습니다.
이 제약이 충족되지 않으면 API 호출이 실패합니다.

## 동작 방식

에이전트 실행 전(`before_agent` 훅)에 전체 메시지 히스토리를 순회하면서:
1. AIMessage에 포함된 각 tool_call에 대해
2. 대응하는 ToolMessage가 이후 메시지에 존재하는지 확인하고
3. 없으면 "취소됨" 내용의 ToolMessage를 삽입합니다.

패치된 메시지 리스트는 `Overwrite`로 상태에 반영되어, 기존 메시지를 완전히 대체합니다.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """매달린(dangling) 도구 호출을 자동으로 패치하는 미들웨어.

    에이전트가 실행되기 전에 메시지 히스토리를 검사하여, AIMessage의 tool_calls 중
    대응하는 ToolMessage가 없는 것을 찾아 "취소됨" ToolMessage를 삽입합니다.

    이 미들웨어는 `before_agent` 훅만 사용하며, LLM 호출을 가로채지 않습니다.
    (`wrap_model_call`은 오버라이드하지 않음)

    사용 예시:
        ```python
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

        agent = create_agent(
            model="openai:gpt-4o",
            middleware=[PatchToolCallsMiddleware()],
        )
        ```

    동작 원리:
        1. 메시지 리스트를 순회하면서 각 AIMessage의 tool_calls를 검사
        2. 해당 tool_call의 id와 일치하는 ToolMessage가 이후 메시지에 있는지 확인
        3. 없으면 "취소됨" 내용의 ToolMessage를 즉시 삽입
        4. 패치된 전체 메시지 리스트를 Overwrite로 상태에 반영

    주의사항:
        - Overwrite를 사용하므로 기존 메시지 리스트가 완전히 대체됩니다.
        - 메시지가 비어있으면 아무 작업도 수행하지 않습니다 (None 반환).
    """

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """에이전트 실행 전에 매달린 도구 호출을 패치합니다.

        전체 메시지 히스토리를 순회하면서, AIMessage에 포함된 tool_calls 중
        대응하는 ToolMessage가 없는 것을 찾아 "취소됨" 메시지를 삽입합니다.

        Args:
            state: 현재 에이전트 상태. `state["messages"]`에서 메시지 히스토리를 읽습니다.
            runtime: 런타임 컨텍스트. 이 미들웨어에서는 사용하지 않습니다 (ARG002 무시).

        Returns:
            패치된 메시지가 있으면 `{"messages": Overwrite(patched_messages)}` 딕셔너리를 반환합니다.
            메시지가 비어있으면 None을 반환하여 상태 변경이 없음을 나타냅니다.
        """
        messages = state["messages"]

        # 메시지가 비어있으면 패치할 것이 없으므로 조기 반환
        if not messages or len(messages) == 0:
            return None

        patched_messages = []

        # 전체 메시지를 순회하면서 매달린 도구 호출 탐지
        for i, msg in enumerate(messages):
            # 현재 메시지를 결과 리스트에 추가
            patched_messages.append(msg)

            # AIMessage이고 tool_calls가 있는 경우에만 검사
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 현재 위치(i) 이후의 메시지에서 대응하는 ToolMessage를 검색
                    # tool_call_id가 일치하는 ToolMessage가 있는지 확인
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),  # ty: ignore[unresolved-attribute]
                        None,
                    )

                    if corresponding_tool_msg is None:
                        # 대응하는 ToolMessage가 없음 → "취소됨" 메시지를 삽입하여
                        # LLM API의 tool_call ↔ ToolMessage 짝 맞춤 제약을 충족시킴
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        # Overwrite를 사용하여 기존 메시지 리스트를 패치된 버전으로 완전히 대체
        # (일반적인 상태 업데이트는 병합(merge)이지만, Overwrite는 전체 교체를 의미)
        return {"messages": Overwrite(patched_messages)}
