"""CLI 통합 테스트에서 사용되는 결정적 가짜 채팅 모델입니다.

이 모듈의 도우미는 외부 공급자에 의존하지 않고 실제 CLI/서버 스택을 실행하는 테스트를 위해 다시 시작으로부터 안전한 모델 동작을 제공합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


class DeterministicIntegrationChatModel(GenericFakeChatModel):
    """CLI 통합 테스트를 위한 결정적 채팅 모델입니다.

    이는 LangChain의 `GenericFakeChatModel`을 하위 클래스로 지정하여 구현이 핵심 가짜 채팅 모델 테스트 표면과 일치하도록
    유지하는 동시에 실제 CLI 서버 통합 테스트를 위해 프롬프트 기반 및 재시작 안전성을 유지하도록 생성을 재정의합니다.

    기존 `langchain_core` 가짜를 여기에서 재사용할 수 없는 이유:

    1. 모든 코어 페이크(`GenericFakeChatModel`, `FakeListChatModel`,
        `FakeMessagesListChatModel`)은 반복자에서 팝되거나 인덱스를 순환합니다. 실제 프롬프트는 무시됩니다. CLI 통합 테스트는
        서버 프로세스를 시작 및 중지하여 메모리 내 상태를 재설정합니다. 반복자 기반 모델은 `StopIteration`을 발생시키거나 다시 시작한 후
        처음부터 재생하여 잘못되거나 누락된 응답을 생성합니다. 이 모델은 프롬프트 텍스트에서만 출력을 파생하므로 동일한 입력은 프로세스 수명주기에
        관계없이 항상 동일한 출력을 생성합니다.

    2. 에이전트 런타임은 도중에 `model.bind_tools(schemas)`을 호출합니다.
        초기화. 핵심 가짜는 `bind_tools`을 구현하지 않으므로 모든 에이전트 루프 컨텍스트에서 `AttributeError`을 발생시킵니다.
        이 모델은 무작동 패스스루를 제공합니다.

    3. CLI 서버는 기능 협상을 위해 `model.profile`을 읽습니다(예:
        `tool_calling`, `max_input_tokens`). 핵심 가짜에는 그러한 속성이 없으므로 런타임 시 `AttributeError`
        또는 자동 구성 오류가 발생합니다.

    또한 컴팩트 미들웨어는 문제 요약을 통해 중간 대화를 유도합니다. 목록 기반 모델은 정확한 통화 순서에 대한 사전 지식 없이 이를 일반 사용자 차례와
    구별할 수 없는 반면, 이 모델은 프롬프트 내용을 검사하여 요약 요청을 감지합니다.

    """

    model: str = "fake"
    # `GenericFakeChatModel`에 필요하지만 재정의에서는 이를 사용하지 않습니다.
    messages: object = Field(default_factory=lambda: iter(()))
    profile: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "tool_calling": True,
            "max_input_tokens": 8000,
        }
    )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """테스트 중에 에이전트가 도구 스키마를 바인딩할 수 있도록 self를 반환합니다."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """프롬프트 텍스트에서 파생된 결정론적 응답을 생성합니다.

Returns:
            결정론적 콘텐츠가 포함된 단일 메시지 `ChatResult`.

        """
        prompt = "\n".join(
            text
            for message in messages
            if (text := self._stringify_message(message)).strip()
        )
        if self._looks_like_summary_request(prompt):
            content = "integration summary"
        else:
            excerpt = " ".join(prompt.split()[-18:])
            if excerpt:
                content = f"integration reply: {excerpt}"
            else:
                content = "integration reply"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        """LangChain 모델 유형 식별자를 반환합니다."""
        return "deterministic-integration"

    @staticmethod
    def _stringify_message(message: BaseMessage) -> str:
        """결정적 응답을 위해 메시지 콘텐츠를 일반 텍스트로 평면화합니다.

Returns:
            메시지에서 추출된 일반 텍스트 콘텐츠입니다.

        """
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return str(content)

    @staticmethod
    def _looks_like_summary_request(prompt: str) -> bool:
        """미들웨어의 요약 생성 프롬프트를 감지합니다.

Returns:
            `True` 프롬프트가 요약 요청으로 나타날 때.

        """
        lowered = prompt.lower()
        return (
            "messages to summarize" in lowered
            or "condense the following conversation" in lowered
            or "<summary>" in lowered
        )
