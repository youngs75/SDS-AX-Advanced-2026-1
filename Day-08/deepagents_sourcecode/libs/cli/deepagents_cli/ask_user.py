"""`ask_user` 도구 호출을 대화형 CLI 중단으로 연결합니다.

이 모듈은 도구 스키마를 정의하고, 에이전트가 작성한 질문을 검증하고, CLI에서 수집한 답변을 다시 그래프 업데이트로 변환합니다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import InjectedToolCallId
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from deepagents_cli._ask_user_types import AskUserRequest, Question

logger = logging.getLogger(__name__)


ASK_USER_TOOL_DESCRIPTION = """Ask the user one or more questions when you need clarification or input before proceeding.

Each question can be either:
- "text": Free-form text response from the user
- "multiple_choice": User selects from predefined options (an "Other" option is always available)

For multiple choice questions, provide a list of choices. The user can pick one or type a custom answer via the "Other" option.

By default all questions are required. Set "required" to false for optional questions that the user can skip.

Use this tool when:
- You need clarification on ambiguous requirements
- You want the user to choose between multiple valid approaches
- You need specific information only the user can provide
- You want to confirm a plan before executing it

Do NOT use this tool for:
- Simple yes/no confirmations (just proceed with your best judgment)
- Questions you can answer yourself from context
- Trivial decisions that don't meaningfully affect the outcome"""  # noqa: E501

ASK_USER_SYSTEM_PROMPT = """## `ask_user`

You have access to the `ask_user` tool to ask the user questions when you need clarification or input.
Use this tool sparingly - only when you genuinely need information from the user that you cannot determine from context.

When using `ask_user`:
- Be concise and specific with your questions
- Use multiple choice when there are clear options to choose from
- Use text input when you need free-form responses
- Group related questions into a single ask_user call rather than making multiple calls
- Never ask questions you can answer yourself from the available context"""  # noqa: E501


def _validate_questions(questions: list[Question]) -> None:
    """중단하기 전에 Ask_user 질문 구조를 확인하세요.

Args:
        questions: `ask_user` 도구에 제공되는 질문 정의입니다.

Raises:
        ValueError: 질문 목록 또는 개별 질문이 유효하지 않은 경우.

    """
    if not questions:
        msg = "ask_user requires at least one question"
        raise ValueError(msg)

    for q in questions:
        question_text = q.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            msg = "ask_user questions must have non-empty 'question' text"
            raise ValueError(msg)

        question_type = q.get("type")
        if question_type not in {"text", "multiple_choice"}:
            msg = f"unsupported ask_user question type: {question_type!r}"
            raise ValueError(msg)

        if question_type == "multiple_choice" and not q.get("choices"):
            msg = (
                f"multiple_choice question "
                f"{q.get('question')!r} requires a "
                f"non-empty 'choices' list"
            )
            raise ValueError(msg)

        if question_type == "text" and q.get("choices"):
            msg = f"text question {q.get('question')!r} must not define 'choices'"
            raise ValueError(msg)


def _parse_answers(
    response: object,
    questions: list[Question],
    tool_call_id: str,
) -> Command[Any]:
    """`ToolMessage`을 사용하여 인터럽트 응답을 `Command`로 구문 분석합니다.

    어댑터의 명시적 상태 신호를 지원합니다.

    - `answered`(기본값): 제공된 `answers` 사용 - `cancelled`: `(cancelled)` 답변 종합 - `error`:
    `(error: ...)` 답변 종합

    잘못된 페이로드는 자동으로 `(no answer)`을 기본값으로 설정하는 대신 명시적인 오류 답변으로 변환됩니다.

Args:
        response: `interrupt()`에서 반환된 원시 값입니다.
        questions: 받은 질문.
        tool_call_id: `ToolMessage`에 대한 원래 도구 호출 ID입니다.

Returns:
        `Command`에는 Q&A 쌍이 포함된 형식화된 `ToolMessage`이 포함되어 있습니다.

    """
    status: str = "answered"
    error_text: str | None = None
    answers: list[str]
    if not isinstance(response, dict):
        logger.error(
            "ask_user received malformed resume payload "
            "(expected dict, got %s); returning explicit error answers",
            type(response).__name__,
        )
        answers = []
        status = "error"
        error_text = "invalid ask_user response payload"
    else:
        response_dict = cast("dict[str, Any]", response)
        response_status = response_dict.get("status")
        if isinstance(response_status, str):
            status = response_status

        if "answers" not in response_dict:
            if status == "answered":
                logger.error(
                    "ask_user received resume payload without 'answers'; "
                    "returning explicit error answers"
                )
                answers = []
                status = "error"
                error_text = "missing ask_user answers payload"
            else:
                answers = []
        else:
            raw_answers = response_dict["answers"]
            if isinstance(raw_answers, list):
                answers = [str(answer) for answer in raw_answers]
            else:
                logger.error(
                    "ask_user received non-list 'answers' payload (%s); "
                    "returning explicit error answers",
                    type(raw_answers).__name__,
                )
                answers = []
                status = "error"
                error_text = "invalid ask_user answers payload"

        if status == "error":
            response_error = response_dict.get("error")
            if isinstance(response_error, str) and response_error:
                error_text = response_error
        elif status == "cancelled":
            answers = ["(cancelled)" for _ in questions]
        elif status == "answered":
            if len(answers) != len(questions):
                logger.warning(
                    "ask_user answer count mismatch: expected %d, got %d",
                    len(questions),
                    len(answers),
                )
        else:
            logger.error(
                "ask_user received unknown status %r; returning explicit error answers",
                status,
            )
            answers = []
            status = "error"
            error_text = "invalid ask_user response status"

    if status == "error":
        detail = error_text or "ask_user interaction failed"
        answers = [f"(error: {detail})" for _ in questions]

    formatted_answers = []
    for i, q in enumerate(questions):
        answer = answers[i] if i < len(answers) else "(no answer)"
        formatted_answers.append(f"Q: {q['question']}\nA: {answer}")
    result_text = "\n\n".join(formatted_answers)
    return Command(
        update={
            "messages": [ToolMessage(result_text, tool_call_id=tool_call_id)],
        }
    )


class AskUserMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """대화형 질문을 위한 Ask_user 도구를 제공하는 미들웨어입니다.

    이 미들웨어는 에이전트가 실행 중에 사용자에게 질문할 수 있는 `ask_user` 도구를 추가합니다. 질문은 자유 형식 텍스트이거나 객관식일 수
    있습니다. 이 도구는 LangGraph 인터럽트를 사용하여 실행을 일시 중지하고 사용자 입력을 기다립니다.

    """

    def __init__(
        self,
        *,
        system_prompt: str = ASK_USER_SYSTEM_PROMPT,
        tool_description: str = ASK_USER_TOOL_DESCRIPTION,
    ) -> None:
        """AskUserMiddleware를 초기화합니다.

Args:
            system_prompt: `ask_user` 사용법을 안내하기 위해 모든 LLM 요청에 시스템 수준 지침이 삽입되었습니다.
            tool_description: 도구 스키마의 LLM에 표시되는 `ask_user` 도구 데코레이터에 전달된 설명 문자열입니다.

        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        @tool(description=self.tool_description)
        def _ask_user(
            questions: list[Question],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command[Any]:
            """사용자에게 하나 이상의 질문을 하십시오.

Args:
                questions: 사용자에게 제시할 질문입니다.
                tool_call_id: LangChain이 주입한 도구 호출 식별자입니다.

Returns:
                구문 분석된 사용자 응답을 `ToolMessage`로 포함하는 `Command`.

            """
            _validate_questions(questions)
            ask_request = AskUserRequest(
                type="ask_user",
                questions=questions,
                tool_call_id=tool_call_id,
            )
            response = interrupt(ask_request)
            return _parse_answers(response, questions, tool_call_id)

        _ask_user.name = "ask_user"
        self.tools = [_ask_user]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Ask_user 시스템 프롬프트를 삽입합니다.

Returns:
            래핑된 핸들러의 모델 응답입니다.

        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Ask_user 시스템 프롬프트(비동기)를 삽입합니다.

Returns:
            래핑된 핸들러의 모델 응답입니다.

        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
