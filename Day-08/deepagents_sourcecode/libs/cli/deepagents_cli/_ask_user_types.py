"""사용자 인터럽트 요청 프로토콜의 경량 유형입니다.

`ask_user`에서 추출되므로 `textual_adapter`은 모듈 수준에서 `AskUserRequest`을 가져올 수 있고 `app`은
langchain 미들웨어 스택을 가져오지 않고도 유형 확인 시 유형을 참조할 수 있습니다.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict


class Choice(TypedDict):
    """객관식 질문에 대한 단일 선택 옵션입니다."""

    value: Annotated[str, Field(description="The display label for this choice.")]


class Question(TypedDict):
    """사용자에게 묻는 질문입니다."""

    question: Annotated[str, Field(description="The question text to display.")]

    type: Annotated[
        Literal["text", "multiple_choice"],
        Field(
            description=(
                "Question type. 'text' for free-form input, 'multiple_choice' for "
                "predefined options."
            )
        ),
    ]

    choices: NotRequired[
        Annotated[
            list[Choice],
            Field(
                description=(
                    "Options for multiple_choice questions. An 'Other' free-form "
                    "option is always appended automatically."
                )
            ),
        ]
    ]

    required: NotRequired[
        Annotated[
            bool,
            Field(
                description="Whether the user must answer. Defaults to true if omitted."
            ),
        ]
    ]


class AskUserRequest(TypedDict):
    """사용자에게 질문할 때 인터럽트를 통해 요청 페이로드가 전송됩니다."""

    type: Literal["ask_user"]
    """판별자 태그는 항상 `'ask_user'`입니다."""

    questions: list[Question]
    """사용자에게 제시할 질문입니다."""

    tool_call_id: str
    """응답을 다시 라우팅하는 데 사용되는 원래 도구 호출의 ID입니다."""


class AskUserAnswered(TypedDict):
    """사용자가 답변을 제출하면 위젯 결과가 표시됩니다."""

    type: Literal["answered"]
    """판별자 태그는 항상 `'answered'`입니다."""

    answers: list[str]
    """사용자가 제공한 답변(질문당 하나씩)"""


class AskUserCancelled(TypedDict):
    """사용자가 프롬프트를 취소하면 위젯 결과가 표시됩니다."""

    type: Literal["cancelled"]
    """판별자 태그는 항상 `'cancelled'`입니다."""


AskUserWidgetResult = AskUserAnswered | AskUserCancelled
"""Ask_user 위젯의 향후 결과에 대한 식별된 공용체입니다."""
