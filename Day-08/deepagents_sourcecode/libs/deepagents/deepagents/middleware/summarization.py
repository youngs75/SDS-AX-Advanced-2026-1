"""자동 및 도구 기반 대화 압축(compaction)을 위한 요약 미들웨어 모듈.

이 모듈은 두 가지 미들웨어 클래스와 편의 팩토리 함수를 제공합니다:

- `SummarizationMiddleware` — 토큰 사용량이 설정 가능한 임계값을 초과하면
    자동으로 대화를 압축합니다.
    오래된 메시지는 LLM 호출로 요약되고, 전체 히스토리는
    나중에 검색할 수 있도록 백엔드에 오프로딩됩니다.

- `SummarizationToolMiddleware` — 에이전트(또는 HIL 승인 흐름)가
    요청 시 압축을 트리거할 수 있는 `compact_conversation` 도구를 노출합니다.
    `SummarizationMiddleware` 인스턴스와 조합하여 그 요약 엔진을 재사용합니다.

- `create_summarization_tool_middleware` — 모델 인식 기본값으로 두 미들웨어
    계층을 함께 생성하는 편의 팩토리 함수입니다.

## 핵심 개념

### 요약 트리거 (trigger)
토큰/메시지/비율 기반 임계값으로 자동 요약을 트리거합니다.
- `("tokens", 170000)`: 토큰 수가 17만 이상이면 트리거
- `("fraction", 0.85)`: 모델 최대 토큰의 85% 이상이면 트리거
- `("messages", 50)`: 메시지 50개 이상이면 트리거

### 유지 정책 (keep)
요약 후 보존할 최근 메시지 범위를 지정합니다.
- `("messages", 6)`: 최근 6개 메시지 보존
- `("fraction", 0.10)`: 모델 최대 토큰의 10%에 해당하는 최근 메시지 보존

### 인수 절삭 (truncate_args)
요약 전 경량 최적화로, 오래된 메시지의 write_file/edit_file 등
대형 도구 인수(args)를 절삭합니다. 요약보다 낮은 임계값에서 작동합니다.

### ContextOverflowError 폴백
임계값 아래여서 요약을 건너뛰었지만 모델 호출이 ContextOverflowError를
발생시키면, 즉시 요약 경로로 폴백하여 재시도합니다.

## 사용 예시

```python
from deepagents import create_deep_agent
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="/data")

summ = SummarizationMiddleware(
    model="gpt-4o-mini",
    backend=backend,
    trigger=("fraction", 0.85),
    keep=("fraction", 0.10),
)
tool_mw = SummarizationToolMiddleware(summ)

agent = create_deep_agent(middleware=[summ, tool_mw])
```

## 스토리지

오프로딩된 메시지는 `/conversation_history/{thread_id}.md`에 마크다운으로 저장됩니다.
각 요약 이벤트는 이 파일에 새 섹션을 추가하여, 퇴거된 모든 메시지의
누적 로그를 생성합니다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from langchain.agents.middleware.summarization import (
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
    DEFAULT_SUMMARY_PROMPT,
    ContextSize,
    SummarizationMiddleware as LCSummarizationMiddleware,
    TokenCounter,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ExtendedModelResponse, PrivateStateAttr
from langchain.tools import ToolRuntime
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_config
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from deepagents.middleware._utils import append_to_system_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.chat_models import BaseChatModel
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

logger = logging.getLogger(__name__)


class CompactConversationSchema(BaseModel):
    """compact_conversation 도구의 입력 스키마. 인자 없음 (빈 스키마)."""


SUMMARIZATION_SYSTEM_PROMPT = """## Compact conversation Tool `compact_conversation`

You have access to a `compact_conversation` tool. This tool refreshes your context window to reduce context bloat and costs.

You should use the tool when:
- The user asks to move on to a completely new task for which previous context is likely irrelevant.
- You have finished extracting or synthesizing a result and previous working context is no longer needed.
"""


class SummarizationEvent(TypedDict):
    """요약 이벤트를 나타내는 TypedDict.

    하나의 요약 이벤트는 "어디까지 요약했는지(cutoff_index)",
    "요약 내용(summary_message)", "원본 저장 위치(file_path)"를 기록합니다.
    이 정보는 에이전트 상태의 `_summarization_event`에 저장되어,
    후속 턴에서 올바른 effective message를 재구성하는 데 사용됩니다.

    Attributes:
        cutoff_index: 메시지 리스트에서 요약이 발생한 인덱스 (이 인덱스 이전이 요약됨).
        summary_message: 요약 내용을 담은 HumanMessage.
        file_path: 대화 히스토리가 오프로딩된 경로. 오프로딩 실패 시 None.
    """

    cutoff_index: int
    summary_message: HumanMessage
    file_path: str | None


class TruncateArgsSettings(TypedDict, total=False):
    """오래된 메시지의 대형 도구 호출 인수(args)를 절삭하기 위한 설정.

    전체 대화 압축보다 낮은 토큰 임계값에서 작동하는 경량의 사전 최적화입니다.
    트리거되면, 유지 윈도우(keep window) *이전* 메시지의
    `AIMessage.tool_calls`에 있는 `args` 값만 단축됩니다 —
    최근 메시지는 그대로 유지됩니다.

    절삭 대상이 되는 대표적인 대형 인수:
    - `write_file`의 content (파일 전체 내용)
    - `edit_file`의 old_string/new_string (패치)

    Args:
        trigger: 절삭을 활성화하는 토큰/메시지/비율 임계값.
            요약 트리거와 동일한 `ContextSize` 형식 사용.
            None이면 절삭 비활성화.
        keep: 절삭하지 않을 최근 메시지 범위 (토큰/메시지 수/비율).
        max_length: 인수 값당 문자 제한 (초과 시 클리핑).
        truncation_text: 절삭된 인수의 첫 20자 뒤에 추가되는 대체 접미사.
    """

    trigger: ContextSize | None
    keep: ContextSize
    max_length: int
    truncation_text: str


class SummarizationState(AgentState):
    """요약 미들웨어의 상태 스키마.

    AgentState를 확장하여 요약 이벤트를 추적하는 비공개(private) 필드를 추가합니다.
    PrivateStateAttr로 표시되어 부모 에이전트에게 전파되지 않습니다.
    """

    _summarization_event: Annotated[NotRequired[SummarizationEvent | None], PrivateStateAttr]
    """가장 최근 요약 이벤트를 저장하는 비공개 필드.
    이전 요약의 cutoff_index를 기억하여 후속 턴에서 올바른 메시지를 재구성합니다."""


class SummarizationDefaults(TypedDict):
    """모델 프로파일에서 계산된 기본 요약 설정."""

    trigger: ContextSize
    keep: ContextSize
    truncate_args_settings: TruncateArgsSettings


def compute_summarization_defaults(model: BaseChatModel) -> SummarizationDefaults:
    """모델 프로파일에 기반하여 기본 요약 설정을 계산합니다.

    모델에 `max_input_tokens` 프로파일이 있으면 비율(fraction) 기반 설정을 사용하고,
    없으면 고정 토큰/메시지 수를 사용합니다.

    비율 기반 설정 (프로파일 있는 모델):
    - trigger: 컨텍스트의 85% 사용 시 요약 트리거
    - keep: 요약 후 컨텍스트의 10%에 해당하는 최근 메시지 보존

    고정값 설정 (프로파일 없는 모델):
    - trigger: 17만 토큰 시 요약 트리거
    - keep: 최근 6개 메시지 보존
    (보수적으로 설정하여 컨텍스트 한도 초과를 방지)

    Args:
        model: 해석된 채팅 모델 인스턴스.

    Returns:
        trigger, keep, truncate_args_settings 기본 설정.
    """
    has_profile = (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    )

    if has_profile:
        return {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.10),
            "truncate_args_settings": {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            },
        }

    # Defaults for models without profile info are more conservative to avoid
    # overshooting context limits.
    return {
        "trigger": ("tokens", 170000),
        "keep": ("messages", 6),
        "truncate_args_settings": {
            "trigger": ("messages", 20),
            "keep": ("messages", 20),
        },
    }


class _DeepAgentsSummarizationMiddleware(AgentMiddleware):
    """대화 히스토리 오프로딩을 지원하는 요약 미들웨어.

    이 미들웨어는 두 가지 핵심 기능을 제공합니다:

    1. **자동 요약**: 토큰 사용량이 임계값을 초과하면 오래된 메시지를 LLM으로 요약하고,
       원본을 백엔드��� 오프로딩한 후, 요약 메시지 + 최근 메��지로 대체
    2. **인수 절삭**: 요약보다 낮은 임계값에서, 오래된 메시지의 대형 도구 인수를 절삭

    동작 흐름 (wrap_model_call):
        1. 이전 요약 이벤트가 있으면 effective messages 재구성
        2. 인수 절삭 적용 (설정된 경우)
        3. 토큰 수 계산 → 요약 필��� 여부 판단
        4. 요약 불필요 시: 모델 호출 시도 → ContextOverflowError 시 요약 폴백
        5. 요약 필요 시: cutoff 결정 → 오프로딩 → LLM 요약 → 메시지 교체
        6. _summarization_event를 ExtendedModelResponse로 상태에 저장

    LangChain의 `LCSummarizationMiddleware`에 핵심 요약 로직을 위임하고,
    Deep Agents 고유의 백엔드 오프로딩과 인수 절삭 기능을 추가합니다.
    """

    state_schema = SummarizationState

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BACKEND_TYPES,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        history_path_prefix: str = "/conversation_history",
        truncate_args_settings: TruncateArgsSettings | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """백엔드 지원이 포함된 요약 미들웨어를 초기화합니다.

        내부적으로 LangChain의 LCSummarizationMiddleware에 핵심 요약 로직을 위임하고,
        Deep Agents 고유의 백엔드 오프로딩과 인수 절삭 설정을 추가합니다.

        Args:
            model: 요약 생성에 사용할 언어 모델.
            backend: 대화 히스토리 영구 저장을 위한 백엔드 인스턴스 또는 팩토리.
            trigger: 요약을 트리거하는 임계값.
                `("tokens", N)`, `("fraction", F)`, `("messages", M)` 형식.
            keep: 요약 후 컨텍스트 보존 정책. 기본값: 최근 20개 메시지 보존.
            token_counter: 메���지의 토큰 수를 계산하는 함수.
            summary_prompt: 요약 생성을 위한 프롬프트 템플릿.
            trim_tokens_to_summarize: 요약 생성 시 포함할 최대 토큰 수. 기본 4000.
            truncate_args_settings: 오래된 메시지의 대형 도구 인수 절삭 설정.
                None이면 인수 절삭 비활성화.
            history_path_prefix: 대화 히스토리 저장 경로 접두사.

        사용 예시:
            ```python
            from deepagents.middleware.summarization import SummarizationMiddleware
            from deepagents.backends import StateBackend

            middleware = SummarizationMiddleware(
                model="gpt-4o-mini",
                backend=StateBackend(),
                trigger=("tokens", 100000),  # 10만 토큰 초과 시 요약
                keep=("messages", 20),       # 최근 20개 메시지 보존
            )
            ```
        """
        # Initialize langchain helper for core summarization logic
        self._lc_helper = LCSummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            **deprecated_kwargs,
        )

        # Deep Agents specific attributes
        self._backend = backend
        self._history_path_prefix = history_path_prefix

        # Parse truncate_args_settings
        if truncate_args_settings is None:
            self._truncate_args_trigger = None
            self._truncate_args_keep: ContextSize = ("messages", 20)
            self._max_arg_length = 2000
            self._truncation_text = "...(argument truncated)"
        else:
            self._truncate_args_trigger = truncate_args_settings.get("trigger")
            self._truncate_args_keep = truncate_args_settings.get("keep", ("messages", 20))
            self._max_arg_length = truncate_args_settings.get("max_length", 2000)
            self._truncation_text = truncate_args_settings.get("truncation_text", "...(argument truncated)")

    # Delegated properties and methods from langchain helper
    @property
    def model(self) -> BaseChatModel:
        """The language model used for generating summaries."""
        return self._lc_helper.model

    @property
    def token_counter(self) -> TokenCounter:
        """Function to count tokens in messages."""
        return self._lc_helper.token_counter

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
        return self._lc_helper._get_profile_limits()

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage."""
        return self._lc_helper._should_summarize(messages, total_tokens)

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        return self._lc_helper._determine_cutoff_index(messages)

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        return self._lc_helper._partition_messages(conversation_messages, cutoff_index)

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages."""
        return self._lc_helper._create_summary(messages_to_summarize)

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages (async)."""
        return await self._lc_helper._acreate_summary(messages_to_summarize)

    def _get_backend(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            # Because we're using `before_model`, which doesn't receive `config` as a
            # parameter, we access it via `runtime.config` instead.
            # Cast is safe: empty dict `{}` is a valid `RunnableConfig` (all fields are
            # optional in TypedDict).
            config = cast("RunnableConfig", getattr(runtime, "config", {}))

            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
        return self._backend

    def _get_thread_id(self) -> str:
        """Extract `thread_id` from langgraph config.

        Uses `get_config()` to access the `RunnableConfig` from langgraph's
        `contextvar`. Falls back to a generated session ID if not available.

        Returns:
            Thread ID string from config, or a generated session ID
                (e.g., `'session_a1b2c3d4'`) if not in a runnable context.
        """
        try:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
        except RuntimeError:
            # Not in a runnable context
            pass

        # Fallback: generate session ID
        generated_id = f"session_{uuid.uuid4().hex[:8]}"
        logger.debug("No thread_id found, using generated session ID: %s", generated_id)
        return generated_id

    def _get_history_path(self) -> str:
        """Generate path for storing conversation history.

        Returns a single file per thread that gets appended to over time.

        Returns:
            Path string like `'/conversation_history/{thread_id}.md'`
        """
        thread_id = self._get_thread_id()
        return f"{self._history_path_prefix}/{thread_id}.md"

    def _is_summary_message(self, msg: AnyMessage) -> bool:
        """Check if a message is a previous summarization message.

        Summary messages are `HumanMessage` objects with `lc_source='summarization'` in
        `additional_kwargs`. These should be filtered from offloads to avoid redundant
        storage during chained summarization.

        Args:
            msg: Message to check.

        Returns:
            Whether this is a summary `HumanMessage` from a previous summarization.
        """
        if not isinstance(msg, HumanMessage):
            return False
        return msg.additional_kwargs.get("lc_source") == "summarization"

    def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Filter out previous summary messages from a message list.

        When chained summarization occurs, we don't want to re-offload the previous
        summary `HumanMessage` since the original messages are already stored in the
        backend.

        Args:
            messages: List of messages to filter.

        Returns:
            Messages without previous summary `HumanMessage` objects.
        """
        return [msg for msg in messages if not self._is_summary_message(msg)]

    def _build_new_messages_with_path(self, summary: str, file_path: str | None) -> list[AnyMessage]:
        """Build the summary message with optional file path reference.

        Args:
            summary: The generated summary text.
            file_path: Path where conversation history was stored, or `None`.

                Optional since offloading may fail.

        Returns:
            List containing the summary `HumanMessage`.
        """
        if file_path is not None:
            content = f"""\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"

        return [
            HumanMessage(
                content=content,
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    def _get_effective_messages(self, request: ModelRequest) -> list[AnyMessage]:
        """Generate effective messages for model call based on summarization event.

        Delegates to `_apply_event_to_messages` so the defensive checks
        (malformed event, out-of-bounds cutoff) are shared with the compact
        tool path.

        Args:
            request: The model request with messages from state.

        Returns:
            The effective message list to use for the model call.
        """
        event = request.state.get("_summarization_event")
        return self._apply_event_to_messages(request.messages, event)

    @staticmethod
    def _apply_event_to_messages(
        messages: list[AnyMessage],
        event: SummarizationEvent | None,
    ) -> list[AnyMessage]:
        """Reconstruct effective messages from raw state messages and a summarization event.

        When a prior summarization event exists, the effective conversation is
        the summary message followed by all messages from `cutoff_index` onward.

        Args:
            messages: Full message list from state.
            event: The `_summarization_event` dict, or `None`.

        Returns:
            The effective message list the model would see.
        """
        if event is None:
            return list(messages)

        try:
            summary_msg = event["summary_message"]
            cutoff_idx = event["cutoff_index"]
        except (KeyError, TypeError) as exc:
            logger.warning("Malformed _summarization_event (missing keys): %s", exc)
            return list(messages)

        if cutoff_idx > len(messages):
            logger.warning(
                "Summarization cutoff_index %d exceeds message count %d; remaining slice will be empty",
                cutoff_idx,
                len(messages),
            )
            return [summary_msg]

        result: list[AnyMessage] = [summary_msg]
        result.extend(messages[cutoff_idx:])
        return result

    @staticmethod
    def _compute_state_cutoff(
        event: SummarizationEvent | None,
        effective_cutoff: int,
    ) -> int:
        """Translate an effective-list cutoff index to an absolute state index.

        When a prior summarization event exists, the effective message list
        starts with the summary message at index 0. The -1 accounts for the
        summary message at effective index 0, which does not correspond to a
        real state message -- the effective cutoff already counts it, so we
        subtract 1 to avoid double-counting.

        Args:
            event: The prior `_summarization_event`, or `None`.
            effective_cutoff: Cutoff index within the effective message list.

        Returns:
            The absolute cutoff index for the state.
        """
        if event is None:
            return effective_cutoff
        prior_cutoff = event.get("cutoff_index")
        if not isinstance(prior_cutoff, int):
            logger.warning("Malformed _summarization_event: missing cutoff_index")
            return effective_cutoff
        return prior_cutoff + effective_cutoff - 1

    def _should_truncate_args(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Check if argument truncation should be triggered.

        Args:
            messages: Current message history.
            total_tokens: Total token count of messages.

        Returns:
            True if truncation should occur, False otherwise.
        """
        if self._truncate_args_trigger is None:
            return False

        trigger_type, trigger_value = self._truncate_args_trigger

        if trigger_type == "messages":
            return len(messages) >= trigger_value
        if trigger_type == "tokens":
            return total_tokens >= trigger_value
        if trigger_type == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return False
            threshold = int(max_input_tokens * trigger_value)
            if threshold <= 0:
                threshold = 1
            return total_tokens >= threshold

        return False

    def _determine_truncate_cutoff_index(self, messages: list[AnyMessage]) -> int:  # noqa: PLR0911
        """Determine the cutoff index for argument truncation based on keep policy.

        Messages at index >= cutoff should be preserved without truncation.
        Messages at index < cutoff can have their tool args truncated.

        Args:
            messages: Current message history.

        Returns:
            Index where truncation cutoff occurs. Messages before this index
            should have args truncated, messages at/after should be preserved.
        """
        keep_type, keep_value = self._truncate_args_keep

        if keep_type == "messages":
            # Keep the most recent N messages
            if len(messages) <= keep_value:
                return len(messages)  # All messages are recent
            return int(len(messages) - keep_value)

        if keep_type in {"tokens", "fraction"}:
            # Calculate target token count
            if keep_type == "fraction":
                max_input_tokens = self._get_profile_limits()
                if max_input_tokens is None:
                    # Fallback to message count if profile not available
                    messages_to_keep = 20
                    if len(messages) <= messages_to_keep:
                        return len(messages)
                    return len(messages) - messages_to_keep
                target_token_count = int(max_input_tokens * keep_value)
            else:
                target_token_count = int(keep_value)

            if target_token_count <= 0:
                target_token_count = 1

            # Keep recent messages up to token limit
            tokens_kept = 0
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = self._lc_helper._partial_token_counter([messages[i]])
                if tokens_kept + msg_tokens > target_token_count:
                    return i + 1
                tokens_kept += msg_tokens
            return 0  # All messages are within token limit

        return len(messages)

    def _truncate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Truncate large arguments in a single tool call.

        Args:
            tool_call: The tool call dictionary to truncate.

        Returns:
            A copy of the tool call with large arguments truncated.
        """
        args = tool_call.get("args", {})

        truncated_args = {}
        modified = False

        for key, value in args.items():
            if isinstance(value, str) and len(value) > self._max_arg_length:
                truncated_args[key] = value[:20] + self._truncation_text
                modified = True
            else:
                truncated_args[key] = value

        if modified:
            return {
                **tool_call,
                "args": truncated_args,
            }
        return tool_call

    def _truncate_args(
        self,
        messages: list[AnyMessage],
        system_message: SystemMessage | None,
        tools: list[BaseTool | dict[str, Any]] | None,
    ) -> tuple[list[AnyMessage], bool]:
        """Truncate large tool call arguments in old messages.

        Args:
            messages: Messages to potentially truncate.
            system_message: Optional system message for token counting.
            tools: Optional tools for token counting.

        Returns:
            Tuple of (truncated_messages, modified). If modified is False,
            truncated_messages is the same as input messages.
        """
        counted_messages = [system_message, *messages] if system_message is not None else messages
        try:
            total_tokens = self.token_counter(counted_messages, tools=tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        if not self._should_truncate_args(messages, total_tokens):
            return messages, False

        cutoff_index = self._determine_truncate_cutoff_index(messages)
        if cutoff_index >= len(messages):
            return messages, False

        # Process messages before the cutoff
        truncated_messages = []
        modified = False

        for i, msg in enumerate(messages):
            if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
                # Check if this AIMessage has tool calls we need to truncate
                truncated_tool_calls = []
                msg_modified = False

                for tool_call in msg.tool_calls:
                    if tool_call["name"] in {"write_file", "edit_file"}:
                        truncated_call = self._truncate_tool_call(tool_call)  # ty: ignore[invalid-argument-type]
                        if truncated_call != tool_call:
                            msg_modified = True
                        truncated_tool_calls.append(truncated_call)
                    else:
                        truncated_tool_calls.append(tool_call)

                if msg_modified:
                    # Create a new AIMessage with truncated tool calls
                    truncated_msg = msg.model_copy()
                    truncated_msg.tool_calls = truncated_tool_calls
                    truncated_messages.append(truncated_msg)
                    modified = True
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)

        return truncated_messages, modified

    def _offload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
    ) -> str | None:
        """Persist messages to backend before summarization.

        Appends evicted messages to a single markdown file per thread. Each
        summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        A ``None`` return is non-fatal; callers may proceed without the
        offloaded history.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.

        Returns:
            The file path where history was offloaded, or ``None`` on failure.
        """
        path = self._get_history_path()

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat()
        new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages)}\n\n"

        # Read existing content (if any) and append.
        # Note: We use download_files() instead of read() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        existing_content = ""
        try:
            responses = backend.download_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = backend.edit(path, existing_content, combined_content) if existing_content else backend.write(path, combined_content)
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return path

    async def _aoffload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
    ) -> str | None:
        """Persist messages to backend before summarization (async).

        Appends evicted messages to a single markdown file per thread. Each
        summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        A ``None`` return is non-fatal; callers may proceed without the
        offloaded history.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.

        Returns:
            The file path where history was offloaded, or ``None`` on failure.
        """
        path = self._get_history_path()

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat()
        new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages)}\n\n"

        # Read existing content (if any) and append.
        # Note: We use adownload_files() instead of aread() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        existing_content = ""
        try:
            responses = await backend.adownload_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = (
                await backend.aedit(path, existing_content, combined_content) if existing_content else await backend.awrite(path, combined_content)
            )
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return path

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Process messages before model invocation, with history offloading and arg truncation.

        First applies any previous summarization events to reconstruct the effective message list.
        Then truncates large tool arguments in old messages if configured.
        Finally offloads messages to backend before summarization if thresholds are met.

        Control flow details:

        - If thresholds say "do not summarize", we still attempt one normal
            model call with the current effective/truncated messages.
        - If that call raises `ContextOverflowError`, we immediately fall back to
            the summarization path and retry the model call with
            `summary_message + preserved_recent_messages`.

        Unlike the legacy `before_model` approach, this does NOT modify the LangGraph state.
        Instead, it tracks summarization events in middleware state and modifies the model
        request directly.

        Args:
            request: The model request to process.
            handler: The handler to call with the (possibly modified) request.

        Returns:
            A plain `ModelResponse` when no summarization event is created, or
                an `ExtendedModelResponse` that updates `_summarization_event`
                with `cutoff_index`, `summary_message`, and `file_path`.

                If `cutoff_index <= 0`, no compaction occurs and no
                `_summarization_event` update is emitted.
        """
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Step 1: Truncate args if configured
        truncated_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 2: Check if summarization should happen
        counted_messages = [request.system_message, *truncated_messages] if request.system_message is not None else truncated_messages
        try:
            total_tokens = self.token_counter(counted_messages, tools=request.tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        if not should_summarize:
            try:
                return handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                pass
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend first so history is preserved before summarization.
        # If offload fails, summarization still proceeds (with file_path=None).
        backend = self._get_backend(request.state, request.runtime)
        file_path = self._offload_to_backend(backend, messages_to_summarize)
        if file_path is None:
            msg = "Offloading conversation history to backend failed during summarization. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)

        # Generate summary
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = handler(request.override(messages=modified_messages))

        # Return ExtendedModelResponse with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Process messages before model invocation, with history offloading and arg truncation (async).

        First applies any previous summarization events to reconstruct the effective message list.
        Then truncates large tool arguments in old messages if configured.
        Finally offloads messages to backend before summarization if thresholds are met.

        Control flow details:

        - If thresholds say "do not summarize", we still attempt one normal
            model call with the current effective/truncated messages.
        - If that call raises `ContextOverflowError`, we immediately fall back
            to the summarization path and retry the model call with
            `summary_message + preserved_recent_messages`.

        Unlike the legacy `abefore_model` approach, this does NOT modify the LangGraph state.
        Instead, it tracks summarization events in middleware state and modifies the model
        request directly.

        Args:
            request: The model request to process.
            handler: The handler to call with the (possibly modified) request.

        Returns:
            A plain `ModelResponse` when no summarization event is created, or
                an `ExtendedModelResponse` that updates `_summarization_event`
                with `cutoff_index`, `summary_message`, and `file_path`.

                If `cutoff_index <= 0`, no compaction occurs and no
                `_summarization_event` update is emitted.
        """
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Step 1: Truncate args if configured
        truncated_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 2: Check if summarization should happen
        counted_messages = [request.system_message, *truncated_messages] if request.system_message is not None else truncated_messages
        try:
            total_tokens = self.token_counter(counted_messages, tools=request.tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        if not should_summarize:
            try:
                return await handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                pass
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return await handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend and generate summary concurrently -- they are independent.
        # If offload fails, summarization still proceeds (with file_path=None).
        backend = self._get_backend(request.state, request.runtime)
        file_path, summary = await asyncio.gather(
            self._aoffload_to_backend(backend, messages_to_summarize),
            self._acreate_summary(messages_to_summarize),
        )
        if file_path is None:
            msg = "Offloading conversation history to backend failed during summarization. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = await handler(request.override(messages=modified_messages))

        # Return ExtendedModelResponse with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )


# Public alias
SummarizationMiddleware = _DeepAgentsSummarizationMiddleware


def create_summarization_middleware(
    model: BaseChatModel,
    backend: BACKEND_TYPES,
) -> _DeepAgentsSummarizationMiddleware:
    """모델 인식 기본값으로 SummarizationMiddleware를 생성합니다.

    모델 프로파일에서 trigger, keep, 절삭 설정을 계산하고
    (프로파일이 없으면 고정 토큰 폴백 사용) 설정된 미들웨어를 반환합니다.

    Args:
        model: 해석된 채팅 모델 인스턴스.
        backend: 대화 히스토리 영구 저장을 위한 백엔드 인스턴스 또는 팩토리.

    Returns:
        설정된 SummarizationMiddleware 인스턴스.
    """
    from langchain.chat_models import BaseChatModel as RuntimeBaseChatModel  # noqa: PLC0415

    if not isinstance(model, RuntimeBaseChatModel):
        msg = "`create_summarization_middleware` expects `model` to be a `BaseChatModel` instance."
        raise TypeError(msg)

    defaults = compute_summarization_defaults(model)
    return SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=defaults["trigger"],
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
        truncate_args_settings=defaults["truncate_args_settings"],
    )


def create_summarization_tool_middleware(
    model: str | BaseChatModel,
    backend: BACKEND_TYPES,
) -> SummarizationToolMiddleware:
    """모델 인식 기본값으로 SummarizationToolMiddleware를 생성하는 편의 팩토리.

    `create_summarization_middleware`를 통해 `SummarizationMiddleware`를 생성하고,
    그것을 `SummarizationToolMiddleware`로 래핑합니다.

    Args:
        model: Chat model instance or model string (e.g., `"anthropic:claude-sonnet-4-20250514"`).
        backend: Backend instance or factory for persisting conversation history.

    Returns:
        Configured `SummarizationToolMiddleware` instance.

    Example:
        Using the default `StateBackend`:

        ```python
        from deepagents import create_deep_agent
        from deepagents.backends import StateBackend
        from deepagents.middleware.summarization import (
            create_summarization_tool_middleware,
        )

        model = "openai:gpt-5.4"
        agent = create_deep_agent(
            model=model,
            middleware=[
                create_summarization_tool_middleware(model, StateBackend),
            ],
        )
        ```

        Using a custom backend instance (e.g., Daytona Sandbox):

        ```python
        from daytona import Daytona
        from deepagents import create_deep_agent
        from deepagents.middleware.summarization import (
            create_summarization_tool_middleware,
        )
        from langchain_daytona import DaytonaSandbox

        sandbox = Daytona().create()
        backend = DaytonaSandbox(sandbox=sandbox)
        model = "openai:gpt-5.4"
        agent = create_deep_agent(
            model=model,
            backend=backend,
            middleware=[
                create_summarization_tool_middleware(model, backend),
            ],
        )
        ```
    """
    from deepagents._models import resolve_model  # noqa: PLC0415

    if isinstance(model, str):
        model = resolve_model(model)
    summarization = create_summarization_middleware(model, backend)
    return SummarizationToolMiddleware(summarization)


class SummarizationToolMiddleware(AgentMiddleware):
    """수동 압축을 위한 `compact_conversation` 도구를 제공하는 미들웨어.

    `SummarizationMiddleware` 인스턴스와 조합하여 그 요약 ��진
    (모델, 백엔드, 트리거 임계값)을 재사용합니다.
    에이전트가 자신의 컨텍스트 윈도우를 직접 압축할 수 있게 합니다.

    이 미들웨어는 **자동으로 압축하지 않습니다**. 압축은 `compact_conversation`이
    일반 도구 호출로 실행될 때(모델에 의해 또는 사용자의 명시적 요청으로)만 발생합니다.

    너무 이른 압축을 방지하기 위해, 도구 실행은 `_is_eligible_for_compaction`으로
    게이팅됩니다. 보��된 사용량이 자동 요약 트리거의 약 50%에 도달해야 실행됩니다.

    도구와 자동 요약은 동일한 `_summarization_event` 상태 키를 공유하므로
    올바르게 상호 운용됩니다.

    더 간단한 설정을 위해 `create_summarization_tool_middleware`를 사용하면
    두 단계를 모두 처리합니다.

    사용 예시:
        ```python
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
            SummarizationToolMiddleware,
        )

        summ = SummarizationMiddleware(model="gpt-4o-mini", backend=backend)
        tool_mw = SummarizationToolMiddleware(summ)

        agent = create_deep_agent(middleware=[summ, tool_mw])
        ```
    """

    state_schema = SummarizationState

    def __init__(self, summarization: _DeepAgentsSummarizationMiddleware) -> None:
        """Initialize with a reference to the summarization middleware.

        Args:
            summarization: The `SummarizationMiddleware` instance whose
                summarization engine this tool will delegate to.
        """
        self._summarization = summarization
        self.tools: list[BaseTool] = [self._create_compact_tool()]

    def _resolve_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Resolve backend from instance or factory using a `ToolRuntime`.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        backend = self._summarization._backend
        if callable(backend):
            return backend(runtime)  # ty: ignore[call-top-callable]
        return backend

    def _create_compact_tool(self) -> BaseTool:
        """Create the `compact_conversation` structured tool.

        Returns:
            A `StructuredTool` with both sync and async implementations.
        """
        from langchain_core.tools import StructuredTool  # noqa: PLC0415

        mw = self

        def sync_compact(runtime: ToolRuntime) -> Command:
            return mw._run_compact(runtime)

        async def async_compact(runtime: ToolRuntime) -> Command:
            return await mw._arun_compact(runtime)

        return StructuredTool.from_function(
            name="compact_conversation",
            description=(
                "Compact the conversation by summarizing older messages "
                "into a concise summary. Use this proactively when the "
                "conversation is getting long to free up context window "
                "space. This tool takes no arguments."
            ),
            func=sync_compact,
            coroutine=async_compact,
            # infer_schema=False,  # noqa: ERA001
            # args_schema=CompactConversationSchema,  # noqa: ERA001
        )

    def _build_compact_result(
        self,
        runtime: ToolRuntime,
        to_summarize: list[AnyMessage],
        summary: str,
        file_path: str | None,
        event: SummarizationEvent | None,
        cutoff: int,
    ) -> Command:
        """Build the `Command` result for a successful compact operation.

        Shared by both sync and async compact paths to avoid duplicating
        the event construction and cutoff arithmetic.

        Args:
            runtime: The tool runtime context.
            to_summarize: Messages that were summarized.
            summary: The generated summary text.
            file_path: Backend path where history was offloaded, or ``None``.
            event: The prior `_summarization_event`, or ``None``.
            cutoff: The cutoff index within the effective message list.

        Returns:
            A `Command` with `_summarization_event` state update and a
            confirmation `ToolMessage`.
        """
        s = self._summarization
        summary_msg = s._build_new_messages_with_path(summary, file_path)[0]
        state_cutoff = s._compute_state_cutoff(event, cutoff)

        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff,
            "summary_message": summary_msg,  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        return Command(
            update={
                "_summarization_event": new_event,
                "messages": [
                    ToolMessage(
                        content=f"Conversation compacted. Summarized {len(to_summarize)} messages into a concise summary.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    @staticmethod
    def _nothing_to_compact(tool_call_id: str) -> Command:
        """Return a "nothing to compact" result for the compact tool.

        Args:
            tool_call_id: The originating tool call ID.

        Returns:
            A `Command` with a descriptive `ToolMessage`.
        """
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Nothing to compact yet \u2014 conversation is within the token budget.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    @staticmethod
    def _compact_error(tool_call_id: str, exc: BaseException) -> Command:
        """Return an error result for the compact tool.

        Args:
            tool_call_id: The originating tool call ID.
            exc: The exception that caused the failure.

        Returns:
            A `Command` with an error `ToolMessage`.
        """
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "Compaction failed: an error occurred while "
                            f"generating the summary ({type(exc).__name__}: "
                            f"{exc}). The conversation has not been compacted "
                            "— no messages were summarized or removed."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    def _is_eligible_for_compaction(self, messages: list[AnyMessage]) -> bool:
        """Check if manual compaction is currently allowed.

        This is an eligibility gate for `compact_conversation` tool calls, not a
        background trigger. The conversation must be at or above about 50% of
        the configured auto-summarization trigger:

        - For `("tokens", N)`, eligibility starts at `0.5 * N`.
        - For `("fraction", F)`, eligibility starts at `0.5 * F` of model max
            input tokens.

        Uses reported usage metadata when available.
        """
        lc = self._summarization._lc_helper
        trigger_conditions = lc._trigger_conditions
        if not trigger_conditions:
            return False

        for kind, value in trigger_conditions:
            if kind == "tokens":
                threshold = int(value * 0.5)
                if threshold <= 0:
                    threshold = 1
                if lc._should_summarize_based_on_reported_tokens(messages, threshold):
                    return True
            elif kind == "fraction":
                max_input_tokens = lc._get_profile_limits()
                if max_input_tokens is None:
                    continue
                threshold = int(max_input_tokens * value * 0.5)
                if threshold <= 0:
                    threshold = 1
                if lc._should_summarize_based_on_reported_tokens(messages, threshold):
                    return True
        return False

    def _run_compact(self, runtime: ToolRuntime) -> Command:
        """Synchronous compact implementation called by the compact tool.

        Args:
            runtime: The `ToolRuntime` injected by the tool node.

        Returns:
            A `Command` with `_summarization_event` state update, or a
                `Command` with a "nothing to compact" or error `ToolMessage`.
        """
        s = self._summarization
        tool_call_id = runtime.tool_call_id or ""
        messages = runtime.state.get("messages", [])
        event = runtime.state.get("_summarization_event")
        effective = s._apply_event_to_messages(messages, event)

        if not self._is_eligible_for_compaction(effective):
            return self._nothing_to_compact(tool_call_id)

        cutoff = s._determine_cutoff_index(effective)
        if cutoff == 0:
            return self._nothing_to_compact(tool_call_id)

        try:
            to_summarize, _ = s._partition_messages(effective, cutoff)
            summary = s._create_summary(to_summarize)
            backend = self._resolve_backend(runtime)
            file_path = s._offload_to_backend(backend, to_summarize)
        except Exception as exc:  # tool must return a ToolMessage, not raise
            logger.exception("compact_conversation tool failed")
            return self._compact_error(tool_call_id, exc)

        return self._build_compact_result(runtime, to_summarize, summary, file_path, event, cutoff)

    async def _arun_compact(self, runtime: ToolRuntime) -> Command:
        """Async variant of `_run_compact`. See that method for details.

        Args:
            runtime: The `ToolRuntime` injected by the tool node.

        Returns:
            A `Command` with `_summarization_event` state update, or a
                `Command` with a "nothing to compact" or error `ToolMessage`.
        """
        s = self._summarization
        tool_call_id = runtime.tool_call_id or ""
        messages = runtime.state.get("messages", [])
        event = runtime.state.get("_summarization_event")
        effective = s._apply_event_to_messages(messages, event)

        if not self._is_eligible_for_compaction(effective):
            return self._nothing_to_compact(tool_call_id)

        cutoff = s._determine_cutoff_index(effective)
        if cutoff == 0:
            return self._nothing_to_compact(tool_call_id)

        try:
            to_summarize, _ = s._partition_messages(effective, cutoff)
            summary = await s._acreate_summary(to_summarize)
            backend = self._resolve_backend(runtime)
            file_path = await s._aoffload_to_backend(backend, to_summarize)
        except Exception as exc:  # tool must return a ToolMessage, not raise
            logger.exception("compact_conversation tool failed")
            return self._compact_error(tool_call_id, exc)

        return self._build_compact_result(runtime, to_summarize, summary, file_path, event, cutoff)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject a compact-tool usage nudge into the system prompt.

        This only updates prompt text so the model can decide whether to call
        `compact_conversation` earlier in long sessions. It does not execute the
        tool automatically.

        Args:
            request: The model request to process.
            handler: The handler to call with the modified request.

        Returns:
            The model response from the handler.
        """
        new_system_message = append_to_system_message(request.system_message, SUMMARIZATION_SYSTEM_PROMPT)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject a compact-tool usage nudge into the system prompt (async).

        This only updates prompt text so the model can decide whether to call
        `compact_conversation` earlier in long sessions. It does not execute the
        tool automatically.

        Args:
            request: The model request to process.
            handler: The handler to call with the modified request.

        Returns:
            The model response from the handler.
        """
        new_system_message = append_to_system_message(request.system_message, SUMMARIZATION_SYSTEM_PROMPT)
        return await handler(request.override(system_message=new_system_message))
