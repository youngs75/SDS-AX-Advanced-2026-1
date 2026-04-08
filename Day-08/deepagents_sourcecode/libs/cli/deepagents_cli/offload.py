"""/offload 명령에 대한 비즈니스 논리입니다.

텍스트 앱과 독립적으로 테스트할 수 있도록 UI 계층에서 핵심 오프로드 워크플로를 추출합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately

from deepagents_cli.config import create_model
from deepagents_cli.textual_adapter import format_token_count

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.summarization import (
        SummarizationEvent,
        SummarizationMiddleware,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffloadResult:
    """성공적인 오프로드 결과입니다."""

    new_event: SummarizationEvent
    """에이전트 상태에 쓸 요약 이벤트입니다."""

    messages_offloaded: int
    """오프로드된 이전 메시지 수입니다."""

    messages_kept: int
    """컨텍스트에 보관된 최근 메시지 수입니다."""

    tokens_before: int
    """오프로드 전 대화의 대략적인 토큰 수입니다."""

    tokens_after: int
    """오프로드 후 대화의 대략적인 토큰 수입니다."""

    pct_decrease: int
    """토큰 사용량이 백분율로 감소합니다."""

    offload_warning: str | None
    """백엔드 쓰기가 실패한 경우(치명적이지 않음) `None`이 아닙니다."""


@dataclass(frozen=True)
class OffloadThresholdNotMet:
    """오프로드는 아무 작업도 하지 않았고, 대화는 보존 예산 범위 안에 있습니다."""

    conversation_tokens: int
    """대화 메시지만의 대략적인 토큰 수입니다."""

    total_context_tokens: int
    """시스템 오버헤드를 포함한 총 컨텍스트 토큰 수 또는 그렇지 않은 경우 `0`
    토큰 추적기를 사용할 수 있습니다.
    """

    context_limit: int | None
    """모델 컨텍스트 창 제한(사용 가능한 경우)"""

    budget_str: str
    """사람이 읽을 수 있는 보존 예산(예: "20.0K 토큰")"""


class OffloadModelError(Exception):
    """오프로드용 모델을 생성할 수 없는 경우 발생합니다."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_offload_limit(
    keep: tuple[str, int | float], context_limit: int | None
) -> str:
    """오프로드 보존 설정의 형식을 사람이 읽을 수 있는 제한 문자열로 지정합니다.

    Args:
        keep: 요약 기본값의 보존 정책 튜플 `(type, value)`. 여기서 `type`은 `"messages"`, `"tokens"` 또는
              `"fraction"` 중 하나입니다.
        context_limit: 사용 가능한 경우 모델 컨텍스트 제한.

    Returns:
        오프로드 보존 제한을 설명하는 짧은 표시 문자열입니다.

    """
    keep_type, keep_value = keep

    if keep_type == "messages":
        count = int(keep_value)
        noun = "message" if count == 1 else "messages"
        return f"last {count} {noun}"

    if keep_type == "tokens":
        return f"{format_token_count(int(keep_value))} tokens"

    if keep_type == "fraction":
        percent = float(keep_value) * 100
        if context_limit is not None:
            token_limit = max(1, int(context_limit * float(keep_value)))
            return f"{format_token_count(token_limit)} tokens"
        return f"{percent:.0f}% of context window"

    return "current retention threshold"


async def offload_messages_to_backend(
    messages: list[Any],
    middleware: SummarizationMiddleware,
    *,
    thread_id: str,
    backend: BackendProtocol,
) -> str | None:
    """오프로드하기 전에 백엔드 스토리지에 메시지를 쓰세요.

    `SummarizationMiddleware` 오프로드 패턴과 일치하여 대화 기록 파일에 타임스탬프가 표시된 마크다운 섹션으로 메시지를 추가합니다.

    요약 요약 저장을 피하기 위해 미들웨어의 `_filter_summary_messages`을 사용하여 이전 요약 메시지를 필터링합니다.

    Args:
        messages: 오프로드할 메시지입니다.
        middleware: 필터링을 위한 `SummarizationMiddleware` 인스턴스.
        thread_id: 저장소 경로를 파생하는 데 사용되는 스레드 식별자입니다.
        backend: 대화 기록을 유지할 백엔드입니다.

    Returns:
        기록이 저장된 파일 경로, 기록이 없는 경우 `""`(빈 문자열)
            오프로드할 비요약 메시지(오류 아님) 또는 쓰기가 실패한 경우 `None`입니다.

    """
    path = f"/conversation_history/{thread_id}.md"

    # Exclude prior summaries so the offloaded history contains only
    # original messages
    filtered = middleware._filter_summary_messages(messages)
    if not filtered:
        return ""

    timestamp = datetime.now(UTC).isoformat()
    buf = get_buffer_string(filtered)
    new_section = f"## Offloaded at {timestamp}\n\n{buf}\n\n"

    existing_content = ""
    try:
        responses = await backend.adownload_files([path])
        resp = responses[0] if responses else None
        if resp and resp.content is not None and resp.error is None:
            existing_content = resp.content.decode("utf-8")
    except Exception as exc:  # abort write on read failure
        logger.warning(
            "Failed to read existing history at %s; aborting offload to "
            "avoid overwriting prior history: %s",
            path,
            exc,
            exc_info=True,
        )
        return None

    combined = existing_content + new_section

    try:
        result = (
            await backend.aedit(path, existing_content, combined)
            if existing_content
            else await backend.awrite(path, combined)
        )
        if result is None or result.error:
            error_detail = result.error if result else "backend returned None"
            logger.warning(
                "Failed to offload conversation history to %s: %s",
                path,
                error_detail,
            )
            return None
    except Exception as exc:  # defensive: surface write failures gracefully
        logger.warning(
            "Exception offloading conversation history to %s: %s",
            path,
            exc,
            exc_info=True,
        )
        return None

    logger.debug("Offloaded %d messages to %s", len(filtered), path)
    return path


# ---------------------------------------------------------------------------
# Core offload workflow
# ---------------------------------------------------------------------------


async def perform_offload(
    *,
    messages: list[Any],
    prior_event: SummarizationEvent | None,
    thread_id: str,
    model_spec: str,
    profile_overrides: dict[str, Any] | None,
    context_limit: int | None,
    total_context_tokens: int,
    backend: BackendProtocol | None,
) -> OffloadResult | OffloadThresholdNotMet:
    """오프로드 워크플로를 실행합니다. 오래된 메시지와 자유 컨텍스트를 요약합니다.

    Args:
        messages: 에이전트 상태의 현재 대화 메시지입니다.
        prior_event: 기존 `_summarization_event`(있는 경우).
        thread_id: 백엔드 저장소의 스레드 식별자입니다.
        model_spec: 모델 사양 문자열(예: "openai:gpt-4")
        profile_overrides: CLI 플래그에서 선택적 프로필 재정의.
        context_limit: 설정의 모델 컨텍스트 제한입니다.
        total_context_tokens: 현재 총 컨텍스트 토큰 수 또는 토큰 추적기를 사용할 수 없는 경우 `0`입니다.
        backend: 오프로드된 기록을 유지하기 위한 백엔드입니다.

    Returns:
        성공 시 `OffloadResult`, 성공 시 `OffloadThresholdNotMet`
            대화는 보존 예산 내에 있습니다.

    Raises:
        OffloadModelError: 모델을 생성할 수 없는 경우.

    """
    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    try:
        result = create_model(model_spec, profile_overrides=profile_overrides)
        model = result.model
    except Exception as exc:
        msg = f"Offload requires a working model configuration: {exc}"
        raise OffloadModelError(msg) from exc

    # Patch context limit into model profile when it differs from the native
    # value (e.g. set via --profile-override or runtime config).
    if context_limit is not None:
        profile = getattr(model, "profile", None)
        native = profile.get("max_input_tokens") if isinstance(profile, dict) else None
        if native != context_limit:
            merged = (
                {**profile, "max_input_tokens": context_limit}
                if isinstance(profile, dict)
                else {"max_input_tokens": context_limit}
            )
            try:
                model.profile = merged  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "Could not patch context limit (%d) into model profile; "
                    "offload budget will use the model's native context window",
                    context_limit,
                    exc_info=True,
                )

    defaults = compute_summarization_defaults(model)
    offload_backend = backend
    if offload_backend is None:
        from deepagents.backends.filesystem import FilesystemBackend

        offload_backend = FilesystemBackend()
        logger.info("Using local FilesystemBackend for offload")

    middleware = SummarizationMiddleware(
        model=model,
        backend=offload_backend,
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
    )

    # Rebuild the message list the model would see, accounting for
    # any prior offload
    effective = middleware._apply_event_to_messages(messages, prior_event)
    cutoff = middleware._determine_cutoff_index(effective)
    budget_str = format_offload_limit(defaults["keep"], context_limit)

    if cutoff == 0:
        return OffloadThresholdNotMet(
            conversation_tokens=count_tokens_approximately(effective),
            total_context_tokens=total_context_tokens,
            context_limit=context_limit,
            budget_str=budget_str,
        )

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)

    tokens_summarized = count_tokens_approximately(to_summarize)
    tokens_kept = count_tokens_approximately(to_keep)
    tokens_before = tokens_summarized + tokens_kept

    # Generate summary first so no side effects occur if the LLM fails
    summary = await middleware._acreate_summary(to_summarize)

    backend_path = await offload_messages_to_backend(
        to_summarize,
        middleware,
        thread_id=thread_id,
        backend=offload_backend,
    )
    offload_warning: str | None = None
    if backend_path is None:
        offload_warning = (
            "Warning: conversation history could not be saved to "
            "storage. Older messages will not be recoverable. "
            "Check logs for details."
        )
        logger.error(
            "Backend write failed for thread %s; offloading will proceed "
            "but messages are not recoverable",
            thread_id,
        )
    file_path = backend_path or None

    summary_msg = middleware._build_new_messages_with_path(summary, file_path)[0]

    # Append token savings note so the model is aware of how much context
    # was reclaimed.
    tokens_summary = count_tokens_approximately([summary_msg])
    tokens_after = tokens_summary + tokens_kept
    pct = (
        round((tokens_before - tokens_after) / tokens_before * 100)
        if tokens_before > 0
        else 0
    )
    summarized_before = format_token_count(tokens_summarized)
    summarized_after = format_token_count(tokens_summary)
    savings_note = (
        f"\n\n{len(to_summarize)} messages were offloaded "
        f"({summarized_before} \u2192 {summarized_after} tokens). "
        f"Total context: {format_token_count(tokens_before)} \u2192 "
        f"{format_token_count(tokens_after)} tokens "
        f"({pct}% decrease), "
        f"{len(to_keep)} messages unchanged."
    )
    summary_msg.content += savings_note

    state_cutoff = middleware._compute_state_cutoff(prior_event, cutoff)

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,  # ty: ignore[invalid-argument-type]
        "file_path": file_path,
    }

    return OffloadResult(
        new_event=new_event,
        messages_offloaded=len(to_summarize),
        messages_kept=len(to_keep),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        pct_decrease=pct,
        offload_warning=offload_warning,
    )
