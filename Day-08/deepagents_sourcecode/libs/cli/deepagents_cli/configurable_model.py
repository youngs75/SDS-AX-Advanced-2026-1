"""LangGraph 런타임 컨텍스트를 통한 런타임 모델 선택을 위한 CLI 미들웨어입니다.

그래프를 다시 컴파일하지 않고 `agent.astream()` / `agent.invoke()`의 `context=`을 통해 `CLIContext`을 전달하여
호출별로 모델을 전환할 수 있습니다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents._models import model_matches_spec  # noqa: PLC2701
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


def _is_anthropic_model(model: object) -> bool:
    """해결된 모델이 `'anthropic'`을(를) 공급자로 보고하는지 확인하세요.

    `BaseChatModel`의 `_get_ls_params`을(를) 사용하여 공급자 이름을 읽습니다.

Returns:
        모델의 `ls_provider`이(가) `'anthropic'`인 경우 `True`입니다.

    """
    try:
        ls_params = model._get_ls_params()  # type: ignore[attr-defined]
    except (AttributeError, TypeError, RuntimeError):
        logger.debug(
            "_get_ls_params raised for %s; assuming non-Anthropic",
            type(model).__name__,
        )
        return False
    return isinstance(ls_params, dict) and ls_params.get("ls_provider") == "anthropic"


_ANTHROPIC_ONLY_SETTINGS: set[str] = {"cache_control"}
"""Anthropic 특정 미들웨어(예: `AnthropicPromptCachingMiddleware`)에 의해 삽입된 키로서 다른 공급자가 허용하지 않으며
공급자 간 스왑 시 제거해야 합니다.
"""


def _apply_overrides(request: ModelRequest) -> ModelRequest:
    """런타임 시 `CLIContext`의 모델/매개변수 재정의를 적용합니다.

Returns:
        `CLIContext`이 없거나 `CLIContext`이 없으면 원래 요청은 변경되지 않습니다.
            재정의가 없습니다. 그렇지 않으면 재정의가 포함된 새 요청입니다.

    """
    runtime = request.runtime
    if runtime is None:
        return request

    ctx = runtime.context
    if not isinstance(ctx, dict):
        return request

    overrides: dict[str, Any] = {}

    # Model swap
    new_model = None
    model = ctx.get("model")
    if model and not model_matches_spec(request.model, model):
        from deepagents_cli.config import create_model
        from deepagents_cli.model_config import ModelConfigError

        logger.debug("Overriding model to %s", model)
        try:
            model_result = create_model(model)
            new_model = model_result.model
        except ModelConfigError:
            logger.exception(
                "Failed to resolve runtime model override '%s'; "
                "continuing with current model",
                model,
            )
            return request
        overrides["model"] = new_model

    # Param merge
    model_params = ctx.get("model_params", {})
    if model_params:
        overrides["model_settings"] = {**request.model_settings, **model_params}

    if not overrides:
        return request

    # When switching away from Anthropic, strip provider-specific settings
    # that would cause errors on other providers (e.g. cache_control passed
    # to the OpenAI SDK raises TypeError).
    if new_model is not None and not _is_anthropic_model(new_model):
        settings = overrides.get("model_settings", request.model_settings)
        dropped = settings.keys() & _ANTHROPIC_ONLY_SETTINGS
        if dropped:
            logger.debug(
                "Stripped Anthropic-only settings %s for non-Anthropic model",
                dropped,
            )
            overrides["model_settings"] = {
                k: v for k, v in settings.items() if k not in dropped
            }

    # Patch the Model Identity section in the system prompt so the new model
    # sees its own name/provider/context-limit, not the original's.
    # We read metadata from model_result (not the CLI settings singleton)
    # because the middleware runs in the server subprocess where settings
    # are never updated by /model.
    if new_model is not None and request.system_prompt:
        from deepagents_cli.agent import (
            MODEL_IDENTITY_RE,
            build_model_identity_section,
        )

        prompt = request.system_prompt
        new_identity = build_model_identity_section(
            model_result.model_name,
            provider=model_result.provider,
            context_limit=model_result.context_limit,
            unsupported_modalities=model_result.unsupported_modalities,
        )
        patched = MODEL_IDENTITY_RE.sub(new_identity, prompt, count=1)
        if patched != prompt:
            overrides["system_prompt"] = patched
        elif "### Model Identity" in prompt:
            logger.warning(
                "System prompt contains '### Model Identity' but regex "
                "did not match; identity section was NOT updated for "
                "model '%s'. The regex may be out of sync with the "
                "prompt template.",
                model_result.model_name,
            )

    return request.override(**overrides)


class ConfigurableModelMiddleware(AgentMiddleware):
    """`runtime.context`에서 모델 또는 호출별 설정을 바꾸세요."""

    def wrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """런타임 재정의를 적용하고 다음 핸들러에 위임합니다."""  # noqa: DOC201
        return handler(_apply_overrides(request))

    async def awrap_model_call(  # noqa: PLR6301
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """런타임 재정의를 적용하고 다음 비동기 처리기에 위임합니다."""  # noqa: DOC201
        return await handler(_apply_overrides(request))
