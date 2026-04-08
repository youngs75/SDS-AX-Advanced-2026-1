"""모델 구성 관리.

TOML 파일에서 모델 구성 로드 및 저장을 처리하여 사용 가능한 모델 및 공급자를 정의하는 구조화된 방법을 제공합니다.
"""


from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import tempfile
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import tomli_w

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_ENV_PREFIX = "DEEPAGENTS_CLI_"


def resolve_env_var(name: str) -> str | None:
    """`DEEPAGENTS_CLI_` 접두사 재정의를 사용하여 환경 변수를 찾습니다.

    `DEEPAGENTS_CLI_{name}`을 먼저 확인한 다음 `{name}`로 대체합니다.

    접두사가 붙은 변수가 환경에 *존재*하는 경우(빈 문자열이라도) 정식 변수는 참조되지 않습니다. 이를 통해 사용자는
    `DEEPAGENTS_CLI_X=""`을 설정하여 정식으로 설정된 키를 숨길 수 있습니다. 함수는 `None`(빈 문자열이 `None`로
    정규화되므로)을 반환하여 정식 값을 효과적으로 억제합니다.

    `name`에 이미 접두사가 있는 경우 무의미한 `DEEPAGENTS_CLI_DEEPAGENTS_CLI_*` 읽기를 방지하기 위해 이중 접두사가 있는
    조회를 건너뜁니다(예: 이름이 사용자의 `config.toml`에서 오는 경우).

    Args:
        name: 표준 환경 변수 이름(예: `ANTHROPIC_API_KEY`)입니다.

    Returns:
        해결된 값 또는 없거나 비어 있는 경우 `None`입니다.

    """

    if not name.startswith(_ENV_PREFIX):
        prefixed = f"{_ENV_PREFIX}{name}"
        if prefixed in os.environ:
            val = os.environ[prefixed]
            if not val and os.environ.get(name):
                logger.debug(
                    "%s is set but empty, blocking non-empty %s. "
                    "Unset %s to use the canonical variable.",
                    prefixed,
                    name,
                    prefixed,
                )
            return val or None
    return os.environ.get(name) or None


class ModelConfigError(Exception):
    """모델 구성 또는 생성이 실패하면 발생합니다."""



@dataclass(frozen=True)
class ModelSpec:
    """`provider:model` 형식의 모델 사양입니다.

    Examples:
        >>> spec = ModelSpec.parse("anthropic:claude-sonnet-4-5")
        >>> spec.provider
        '인류적'
        >>> spec.model
        '클로드 소네트-4-5'
        >>> str(spec)
        '인류:클로드-소네트-4-5'

    """


    provider: str
    """공급자 이름(예: `'anthropic'`, `'openai'`)"""


    model: str
    """모델 식별자(예: `'claude-sonnet-4-5'`, `'gpt-4o'`)입니다."""


    def __post_init__(self) -> None:
        """초기화 후 모델 사양을 검증합니다.

        Raises:
            ValueError: 공급자 또는 모델이 비어 있는 경우.

        """

        if not self.provider:
            msg = "Provider cannot be empty"
            raise ValueError(msg)
        if not self.model:
            msg = "Model cannot be empty"
            raise ValueError(msg)

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        """모델 사양 문자열을 구문 분석합니다.

        Args:
            spec: `'provider:model'` 형식의 모델 사양입니다.

        Returns:
            구문 분석된 ModelSpec 인스턴스.

        Raises:
            ValueError: 사양이 유효한 `'provider:model'` 형식이 아닌 경우.

        """

        if ":" not in spec:
            msg = (
                f"Invalid model spec '{spec}': must be in provider:model format "
                "(e.g., 'anthropic:claude-sonnet-4-5')"
            )
            raise ValueError(msg)
        provider, model = spec.split(":", 1)
        return cls(provider=provider, model=model)

    @classmethod
    def try_parse(cls, spec: str) -> ModelSpec | None:
        """`parse`의 비모금 변형입니다.

        Args:
            spec: `provider:model` 형식의 모델 사양입니다.

        Returns:
            *사양*이 유효하지 않은 경우 `ModelSpec` 또는 `None`을 구문 분석했습니다.

        """

        try:
            return cls.parse(spec)
        except ValueError:
            return None

    def __str__(self) -> str:
        """모델 사양을 `provider:model` 형식의 문자열로 반환합니다."""

        return f"{self.provider}:{self.model}"


class ModelProfileEntry(TypedDict):
    """재정의 추적이 포함된 모델의 프로필 데이터입니다."""


    profile: dict[str, Any]
    """병합된 프로필 사전(업스트림 기본값 + config.toml 재정의)

    키는 제공업체에 따라 다릅니다(예: `max_input_tokens`, `tool_calling`).

    """


    overridden_keys: frozenset[str]
    """값이 config.toml에서 나온 `profile`의 키
    업스트림 공급자 패키지.
    """



class ProviderConfig(TypedDict, total=False):
    """모델 제공자를 위한 구성입니다.

    선택적 `class_path` 필드를 사용하면 `init_chat_model`을 완전히 우회하고 importlib를 통해 임의의
    `BaseChatModel` 하위 클래스를 인스턴스화할 수 있습니다.

    !!! 경고

        `class_path`을 설정하면 사용자 구성 파일에서 임의의 Python 코드가 실행됩니다. 이는 `pyproject.toml` 빌드
        스크립트와 동일한 신뢰 모델을 가지고 있습니다. 즉, 사용자가 자신의 시스템을 제어합니다.

    """


    enabled: bool
    """이 공급자가 모델 전환기에 표시되는지 여부입니다.

    기본값은 `True`입니다. `/model` 선택기에서 패키지 검색 공급자와 해당 모델을 모두 숨기려면 `False`로 설정합니다. LangChain
    공급자 패키지가 전이적 종속성으로 설치되었지만 사용자에게 표시되어서는 안 되는 경우에 유용합니다.

    """


    models: list[str]
    """이 공급자가 제공하는 모델 식별자 목록입니다."""


    api_key_env: str
    """API 키가 포함된 환경 변수 이름입니다."""


    base_url: str
    """사용자 정의 기본 URL."""


    # Level 2: arbitrary BaseChatModel classes

    class_path: str
    """`module.path:ClassName` 형식의 정규화된 Python 클래스입니다.

    설정되면 `create_model`은 이 클래스를 가져오고 `init_chat_model`을 호출하는 대신 직접 인스턴스화합니다.

    """


    params: dict[str, Any]
    """모델 생성자에 추가 키워드 인수가 전달됩니다.

    플랫 키(예: `temperature = 0`)는 이 공급자의 모든 모델에 적용되는 공급자 전체 기본값입니다. 모델 키 하위 테이블(예:
    `[params."qwen3:4b"]`)은 해당 모델의 개별 값만 재정의합니다. 병합이 얕습니다(충돌 시 모델이 승리함).

    """


    profile: dict[str, Any]
    """모델의 런타임 프로필 사전에 재정의가 병합되었습니다.

    단순 키(예: `max_input_tokens = 4096`)는 공급자 전체 기본값입니다. 모델 키 하위 테이블(예:
    `[profile."claude-sonnet-4-5"]`)은 해당 모델의 개별 값만 재정의합니다. 병합이 얕습니다.

    """



DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
"""사용자 수준 Deep Agent 구성을 위한 디렉터리(`~/.deepagents`)입니다."""


DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
"""사용자의 모델 구성 파일(`~/.deepagents/config.toml`)에 대한 경로입니다."""


PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
    "groq": "GROQ_API_KEY",
    "huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "ibm": "WATSONX_APIKEY",
    "litellm": "LITELLM_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "perplexity": "PPLX_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}
"""API 키를 보유하는 env var에 매핑된 잘 알려진 공급자입니다.

`has_provider_credentials`에서 모델 생성 *전에* 자격 증명을 확인하는 데 사용됩니다. 따라서 호출 시 공급자가 실패하도록 하는 대신
UI에 경고 아이콘과 특정 오류 메시지(예: "ANTHROPIC_API_KEY가 설정되지 않음")를 표시할 수 있습니다.

여기에 나열되지 않은 공급자는 구성 파일 검사 또는 langchain 레지스트리 폴백을 거치게 됩니다.
"""



# Module-level caches — cleared by `clear_caches()`.
_available_models_cache: dict[str, list[str]] | None = None
_builtin_providers_cache: dict[str, Any] | None = None
_default_config_cache: ModelConfig | None = None
_provider_profiles_cache: dict[str, dict[str, Any]] = {}
_provider_profiles_lock = threading.Lock()
_profiles_cache: Mapping[str, ModelProfileEntry] | None = None
_profiles_override_cache: tuple[int, Mapping[str, ModelProfileEntry]] | None = None


def clear_caches() -> None:
    """다음 호출이 처음부터 다시 계산되도록 모듈 수준 캐시를 재설정합니다.

    테스트 및 `/reload` 명령용입니다.

    """

    global _available_models_cache, _builtin_providers_cache, _default_config_cache, _profiles_cache, _profiles_override_cache  # noqa: PLW0603, E501  # Module-level caches require global statement
    _available_models_cache = None
    _builtin_providers_cache = None
    _default_config_cache = None
    _provider_profiles_cache.clear()
    _profiles_cache = None
    _profiles_override_cache = None
    invalidate_thread_config_cache()


def _get_builtin_providers() -> dict[str, Any]:
    """langchain의 내장 공급자 레지스트리를 반환합니다.

    최신 `_BUILTIN_PROVIDERS` 이름을 먼저 시도한 다음 이전 langchain 버전의 경우 레거시
    `_SUPPORTED_PROVIDERS`로 대체합니다.

    결과는 첫 번째 호출 후에 캐시됩니다. 재설정하려면 `clear_caches()`을(를) 사용하세요.

    Returns:
        `langchain.chat_models.base`의 공급자 레지스트리 dict입니다.

    """

    global _builtin_providers_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _builtin_providers_cache is not None:
        return _builtin_providers_cache

    # Deferred: langchain.chat_models pulls in heavy provider registry,
    # only needed when resolving provider names for model config.
    from langchain.chat_models import base

    registry: dict[str, Any] | None = getattr(base, "_BUILTIN_PROVIDERS", None)
    if registry is None:
        registry = getattr(base, "_SUPPORTED_PROVIDERS", None)
    _builtin_providers_cache = registry if registry is not None else {}
    return _builtin_providers_cache


def _get_provider_profile_modules() -> list[tuple[str, str]]:
    """langchain의 공급자 레지스트리에서 `(provider, profile_module)` 목록을 구축하세요.

    `langchain.chat_models.base`에서 내장 공급자 레지스트리를 읽어 `init_chat_model`이 알고 있는 모든 공급자를 검색한
    다음 각각에 대한 `<package>.data._profiles` 모듈 경로를 파생합니다.

    Returns:
        `(provider_name, profile_module_path)` 튜플 목록입니다.

    """

    providers = _get_builtin_providers()

    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for provider_name, (module_path, *_rest) in providers.items():
        package_root = module_path.split(".", maxsplit=1)[0]
        profile_module = f"{package_root}.data._profiles"
        key = (provider_name, profile_module)
        if key not in seen:
            seen.add(key)
            result.append((provider_name, profile_module))

    return result


def _load_provider_profiles(module_path: str) -> dict[str, Any]:
    """공급자의 데이터 모듈에서 `_PROFILES`을 로드합니다.

    결과는 `module_path`에 의해 캐시되므로 반복 호출(예: `get_available_models` 및 `get_model_profiles`
    모두에서)은 동일한 사전을 재사용합니다. 재설정하려면 `clear_caches()`을(를) 사용하세요.

    `importlib.util.find_spec`을 사용하여 디스크에서 패키지를 찾고 `spec_from_file_location`를 통해
    `_profiles.py` 파일 *만* 로드합니다.

    Args:
        module_path: 점으로 구분된 모듈 경로(예: `"langchain_openai.data._profiles"`).

    Returns:
        모듈의 `_PROFILES` 사전 또는 빈 dict인 경우
            모듈에는 그러한 속성이 없습니다.

    Raises:
        ImportError: 패키지가 설치되지 않았거나 디스크에서 프로필 모듈을 찾을 수 없는 경우.

    """

    with _provider_profiles_lock:
        cached = _provider_profiles_cache.get(module_path)
        if cached is not None:  # `is not None` so empty profile dicts are cached
            return cached

        parts = module_path.split(".")
        package_root = parts[0]

        spec = importlib.util.find_spec(package_root)
        if spec is None:
            msg = f"Package {package_root} is not installed"
            raise ImportError(msg)

        # Determine the package directory from the spec.
        if spec.origin:
            package_dir = Path(spec.origin).parent
        elif spec.submodule_search_locations:
            package_dir = Path(next(iter(spec.submodule_search_locations)))
        else:
            msg = f"Cannot determine location for {package_root}"
            raise ImportError(msg)

        # Build the path to the target file (e.g., data/_profiles.py).
        relative_parts = parts[1:]  # ["data", "_profiles"]
        profiles_path = package_dir.joinpath(
            *relative_parts[:-1], f"{relative_parts[-1]}.py"
        )

        if not profiles_path.exists():
            msg = f"Profile module not found: {profiles_path}"
            raise ImportError(msg)

        file_spec = importlib.util.spec_from_file_location(module_path, profiles_path)
        if file_spec is None or file_spec.loader is None:
            msg = f"Could not create module spec for {profiles_path}"
            raise ImportError(msg)

        module = importlib.util.module_from_spec(file_spec)
        file_spec.loader.exec_module(module)
        profiles = getattr(module, "_PROFILES", {})
        _provider_profiles_cache[module_path] = profiles
        return profiles


def _profile_module_from_class_path(class_path: str) -> str | None:
    """`class_path` 구성 값에서 프로필 모듈 경로를 파생시킵니다.

    Args:
        class_path: `module.path:ClassName` 형식의 정규화된 클래스입니다.

    Returns:
        `langchain_baseten.data._profiles`과 같은 점으로 구분된 모듈 경로 또는 없음
            `class_path`의 형식이 잘못된 경우.

    """

    if ":" not in class_path:
        return None
    module_part, _ = class_path.split(":", 1)
    package_root = module_part.split(".", maxsplit=1)[0]
    if not package_root:
        return None
    return f"{package_root}.data._profiles"


def get_available_models() -> dict[str, list[str]]:
    """설치된 LangChain 공급자 패키지에서 사용 가능한 모델을 동적으로 가져옵니다.

    각 공급자 패키지에서 모델 프로필을 가져오고 모델 이름을 추출합니다.

    결과는 첫 번째 호출 후에 캐시됩니다. 재설정하려면 `clear_caches()`을(를) 사용하세요.

    Returns:
        공급자 이름을 모델 식별자 목록에 매핑하는 사전입니다.
            langchain 레지스트리의 공급자, 명시적인 모델 목록이 있는 구성 파일 공급자, 패키지가 `_profiles` 모듈을 노출하는
            `class_path` 공급자를 포함합니다.

    """

    global _available_models_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _available_models_cache is not None:
        return _available_models_cache

    available: dict[str, list[str]] = {}
    config = ModelConfig.load()

    # Try to load from langchain provider profile data.
    # Build the list dynamically from langchain's supported-provider registry
    # so new providers are picked up automatically when langchain adds them.
    provider_modules = _get_provider_profile_modules()
    registry_providers: set[str] = set()

    for provider, module_path in provider_modules:
        registry_providers.add(provider)
        # Skip providers explicitly disabled in config.
        if not config.is_provider_enabled(provider):
            logger.debug(
                "Provider '%s' is disabled in config; skipping registry discovery",
                provider,
            )
            continue
        try:
            profiles = _load_provider_profiles(module_path)
        except ImportError:
            logger.debug(
                "Could not import profiles from %s (package may not be installed)",
                module_path,
            )
            continue
        except Exception:
            logger.warning(
                "Failed to load profiles from %s, skipping provider '%s'",
                module_path,
                provider,
                exc_info=True,
            )
            continue

        # Filter to models that support tool calling and text I/O.
        models = [
            name
            for name, profile in profiles.items()
            if profile.get("tool_calling", False)
            and profile.get("text_inputs", True) is not False
            and profile.get("text_outputs", True) is not False
        ]

        models.sort()
        if models:
            available[provider] = models

    # Merge in models from config file (custom providers like ollama, fireworks)
    for provider_name, provider_config in config.providers.items():
        # Respect enabled = false (hide provider entirely).
        if not config.is_provider_enabled(provider_name):
            logger.debug(
                "Provider '%s' is disabled in config; skipping",
                provider_name,
            )
            continue

        config_models = list(provider_config.get("models", []))

        # For class_path providers not in the built-in registry, auto-discover
        # models from the package's _profiles.py when no explicit models list.
        if (
            not config_models
            and provider_name not in registry_providers
            and provider_name not in available
        ):
            class_path = provider_config.get("class_path", "")
            profile_module = _profile_module_from_class_path(class_path)
            if profile_module:
                try:
                    profiles = _load_provider_profiles(profile_module)
                except ImportError:
                    logger.debug(
                        "Could not import profiles from %s for class_path "
                        "provider '%s' (package may not be installed)",
                        profile_module,
                        provider_name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to load profiles from %s for class_path provider '%s'",
                        profile_module,
                        provider_name,
                        exc_info=True,
                    )
                else:
                    config_models = sorted(
                        name
                        for name, profile in profiles.items()
                        if profile.get("tool_calling", False)
                        and profile.get("text_inputs", True) is not False
                        and profile.get("text_outputs", True) is not False
                    )

        if provider_name not in available:
            if config_models:
                available[provider_name] = config_models
        else:
            # Append any config models not already discovered
            existing = set(available[provider_name])
            for model in config_models:
                if model not in existing:
                    available[provider_name].append(model)

    _available_models_cache = available
    return available


def _build_entry(
    base: dict[str, Any],
    overrides: dict[str, Any],
    cli_override: dict[str, Any] | None,
) -> ModelProfileEntry:
    """기본, 재정의 및 CLI 재정의를 병합하여 프로필 항목을 만듭니다.

    Args:
        base: 업스트림 프로필 사전(구성 전용 모델의 경우 비어 있음)
        overrides: `config.toml` 프로필이 재정의됩니다.
        cli_override: `--profile-override`의 추가 필드입니다.

    Returns:
        병합된 데이터 및 재정의 추적이 포함된 프로필 항목입니다.

    """

    merged = {**base, **overrides}
    overridden_keys = set(overrides)
    if cli_override:
        merged = {**merged, **cli_override}
        overridden_keys |= set(cli_override)
    return ModelProfileEntry(
        profile=merged,
        overridden_keys=frozenset(overridden_keys),
    )


def get_model_profiles(
    *,
    cli_override: dict[str, Any] | None = None,
) -> Mapping[str, ModelProfileEntry]:
    """config.toml 재정의와 병합된 업스트림 프로필을 로드합니다.

    `provider:model` 사양 문자열로 입력됩니다. 각 항목에는 병합된 프로필 사전과 config.toml에 의해 재정의된 키 세트가 포함되어
    있습니다.

    `get_available_models()`과 달리 여기에는 기능 필터(도구 호출, 텍스트 I/O)에 관계없이 업스트림 프로필의 모든 모델이
    포함됩니다.

    결과는 캐시됩니다. 재설정하려면 `clear_caches()`을(를) 사용하세요. `cli_override`이 제공되면 결과는
    `id(cli_override)`로 키가 지정된 단일 슬롯 캐시에 저장됩니다. 이는 세션에 대해 동일한 dict 객체를 유지하는 호출자에
    의존합니다(CLI는 이를 앱 인스턴스에 한 번 저장합니다). 동일한 내용을 가진 다른 사전을 전달하면 캐시를 우회하고 이전 항목을 덮어씁니다.

    Args:
        cli_override: `--profile-override`의 추가 프로필 필드입니다.

            제공되면 모든 프로필 항목(upstream + config.toml 이후) 위에 병합되고 해당 키가 `overridden_keys`에
            추가됩니다.

    Returns:
        사양 문자열을 프로필 항목에 대한 읽기 전용 매핑입니다.

    """

    global _profiles_cache, _profiles_override_cache  # noqa: PLW0603  # Module-level caches require global statement
    if cli_override is None and _profiles_cache is not None:
        return _profiles_cache
    if cli_override is not None and _profiles_override_cache is not None:
        cached_id, cached_result = _profiles_override_cache
        if cached_id == id(cli_override):
            return cached_result

    result: dict[str, ModelProfileEntry] = {}
    config = ModelConfig.load()

    # Collect upstream profiles from provider packages.
    seen_specs: set[str] = set()
    provider_modules = _get_provider_profile_modules()
    registry_providers: set[str] = set()
    for provider, module_path in provider_modules:
        registry_providers.add(provider)
        # Skip providers explicitly disabled in config.
        if not config.is_provider_enabled(provider):
            logger.debug(
                "Provider '%s' is disabled in config; skipping profiles",
                provider,
            )
            continue
        try:
            profiles = _load_provider_profiles(module_path)
        except ImportError:
            logger.debug(
                "Could not import profiles from %s for provider '%s'",
                module_path,
                provider,
            )
            continue
        except Exception:
            logger.warning(
                "Failed to load profiles from %s for provider '%s'",
                module_path,
                provider,
                exc_info=True,
            )
            continue

        for model_name, upstream_profile in profiles.items():
            spec = f"{provider}:{model_name}"
            seen_specs.add(spec)
            overrides = config.get_profile_overrides(provider, model_name=model_name)
            result[spec] = _build_entry(upstream_profile, overrides, cli_override)

    # Add config-only models and class_path provider profiles.
    for provider_name, provider_config in config.providers.items():
        if not config.is_provider_enabled(provider_name):
            logger.debug(
                "Provider '%s' is disabled in config; skipping profiles",
                provider_name,
            )
            continue
        # For class_path providers not in the built-in registry, load
        # upstream profiles from the package's _profiles.py.
        if provider_name not in registry_providers:
            class_path = provider_config.get("class_path", "")
            profile_module = _profile_module_from_class_path(class_path)
            if profile_module:
                try:
                    pkg_profiles = _load_provider_profiles(profile_module)
                except ImportError:
                    logger.debug(
                        "Could not import profiles from %s for class_path "
                        "provider '%s' (package may not be installed)",
                        profile_module,
                        provider_name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to load profiles from %s for class_path provider '%s'",
                        profile_module,
                        provider_name,
                        exc_info=True,
                    )
                else:
                    for model_name, upstream_profile in pkg_profiles.items():
                        spec = f"{provider_name}:{model_name}"
                        seen_specs.add(spec)
                        overrides = config.get_profile_overrides(
                            provider_name, model_name=model_name
                        )
                        result[spec] = _build_entry(
                            upstream_profile, overrides, cli_override
                        )

        config_models = provider_config.get("models", [])
        for model_name in config_models:
            spec = f"{provider_name}:{model_name}"
            if spec not in seen_specs:
                overrides = config.get_profile_overrides(
                    provider_name, model_name=model_name
                )
                result[spec] = _build_entry({}, overrides, cli_override)

    frozen = MappingProxyType(result)
    if cli_override is None:
        _profiles_cache = frozen
    else:
        _profiles_override_cache = (id(cli_override), frozen)
    return frozen


def has_provider_credentials(provider: str) -> bool | None:
    """공급자에 대한 자격 증명을 사용할 수 있는지 확인하세요.

    해결 순서:

    1. `api_key_env`이 있는 구성 파일 공급자(`config.toml`) — 소요
        우선순위를 지정하므로 사용자 재정의가 존중됩니다.
    2. `class_path`은 있지만 `api_key_env`은 없는 구성 파일 제공자 —
        자체 인증(예: 사용자 정의 헤더, JWT, mTLS)을 관리한다고 가정합니다.
    3. 하드코딩된 `PROVIDER_API_KEY_ENV` 매핑(anthropic, openai 등). 4. 기타 제공업체(예: 제3자 랭체인 제공업체)
        패키지), 자격 증명 상태를 알 수 없습니다. 공급자 자체는 모델 생성 시 인증 실패를 보고합니다.

    Args:
        provider: 공급자 이름.

    Returns:
        자격 증명이 사용 가능한 것으로 확인되었거나 공급자가 다음인 경우 True입니다.
            자체 인증(예: `class_path` 공급자)을 관리해야 하며 누락이 확인된 경우 False, 자격 증명 상태를 확인할 수 없는 경우
            None입니다.

    """

    # Config-file providers take priority when api_key_env is specified.
    config = ModelConfig.load()
    provider_config = config.providers.get(provider)
    if provider_config:
        result = config.has_credentials(provider)
        if result is not None:
            return result
        # class_path providers that omit api_key_env manage their own auth
        # (e.g., custom headers, JWT, mTLS) — treat as available.
        if provider_config.get("class_path"):
            return True
        # No api_key_env in config — fall through to hardcoded map.

    # Fall back to hardcoded well-known providers.
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if env_var:
        return bool(resolve_env_var(env_var))

    # Provider not found in config or hardcoded map — credential status is
    # unknown. The provider itself will report auth failures at
    # model-creation time.
    logger.debug(
        "No credential information for provider '%s'; deferring auth to provider",
        provider,
    )
    return None


def get_credential_env_var(provider: str) -> str | None:
    """공급자에 대한 자격 증명을 보유하는 환경 변수 이름을 반환합니다.

    먼저 구성 파일을 확인한 다음(사용자 재정의) 하드코딩된 `PROVIDER_API_KEY_ENV` 맵으로 대체합니다.

    Args:
        provider: 공급자 이름.

    Returns:
        환경 변수 이름 또는 알 수 없는 경우 없음입니다.

    """

    config = ModelConfig.load()
    config_env = config.get_api_key_env(provider)
    if config_env:
        return config_env
    return PROVIDER_API_KEY_ENV.get(provider)


@dataclass(frozen=True)
class ModelConfig:
    """`config.toml`에서 모델 구성을 구문 분석했습니다.

    인스턴스는 일단 생성되면 변경할 수 없습니다. `providers` 매핑은 `MappingProxyType`에 래핑되어 `load()`에서 반환된
    전역적으로 캐시된 싱글톤이 실수로 변경되는 것을 방지합니다.

    """


    default_model: str | None = None
    """사용자의 의도적인 기본 모델(구성 파일 `[models].default`에서)."""


    recent_model: str | None = None
    """가장 최근에 전환된 모델(구성 파일 `[models].recent`에서)"""


    providers: Mapping[str, ProviderConfig] = field(default_factory=dict)
    """공급자 이름을 해당 구성에 대한 읽기 전용 매핑입니다."""


    def __post_init__(self) -> None:
        """공급자 사전을 읽기 전용 프록시로 고정합니다."""

        if not isinstance(self.providers, MappingProxyType):
            object.__setattr__(self, "providers", MappingProxyType(self.providers))

    @classmethod
    def load(cls, config_path: Path | None = None) -> ModelConfig:
        """파일에서 구성을 로드합니다.

        기본 경로로 호출하면 프로세스 수명 동안 결과가 캐시됩니다. 재설정하려면 `clear_caches()`을(를) 사용하세요.

        Args:
            config_path: 구성 파일의 경로입니다. 기본값은 ~/.deepagents/config.toml입니다.

        Returns:
            `ModelConfig` 인스턴스를 구문 분석했습니다.
                파일이 없거나 읽을 수 없거나 잘못된 TOML 구문이 포함된 경우 빈 구성을 반환합니다.

        """

        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        is_default = config_path is None
        if is_default and _default_config_cache is not None:
            return _default_config_cache

        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        if not config_path.exists():
            fallback = cls()
            if is_default:
                _default_config_cache = fallback
            return fallback

        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            logger.warning(
                "Config file %s has invalid TOML syntax: %s. "
                "Ignoring config file. Fix the file or delete it to reset.",
                config_path,
                e,
            )
            fallback = cls()
            if is_default:
                _default_config_cache = fallback
            return fallback
        except (PermissionError, OSError) as e:
            logger.warning("Could not read config file %s: %s", config_path, e)
            fallback = cls()
            if is_default:
                _default_config_cache = fallback
            return fallback

        models_section = data.get("models", {})
        config = cls(
            default_model=models_section.get("default"),
            recent_model=models_section.get("recent"),
            providers=models_section.get("providers", {}),
        )

        # Validate config consistency
        config._validate()

        if is_default:
            _default_config_cache = config

        return config

    def _validate(self) -> None:
        """구성의 내부 일관성을 검증합니다.

        잘못된 구성에 대한 경고를 표시하지만 예외를 발생시키지 않으므로 앱이 잠재적으로 저하된 기능을 계속 사용할 수 있습니다.

        """

        # Warn if default_model is set but doesn't use provider:model format
        if self.default_model and ":" not in self.default_model:
            logger.warning(
                "default_model '%s' should use provider:model format "
                "(e.g., 'anthropic:claude-sonnet-4-5')",
                self.default_model,
            )

        # Warn if recent_model is set but doesn't use provider:model format
        if self.recent_model and ":" not in self.recent_model:
            logger.warning(
                "recent_model '%s' should use provider:model format "
                "(e.g., 'anthropic:claude-sonnet-4-5')",
                self.recent_model,
            )

        # Validate enabled field type and class_path format / params references
        for name, provider in self.providers.items():
            enabled = provider.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                logger.warning(
                    "Provider '%s' has non-boolean 'enabled' value %r "
                    "(expected true/false). Provider will remain visible.",
                    name,
                    enabled,
                )

            class_path = provider.get("class_path")
            if class_path and ":" not in class_path:
                logger.warning(
                    "Provider '%s' has invalid class_path '%s': "
                    "must be in module.path:ClassName format "
                    "(e.g., 'my_package.models:MyChatModel')",
                    name,
                    class_path,
                )

            models = set(provider.get("models", []))

            params = provider.get("params", {})
            for key, value in params.items():
                if isinstance(value, dict) and key not in models:
                    logger.warning(
                        "Provider '%s' has params for '%s' "
                        "which is not in its models list",
                        name,
                        key,
                    )

    def is_provider_enabled(self, provider_name: str) -> bool:
        """공급자가 모델 전환기에 표시되어야 하는지 확인하세요.

        구성이 명시적으로 `enabled = false`을 설정하면 공급자가 비활성화됩니다. 구성 파일에 없는 공급자는 항상 활성화된 것으로
        간주됩니다.

        Args:
            provider_name: 확인할 공급자입니다.

        Returns:
            공급자가 명시적으로 비활성화된 경우 `False`, 그렇지 않은 경우 `True`.

        """

        provider = self.providers.get(provider_name)
        if not provider:
            return True
        return provider.get("enabled") is not False

    def get_all_models(self) -> list[tuple[str, str]]:
        """모든 모델을 `(model_name, provider_name)` 튜플로 가져옵니다.

        원시 구성 데이터를 반환합니다. `is_provider_enabled`으로 필터링하지 않습니다. 모델 전환기에 표시된 필터링된 집합의 경우
        `get_available_models()`을 사용하세요.

        Returns:
            `(model_name, provider_name)`을 포함하는 튜플 목록입니다.

        """

        return [
            (model, provider_name)
            for provider_name, provider_config in self.providers.items()
            for model in provider_config.get("models", [])
        ]

    def get_provider_for_model(self, model_name: str) -> str | None:
        """이 모델이 포함된 공급자를 찾으세요.

        원시 구성 데이터를 반환합니다. `is_provider_enabled`으로 필터링하지 않습니다.

        Args:
            model_name: 조회할 모델 식별자입니다.

        Returns:
            발견된 경우 공급자 이름, 그렇지 않은 경우 없음.

        """

        for provider_name, provider_config in self.providers.items():
            if model_name in provider_config.get("models", []):
                return provider_name
        return None

    def has_credentials(self, provider_name: str) -> bool | None:
        """공급자에 대한 자격 증명을 사용할 수 있는지 확인하세요.

        이는 사용자 정의 공급자(예: 키가 필요하지 않은 로컬 Ollama)를 지원하는 구성 파일 기반 자격 증명 확인입니다. 핫스왑 경로에 사용되는
        하드코딩된 `PROVIDER_API_KEY_ENV` 기반 검사는 모듈 수준 `has_provider_credentials()`을 참조하세요.

        Args:
            provider_name: 확인할 공급자입니다.

        Returns:
            자격 증명이 사용 가능한 것으로 확인되면 True, 확인되면 False
                누락되거나, `api_key_env`이 구성되지 않고 자격 증명 상태를 확인할 수 없는 경우 없음입니다.

        """

        provider = self.providers.get(provider_name)
        if not provider:
            return False
        env_var = provider.get("api_key_env")
        if not env_var:
            return None  # No key configured — can't verify
        return bool(resolve_env_var(env_var))

    def get_base_url(self, provider_name: str) -> str | None:
        """맞춤 기본 URL을 받으세요.

        Args:
            provider_name: 기본 URL을 가져올 공급자입니다.

        Returns:
            구성된 경우 기본 URL이고, 그렇지 않으면 없음입니다.

        """

        provider = self.providers.get(provider_name)
        return provider.get("base_url") if provider else None

    def get_api_key_env(self, provider_name: str) -> str | None:
        """공급자의 API 키에 대한 환경 변수 이름을 가져옵니다.

        Args:
            provider_name: API 키 env var를 가져올 공급자입니다.

        Returns:
            구성된 경우 환경 변수 이름이고, 그렇지 않으면 없음입니다.

        """

        provider = self.providers.get(provider_name)
        return provider.get("api_key_env") if provider else None

    def get_class_path(self, provider_name: str) -> str | None:
        """공급자의 사용자 정의 클래스 경로를 가져옵니다.

        Args:
            provider_name: 조회할 공급자입니다.

        Returns:
            Class path in `module.path: ClassName` 형식 또는 없음.

        """

        provider = self.providers.get(provider_name)
        return provider.get("class_path") if provider else None

    def get_kwargs(
        self, provider_name: str, *, model_name: str | None = None
    ) -> dict[str, Any]:
        """공급자에 대한 추가 생성자 kwargs를 가져옵니다.

        공급자 구성에서 `params` 테이블을 읽습니다. 플랫 키는 공급자 전체의 기본값입니다. 모델 키 하위 테이블은 상단에서 얕은 병합을 수행하는
        모델별 재정의입니다(충돌 시 모델이 승리함).

        Args:
            provider_name: 조회할 공급자입니다.
            model_name: 모델별 재정의를 위한 선택적 모델 이름입니다.

        Returns:
            추가 kwargs 사전(구성된 것이 없으면 비어 있음)

        """

        provider = self.providers.get(provider_name)
        if not provider:
            return {}
        params = provider.get("params", {})
        result = {k: v for k, v in params.items() if not isinstance(v, dict)}
        if model_name:
            overrides = params.get(model_name)
            if isinstance(overrides, dict):
                result.update(overrides)
        return result

    def get_profile_overrides(
        self, provider_name: str, *, model_name: str | None = None
    ) -> dict[str, Any]:
        """공급자에 대한 프로필 재정의를 받으세요.

        공급자 구성에서 `profile` 테이블을 읽습니다. 플랫 키는 공급자 전체의 기본값입니다. 모델 키 하위 테이블은 상단에서 얕은 병합을
        수행하는 모델별 재정의입니다(충돌 시 모델이 승리함).

        Args:
            provider_name: 조회할 공급자입니다.
            model_name: 모델별 재정의를 위한 선택적 모델 이름입니다.

        Returns:
            프로필 재정의 사전(구성된 것이 없으면 비어 있음)

        """

        provider = self.providers.get(provider_name)
        if not provider:
            return {}
        profile = provider.get("profile", {})
        result = {k: v for k, v in profile.items() if not isinstance(v, dict)}
        if model_name:
            overrides = profile.get(model_name)
            if isinstance(overrides, dict):
                result.update(overrides)
        return result


def _save_model_field(
    field: str, model_spec: str, config_path: Path | None = None
) -> bool:
    """구성 파일에서 `[models].<field>` 키를 읽고 수정하고 씁니다.

    Args:
        field: `[models]` 테이블 아래의 키 이름(예: `'default'` 또는 `'recent'`)
        model_spec: `provider:model` 형식으로 저장할 모델입니다.
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

    Returns:
        저장에 성공하면 True이고, I/O 오류로 인해 실패하면 False입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing config or start fresh
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        if "models" not in data:
            data["models"] = {}
        data["models"][field] = model_spec

        # Write to temp file then rename to prevent corruption if write is interrupted
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            # Clean up temp file on any failure
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save %s model preference", field)
        return False
    else:
        # Invalidate config cache so the next load() picks up the change.
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        _default_config_cache = None
        return True


def save_default_model(model_spec: str, config_path: Path | None = None) -> bool:
    """구성 파일에서 기본 모델을 업데이트합니다.

    기존 구성(있는 경우)을 읽고, `[models].default`을 업데이트하고, 적절한 TOML 직렬화를 사용하여 다시 씁니다.

    Args:
        model_spec: `provider:model` 형식으로 기본값으로 설정할 모델입니다.
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

    Returns:
        저장에 성공하면 True이고, I/O 오류로 인해 실패하면 False입니다.

    Note:
        이 기능은 구성 파일의 주석을 유지하지 않습니다.

    """

    return _save_model_field("default", model_spec, config_path)


def clear_default_model(config_path: Path | None = None) -> bool:
    """구성 파일에서 기본 모델을 제거합니다.

    향후 실행이 `[models].recent` 또는 환경 자동 감지로 대체되도록 `[models].default` 키를 삭제합니다.

    Args:
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

    Returns:
        키가 제거된 경우(또는 이미 없는 경우) True이고, I/O 오류인 경우 False입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return True  # Nothing to clear

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)

        models_section = data.get("models")
        if not isinstance(models_section, dict) or "default" not in models_section:
            return True  # Already absent

        del models_section["default"]

        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not clear default model preference")
        return False
    else:
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        _default_config_cache = None
        return True


def is_warning_suppressed(key: str, config_path: Path | None = None) -> bool:
    """구성 파일에 경고 키가 표시되지 않는지 확인하세요.

    `config.toml`에서 `[warnings].suppress` 목록을 읽고 `key`이 있는지 확인합니다.

    Args:
        key: 확인할 경고 식별자(예: `'ripgrep'`)입니다.
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

    Returns:
        경고가 표시되지 않으면 `True`, 그렇지 않으면 `False`(포함
            파일이 없거나, 읽을 수 없거나, `[warnings]` 섹션이 없는 경우).

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        if not config_path.exists():
            return False
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug(
            "Could not read config file %s for warning suppression check",
            config_path,
            exc_info=True,
        )
        return False

    suppress_list = data.get("warnings", {}).get("suppress", [])
    if not isinstance(suppress_list, list):
        logger.debug(
            "[warnings].suppress in %s should be a list, got %s",
            config_path,
            type(suppress_list).__name__,
        )
        return False
    return key in suppress_list


def suppress_warning(key: str, config_path: Path | None = None) -> bool:
    """구성 파일의 금지 목록에 경고 키를 추가합니다.

    기존 구성(있는 경우)을 읽고 `key`을 `[warnings].suppress`에 추가한 다음 원자 임시 파일 이름 바꾸기를 사용하여 다시 씁니다.
    항목을 중복 제거합니다.

    Args:
        key: 억제할 경고 식별자(예: `'ripgrep'`)입니다.
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

    Returns:
        저장에 성공한 경우 `True`, I/O 오류로 인해 실패한 경우 `False`입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        if "warnings" not in data:
            data["warnings"] = {}
        suppress_list: list[str] = data["warnings"].get("suppress", [])
        if key not in suppress_list:
            suppress_list.append(key)
        data["warnings"]["suppress"] = suppress_list

        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save warning suppression for '%s'", key)
        return False
    return True


THREAD_COLUMN_DEFAULTS: dict[str, bool] = {
    "thread_id": False,
    "messages": True,
    "created_at": True,
    "updated_at": True,
    "git_branch": False,
    "cwd": False,
    "initial_prompt": True,
    "agent_name": False,
}
"""스레드 선택기 열의 기본 가시성입니다."""



class ThreadConfig(NamedTuple):
    """단일 TOML 구문 분석에서 읽은 통합된 스레드 선택기 구성입니다."""


    columns: dict[str, bool]
    """열 가시성 설정."""


    relative_time: bool
    """타임스탬프를 상대 시간으로 표시할지 여부입니다."""


    sort_order: str
    """`'updated_at'` 또는 `'created_at'`."""



_thread_config_cache: ThreadConfig | None = None


def load_thread_config(config_path: Path | None = None) -> ThreadConfig:
    """하나의 구성 파일 읽기에서 모든 스레드 선택기 설정을 로드합니다.

    기본 구성 경로를 읽을 때 캐시된 결과를 반환합니다. Prewarm 작업자는 시작 시 이를 호출하므로 이후에 `/threads` 모달을 열면 디스크
    I/O가 완전히 방지됩니다.

    Args:
        config_path: 구성 파일의 경로입니다.

    Returns:
        통합된 스레드 구성.

    """

    global _thread_config_cache  # noqa: PLW0603  # Module-level cache requires global statement

    if config_path is None:
        if _thread_config_cache is not None:
            return _thread_config_cache
        config_path = DEFAULT_CONFIG_PATH
    use_default = config_path == DEFAULT_CONFIG_PATH

    columns = dict(THREAD_COLUMN_DEFAULTS)
    relative_time = True
    sort_order = "updated_at"

    try:
        if not config_path.exists():
            result = ThreadConfig(columns, relative_time, sort_order)
            if use_default:
                _thread_config_cache = result
            return result
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        threads_section = data.get("threads", {})

        # columns
        raw_columns = threads_section.get("columns", {})
        if isinstance(raw_columns, dict):
            for key in columns:
                if key in raw_columns and isinstance(raw_columns[key], bool):
                    columns[key] = raw_columns[key]

        # relative_time
        rt_value = threads_section.get("relative_time")
        if isinstance(rt_value, bool):
            relative_time = rt_value

        # sort_order
        so_value = threads_section.get("sort_order")
        if so_value in {"updated_at", "created_at"}:
            sort_order = so_value
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read thread config; using defaults", exc_info=True)
        # Do not cache on error — allow retry on next call in case the
        # file is fixed or permissions are restored.
        return ThreadConfig(columns, relative_time, sort_order)

    result = ThreadConfig(columns, relative_time, sort_order)
    if use_default:
        _thread_config_cache = result
    return result


def invalidate_thread_config_cache() -> None:
    """다음 로드 시 디스크를 다시 읽을 수 있도록 캐시된 `ThreadConfig`을 지웁니다."""

    global _thread_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _thread_config_cache = None


def load_thread_columns(config_path: Path | None = None) -> dict[str, bool]:
    """구성 파일에서 스레드 열 가시성을 로드합니다.

    Args:
        config_path: 구성 파일의 경로입니다.

    Returns:
        열 이름을 가시성 부울에 매핑하는 사전입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    result = dict(THREAD_COLUMN_DEFAULTS)
    try:
        if not config_path.exists():
            return result
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        columns = data.get("threads", {}).get("columns", {})
        if isinstance(columns, dict):
            for key in result:
                if key in columns and isinstance(columns[key], bool):
                    result[key] = columns[key]
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read thread column config", exc_info=True)
    return result


def save_thread_columns(
    columns: dict[str, bool], config_path: Path | None = None
) -> bool:
    """스레드 열 가시성을 구성 파일에 저장합니다.

    Args:
        columns: 열 이름을 가시성 부울에 매핑하는 사전입니다.
        config_path: 구성 파일의 경로입니다.

    Returns:
        저장이 성공하면 True이고, I/O 오류가 있으면 False입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        if "threads" not in data:
            data["threads"] = {}
        data["threads"]["columns"] = columns

        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save thread column preferences")
        return False
    invalidate_thread_config_cache()
    return True


def load_thread_relative_time(config_path: Path | None = None) -> bool:
    """스레드 타임스탬프에 대한 상대 시간 표시 기본 설정을 로드합니다.

    Args:
        config_path: 구성 파일의 경로입니다.

    Returns:
        타임스탬프가 상대 시간으로 표시되어야 하는 경우 참입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        if not config_path.exists():
            return True
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        value = data.get("threads", {}).get("relative_time")
        if isinstance(value, bool):
            return value
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read thread relative_time config", exc_info=True)
    return True


def save_thread_relative_time(enabled: bool, config_path: Path | None = None) -> bool:
    """스레드 타임스탬프에 대한 상대 시간 표시 기본 설정을 저장합니다.

    Args:
        enabled: 상대 타임스탬프를 표시할지 여부입니다.
        config_path: 구성 파일의 경로입니다.

    Returns:
        저장이 성공하면 True이고, I/O 오류가 있으면 False입니다.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}
        if "threads" not in data:
            data["threads"] = {}
        data["threads"]["relative_time"] = enabled
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save thread relative_time preference")
        return False
    invalidate_thread_config_cache()
    return True


def load_thread_sort_order(config_path: Path | None = None) -> str:
    """스레드 선택기에 대한 정렬 순서 기본 설정을 로드합니다.

    Args:
        config_path: 구성 파일의 경로입니다.

    Returns:
        `"updated_at"` 또는 `"created_at"`.

    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        if not config_path.exists():
            return "updated_at"
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        value = data.get("threads", {}).get("sort_order")
        if value in {"updated_at", "created_at"}:
            return value
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read thread sort_order config", exc_info=True)
    return "updated_at"


def save_thread_sort_order(sort_order: str, config_path: Path | None = None) -> bool:
    """스레드 선택기의 정렬 순서 기본 설정을 저장합니다.

    Args:
        sort_order: `"updated_at"` 또는 `"created_at"`.
        config_path: 구성 파일의 경로입니다.

    Returns:
        저장이 성공하면 True이고, I/O 오류가 있으면 False입니다.

    Raises:
        ValueError: `sort_order`이 인식된 값이 아닌 경우.

    """

    if sort_order not in {"updated_at", "created_at"}:
        msg = (
            f"Invalid sort_order {sort_order!r}; expected 'updated_at' or 'created_at'"
        )
        raise ValueError(msg)
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}
        if "threads" not in data:
            data["threads"] = {}
        data["threads"]["sort_order"] = sort_order
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except Exception:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save thread sort_order preference")
        return False
    invalidate_thread_config_cache()
    return True


def save_recent_model(model_spec: str, config_path: Path | None = None) -> bool:
    """구성 파일에서 최근에 사용된 모델을 업데이트합니다.

    `[models].default` 대신 `[models].recent`에 쓰므로 `/model` 스위치가 사용자의 의도적인 기본값을 덮어쓰지 않습니다.

    Args:
        model_spec: `provider:model` 형식으로 저장할 모델입니다.
        config_path: 구성 파일의 경로입니다.

            기본값은 `~/.deepagents/config.toml`입니다.

    Returns:
        저장에 성공하면 True이고, I/O 오류로 인해 실패하면 False입니다.

    Note:
        이 기능은 구성 파일의 주석을 유지하지 않습니다.

    """

    return _save_model_field("recent", model_spec, config_path)
