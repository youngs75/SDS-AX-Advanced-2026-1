"""평가 파이프라인 설정 모듈.

Pydantic BaseSettings를 사용하여 환경변수 기반 설정을 관리합니다.
youngs75_a2a/.env 파일에서 자동으로 값을 읽어오며, 환경변수가 우선합니다.

사용 예시:
    from youngs75_a2a.eval_pipeline.settings import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# youngs75_a2a 패키지 루트의 .env 경로
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PACKAGE_ROOT / ".env"


class Settings(BaseSettings):
    """평가 파이프라인 중앙 설정 클래스."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 런타임 식별 ──
    env: str = Field(default="local", alias="ENV")
    service_name: str = Field(default="youngs75-a2a", alias="SERVICE_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")

    # ── Langfuse (관측성) ──
    langfuse_host: str = Field(default="", alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_tracing_enabled: bool = Field(default=True, alias="LANGFUSE_TRACING_ENABLED")
    langfuse_sample_rate: float = Field(default=1.0, alias="LANGFUSE_SAMPLE_RATE")

    # ── 데이터 경로 ──
    data_dir: Path = Field(default=_PACKAGE_ROOT / "data", alias="DATA_DIR")
    local_corpus_dir: Path = Field(default=_PACKAGE_ROOT / "data" / "corpus", alias="LOCAL_CORPUS_DIR")

    # ── 하위 호환성 (OpenRouter → OpenAI 전환 과정) ──
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model_name: str = Field(default="gpt-5.4", alias="OPENROUTER_MODEL_NAME")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Settings 싱글턴 인스턴스를 반환합니다."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
