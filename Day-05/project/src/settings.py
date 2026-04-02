"""Day3 프로젝트 설정 모듈.

Pydantic BaseSettings를 사용하여 환경변수 기반 설정을 관리합니다.
.env 파일에서 자동으로 값을 읽어오며, 환경변수가 우선합니다.

사용 예시:
    from src.settings import get_settings
    settings = get_settings()
    print(settings.openrouter_api_key)  # .env의 OPENROUTER_API_KEY 값

Day2와의 차이점:
    - vLLM 대신 OpenRouter를 단일 LLM 게이트웨이로 사용
    - Day3 전용 필드 (openrouter_api_key, openrouter_model_name)
    - 싱글턴 패턴으로 Settings 인스턴스를 캐싱
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Day3 프로젝트의 중앙 설정 클래스.

    모든 설정값은 환경변수 또는 .env 파일에서 읽어옵니다.
    alias를 통해 환경변수명과 Python 속성명을 매핑합니다.

    Attributes:
        env: 실행 환경 (local, staging, prod)
        service_name: 서비스 식별자 (Langfuse 태깅용)
        app_version: 애플리케이션 버전
        openrouter_api_key: OpenRouter API 키 (모든 LLM 호출에 사용)
        openrouter_model_name: 기본 모델명 (예: openai/gpt-4.1)
        langfuse_host: Langfuse 서버 URL
        langfuse_public_key: Langfuse Public Key
        langfuse_secret_key: Langfuse Secret Key
        langfuse_tracing_enabled: 트레이싱 활성화 여부
        langfuse_sample_rate: 트레이스 샘플링 비율 (0.0~1.0)
        data_dir: 데이터 루트 디렉토리 경로
        local_corpus_dir: 소스 문서(corpus) 디렉토리 경로
    """

    # SettingsConfigDict: .env 파일 자동 로드, 알 수 없는 필드는 무시
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── 런타임 식별 ──────────────────────────────────────────
    env: str = Field(default="local", alias="ENV")
    service_name: str = Field(default="agent-ops-day3", alias="SERVICE_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")

    # ── OpenRouter (LLM 단일 게이트웨이) ─────────────────────
    # OpenRouter는 다양한 LLM 프로바이더를 하나의 API로 통합합니다.
    # base_url: https://openrouter.ai/api/v1 (OpenAI SDK 호환)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model_name: str = Field(default="openai/gpt-4.1", alias="OPENROUTER_MODEL_NAME")

    # ── Langfuse (관측성 플랫폼) ─────────────────────────────
    # Langfuse가 설정되지 않으면 관측성 기능은 자동으로 비활성화됩니다.
    langfuse_host: str = Field(default="", alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_tracing_enabled: bool = Field(default=True, alias="LANGFUSE_TRACING_ENABLED")
    langfuse_sample_rate: float = Field(default=1.0, alias="LANGFUSE_SAMPLE_RATE")

    # ── 데이터 경로 ──────────────────────────────────────────
    # corpus: Synthesizer의 소스 문서, synthetic/review/golden: Loop 1 단계별 데이터
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    local_corpus_dir: Path = Field(default=Path("./data/corpus"), alias="LOCAL_CORPUS_DIR")


# 싱글턴 캐시: get_settings()를 여러 번 호출해도 동일한 인스턴스를 반환
_settings: Settings | None = None


def get_settings() -> Settings:
    """Settings 싱글턴 인스턴스를 반환합니다.

    최초 호출 시 .env 파일을 읽어 Settings를 생성하고,
    이후 호출에서는 캐싱된 인스턴스를 재사용합니다.

    Returns:
        Settings: 프로젝트 설정 인스턴스
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
