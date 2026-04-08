"""`deepagents-cli`에 대한 버전 상수 및 가져오기 안전 메타데이터입니다.

이 모듈을 종속성 없이 유지하면 더 무거운 CLI 런타임을 트리거하지 않고도 버전 확인, 도움말 화면 및 업데이트 로직을 가져올 수 있습니다.
"""

__version__ = "0.0.34"  # x-릴리스-제발-버전

DOCS_URL = "https://docs.langchain.com/oss/python/deepagents/cli"
"""`deepagents-cli` 문서의 URL입니다."""

PYPI_URL = "https://pypi.org/pypi/deepagents-cli/json"
"""버전 확인을 위한 PyPI JSON API 엔드포인트."""

CHANGELOG_URL = (
    "https://github.com/langchain-ai/deepagents/blob/main/libs/cli/CHANGELOG.md"
)
"""전체 변경 로그의 URL입니다."""

USER_AGENT = f"deepagents-cli/{__version__} update-check"
"""PyPI 요청과 함께 전송된 User-Agent 헤더."""
