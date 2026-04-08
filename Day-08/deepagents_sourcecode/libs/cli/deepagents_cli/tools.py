"""핵심 Deep Agents 패키지에서 제공되지 않는 CLI 관련 도구입니다.

현재 이 모듈은 대부분 선택적 웹 검색 기능을 래핑하고 공급자 클라이언트를 지연 초기화 상태로 유지하므로 도구를 사용하지 않을 때 시작 비용이 저렴하게
유지됩니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tavily import TavilyClient

_UNSET = object()
_tavily_client: TavilyClient | object | None = _UNSET


def _get_tavily_client() -> TavilyClient | None:
    """게으른 Tavily 클라이언트 싱글톤을 가져오거나 초기화합니다.

Returns:
        TavilyClient 인스턴스 또는 API 키가 구성되지 않은 경우 None입니다.

    """
    global _tavily_client  # noqa: PLW0603  # Module-level cache requires global statement
    if _tavily_client is not _UNSET:
        return _tavily_client  # type: ignore[return-value]  # narrowed by sentinel check

    from deepagents_cli.config import settings

    if settings.has_tavily:
        from tavily import TavilyClient as _TavilyClient

        _tavily_client = _TavilyClient(api_key=settings.tavily_api_key)
    else:
        _tavily_client = None
    return _tavily_client


def web_search(  # noqa: ANN201  # Return type depends on dynamic tool configuration
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """최신 정보와 문서를 보려면 Tavily를 사용하여 웹을 검색하세요.

    이 도구는 웹을 검색하고 관련 결과를 반환합니다. 결과를 받은 후에는 정보를 사용자에게 자연스럽고 유용한 응답으로 종합해야 합니다.

Args:
        query: 검색어(구체적이고 상세함)
        max_results: 반환할 결과 수(기본값: 5)
        topic: 검색 주제 유형 - 대부분의 검색어는 '일반', 시사는 '뉴스'
        include_raw_content: 전체 페이지 콘텐츠 포함(경고: 더 많은 토큰을 사용함)

Returns:
        Dictionary containing:
        - results: 검색 결과 목록(각 항목: - 제목: 페이지 제목 - url: 페이지 URL - 내용: 페이지에서 관련 발췌 - 점수:
                   관련성 점수(0-1))
        - query: 원래 검색어

    IMPORTANT: 이 도구를 사용한 후:
    1. 각 결과의 '콘텐츠' 필드를 읽습니다. 2. 사용자 질문에 답변하는 관련 정보를 추출합니다. 3. 이를 명확하고 자연스러운 언어 응답으로
    합성합니다. 4. 페이지 제목이나 URL을 언급하여 출처를 인용합니다. 5. 원시 JSON을 사용자에게 절대 표시하지 않습니다. 항상 형식화된 응답을
    제공합니다.

    """
    try:
        import requests
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return {
            "error": f"Required package not installed: {exc.name}. "
            "Install with: pip install 'deepagents[cli]'",
            "query": query,
        }

    client = _get_tavily_client()
    if client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        # Tavily-specific exceptions
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """URL에서 콘텐츠를 가져와 HTML을 마크다운 형식으로 변환합니다.

    이 도구는 웹페이지 콘텐츠를 가져와서 깔끔한 마크다운 텍스트로 변환하므로 HTML 콘텐츠를 쉽게 읽고 처리할 수 있습니다. 마크다운을 받은 후에는
    정보를 사용자에게 자연스럽고 유용한 응답으로 합성해야 합니다.

Args:
        url: 가져올 URL(유효한 HTTP/HTTPS URL이어야 함)
        timeout: 요청 제한 시간(초)(기본값: 30)

Returns:
        Dictionary containing:
        - success: 요청 성공 여부
        - url: 리디렉션 후 최종 URL
        - markdown_content: 마크다운으로 변환된 페이지 콘텐츠
        - status_code: HTTP 상태 코드
        - content_length: 마크다운 콘텐츠의 문자 길이

    IMPORTANT: 이 도구를 사용한 후:
    1. 마크다운 콘텐츠를 읽습니다. 2. 사용자의 질문에 답하는 관련 정보를 추출합니다. 3. 이를 명확하고 자연스러운 언어 응답으로 합성합니다. 4.
    특별히 요청하지 않는 한 사용자에게 원시 마크다운을 표시하지 마세요.

    """
    try:
        import requests
        from markdownify import markdownify
    except ImportError as exc:
        return {
            "error": f"Required package not installed: {exc.name}. "
            "Install with: pip install 'deepagents[cli]'",
            "url": url,
        }

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}
