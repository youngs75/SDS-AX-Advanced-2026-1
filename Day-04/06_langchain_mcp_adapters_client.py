"""LangChain-LangGraph 와 MCP Server 통합.

## 학습 목표
- langchain-mcp-adapters를 사용하여 MCP 서버를 LangChain 도구로 변환
- LangChain ReAct Agent에 MCP 도구 통합
- 여러 MCP 서버를 동시에 사용
- OpenRouter LLM으로 복합 작업 수행
- 실전 시나리오: 검색 → 분석 → 요약

## 핵심 개념

### LangChain MCP Adapters
- MCP 도구를 LangChain Tool로 변환
- 여러 MCP 서버를 하나의 Agent에 통합
- Tool Calling과 완벽한 호환성

### MultiServerMCPClient
- 여러 MCP 서버를 동시에 관리
- 각 서버의 도구를 통합
- stdio, streamable-http 모두 지원

## 참고 문서
- LangChain MCP Docs: https://docs.langchain.com/oss/python/langchain/mcp
- langchain-mcp-adapters: https://github.com/langchain-ai/langchain-mcp-adapters
- MCP Specification: https://modelcontextprotocol.io/
"""

import asyncio
import logging

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from openrouter_llm import create_openrouter_llm


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_real_agent() -> None:
    """LangChain MCP Adapters 클라이언트를 사용하여 LangChain Agent 생성."""
    llm = create_openrouter_llm(
        model="openai/gpt-4.1",
        temperature=0.3,
    )

    # 클라이언트 생성 전에 MCP 서버 실행이 필요합니다.
    client = MultiServerMCPClient(
        {
            # TODO: 원격 또는 직접 MCP 서버 실행 후 추가 필요
            # 1. LangChain Docs (직접 주소: https://docs.langchain.com/mcp)
            # 2. Notion MCP(문서: https://developers.notion.com/docs/mcp)
            # 3. **DesktopCommander MCP(깃헙: https://github.com/wonderwhy-er/DesktopCommanderMCP)**
            # 주의!! 굉장히 위험한 MCP 서버입니다. 주의해서 사용해야합니다.
            # 4. DeekWiki MCP(문서: https://docs.devin.ai/work-with-devin/deepwiki-mcp)
        }
    )

    agent = create_agent(
        model=llm,
        tools=client.get_tools(),
        system_prompt="""당신은 다양한 도구를 사용할 수 있는 어시스턴트입니다.
사용 가능한 도구를 최대한 활용하여 사용자의 요청을 처리하세요.
필요한 경우 여러 도구를 순차적으로 사용하세요.""",
    )

    scenarios = [
        "LangChain v1.0 에서 Middleware 에 대해 설명해주면서 어떤 장점이 있는지, 특히 Built-in Middleware 에 대해 자세히 설명해주세요. 그리고 그렇게 정리한 자료를 Notion Page 에 이쁘게 보고서 형태로 정리해주세요. 단, 이모지는 사용하지 마세요.",
        # TODO: 경로를 추가해야합니다.
        "SerenaMCP 에 대해서 자세히 조사 후 깊게 설명해주길 바라고, 설명해주는 자료들을 '' 경로의 Markdown 파일로 저장해주세요.",
    ]

    for scenario in scenarios:
        response = await agent.ainvoke(
            {"messages": {"role": "user", "content": scenario}}
        )
        response.pretty_print()


async def main():
    """메인 함수."""
    return await create_real_agent()


if __name__ == "__main__":
    asyncio.run(main())
