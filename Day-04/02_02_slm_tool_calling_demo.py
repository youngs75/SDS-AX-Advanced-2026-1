"""Small Language Model Tool Calling 이슈 시연.

## 학습 목표
- Small Language Model (SLM)의 Tool Calling 불안정성 이해
- 다양한 모델 간 Tool Calling 성능 비교
- Structured Output Fallback 전략의 필요성

## 핵심 인사이트

### SLM의 Tool Calling 문제
1. **형식 오류**: JSON 스키마를 정확히 따르지 않음
2. **필수 필드 누락**: required 필드를 빠뜨림
3. **타입 불일치**: 문자열을 숫자로 보내거나 그 반대
4. **환각(Hallucination)**: 존재하지 않는 Tool 호출 시도
5. **Routing 오류**: 여러 Tool 중 잘못된 Tool 선택

### Structured Output Fallback이 필요한 이유
- SLM은 Function Calling 학습이 부족
- Structured Output은 JSON Schema로 강제하여 더 안정적
- Fallback 전략으로 20-40% 성공률 향상
"""

import asyncio
import logging
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from langchain.tools import ToolRuntime, tool
from langchain_community.retrievers import ArxivRetriever
from langchain_tavily import TavilySearch
from openrouter_llm import create_openrouter_llm
from pydantic import BaseModel, Field


load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WeatherArgs(BaseModel):
    city: str = Field(..., description="도시 이름")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="온도 단위"
    )


@tool(args_schema=WeatherArgs)
async def get_weather(
    city: str, units: Literal["celsius", "fahrenheit"] = "celsius"
) -> str:
    """도시의 날씨를 조회합니다."""
    return f"{city}의 날씨는 {units.capitalize()} 단위로 맑습니다."


class CalculateArgs(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        ..., description="연산 종류"
    )
    first_number: float = Field(..., description="첫 번째 숫자")
    second_number: float = Field(..., description="두 번째 숫자")


@tool(args_schema=CalculateArgs)
async def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    first_number: float,
    second_number: float,
) -> float | str:
    """계산 결과를 반환합니다."""
    if operation == "add":
        return first_number + second_number
    if operation == "subtract":
        return first_number - second_number
    if operation == "multiply":
        return first_number * second_number
    if operation == "divide":
        return first_number / second_number
    return "Invalid operation"


tavily_search = TavilySearch(max_results=10, search_depth="advanced")


@tool
async def arxiv_search(query: str, runtime: ToolRuntime) -> str:
    """논문 검색 사이트 Arxiv에서 검색합니다.

    Args:
        query: 검색 쿼리

    Returns:
        검색 결과
    """
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_full_documents=True,
    )
    docs = await retriever.ainvoke(query)

    return (
        "\n".join(
            [
                f"page_content: {doc.page_content}\n\n page_metadata: {doc.metadata}"
                for doc in docs
            ]
        )
        if docs
        else "No results found"
    )


TEST_MODELS = [
    # 높은 성능의 Frontier 모델
    "openai/gpt-4.1-mini",
    "anthropic/claude-haiku-4.5",
    # Small Language Models (SLM)
    "meta-llama/llama-4-scout",
    "openai/gpt-oss-20b",
    "qwen/qwen3-8b",
]


async def test_tool_calling_with_model(
    model_name: str, test_case: dict[str, Any]
) -> dict[str, Any]:
    """특정 모델로 Tool Calling 테스트.

    Args:
        model_name: 모델 이름
        test_case: 테스트 케이스 (prompt, expected_tool, expected_args)

    Returns:
        테스트 결과
    """
    logger.info("테스트 중: %s - %s", model_name, test_case["name"])

    try:
        # LLM 생성
        llm = create_openrouter_llm(
            model=model_name,
            temperature=0,
        )

        tools = [
            get_weather,
            calculate,
            tavily_search,
            arxiv_search,
        ]

        # LLM에 tools 바인딩
        llm_with_tools = llm.bind_tools(tools)

        # LLM 호출 (Tool Calling)
        response = llm_with_tools.invoke(test_case["prompt"])

        # 응답 분석
        # TODO: 만약 tool_calls 가 없다면?
        # 1. content 영역에 들어오는 경우
        # 2. tool_calls 에 오긴 왔는데, Parameter 가 잘못된...
        # 3. content 에도 없고, tool_calls 에도 없음 => 빈 메시지로 오는 경우
        # Example:
        #     [
        #         {"role": "sytem", "content": "You are a Helpful AI Assistant. USE TOOLS."},
        #         {"role": "human", "content": "어떤 질문"},
        #         {"role": "assistant", "content": "", "tool_calls": {"tool_call_id": "1", ""}},
        #         {"role": "tool", "content": "이 도구는 도구 호출의 정상 여부를 검증하는 도구로 매 도구 호출 전 1 회 실행됩니다.", "tool_call_id": "1"},
        #         {"role": "human", "content": "어떤 질문" + "tool_calls 를 꼭 채우도록 하십시오"},
        #     ]
        # 4. 아예 에러인 경우
        """
        파인튜닝된 모델 GPT-OSS
        > Chat Template
        {%- macro render_tool_namespace(namespace_name, tools) -%}
        {{- "## " + namespace_name + "\n\n" }}
        {{- "namespace " + namespace_name + " {\n\n" }}
        {%- for tool in tools %}
        {{- "# Tools\n\n" }}
        {{- render_tool_namespace("functions", tools) }}

        System(Developer) Message:
            ## namespace: funtions 를 아래 제공된 것들로만 사용해주세요.
            tool_name_1
            tool_name_2
            tool_name_3

            # Tools\n\n
            Keep focus on this tools ONLY.
            tool_name_1
            tool_name_2
            tool_name_3

        """

        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []

        if not tool_calls:
            return {
                "model": model_name,
                "test_case": test_case["name"],
                "prompt": test_case["prompt"],
                "expected_tool": test_case["expected_tool"],
                "actual_tool": "없음",
                "tool_correct": False,
                "success": False,
                "error": "Tool 호출 안됨",
            }

        # 첫 번째 Tool Call 확인
        tool_call = tool_calls[0]
        tool_name = (
            tool_call.get("name")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "name", None)
        )

        # tool_name과 expected_tool 비교
        expected_tool = test_case["expected_tool"]
        tool_correct = tool_name == expected_tool
        success = tool_correct

        return {
            "model": model_name,
            "test_case": test_case["name"],
            "prompt": test_case["prompt"],
            "expected_tool": expected_tool,
            "actual_tool": tool_name,
            "tool_correct": tool_correct,
            "success": success,
            "error": None,
        }

    except Exception as e:
        logger.exception("  오류 발생: %s", e)
        return {
            "model": model_name,
            "test_case": test_case["name"],
            "prompt": test_case["prompt"],
            "expected_tool": test_case["expected_tool"],
            "actual_tool": "오류",
            "tool_correct": False,
            "success": False,
            "error": str(e),
        }


TEST_CASES = [
    {
        "name": "날씨 조회 (간단)",
        "prompt": "오늘 GPT5.1 이 나왔는데, 성능은 꽤 괜찮은 것 같고 수능날인데 오늘 날씨 왜이래? 그래서 지금 여기 날씨를 알려줘. 아, 참고로 여기는 나주야.",
        "expected_tool": "get_weather",
    },
    # {
    #     "name": "날씨 조회 (단위 지정)",
    #     "prompt": "뉴욕의 날씨를 화씨로 알려주세요.",
    #     "expected_tool": "get_weather",
    # },
    # {
    #     "name": "논문 검색",
    #     "prompt": "LLM 및 Agent 에 대한 2025년 11월 기준 최근 논문만을 검색해서 정리해주세요.",
    #     "expected_tool": "arxiv_search",
    # },
    # {
    #     "name": "계산 (덧셈)",
    #     "prompt": "123 더하기 456은 얼마인가요?",
    #     "expected_tool": "calculate",
    # },
    # {
    #     "name": "계산 (곱셈)",
    #     "prompt": "25 곱하기 4는?",
    #     "expected_tool": "calculate",
    # },
    # {
    #     "name": "웹 검색",
    #     "prompt": "LLM 및 Agent 에 대한 2025년 11월 기준 최근 이슈 내용을 검색해서 정리해주세요.",
    #     "expected_tool": "tavily_search",
    # }
]


async def run_all_tests() -> None:
    all_results = []

    for model in TEST_MODELS:
        logger.info("테스트 중: %s", model)

        for test_case in TEST_CASES:
            result = await test_tool_calling_with_model(model, test_case)
            all_results.append(result)

            # Rate limiting 방지
            await asyncio.sleep(0.3)

    # 결과 분석
    print_results_summary(all_results)


def print_results_summary(results: list[dict[str, Any]]) -> None:
    """테스트 결과 요약 출력."""
    df = pd.DataFrame(results)
    df["model_short"] = df["model"].apply(lambda x: x.split("/")[-1])

    # 성공/실패를 이모지로
    df["결과"] = df["success"].apply(lambda x: "✅" if x else "❌")
    df["Tool"] = df["tool_correct"].apply(lambda x: "✅" if x else "❌")

    display_df = df[
        ["model_short", "test_case", "결과", "Tool", "expected_tool", "actual_tool"]
    ].copy()
    display_df.columns = [
        "모델",
        "테스트 케이스",
        "전체 결과",
        "Tool 정확",
        "기대 Tool",
        "호출된 Tool",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    summary = (
        df.groupby("model_short")
        .agg(
            총_테스트=("success", "count"),
            성공_횟수=("success", "sum"),
            Tool_정확=("tool_correct", "sum"),
        )
        .reset_index()
    )

    summary["성공률"] = (summary["성공_횟수"] / summary["총_테스트"] * 100).round(1)
    summary["Tool_정확률"] = (summary["Tool_정확"] / summary["총_테스트"] * 100).round(
        1
    )

    summary_display = summary[
        ["model_short", "총_테스트", "성공률", "Tool_정확률"]
    ].copy()
    summary_display.columns = ["모델", "총 테스트", "성공률 (%)", "Tool 정확률 (%)"]

    # 성공률 순으로 정렬
    summary_display = summary_display.sort_values("성공률 (%)", ascending=False)


async def main() -> None:
    logger.info("=" * 80)
    logger.info("SLM Tool Calling 이슈 시연")
    logger.info("=" * 80)
    logger.info("테스트할 모델 수: %s", len(TEST_MODELS))
    logger.info("테스트 케이스 수: %s", len(TEST_CASES))
    logger.info("총 테스트 수: %s\n", len(TEST_MODELS) * len(TEST_CASES))

    # 전체 테스트 실행
    await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
