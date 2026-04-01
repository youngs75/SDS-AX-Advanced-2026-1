"""기본 MCP 서버 구현.

## 학습 목표
- MCP 서버의 핵심 개념 3가지 이해: Tool, Resource, Prompt
- FastMCP 프레임워크를 사용한 MCP 서버 구현
- stdio transport를 사용한 기본 통신 방식 이해
- Pydantic을 활용한 타입 안전 스키마 정의

## 핵심 개념

### Tool (도구)
- LLM이 발견하고 실행할 수 있는 함수
- 상태 변경 가능 (Side effects 허용)
- 사용자 승인 필요 (Human-in-the-loop)
- 예: API 호출, 파일 쓰기, 계산 수행

### Resource (리소스)
- LLM에게 컨텍스트 데이터를 제공하는 읽기 전용 자원
- Side effect 없음
- 사용자 승인 불필요
- URI 템플릿 사용 (file://, https://, custom://)
- 예: 파일 내용, 문서, 데이터베이스 스키마

### Prompt (프롬프트)
- 재사용 가능한 프롬프트 템플릿
- 변수를 포함한 구조화된 프롬프트
- LLM이 선택하여 사용 가능
- 예: 코드 리뷰, 문서 작성, 분석 템플릿

## 참고 문서
- MCP 공식 문서: https://modelcontextprotocol.io/
- FastMCP GitHub: https://github.com/jlowin/fastmcp
- Pydantic 문서: https://docs.pydantic.dev/
"""

import asyncio
import ast
import json
import logging
import operator
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # MCP는 stdout을 통신에 사용하므로 stderr에 로깅
)
logger = logging.getLogger(__name__)


# ============================================================================
# MCP 서버 초기화
# ============================================================================

mcp = FastMCP(
    name="BasicMCPServer",
    instructions="MCP 핵심 개념 학습을 위한 기본 서버 (Tool, Resource, Prompt 포함)",
)


# ============================================================================
# Tool 정의: 상태 변경 가능한 기능
# ============================================================================


class WeatherQuery(BaseModel):
    """날씨 조회 요청 스키마."""

    city: str = Field(..., description="조회할 도시 이름 (예: Seoul, Tokyo, New York)")
    units: str = Field(
        default="metric",
        description="온도 단위: 'metric' (섭씨) 또는 'imperial' (화씨)",
        pattern="^(metric|imperial)$",
    )


class CalculationRequest(BaseModel):
    """계산 요청 스키마."""

    expression: str = Field(..., description="계산할 수식 (예: '2 + 3 * 4')")


def safe_eval(expression: str) -> float | int:
    """AST를 사용한 안전한 수식 평가.

    Args:
        expression: 평가할 수학 표현식

    Returns:
        계산 결과

    Raises:
        ValueError: 허용되지 않는 연산이 포함된 경우
    """
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                msg = f"허용되지 않는 연산자: {op_type.__name__}"
                raise ValueError(msg)
            return allowed_operators[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                msg = f"허용되지 않는 연산자: {op_type.__name__}"
                raise ValueError(msg)
            return allowed_operators[op_type](operand)
        msg = f"허용되지 않는 표현식: {ast.dump(node)}"
        raise ValueError(msg)

    tree = ast.parse(expression, mode="eval")
    return eval_node(tree.body)


@mcp.tool(structured_output=True)
def get_weather(query: WeatherQuery) -> dict[str, Any]:
    """지정된 도시의 날씨 정보를 조회합니다.

    Args:
        query: 날씨 조회 요청 (도시명, 온도 단위 포함)

    Returns:
        날씨 정보를 포함한 딕셔너리

    Note:
        이것은 시뮬레이션된 응답입니다.
        실제 환경에서는 외부 날씨 API를 호출하게 됩니다.
    """
    logger.info("Tool 호출: get_weather(city=%s, units=%s)", query.city, query.units)

    # 시뮬레이션된 날씨 데이터
    weather_data = {
        "city": query.city,
        "temperature": 22 if query.units == "metric" else 72,
        # "unit": "°C" if query.units == "metric" else "°F",
        "condition": "맑음",
        "humidity": 60,
        "wind_speed": 12,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "날씨 조회 완료: %s - %s%s",
        query.city,
        weather_data["temperature"],
        weather_data["unit"],
    )

    return {
        "success": True,
        "data": weather_data,
        # "message": f"{query.city}의 현재 날씨는 {weather_data['condition']}입니다. "
        # f"온도: {weather_data['temperature']}{weather_data['unit']}, "
        # f"습도: {weather_data['humidity']}%",
    }


@mcp.tool()
def calculate(request: CalculationRequest) -> dict[str, Any]:
    """수학 표현식을 계산합니다.

    Args:
        request: 계산 요청 (수식 포함)

    Returns:
        계산 결과를 포함한 딕셔너리

    Note:
        보안상 AST를 사용하여 제한된 연산자만 허용합니다.
    """
    logger.info("Tool 호출: calculate(expression=%s)", request.expression)

    try:
        # AST를 사용한 안전한 수식 평가
        result = safe_eval(request.expression)
        logger.info("계산 완료: %s = %s", request.expression, result)

        return {
            "success": 0,
            # "expression": request.expression,
            "result": result,
            # "message": f"{request.expression} = {result}",
        }

    except Exception as e:
        logger.exception("계산 오류: %s", e)
        return {
            "success": False,
            "error": f"계산 중 오류 발생: {e!s}",
        }


@mcp.tool()
def write_note(content: str, filename: str = "note.txt") -> dict[str, str]:
    """간단한 텍스트 노트를 파일로 저장합니다.

    Args:
        content: 저장할 노트 내용
        filename: 저장할 파일명 (기본값: note.txt)

    Returns:
        저장 결과 메시지

    Note:
        실제 파일 시스템에 저장됩니다. (Side effect 있음)
    """
    logger.info(
        "Tool 호출: write_note(filename=%s, content_length=%d)", filename, len(content)
    )

    try:
        # 임시 디렉토리에 저장
        notes_dir = Path("./mcp_notes")
        notes_dir.mkdir(exist_ok=True)

        filepath = notes_dir / filename
        filepath.write_text(content, encoding="utf-8")

        logger.info("노트 저장 완료: %s", filepath)

        return {
            "success": True,
            "path": str(filepath.absolute()),
            "message": f"노트가 저장되었습니다: {filepath}",
        }

    except Exception:
        logger.exception("노트 저장 오류")
        return {
            "success": False,
            "error": f"노트 저장 중 오류 발생: {e!s}",
        }


# ============================================================================
# Resource 정의: 읽기 전용 컨텍스트 제공
# ============================================================================


@mcp.resource("file://notes/{filename}")
def get_note_content(filename: str) -> str:
    """저장된 노트 파일의 내용을 읽습니다.

    Args:
        filename: 읽을 노트 파일명

    Returns:
        파일 내용 (텍스트)

    Note:
        Resource는 읽기 전용이며 side effect가 없습니다.
        URI 템플릿: file://notes/{filename}
    """
    logger.info("Resource 요청: file://notes/%s", filename)

    try:
        notes_dir = Path("./mcp_notes")
        filepath = notes_dir / filename

        if not filepath.exists():
            return f"Error: 파일을 찾을 수 없습니다: {filename}"

        content = filepath.read_text(encoding="utf-8")
        logger.info("노트 읽기 완료: %s (%d bytes)", filepath, len(content))

        return content

    except Exception:
        logger.exception("노트 읽기 오류")
        return f"Error: 노트 읽기 중 오류 발생: {e!s}"


@mcp.resource("config://server")
def get_server_config() -> dict[str, Any]:
    """MCP 서버 설정 정보를 반환합니다.

    Returns:
        서버 설정 딕셔너리

    Note:
        custom URI scheme 사용 예시: config://server
    """
    logger.info("Resource 요청: config://server")

    return {
        "server_name": mcp.name,
        "version": mcp.version,
        "description": mcp.description,
        "transport": "stdio",
        "tools_count": len(mcp._tools),
        "resources_count": len(mcp._resources),
        "prompts_count": len(mcp._prompts),
        "uptime": datetime.now(timezone.utc).isoformat(),
    }


@mcp.resource("docs://readme")
def get_readme() -> str:
    """서버 사용 설명서를 반환합니다.

    Returns:
        README 텍스트 (Markdown)
    """
    logger.info("Resource 요청: docs://readme")

    return """# Basic MCP Server - 사용 가이드

## 개요
이 MCP 서버는 Tool, Resource, Prompt 세 가지 핵심 개념을 학습하기 위한 기본 서버입니다.

## 사용 가능한 Tool
1. **get_weather** - 도시의 날씨 정보 조회
2. **calculate** - 수학 표현식 계산
3. **write_note** - 텍스트 노트 저장

## 사용 가능한 Resource
1. **file://notes/{filename}** - 저장된 노트 파일 읽기
2. **config://server** - 서버 설정 정보
3. **docs://readme** - 사용 설명서 (현재 문서)

## 사용 가능한 Prompt
1. **code_review** - 코드 리뷰 템플릿
2. **bug_analysis** - 버그 분석 템플릿

## 예제
```python
# Tool 호출
result = call_tool("get_weather", {"city": "Seoul", "units": "metric"})

# Resource 읽기
content = read_resource("file://notes/note.txt")

# Prompt 사용
prompt = get_prompt("code_review", {"code": "...", "language": "python"})
```
"""


# ============================================================================
# Prompt 정의: 재사용 가능한 프롬프트 템플릿
# ============================================================================


@mcp.prompt()
def code_review(code: str, language: str = "python") -> list[dict[str, str]]:
    """코드 리뷰를 위한 프롬프트 템플릿.

    Args:
        code: 리뷰할 코드
        language: 프로그래밍 언어 (기본값: python)

    Returns:
        프롬프트 메시지 리스트
    """
    logger.info(
        "Prompt 요청: code_review(language=%s, code_length=%d)", language, len(code)
    )

    return [
        {
            "role": "system",
            "content": f"""당신은 {language} 코드 리뷰 전문가입니다.
다음 기준으로 코드를 분석하고 피드백을 제공하세요:

1. 코드 품질 및 가독성
2. 잠재적 버그 및 에러 처리
3. 성능 최적화 기회
4. 보안 취약점
5. 모범 사례 준수 여부

각 항목에 대해 구체적인 개선 제안을 제공하세요.""",
        },
        {
            "role": "user",
            "content": f"""다음 {language} 코드를 리뷰해주세요:

```{language}
{code}
```

위 코드에 대한 상세한 리뷰를 제공해주세요.""",
        },
    ]


@mcp.prompt()
def bug_analysis(
    error_message: str,
    code_context: str = "",
    language: str = "python",
) -> list[dict[str, str]]:
    """버그 분석을 위한 프롬프트 템플릿.

    Args:
        error_message: 에러 메시지
        code_context: 에러가 발생한 코드 문맥 (선택사항)
        language: 프로그래밍 언어 (기본값: python)

    Returns:
        프롬프트 메시지 리스트
    """
    logger.info("Prompt 요청: bug_analysis(language=%s)", language)

    context_section = (
        f"""

관련 코드:
```{language}
{code_context}
```"""
        if code_context
        else ""
    )

    return [
        {
            "role": "system",
            "content": f"""당신은 {language} 디버깅 전문가입니다.
에러 메시지를 분석하고 다음을 제공하세요:

1. 에러의 근본 원인
2. 발생 가능한 시나리오
3. 구체적인 해결 방법
4. 재발 방지를 위한 권장사항""",
        },
        {
            "role": "user",
            "content": f"""다음 에러를 분석해주세요:

에러 메시지:
```
{error_message}
```{context_section}

이 에러의 원인과 해결 방법을 상세히 설명해주세요.""",
        },
    ]


@mcp.prompt()
def data_analysis(
    data_description: str,
    analysis_goal: str,
) -> list[dict[str, str]]:
    """데이터 분석을 위한 프롬프트 템플릿.

    Args:
        data_description: 데이터 설명
        analysis_goal: 분석 목표

    Returns:
        프롬프트 메시지 리스트
    """
    logger.info("Prompt 요청: data_analysis")

    return [
        {
            "role": "system",
            "content": """당신은 데이터 분석 전문가입니다.
주어진 데이터에 대해 다음을 수행하세요:

1. 데이터 특성 파악
2. 적절한 분석 방법 선택
3. 통계적 인사이트 도출
4. 시각화 권장사항
5. 실행 가능한 결론 제시""",
        },
        {
            "role": "user",
            "content": f"""다음 데이터를 분석해주세요:

데이터 설명:
{data_description}

분석 목표:
{analysis_goal}

위 목표를 달성하기 위한 상세한 분석 계획과 방법을 제시해주세요.""",
        },
    ]


# ============================================================================
# 서버 생명주기 관리
# ============================================================================


def print_server_info() -> None:
    """서버 정보 출력."""
    logger.info("=" * 70)
    logger.info("MCP Server: %s v%s", mcp.name, mcp.version)
    logger.info("Description: %s", mcp.description)
    logger.info("=" * 70)
    logger.info("Tools: %s", len(mcp._tools))
    for tool_name in mcp._tools:
        logger.info("  - %s", tool_name)
    logger.info("Resources: %s", len(mcp._resources))
    for resource_uri in mcp._resources:
        logger.info("  - %s", resource_uri)
    logger.info("Prompts: %s", len(mcp._prompts))
    for prompt_name in mcp._prompts:
        logger.info("  - %s", prompt_name)
    logger.info("=" * 70)


async def test_server() -> None:
    """서버 기능 테스트."""
    logger.info("\n테스트 시작: MCP 서버 기능 검증\n")

    # Tool 테스트
    logger.info("=" * 70)
    logger.info("1. Tool 테스트")
    logger.info("=" * 70)

    # 날씨 조회
    weather_result = get_weather(WeatherQuery(city="Seoul", units="metric"))
    logger.info(
        "날씨 조회 결과: %s\n", json.dumps(weather_result, ensure_ascii=False, indent=2)
    )

    # 계산
    calc_result = calculate(CalculationRequest(expression="2 + 3 * 4"))
    logger.info(
        "계산 결과: %s\n", json.dumps(calc_result, ensure_ascii=False, indent=2)
    )

    # 노트 작성
    note_result = write_note(
        content="MCP 서버 테스트 노트입니다.\n날짜: "
        + datetime.now(timezone.utc).isoformat(),
        filename="test_note.txt",
    )
    logger.info(
        "노트 저장 결과: %s\n", json.dumps(note_result, ensure_ascii=False, indent=2)
    )

    # Resource 테스트
    logger.info("=" * 70)
    logger.info("2. Resource 테스트")
    logger.info("=" * 70)

    # 노트 읽기
    note_content = get_note_content("test_note.txt")
    logger.info("노트 내용:\n%s\n", note_content)

    # 서버 설정 읽기
    server_config = get_server_config()
    logger.info(
        "서버 설정: %s\n", json.dumps(server_config, ensure_ascii=False, indent=2)
    )

    # Prompt 테스트
    logger.info("=" * 70)
    logger.info("3. Prompt 테스트")
    logger.info("=" * 70)

    # 코드 리뷰 프롬프트
    code_review_prompt = code_review(
        code='def hello():\n    print("Hello, World!")',
        language="python",
    )
    logger.info(
        "코드 리뷰 프롬프트:\n%s\n",
        json.dumps(code_review_prompt, ensure_ascii=False, indent=2),
    )

    logger.info("=" * 70)
    logger.info("테스트 완료")
    logger.info("=" * 70)


# ============================================================================
# 메인 실행
# ============================================================================


def main() -> None:
    """메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="Basic MCP Server")
    parser.add_argument(
        "--test",
        action="store_true",
        help="서버 기능 테스트 실행",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="전송 프로토콜 (기본값: stdio)",
    )

    args = parser.parse_args()

    print_server_info()

    if args.test:
        # 테스트 모드
        logger.info("\n테스트 모드로 실행 중...\n")
        asyncio.run(test_server())
    else:
        # 서버 모드
        logger.info("\nMCP 서버 시작 중... (Transport: %s)\n", args.transport)
        logger.info("클라이언트 연결 대기 중...\n")

        # stdio transport로 서버 실행
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
