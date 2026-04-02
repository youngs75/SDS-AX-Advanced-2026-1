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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from tavily import TavilyClient

# 환경변수 로드
load_dotenv(Path(__file__).parent.parent / "Day-03" / ".env")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # MCP는 stdout을 통신에 사용하므로 stderr에 로깅
)
logger = logging.getLogger(__name__)

# Tavily 클라이언트 초기화
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================================
# MCP 서버 초기화
# ============================================================================

mcp = FastMCP(
    name="BasicMCPServer",
    instructions="MCP 핵심 개념 학습을 위한 기본 서버 (Tool, Resource, Prompt 포함)",
)


# ============================================================================
# Tool 정의: Discriminated Union 패턴으로 3개 도구를 1개로 통합
# ============================================================================


class AssistantRequest(BaseModel):
    """통합 도구 요청 스키마 (Discriminated Union 패턴).

    action 필드의 값에 따라 필요한 payload 필드가 달라집니다.
    """

    action: Literal["weather", "calculate", "write_note"] = Field(
        ...,
        description=(
            "수행할 작업 종류: "
            "'weather' (날씨 조회), "
            "'calculate' (수식 계산), "
            "'write_note' (노트 저장)"
        ),
    )
    # weather용 필드
    city: str | None = Field(
        default=None,
        description="[weather] 조회할 도시 이름 (예: Seoul, Tokyo, New York)",
    )
    # calculate용 필드
    expression: str | None = Field(
        default=None,
        description="[calculate] 계산할 수식 (예: '2 + 3 * 4')",
    )
    # write_note용 필드
    content: str | None = Field(
        default=None,
        description="[write_note] 저장할 노트 내용",
    )
    filename: str = Field(
        default="note.txt",
        description="[write_note] 저장할 파일명 (기본값: note.txt)",
    )


def safe_eval(expression: str) -> float | int:
    """AST를 사용한 안전한 수식 평가."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
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


def _handle_weather(req: AssistantRequest) -> dict[str, Any]:
    """Tavily를 사용한 실제 날씨 조회."""
    if not req.city:
        return {"success": False, "error": "city 필드가 필요합니다."}

    logger.info("날씨 조회: city=%s", req.city)
    result = tavily.search(
        query=f"{req.city} 현재 날씨 기온 습도",
        max_results=3,
    )
    # Tavily 검색 결과를 요약하여 반환
    answers = [r["content"] for r in result.get("results", [])]
    return {
        "success": True,
        "city": req.city,
        "search_results": answers,
    }


def _handle_calculate(req: AssistantRequest) -> dict[str, Any]:
    """안전한 수식 계산."""
    if not req.expression:
        return {"success": False, "error": "expression 필드가 필요합니다."}

    logger.info("계산: expression=%s", req.expression)
    try:
        result = safe_eval(req.expression)
        return {"success": True, "expression": req.expression, "result": result}
    except Exception as e:
        return {"success": False, "error": f"계산 오류: {e!s}"}


def _handle_write_note(req: AssistantRequest) -> dict[str, Any]:
    """노트 파일 저장."""
    if not req.content:
        return {"success": False, "error": "content 필드가 필요합니다."}

    logger.info("노트 저장: filename=%s", req.filename)
    try:
        notes_dir = Path("./mcp_notes")
        notes_dir.mkdir(exist_ok=True)
        filepath = notes_dir / req.filename
        filepath.write_text(req.content, encoding="utf-8")
        return {
            "success": True,
            "path": str(filepath.absolute()),
            "message": f"노트가 저장되었습니다: {filepath}",
        }
    except Exception as e:
        return {"success": False, "error": f"노트 저장 오류: {e!s}"}


# 액션 → 핸들러 매핑 (라우팅 테이블)
_ACTION_HANDLERS = {
    "weather": _handle_weather,
    "calculate": _handle_calculate,
    "write_note": _handle_write_note,
}


@mcp.tool()
def assistant(req: AssistantRequest) -> dict[str, Any]:
    """통합 도구: 날씨 조회, 수식 계산, 노트 저장을 하나의 도구로 수행합니다.

    action 파라미터로 수행할 작업을 선택하고,
    해당 action에 필요한 필드를 함께 전달하세요.

    예시:
    - 날씨: {"action": "weather", "city": "Seoul"}
    - 계산: {"action": "calculate", "expression": "2 + 3 * 4"}
    - 노트: {"action": "write_note", "content": "메모 내용", "filename": "memo.txt"}
    """
    logger.info("Tool 호출: assistant(action=%s)", req.action)

    handler = _ACTION_HANDLERS[req.action]
    return handler(req)


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
    weather_result = assistant(AssistantRequest(action="weather", city="Seoul"))
    logger.info(
        "날씨 조회 결과: %s\n", json.dumps(weather_result, ensure_ascii=False, indent=2)
    )

    # 계산
    calc_result = assistant(AssistantRequest(action="calculate", expression="2 + 3 * 4"))
    logger.info(
        "계산 결과: %s\n", json.dumps(calc_result, ensure_ascii=False, indent=2)
    )

    # 노트 작성
    note_result = assistant(AssistantRequest(
        action="write_note",
        content="MCP 서버 테스트 노트입니다.\n날짜: "
        + datetime.now(timezone.utc).isoformat(),
        filename="test_note.txt",
    ))
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
