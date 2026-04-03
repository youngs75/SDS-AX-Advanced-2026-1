"""Coding Assistant 상태 스키마.

3노드(parse → execute → verify) 간 데이터를 전달하는 상태 정의.
"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ParseResult(TypedDict, total=False):
    """parse_request 노드 출력."""
    task_type: str          # "generate" | "fix" | "refactor" | "explain"
    description: str        # 작업 설명
    target_files: list[str] # 대상 파일 경로
    requirements: list[str] # 세부 요구사항


class VerifyResult(TypedDict, total=False):
    """verify_result 노드 출력."""
    passed: bool
    issues: list[str]       # 발견된 문제 목록
    suggestions: list[str]  # 개선 제안


class CodingState(TypedDict, total=False):
    """Coding Assistant 에이전트 상태."""
    messages: Annotated[list[BaseMessage], add_messages]

    # parse_request 출력
    parse_result: ParseResult

    # execute_code 출력
    generated_code: str         # 생성/수정된 코드
    execution_log: list[str]    # 실행 과정 로그
    source_files: dict[str, str]  # 원본 파일 내용 (JIT 참조용)

    # verify_result 출력
    verify_result: VerifyResult

    # 반복 제어
    iteration: int              # 현재 반복 횟수
    max_iterations: int         # 최대 반복 횟수
