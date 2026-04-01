import sqlite3
import sys
from typing import Literal

from mcp.server.fastmcp import FastMCP


# TODO:
# 1. SQLite DB:
#   1) DB Init 부분은 서버 실행 시 처리해주세요.
#   2) 테이블: Person, Column: name, age
# 2. CRUD 를 모두 구현하되, Tool 1개로 하셔야됩니다.

server = FastMCP(name="예제", instructions="")

db = None


@server.tool(structured_output=True)
def person_db_crud(type: Literal["C", "R", "U", "D"], query: str) -> dict:
    """Person DB 에 대해 Create, Read, Update, Delete 가능한 도구.

    Args:
        type: Create, Read, Update, Delete 의 앞글자를 딴 Type 선택
        query: SQLite 기반의 SQL

    Returns:
        Dict
    """
    result = {}
    if db is None:
        # TODO: DB Connection 오류(MCP 서버측) 설명
        return result

    if query.upper().startswith("DELETE"):
        # TODO: DELETE 가 안전한지 여부 검사
        pass
    else:
        return {
            "success": False,
            "reason": "DELETE Query 위험(데이터 전체 삭제 가능성)",
        }

    if query.upper().startswith("UPDATE"):
        # TODO: UPDATE 가 안전한지 여부 검사
        pass
    else:
        return {
            "success": False,
            "reason": "UPDATE Query 위험(데이터 전체 변경 가능성)",
        }

    # db_result = db.execute(query) # NOTE: 절대 안됌!
    plan_result = db.execute("EXPLAIN" + " " + query)
    if plan_result is None:
        # TODO: query 문법 오류
        return result

    # TODO: 그럼 이제는 Query 실행이 가능한걸까요?
    db.execute(query)

    # NOTE: db_result 검증?
    # Guardrails
    return {
        "success": True,
        "column_description": [
            {"column_name": "name", "description": "사실 이름아님. ID임."}
        ],
        "data": {"column": ["name()", "age"], "row": [("홍길동", 10), ("이영희", 20)]},
    }


def main() -> None:
    """1. DB 초기화
    2. DB 객체 선언.
    """
    global db
    db_conn = sqlite3.Connection("./sample.db")
    if not db_conn:
        sys.exit(1)
    db = db_conn
    server.run()


if __name__ == "__main__":
    main()
