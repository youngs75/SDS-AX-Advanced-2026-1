"""SQLite CRUD MCP 서버.

Discriminated Union 패턴으로 1개의 Tool에서 CRUD를 처리합니다.
- 구조화된 필드(title, content, id)로 안전한 기본 CRUD
- raw SQL(query)로 유연한 쿼리 + Guardrails 안전 검증
"""

import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# DB 경로
DB_PATH = Path(__file__).parent / "mcp_database.db"

# DB 커넥션 (서버 수명 동안 유지)
db: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    """DB 커넥션을 반환합니다. 초기화 전이면 예외를 발생시킵니다."""
    if db is None:
        msg = "DB 연결이 초기화되지 않았습니다."
        raise RuntimeError(msg)
    return db

# 컬럼 메타데이터: LLM에게 컬럼의 실제 의미를 알려줌
COLUMN_METADATA = [
    {"column_name": "id", "description": "노트 고유 식별자 (자동 생성)"},
    {"column_name": "title", "description": "노트 제목"},
    {"column_name": "content", "description": "노트 본문 내용"},
    {"column_name": "created_at", "description": "생성 시각 (UTC)"},
    {"column_name": "updated_at", "description": "최종 수정 시각 (UTC)"},
]


# ============================================================================
# DB 초기화
# ============================================================================

DUMMY_NOTES = [
    ("프로젝트 회의록", "2026년 1분기 목표 정리. 신규 MCP 서버 개발 및 배포 일정 확정."),
    ("Python 학습 메모", "FastMCP 프레임워크: @mcp.tool() 데코레이터로 도구 등록. Pydantic으로 스키마 정의."),
    ("점심 메뉴 추천", "월: 김치찌개, 화: 된장찌개, 수: 비빔밥, 목: 칼국수, 금: 삼겹살"),
    ("독서 목록", "1. Clean Code 2. Designing Data-Intensive Applications 3. The Pragmatic Programmer"),
    ("운동 루틴", "월수금: 웨이트 (상체/하체 분할), 화목: 러닝 30분, 토: 등산"),
]


def init_db() -> sqlite3.Connection:
    """DB 초기화: 테이블 생성 + 더미 데이터 삽입. 커넥션을 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    if count == 0:
        conn.executemany(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            DUMMY_NOTES,
        )
        logger.info("더미 데이터 %d건 삽입", len(DUMMY_NOTES))
    conn.commit()
    logger.info("DB 초기화 완료: %s", DB_PATH)
    return conn


# ============================================================================
# MCP 서버 초기화
# ============================================================================

mcp = FastMCP(
    name="SQLiteCRUDServer",
    instructions="SQLite DB에 대한 CRUD 작업을 1개의 통합 Tool로 제공하는 MCP 서버",
)


# ============================================================================
# SQL Guardrails (안전 검증)
# ============================================================================


def _validate_sql(query: str) -> dict[str, Any] | None:
    """SQL 쿼리의 안전성을 검증합니다. 위험하면 에러 dict 반환, 안전하면 None."""
    upper = query.strip().upper()

    # DROP / ALTER / TRUNCATE 등 DDL 차단
    if re.match(r"^(DROP|ALTER|TRUNCATE)\s", upper):
        return {"success": False, "error": "DDL 문(DROP/ALTER/TRUNCATE)은 허용되지 않습니다."}

    # DELETE without WHERE → 전체 삭제 위험
    if upper.startswith("DELETE") and "WHERE" not in upper:
        return {"success": False, "error": "DELETE 쿼리에 WHERE 절이 필요합니다 (전체 삭제 방지)."}

    # UPDATE without WHERE → 전체 수정 위험
    if upper.startswith("UPDATE") and "WHERE" not in upper:
        return {"success": False, "error": "UPDATE 쿼리에 WHERE 절이 필요합니다 (전체 수정 방지)."}

    # EXPLAIN으로 문법 검증
    try:
        get_db().execute(f"EXPLAIN {query}")
    except sqlite3.Error as e:
        return {"success": False, "error": f"SQL 문법 오류: {e!s}"}

    return None


# ============================================================================
# Discriminated Union 스키마
# ============================================================================


class CRUDRequest(BaseModel):
    """통합 CRUD 요청 스키마 (Discriminated Union 패턴).

    action 필드의 값에 따라 필요한 필드가 달라집니다.
    - create/read/update/delete: 구조화된 필드 사용 (안전)
    - query: raw SQL 직접 실행 (유연, Guardrails 적용)
    """

    action: Literal["create", "read", "update", "delete", "query"] = Field(
        ...,
        description=(
            "수행할 작업: "
            "'create' (노트 생성), "
            "'read' (노트 조회), "
            "'update' (노트 수정), "
            "'delete' (노트 삭제), "
            "'query' (SQL 직접 실행)"
        ),
    )
    # create / update 용
    title: str | None = Field(
        default=None,
        description="[create, update] 노트 제목",
    )
    content: str | None = Field(
        default=None,
        description="[create, update] 노트 내용",
    )
    # read / update / delete 용
    id: int | None = Field(
        default=None,
        description="[read, update, delete] 노트 ID. read에서 생략하면 전체 목록 조회",
    )
    # query 용
    query: str | None = Field(
        default=None,
        description="[query] 실행할 SQL 쿼리 (SELECT/INSERT/UPDATE/DELETE)",
    )


# ============================================================================
# 핸들러 함수
# ============================================================================


def _handle_create(req: CRUDRequest) -> dict[str, Any]:
    """노트 생성."""
    if not req.title or not req.content:
        return {"success": False, "error": "title과 content 필드가 필요합니다."}

    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO notes (title, content) VALUES (?, ?)",
        (req.title, req.content),
    )
    conn.commit()
    note_id = cursor.lastrowid

    logger.info("노트 생성: id=%s, title=%s", note_id, req.title)
    return {"success": True, "id": note_id, "message": f"노트가 생성되었습니다 (id={note_id})"}


def _handle_read(req: CRUDRequest) -> dict[str, Any]:
    """노트 조회. id가 있으면 단건, 없으면 전체 목록."""
    conn = get_db()
    conn.row_factory = sqlite3.Row

    if req.id is not None:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (req.id,)).fetchone()
        if not row:
            return {"success": False, "error": f"id={req.id} 노트를 찾을 수 없습니다."}
        logger.info("노트 조회: id=%s", req.id)
        return {
            "success": True,
            "column_description": COLUMN_METADATA,
            "note": dict(row),
        }

    rows = conn.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()
    logger.info("전체 노트 조회: %d건", len(rows))
    return {
        "success": True,
        "column_description": COLUMN_METADATA,
        "notes": [dict(r) for r in rows],
        "count": len(rows),
    }


def _handle_update(req: CRUDRequest) -> dict[str, Any]:
    """노트 수정."""
    if req.id is None:
        return {"success": False, "error": "id 필드가 필요합니다."}
    if not req.title and not req.content:
        return {"success": False, "error": "title 또는 content 중 하나 이상 필요합니다."}

    conn = get_db()
    existing = conn.execute("SELECT * FROM notes WHERE id = ?", (req.id,)).fetchone()
    if not existing:
        return {"success": False, "error": f"id={req.id} 노트를 찾을 수 없습니다."}

    updates = []
    params = []
    if req.title is not None:
        updates.append("title = ?")
        params.append(req.title)
    if req.content is not None:
        updates.append("content = ?")
        params.append(req.content)
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(req.id)

    conn.execute(f"UPDATE notes SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()

    logger.info("노트 수정: id=%s", req.id)
    return {"success": True, "message": f"노트가 수정되었습니다 (id={req.id})"}


def _handle_delete(req: CRUDRequest) -> dict[str, Any]:
    """노트 삭제."""
    if req.id is None:
        return {"success": False, "error": "id 필드가 필요합니다."}

    conn = get_db()
    existing = conn.execute("SELECT * FROM notes WHERE id = ?", (req.id,)).fetchone()
    if not existing:
        return {"success": False, "error": f"id={req.id} 노트를 찾을 수 없습니다."}

    conn.execute("DELETE FROM notes WHERE id = ?", (req.id,))
    conn.commit()

    logger.info("노트 삭제: id=%s", req.id)
    return {"success": True, "message": f"노트가 삭제되었습니다 (id={req.id})"}


def _handle_query(req: CRUDRequest) -> dict[str, Any]:
    """raw SQL 실행 (Guardrails 적용)."""
    if not req.query:
        return {"success": False, "error": "query 필드가 필요합니다."}

    # 안전성 검증
    error = _validate_sql(req.query)
    if error:
        return error

    logger.info("SQL 실행: %s", req.query)
    conn = get_db()
    try:
        cursor = conn.execute(req.query)

        # SELECT 계열이면 결과 반환
        if req.query.strip().upper().startswith("SELECT"):
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return {
                "success": True,
                "column_description": COLUMN_METADATA,
                "data": {
                    "columns": columns,
                    "rows": [list(r) for r in rows],
                },
                "count": len(rows),
            }

        # INSERT/UPDATE/DELETE 등은 영향받은 행 수 반환
        conn.commit()
        return {
            "success": True,
            "affected_rows": cursor.rowcount,
            "message": f"쿼리 실행 완료 ({cursor.rowcount}행 영향)",
        }

    except sqlite3.Error as e:
        return {"success": False, "error": f"SQL 실행 오류: {e!s}"}


# 액션 → 핸들러 매핑 (라우팅 테이블)
_ACTION_HANDLERS = {
    "create": _handle_create,
    "read": _handle_read,
    "update": _handle_update,
    "delete": _handle_delete,
    "query": _handle_query,
}


# ============================================================================
# 통합 Tool
# ============================================================================


@mcp.tool(structured_output=True)
def crud(req: CRUDRequest) -> dict[str, Any]:
    """SQLite DB에서 노트를 관리합니다 (생성/조회/수정/삭제/SQL).

    action 파라미터로 수행할 작업을 선택하고,
    해당 action에 필요한 필드를 함께 전달하세요.

    예시:
    - 생성: {"action": "create", "title": "회의록", "content": "내용..."}
    - 전체 조회: {"action": "read"}
    - 단건 조회: {"action": "read", "id": 1}
    - 수정: {"action": "update", "id": 1, "title": "새 제목", "content": "새 내용"}
    - 삭제: {"action": "delete", "id": 1}
    - SQL: {"action": "query", "query": "SELECT * FROM notes WHERE title LIKE '%회의%'"}
    """
    logger.info("Tool 호출: crud(action=%s)", req.action)
    try:
        get_db()  # DB 연결 상태 확인
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

    handler = _ACTION_HANDLERS[req.action]
    return handler(req)


# ============================================================================
# 메인 실행
# ============================================================================


def main() -> None:
    """DB 초기화 후 서버를 실행합니다."""
    global db
    import argparse

    parser = argparse.ArgumentParser(description="SQLite CRUD MCP Server")
    parser.add_argument("--test", action="store_true", help="CRUD 테스트 실행")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="전송 프로토콜 (기본값: stdio)",
    )
    args = parser.parse_args()

    # DB 초기화
    db = init_db()
    if db is None:
        logger.error("DB 초기화 실패")
        sys.exit(1)

    if args.test:
        logger.info("테스트 모드 실행\n")

        # 구조화된 CRUD 테스트
        logger.info("=" * 50)
        logger.info("구조화된 CRUD 테스트")
        logger.info("=" * 50)

        r = crud(CRUDRequest(action="create", title="테스트 노트", content="Hello MCP!"))
        logger.info("CREATE: %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="read"))
        logger.info("READ ALL: %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="update", id=1, content="Updated!"))
        logger.info("UPDATE: %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="read", id=1))
        logger.info("READ ONE: %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        # raw SQL 테스트
        logger.info("=" * 50)
        logger.info("raw SQL 테스트 (Guardrails)")
        logger.info("=" * 50)

        r = crud(CRUDRequest(action="query", query="SELECT * FROM notes WHERE title LIKE '%메모%'"))
        logger.info("QUERY (SELECT): %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="query", query="DELETE FROM notes"))
        logger.info("QUERY (DELETE 차단): %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="query", query="UPDATE notes SET title = 'x'"))
        logger.info("QUERY (UPDATE 차단): %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="query", query="DROP TABLE notes"))
        logger.info("QUERY (DROP 차단): %s\n", json.dumps(r, ensure_ascii=False, indent=2))

        r = crud(CRUDRequest(action="query", query="SELECT * FROM 없는테이블"))
        logger.info("QUERY (문법오류): %s\n", json.dumps(r, ensure_ascii=False, indent=2))
    else:
        logger.info("MCP 서버 시작 (Transport: %s)", args.transport)
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
