"""Agent Protocol 호환 FastAPI 서버.

하나의 DeepAgents 에이전트를 호스팅하며, CLI 인자로 에이전트 유형을 선택합니다.
AsyncSubAgentMiddleware가 호출하는 Agent Protocol 엔드포인트를 구현합니다.

실행 방법:
    # 터미널 1: 복잡한 작업 서버 (GLM, port 2024)
    python task_server.py --name complex --port 2024

    # 터미널 2: 간단한 작업 서버 (Qwen, port 2025)
    python task_server.py --name simple --port 2025

엔드포인트:
    POST /threads                              스레드 생성
    POST /threads/{thread_id}/runs             런 시작 (interrupt 지원)
    GET  /threads/{thread_id}/runs/{run_id}    런 상태 조회
    GET  /threads/{thread_id}                  스레드 상태 (결과 메시지)
    POST /threads/{thread_id}/runs/{run_id}/cancel  런 취소
    GET  /ok                                   헬스체크
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from deepagents import create_deep_agent
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage

from _llm import task_glm, task_qwen

# ── 에이전트 설정 ────────────────────────────────────────────────────────────

AGENT_CONFIGS = {
    "complex": {
        "model": task_glm,
        "system_prompt": """당신은 복잡한 작업을 전문적으로 처리하는 에이전트입니다.

복잡한 작업이란 하위 작업이 2개 이상인 작업을 의미합니다.

작업 처리 절차:
1. 작업을 하위 단계로 분해합니다.
2. 각 하위 단계를 순서대로 수행합니다.
3. 결과를 종합하여 최종 답변을 작성합니다.

항상 체계적이고 구조화된 결과를 제공하세요.""",
    },
    "simple": {
        "model": task_qwen,
        "system_prompt": """당신은 간단한 작업을 빠르고 정확하게 처리하는 에이전트입니다.

간단한 작업이란 하위 작업이 1개인 작업을 의미합니다.

작업 처리 절차:
1. 작업의 핵심을 파악합니다.
2. 즉시 결과를 생성합니다.

군더더기 없이 간결하고 정확한 답변을 제공하세요.""",
    },
}

# ── CLI 인자 파싱 ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Agent Protocol 서버")
parser.add_argument(
    "--name",
    choices=["complex", "simple"],
    required=True,
    help="에이전트 유형: complex (GLM) 또는 simple (Qwen)",
)
parser.add_argument(
    "--port",
    type=int,
    default=2024,
    help="서버 포트 (기본값: 2024)",
)
args = parser.parse_args()

# ── 에이전트 생성 ─────────────────────────────────────────────────────────────

config = AGENT_CONFIGS[args.name]
_agent = create_deep_agent(
    model=config["model"],
    system_prompt=config["system_prompt"],
    tools=[],
)

# ── Database ──────────────────────────────────────────────────────────────────

_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row


def _init_db() -> None:
    """threads와 runs 테이블을 생성합니다."""
    _conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id  TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            messages   TEXT NOT NULL DEFAULT '[]',
            values_    TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL REFERENCES threads(thread_id),
            assistant_id TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            error        TEXT
        );
    """)
    _conn.commit()


# ── DB 헬퍼 ───────────────────────────────────────────────────────────────────


def _get_thread(thread_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT thread_id, created_at, messages, values_ FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": row["thread_id"],
        "created_at": row["created_at"],
        "messages": json.loads(row["messages"]),
        "values": json.loads(row["values_"]),
    }


def _get_run(run_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT run_id, thread_id, assistant_id, status, created_at, error FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


# ── Run 실행기 ────────────────────────────────────────────────────────────────


async def _execute_run(run_id: str, thread_id: str, user_message: str) -> None:
    """에이전트를 호출하고 결과를 저장합니다. fire-and-forget으로 실행됩니다."""
    _conn.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))
    _conn.commit()
    try:
        result = await _agent.ainvoke({"messages": [HumanMessage(user_message)]})
        last = result["messages"][-1]
        output = (
            last.content if isinstance(last.content, str) else json.dumps(last.content)
        )
        assistant_msg = {"role": "assistant", "content": output}

        row = _conn.execute(
            "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        msgs = json.loads(row[0]) if row else []
        msgs.append(assistant_msg)
        serialized = json.dumps(msgs)

        _conn.execute(
            "UPDATE threads SET messages = ?, values_ = ? WHERE thread_id = ?",
            (serialized, json.dumps({"messages": msgs}), thread_id),
        )
        _conn.execute("UPDATE runs SET status = 'success' WHERE run_id = ?", (run_id,))
        _conn.commit()
    except Exception as exc:
        _conn.execute(
            "UPDATE runs SET status = 'error', error = ? WHERE run_id = ?",
            (str(exc), run_id),
        )
        _conn.commit()


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _init_db()
    print(f"[{args.name}] 에이전트 서버 시작 (port {args.port})")
    yield


app = FastAPI(lifespan=_lifespan)


# ── 엔드포인트 ────────────────────────────────────────────────────────────────


@app.get("/ok")
async def health() -> dict[str, bool]:
    """헬스체크."""
    return {"ok": True}


@app.post("/threads")
async def create_thread() -> dict[str, Any]:
    """스레드 생성. start_async_task가 런 생성 전에 호출합니다."""
    thread_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    _conn.execute(
        "INSERT INTO threads (thread_id, created_at) VALUES (?, ?)",
        (thread_id, now),
    )
    _conn.commit()
    return {"thread_id": thread_id, "created_at": now, "messages": [], "values": {}}


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
    """스레드에서 런을 시작합니다.

    start_async_task(새 태스크)와 update_async_task(재실행)에서 호출됩니다.
    multitask_strategy가 'interrupt'이면 기존 실행 중인 런을 취소합니다.
    """
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    body = await request.json()
    multitask_strategy = body.get("multitask_strategy")

    if multitask_strategy == "interrupt":
        _conn.execute(
            "UPDATE runs SET status = 'cancelled' WHERE thread_id = ? AND status = 'running'",
            (thread_id,),
        )
        _conn.execute(
            "UPDATE threads SET values_ = '{}' WHERE thread_id = ?",
            (thread_id,),
        )
        _conn.commit()

    messages = (body.get("input") or {}).get("messages") or []
    user_message = next((m["content"] for m in messages if m.get("role") == "user"), "")

    if user_message:
        existing = json.loads(
            _conn.execute(
                "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]
        )
        existing.append({"role": "user", "content": user_message})
        _conn.execute(
            "UPDATE threads SET messages = ? WHERE thread_id = ?",
            (json.dumps(existing), thread_id),
        )
        _conn.commit()

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    # graph_id가 assistant_id로 전달됨 — 저장하지만 라우팅에는 사용하지 않음 (서버당 에이전트 1개)
    assistant_id = body.get("assistant_id") or args.name
    _conn.execute(
        "INSERT INTO runs (run_id, thread_id, assistant_id, created_at) VALUES (?, ?, ?, ?)",
        (run_id, thread_id, assistant_id, now),
    )
    _conn.commit()

    asyncio.ensure_future(_execute_run(run_id, thread_id, user_message))

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "pending",
        "created_at": now,
        "error": None,
    }


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """런 상태를 조회합니다. check_async_task가 호출합니다."""
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """스레드 상태를 조회합니다. 런이 성공하면 values.messages에서 결과를 추출합니다."""
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """런을 취소합니다. cancel_async_task가 호출합니다."""
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    _conn.execute("UPDATE runs SET status = 'cancelled' WHERE run_id = ?", (run_id,))
    _conn.commit()
    return {**run, "status": "cancelled"}


# ── 메인 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
