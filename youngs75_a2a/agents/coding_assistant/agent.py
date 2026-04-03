"""Coding Assistant 에이전트 — 3노드 구조.

논문 인사이트 기반 설계:
- P1 (Agent-as-a-Judge): 최소 3노드 parse → execute → verify
- P2 (RubricRewards): Generator/Verifier 모델 분리
- P5 (GAM): verify 실패 시 원본 코드 재참조 (JIT)

사용 예:
    agent = CodingAssistantAgent(config=CodingConfig())
    result = await agent.graph.ainvoke({
        "messages": [HumanMessage("파이썬으로 피보나치 함수를 작성해줘")],
        "iteration": 0,
        "max_iterations": 3,
    })
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent

from .config import CodingConfig
from .prompts import EXECUTE_SYSTEM_PROMPT, PARSE_SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT
from .schemas import CodingState


class CodingAssistantAgent(BaseGraphAgent):
    """3노드 Coding Assistant: parse_request → execute_code → verify_result."""

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "PARSE": "parse_request",
        "EXECUTE": "execute_code",
        "VERIFY": "verify_result",
    }

    def __init__(
        self,
        *,
        config: CodingConfig | None = None,
        model: BaseChatModel | None = None,
        **kwargs: Any,
    ) -> None:
        self._coding_config = config or CodingConfig()

        # Generator/Verifier 모델 — lazy init (P2)
        self._explicit_model = model
        self._gen_model: BaseChatModel | None = None
        self._verify_model: BaseChatModel | None = None
        self._parse_model: BaseChatModel | None = None

        kwargs.pop("auto_build", None)
        super().__init__(
            config=self._coding_config,
            model=model,
            state_schema=CodingState,
            agent_name="CodingAssistantAgent",
            auto_build=True,
            **kwargs,
        )

    # ── 모델 lazy init ──────────────────────────────────────

    def _get_parse_model(self) -> BaseChatModel:
        if self._parse_model is None:
            self._parse_model = self._explicit_model or self._coding_config.get_model("default")
        return self._parse_model

    def _get_gen_model(self) -> BaseChatModel:
        if self._gen_model is None:
            self._gen_model = self._coding_config.get_model("generation")
        return self._gen_model

    def _get_verify_model(self) -> BaseChatModel:
        if self._verify_model is None:
            self._verify_model = self._coding_config.get_model("verification")
        return self._verify_model

    # ── 노드 구현 ──────────────────────────────────────────

    async def _parse_request(self, state: CodingState) -> dict[str, Any]:
        """사용자 요청을 분석하여 작업 유형과 요구사항을 추출한다."""
        messages = [
            SystemMessage(content=PARSE_SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = await self._get_parse_model().ainvoke(messages)

        try:
            parse_result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            parse_result = {
                "task_type": "generate",
                "description": response.content,
                "target_files": [],
                "requirements": [],
            }

        return {
            "parse_result": parse_result,
            "execution_log": [f"[parse] task_type={parse_result.get('task_type')}"],
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
        }

    async def _execute_code(self, state: CodingState) -> dict[str, Any]:
        """요구사항에 따라 코드를 생성하거나 수정한다."""
        parse_result = state.get("parse_result", {})
        verify_result = state.get("verify_result")

        # 반복 시 이전 검증 피드백을 포함
        context_parts = [
            f"작업 유형: {parse_result.get('task_type', 'generate')}",
            f"설명: {parse_result.get('description', '')}",
        ]
        if parse_result.get("requirements"):
            context_parts.append(f"요구사항: {', '.join(parse_result['requirements'])}")

        # 검증 실패로 재시도 시, 피드백 반영
        if verify_result and not verify_result.get("passed"):
            issues = verify_result.get("issues", [])
            context_parts.append(f"\n이전 검증에서 발견된 문제:\n- " + "\n- ".join(issues))
            context_parts.append("위 문제를 수정하여 다시 코드를 작성하세요.")

        context_msg = HumanMessage(content="\n".join(context_parts))
        messages = [
            SystemMessage(content=EXECUTE_SYSTEM_PROMPT),
            *state["messages"],
            context_msg,
        ]
        response = await self._get_gen_model().ainvoke(messages)

        iteration = state.get("iteration", 0)
        log = state.get("execution_log", [])
        log.append(f"[execute] iteration={iteration}")

        return {
            "generated_code": response.content,
            "execution_log": log,
            "messages": [AIMessage(content=response.content)],
        }

    async def _verify_result(self, state: CodingState) -> dict[str, Any]:
        """생성된 코드를 검증한다 (검증자 특권 정보 포함)."""
        generated_code = state.get("generated_code", "")
        parse_result = state.get("parse_result", {})

        # 검증자 특권 정보 주입 (P2)
        verify_prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=self._coding_config.max_delete_lines,
            allowed_extensions=", ".join(self._coding_config.allowed_extensions),
        )

        verify_context = (
            f"원래 요청: {parse_result.get('description', '')}\n\n"
            f"생성된 코드:\n{generated_code}"
        )
        messages = [
            SystemMessage(content=verify_prompt),
            HumanMessage(content=verify_context),
        ]
        response = await self._get_verify_model().ainvoke(messages)

        try:
            verify_result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            verify_result = {
                "passed": True,
                "issues": [],
                "suggestions": [],
            }

        log = state.get("execution_log", [])
        log.append(f"[verify] passed={verify_result.get('passed')}")

        return {
            "verify_result": verify_result,
            "execution_log": log,
            "iteration": state.get("iteration", 0) + 1,
        }

    # ── 라우팅 ──────────────────────────────────────────────

    def _should_retry(self, state: CodingState) -> str:
        """검증 실패 시 재시도 여부를 판단한다."""
        verify_result = state.get("verify_result", {})
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if verify_result.get("passed", True):
            return END
        if iteration >= max_iterations:
            return END
        return self.get_node_name("EXECUTE")

    # ── 그래프 구성 ─────────────────────────────────────────

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("PARSE"), self._parse_request)
        graph.add_node(self.get_node_name("EXECUTE"), self._execute_code)
        graph.add_node(self.get_node_name("VERIFY"), self._verify_result)

    def init_edges(self, graph: StateGraph) -> None:
        # parse → execute → verify → (retry or END)
        graph.set_entry_point(self.get_node_name("PARSE"))
        graph.add_edge(
            self.get_node_name("PARSE"),
            self.get_node_name("EXECUTE"),
        )
        graph.add_edge(
            self.get_node_name("EXECUTE"),
            self.get_node_name("VERIFY"),
        )
        graph.add_conditional_edges(
            self.get_node_name("VERIFY"),
            self._should_retry,
        )
