# ruff: noqa: E501  # MEMORY_SYSTEM_PROMPT 내 긴 프롬프트 문자열 허용
"""AGENTS.md 파일에서 에이전트 메모리/컨텍스트를 로드하는 미들웨어 모듈.

이 모듈은 AGENTS.md 명세(https://agents.md/)를 구현하며,
설정 가능한 소스에서 메모리/컨텍스트를 로드하여 시스템 프롬프트에 주입합니다.

## 개요

AGENTS.md 파일은 AI 에이전트가 효과적으로 작업할 수 있도록
프로젝트별 컨텍스트와 지시사항을 제공합니다.
스킬(on-demand 워크플로우)과 달리, 메모리는 **항상 로드**되어 지속적인 컨텍스트를 제공합니다.

## 핵심 개념

- **메모리 소스(Source)**: AGENTS.md 파일의 경로 목록. 순서대로 로드되어 결합됩니다.
- **before_agent 훅**: 에이전트 실행 전에 한 번만 백엔드에서 파일을 다운로드합니다.
- **wrap_model_call 훅**: 매 LLM 호출마다 로드된 메모리 내용을 시스템 프롬프트에 주입합니다.
- **학습 기반 업데이트**: LLM이 사용자 피드백에서 학습하면 edit_file로 AGENTS.md를 직접 수정합니다.

## 사용 예시

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# 보안 주의: FilesystemBackend는 전체 파일시스템 읽기/쓰기를 허용합니다.
# 에이전트가 샌드박스 내에서 실행되거나, 파일 작업에 HIL(Human-in-the-loop)
# 승인을 추가해야 합니다.
backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",     # 사용자 전역 메모리
        "./.deepagents/AGENTS.md",     # 프로젝트별 메모리
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

## 메모리 소스

소스는 단순히 AGENTS.md 파일 경로이며, 순서대로 로드되어 결합됩니다.
여러 소스는 순서대로 연결되며, 나중 소스가 먼저 소스 뒤에 위치합니다.

## 파일 형식

AGENTS.md 파일은 특별한 구조가 필요 없는 표준 마크다운입니다.
일반적인 섹션:
- 프로젝트 개요
- 빌드/테스트 명령어
- 코드 스타일 가이드라인
- 아키텍처 노트
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """MemoryMiddleware의 상태 스키마.

    에이전트 상태를 확장하여 메모리 내용을 저장하는 필드를 추가합니다.

    Attributes:
        memory_contents: 소스 경로를 키로, 로드된 내용을 값으로 하는 딕셔너리.
            PrivateStateAttr로 표시되어 최종 에이전트 상태에는 포함되지 않습니다.
            즉, 부모 에이전트나 외부에서는 이 데이터에 접근할 수 없습니다.
    """

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class MemoryStateUpdate(TypedDict):
    """MemoryMiddleware의 상태 업데이트용 TypedDict.

    before_agent 훅에서 반환하여 메모리 내용을 상태에 저장할 때 사용합니다.
    """

    memory_contents: dict[str, str]


# LLM에게 주입되는 메모리 시스템 프롬프트 템플릿
# {agent_memory} 자리에 실제 AGENTS.md 내용이 삽입됩니다.
# 이 프롬프트는 LLM에게 메모리 업데이트 시점, 방법, 주의사항을 상세히 안내합니다.
MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the `edit_file` tool.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
    - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
    - When user says something is better/worse, capture WHY and encode it as a pattern.
    - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions.
    - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
    - Look for the underlying principle behind corrections, not just the specific mistake.
    - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action (e.g. send a Slack DM, requires a user ID/email) you should explicitly ask the user for this information.
    - It is preferred for you to ask for information, don't assume anything that you do not know!
    - When the user provides information that is useful for future use, you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
    - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
    - When the user gives feedback on your work - capture what was wrong and how to improve
    - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
    - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
    - When you discover new patterns or preferences (coding styles, conventions, workflows)

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
    - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
    - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
    - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
    - When the information is stale or irrelevant in future conversations
    - Never store API keys, access tokens, passwords, or any other credentials in any file, memory, or system prompt.
    - If the user asks where to put API keys or provides an API key, do NOT echo or save it.

    **Examples:**
    Example 1 (remembering user information):
    User: Can you connect to my google account?
    Agent: Sure, I'll connect to your google account, what's your google account email?
    User: john@example.com
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user's google account email is john@example.com

    Example 2 (remembering implicit user preferences):
    User: Can you write me an example for creating a deep agent in LangChain?
    Agent: Sure, I'll write you an example for creating a deep agent in LangChain <example code in Python>
    User: Can you do this in JavaScript
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user prefers to get LangChain code examples in JavaScript
    Agent: Sure, here is the JavaScript example<example code in JavaScript>

    Example 3 (do not remember transient information):
    User: I'm going to play basketball tonight so I will be offline for a few hours.
    Agent: Okay I'll add a block to your calendar.
    Tool Call: create_calendar_event(...) -> just calls a tool, does not commit anything to memory, as it is transient information
</memory_guidelines>
"""


class MemoryMiddleware(AgentMiddleware[MemoryState, ContextT, ResponseT]):
    """AGENTS.md 파일에서 에이전트 메모리를 로드하는 미들웨어.

    설정된 소스에서 메모리 내용을 로드하고 시스템 프롬프트에 주입합니다.
    여러 소스를 지원하며, 순서대로 결합됩니다.

    이 미들웨어의 생명주기:
        1. **before_agent** (최초 1회): 백엔드에서 AGENTS.md 파일들을 다운로드하여 상태에 저장
        2. **wrap_model_call** (매 LLM 호출): 저장된 메모리 내용을 시스템 프롬프트에 주입

    Args:
        backend: 파일 작업을 위한 백엔드 인스턴스 또는 팩토리 함수.
        sources: 메모리 파일 경로 목록 (예: `["~/.deepagents/AGENTS.md"]`).
    """

    # 이 미들웨어가 에이전트 상태에 추가하는 필드를 정의하는 스키마
    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: list[str],
    ) -> None:
        """메모리 미들웨어를 초기화합니다.

        Args:
            backend: 백엔드 인스턴스 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
                     StateBackend를 사용할 때는 팩토리를 사용합니다.
            sources: 로드할 메모리 파일 경로 목록.
                     (예: `["~/.deepagents/AGENTS.md", "./.deepagents/AGENTS.md"]`)
                     표시 이름은 경로에서 자동으로 추출됩니다.
                     소스는 순서대로 로드됩니다.
        """
        self._backend = backend
        self.sources = sources

    def _get_backend(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """백엔드 인스턴스를 해석(resolve)합니다.

        백엔드가 직접 인스턴스이면 그대로 반환하고,
        callable(팩토리 함수)이면 호출하여 인스턴스를 생성합니다.

        팩토리 패턴은 StateBackend처럼 런타임 컨텍스트(상태, 스토어 등)에
        접근해야 하는 백엔드를 위해 필요합니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 팩토리 함수에 전달할 런타임 컨텍스트.
            config: 백엔드 팩토리에 전달할 Runnable 설정.

        Returns:
            해석된 백엔드 인스턴스.
        """
        if callable(self._backend):
            # 팩토리 함수 호출을 위해 인위적인 ToolRuntime 객체를 생성
            # (before_agent 훅에서는 ToolRuntime을 직접 받지 못하므로)
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
        return self._backend

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """메모리 내용을 시스템 프롬프트용 형식으로 포맷합니다.

        각 소스의 경로와 내용을 쌍으로 묶어 <agent_memory> 태그 안에 배치합니다.

        Args:
            contents: 소스 경로를 키로, 파일 내용을 값으로 하는 딕셔너리.

        Returns:
            MEMORY_SYSTEM_PROMPT 템플릿에 메모리 내용이 삽입된 포맷된 문자열.
            내용이 없으면 "(No memory loaded)" 메시지가 포함됩니다.
        """
        if not contents:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        # self.sources 순서를 유지하면서 내용이 있는 소스만 포함
        # (소스 순서가 프롬프트 내 배치 순서를 결정)
        sections = [f"{path}\n{contents[path]}" for path in self.sources if contents.get(path)]

        if not sections:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        # 각 소스 섹션을 빈 줄 2개로 구분하여 결합
        memory_body = "\n\n".join(sections)
        return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)

    def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:  # ty: ignore[invalid-method-override]
        """에이전트 실행 전에 메모리 내용을 로드합니다 (동기 버전).

        모든 설정된 소스에서 메모리를 로드하여 상태에 저장합니다.
        이미 상태에 로드된 경우(이전 턴 또는 체크포인트된 세션에서)는 건너뜁니다.

        백엔드의 `download_files`를 사용하여 모든 소스 파일을 한 번에 배치 다운로드합니다.
        개별 `read` 대신 배치 다운로드를 사용하는 이유:
        - 네트워크 왕복(round-trip)을 최소화
        - `read()`는 줄 번호가 포함된 LLM용 포맷을 반환하지만, 여기서는 원본 내용이 필요

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.
            config: Runnable 설정.

        Returns:
            memory_contents가 채워진 상태 업데이트, 또는 이미 로드된 경우 None.

        Raises:
            ValueError: 파일 다운로드 실패 시 (file_not_found 제외).
        """
        # 이미 로드되었으면 건너뛰기 (중복 로드 방지)
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        # 모든 소스 파일을 배치로 다운로드
        results = backend.download_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                # file_not_found는 정상 — 아직 메모리 파일이 생성되지 않은 경우
                if response.error == "file_not_found":
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                # 바이트를 UTF-8 문자열로 디코딩하여 저장
                contents[path] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    async def abefore_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:  # ty: ignore[invalid-method-override]
        """에이전트 실행 전에 메모리 내용을 로드합니다 (비동기 버전).

        동기 버전 `before_agent`와 동일한 로직이지만, 비동기 백엔드 메서드를 사용합니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.
            config: Runnable 설정.

        Returns:
            memory_contents가 채워진 상태 업데이트, 또는 이미 로드된 경우 None.

        Raises:
            ValueError: 파일 다운로드 실패 시 (file_not_found 제외).
        """
        # 이미 로드되었으면 건너뛰기
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        # 비동기 배치 다운로드
        results = await backend.adownload_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                if response.error == "file_not_found":
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                contents[path] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """메모리 내용을 시스템 메시지에 주입합니다.

        상태에서 메모리 내용을 읽어 포맷한 후, 시스템 메시지에 추가합니다.
        이 메서드는 wrap_model_call에서 호출되며, 동기/비동기 버전에서 공통으로 사용됩니다.

        Args:
            request: 수정할 모델 요청 객체.

        Returns:
            메모리가 시스템 메시지에 주입된 수정된 요청 객체.
        """
        # 상태에서 메모리 내용 가져오기 (없으면 빈 딕셔너리)
        contents = request.state.get("memory_contents", {})

        # 메모리 내용을 시스템 프롬프트 형식으로 포맷
        agent_memory = self._format_agent_memory(contents)

        # 기존 시스템 메시지에 메모리 프롬프트 추가
        new_system_message = append_to_system_message(request.system_message, agent_memory)

        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """모델 호출을 래핑하여 메모리를 시스템 프롬프트에 주입합니다 (동기 버전).

        매 LLM 호출 전에 메모리 내용을 시스템 프롬프트에 추가합니다.
        이를 통해 LLM은 항상 프로젝트별 컨텍스트와 이전 학습 내용을 인식합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """모델 호출을 래핑하여 메모리를 시스템 프롬프트에 주입합니다 (비동기 버전).

        동기 버전 `wrap_model_call`과 동일한 로직이지만, 비동기 핸들러를 await합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 비동기 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
