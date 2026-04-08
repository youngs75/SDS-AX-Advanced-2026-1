"""동기 서브에이전트를 `task` 도구를 통해 제공하는 미들웨어 모듈.

이 모듈은 메인 에이전트가 복잡한 작업을 독립적인 서브에이전트에게 위임할 수 있게 합니다.
서브에이전트는 격리된 컨텍스트 윈도우에서 실행되며, 완료 후 단일 결과 메시지를 반환합니다.

## 핵심 개념

### 서브에이전트 패턴
- **오케스트레이터(메인 에이전트)**: 사용자와 직접 상호작용하며, 복잡한 작업을 분배
- **서브에이전트**: 격리된 환경에서 특정 작업을 수행하고, 결과만 오케스트레이터에게 반환
- **장점**: 컨텍스트 격리(토큰 절약), 병렬 실행 가능, 전문화된 도구/프롬프트 사용

### 서브에이전트 유형
1. **SubAgent**: 선언적 설정(TypedDict). model, tools, system_prompt를 지정하면
   미들웨어가 자동으로 `create_agent()`를 호출하여 실행 가능한 에이전트를 생성합니다.
2. **CompiledSubAgent**: 사전 컴파일된 에이전트(Runnable). 이미 완성된 LangGraph
   그래프를 직접 전달합니다. 상태 스키마에 'messages' 키가 필수입니다.

### 상태 격리
서브에이전트는 부모의 메시지 히스토리를 물려받지 않습니다.
`_EXCLUDED_STATE_KEYS`에 정의된 키(messages, todos, structured_response 등)는
서브에이전트에 전달되지 않으며, 서브에이전트가 반환하는 업데이트에서도 제외됩니다.

## 비동기 서브에이전트와의 차이

| 항목 | SubAgentMiddleware (이 모듈) | AsyncSubAgentMiddleware |
|------|------------------------------|-------------------------|
| 실행 방식 | 동기 — 완료될 때까지 블록 | 비동기 — 즉시 task_id 반환 |
| 실행 위치 | 로컬 (같은 프로세스) | 원격 Agent Protocol 서버 |
| 결과 반환 | 즉시 (도구 호출 결과로) | check_async_task로 폴링 |
| 용도 | 빠른 위임, 컨텍스트 격리 | 장시간 실행, 원격 배포 |
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message


class SubAgent(TypedDict):
    """서브에이전트를 위한 선언적 설정 스펙.

    `create_deep_agent`를 사용할 때, 서브에이전트는 자동으로 기본 미들웨어 스택
    (TodoListMiddleware, FilesystemMiddleware, SummarizationMiddleware 등)을
    이 스펙에 지정된 커스텀 `middleware`보다 먼저 받습니다.

    필수 필드:
        name: 서브에이전트의 고유 식별자.
            메인 에이전트가 `task()` 도구 호출 시 이 이름을 사용합니다.
        description: 서브에이전트가 수행하는 작업 설명.
            구체적이고 행동 지향적으로 작성합니다. 메인 에이전트가 위임 결정 시 참고합니다.
        system_prompt: 서브에이전트에 대한 지시사항.
            도구 사용 가이드와 출력 형식 요구사항을 포함합니다.

    선택 필드:
        tools: 서브에이전트가 사용할 도구 목록.
            지정하지 않으면 `default_tools`를 통해 메인 에이전트의 도구를 상속합니다.
        model: 메인 에이전트의 모델을 오버라이드.
            `'provider:model-name'` 형식을 사용합니다 (예: `'openai:gpt-4o'`).
        middleware: 커스텀 동작, 로깅, 속도 제한을 위한 추가 미들웨어.
        interrupt_on: 특정 도구에 대한 HIL(Human-in-the-loop) 설정.
            체크포인터가 필요합니다.
        skills: SkillsMiddleware용 스킬 소스 경로 목록.
            (예: `["/skills/user/", "/skills/project/"]`)
    """

    name: str
    """서브에이전트의 고유 식별자."""

    description: str
    """서브에이전트의 역할 설명. 메인 에이전트가 위임 판단 시 참고합니다."""

    system_prompt: str
    """서브에이전트에 대한 지시사항."""

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """서브에이전트가 사용할 도구 목록. 미지정 시 메인 에이전트로부터 상속."""

    model: NotRequired[str | BaseChatModel]
    """메인 에이전트 모델 오버라이드. `'provider:model-name'` 형식 사용."""

    middleware: NotRequired[list[AgentMiddleware]]
    """커스텀 동작을 위한 추가 미들웨어."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """특정 도구에 대한 HIL(Human-in-the-loop) 설정."""

    skills: NotRequired[list[str]]
    """SkillsMiddleware용 스킬 소스 경로 목록."""


class CompiledSubAgent(TypedDict):
    """사전 컴파일된 에이전트 스펙.

    이미 완성된 LangGraph 그래프나 create_agent()로 생성된 Runnable을
    직접 서브에이전트로 사용할 때 사용합니다.

    주의:
        runnable의 상태 스키마에 반드시 'messages' 키가 포함되어야 합니다.
        서브에이전트가 완료되면 'messages' 리스트의 마지막 메시지가
        부모 에이전트에게 ToolMessage로 반환됩니다.
    """

    name: str
    """서브에이전트의 고유 식별자."""

    description: str
    """서브에이전트의 역할 설명."""

    runnable: Runnable
    """커스텀 에이전트 구현체.

    다음 중 하나로 생성합니다:
    1. LangChain의 `create_agent()` (권장)
    2. `langgraph`를 사용한 커스텀 그래프

    커스텀 그래프를 사용하는 경우, 상태 스키마에 'messages' 키가 필수입니다.
    이 키를 통해 서브에이전트가 결과를 부모 에이전트에게 전달합니다.
    """


# 서브에이전트의 기본 시스템 프롬프트
# 간결한 기본값 — 실제 사용 시 SubAgent.system_prompt로 오버라이드됩니다.
DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# 서브에이전트 호출/반환 시 상태에서 제외되는 키 목록
#
# 제외 이유:
# 1. messages: 서브에이전트는 자체 메시지 히스토리를 가지므로, 부모의 히스토리와 격리
# 2. todos, structured_response: 리듀서가 정의되지 않았고, 서브에이전트 → 부모 반환 의미가 불명확
# 3. skills_metadata, memory_contents: PrivateStateAttr로 자동 제외되지만,
#    부모 상태가 자식에게 누출되는 것을 방지하기 위해 명시적으로도 필터링
#    (예: 범용 서브에이전트가 SkillsMiddleware를 통해 자체 스킬을 로드해야 하므로)
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "skills_metadata", "memory_contents"}


class TaskToolSchema(BaseModel):
    """task 도구의 입력 스키마.

    LLM이 task 도구를 호출할 때 제공해야 하는 매개변수를 정의합니다.
    Pydantic BaseModel을 사용하여 LLM에게 구조화된 입력 형식을 제공합니다.
    """

    description: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )
    subagent_type: str = Field(description=("The type of subagent to use. Must be one of the available agent types listed in the tool description."))


# task 도구의 상세 설명 템플릿
# {available_agents} 자리에 사용 가능한 서브에이전트 목록이 동적으로 삽입됩니다.
# 이 설명은 LLM에게 task 도구의 사용법, 모범 사례, 예시를 제공합니다.
TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501

# task 도구 사용법에 대한 시스템 프롬프트
# wrap_model_call에서 시스템 메시지에 주입되어, LLM이 task 도구의 존재와 사용 패턴을 인식하게 합니다.
TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""  # noqa: E501


# 범용(general-purpose) 서브에이전트의 기본 설명
# 메인 에이전트와 동일한 도구를 가지며, 컨텍스트 격리가 필요한 모든 작업에 사용됩니다.
DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501

# 범용 서브에이전트 기본 스펙 (호출자가 model, tools, middleware를 추가해야 함)
GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}


class _SubagentSpec(TypedDict):
    """task 도구 빌드를 위한 내부 스펙.

    SubAgent와 CompiledSubAgent를 통합하여 task 도구 빌드에 필요한
    최소 정보(name, description, runnable)만 담는 내부용 TypedDict입니다.
    """

    name: str
    description: str
    runnable: Runnable


def _build_task_tool(  # noqa: C901
    subagents: list[_SubagentSpec],
    task_description: str | None = None,
) -> BaseTool:
    """사전 빌드된 서브에이전트 그래프로부터 task 도구를 생성합니다.

    이 함수는 서브에이전트 스펙 리스트를 받아, LLM이 호출할 수 있는
    `task` StructuredTool을 생성합니다. 동기/비동기 양쪽 구현을 모두 제공합니다.

    Args:
        subagents: name, description, runnable을 포함하는 서브에이전트 스펙 리스트.
        task_description: task 도구의 커스텀 설명. None이면 기본 템플릿 사용.
            `{available_agents}` 플레이스홀더를 지원합니다.

    Returns:
        서브에이전트를 유형별로 호출할 수 있는 StructuredTool.
    """
    # 스펙 리스트에서 이름→그래프 매핑 딕셔너리와 설명 문자열을 생성
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in subagents}
    subagent_description_str = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)

    # 커스텀 설명이 없으면 기본 템플릿 사용, 있으면 플레이스홀더 치환
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        description = task_description.format(available_agents=subagent_description_str)
    else:
        description = task_description

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        """서브에이전트 실행 결과를 Command 객체로 변환합니다.

        서브에이전트의 최종 메시지를 추출하여 ToolMessage로 감싸고,
        제외 키를 필터링한 상태 업데이트를 생성합니다.

        Args:
            result: 서브에이전트 실행 결과 딕셔너리.
            tool_call_id: 원본 도구 호출 ID (ToolMessage에 연결).

        Returns:
            상태 업데이트가 포함된 Command 객체.

        Raises:
            ValueError: 결과에 'messages' 키가 없는 경우.
        """
        # 서브에이전트가 반드시 'messages' 키를 반환해야 함을 검증
        if "messages" not in result:
            error_msg = (
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)

        # 제외 키를 필터링하여 부모에게 전달할 상태 업데이트 생성
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}

        # 후행 공백 제거 — Anthropic API에서 후행 공백이 있으면 오류 발생
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""

        return Command(
            update={
                **state_update,
                # 서브에이전트의 마지막 메시지만 ToolMessage로 변환하여 부모에게 반환
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        """서브에이전트 호출을 위한 상태를 준비합니다.

        부모 상태에서 제외 키를 필터링하고, 서브에이전트용 메시지 히스토리를
        사용자의 task description 단일 메시지로 초기화합니다.

        Args:
            subagent_type: 호출할 서브에이전트 유형 이름.
            description: 서브에이전트에게 전달할 작업 설명.
            runtime: 현재 도구 런타임 (부모 상태 접근용).

        Returns:
            (서브에이전트 Runnable, 준비된 상태 딕셔너리) 튜플.
        """
        subagent = subagent_graphs[subagent_type]

        # 부모 상태를 복사하되, 제외 키는 필터링 (상태 격리)
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}

        # 서브에이전트는 부모의 대화 히스토리 대신, task description을 유일한 메시지로 받음
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """서브에이전트를 동기적으로 실행하는 task 도구 구현 (동기 버전).

        Args:
            description: 서브에이전트가 수행할 작업에 대한 상세 설명.
            subagent_type: 사용할 서브에이전트 유형 이름.
            runtime: 도구 런타임 (상태 접근 및 tool_call_id 제공).

        Returns:
            성공 시 Command 객체 (상태 업데이트 + ToolMessage),
            실패 시 오류 메시지 문자열.

        Raises:
            ValueError: tool_call_id가 없는 경우.
        """
        # 존재하지 않는 서브에이전트 유형이면 오류 메시지 반환
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)

        # 서브에이전트 상태 준비 및 동기 실행
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = subagent.invoke(subagent_state)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """서브에이전트를 비동기적으로 실행하는 task 도구 구현 (비동기 버전).

        동기 버전 `task`와 동일한 로직이지만, `ainvoke`를 사용합니다.

        Args:
            description: 서브에이전트가 수행할 작업에 대한 상세 설명.
            subagent_type: 사용할 서브에이전트 유형 이름.
            runtime: 도구 런타임.

        Returns:
            성공 시 Command 객체, 실패 시 오류 메시지 문자열.

        Raises:
            ValueError: tool_call_id가 없는 경우.
        """
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)

        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,        # Pydantic 스키마를 명시적으로 제공하므로 자동 추론 비활성화
        args_schema=TaskToolSchema,
    )


class SubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """동기 서브에이전트를 `task` 도구를 통해 제공하는 미들웨어.

    이 미들웨어는 에이전트에 `task` 도구를 추가하여 서브에이전트를 호출할 수 있게 합니다.
    서브에이전트는 복잡한 다단계 작업이나 많은 컨텍스트가 필요한 작업을 처리하는 데 유용합니다.

    서브에이전트의 핵심 장점:
    - **컨텍스트 격리**: 서브에이전트는 격리된 컨텍스트 윈도우에서 실행되어 토큰 절약
    - **병렬 실행**: 독립적인 작업을 병렬로 실행하여 성능 향상
    - **전문화**: 각 서브에이전트가 전문 도구와 프롬프트를 사용
    - **깔끔한 결과**: 중간 과정은 숨기고 최종 결과만 반환

    동작 방식:
        1. __init__에서 서브에이전트 스펙을 기반으로 LangGraph 에이전트 그래프를 생성
        2. wrap_model_call에서 시스템 프롬프트에 task 도구 사용법 주입
        3. LLM이 task 도구를 호출하면 해당 서브에이전트를 실행하고 결과 반환

    Args:
        backend: 파일 작업 및 실행을 위한 백엔드.
        subagents: 완전히 지정된 서브에이전트 설정 리스트. 각 SubAgent는
            `model`과 `tools`를 반드시 지정해야 합니다. 개별 서브에이전트의
            `interrupt_on` 설정도 존중됩니다.
        system_prompt: 메인 에이전트의 시스템 프롬프트에 추가될 task 도구 사용 지침.
        task_description: task 도구의 커스텀 설명.

    사용 예시:
        ```python
        from deepagents.middleware import SubAgentMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    backend=my_backend,
                    subagents=[
                        {
                            "name": "researcher",
                            "description": "Research agent",
                            "system_prompt": "You are a researcher.",
                            "model": "openai:gpt-4o",
                            "tools": [search_tool],
                        }
                    ],
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = None,
    ) -> None:
        """SubAgentMiddleware를 초기화합니다.

        서브에이전트 스펙을 검증하고, 각 서브에이전트를 실행 가능한 LangGraph 그래프로
        컴파일한 후, task 도구와 시스템 프롬프트를 생성합니다.

        Args:
            backend: 파일 작업 백엔드.
            subagents: 서브에이전트 설정 리스트 (최소 1개 필수).
            system_prompt: task 도구 사용 지침 (None이면 시스템 프롬프트 주입 건너뜀).
            task_description: task 도구의 커스텀 설명.

        Raises:
            ValueError: 서브에이전트가 비어있는 경우.
        """
        super().__init__()

        if not subagents:
            msg = "At least one subagent must be specified"
            raise ValueError(msg)
        self._backend = backend
        self._subagents = subagents

        # 서브에이전트 스펙을 실행 가능한 그래프로 컴파일
        subagent_specs = self._get_subagents()

        # task 도구 생성
        task_tool = _build_task_tool(subagent_specs, task_description)

        # 사용 가능한 서브에이전트 목록을 시스템 프롬프트에 추가
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagent_specs)
            self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

        # 미들웨어가 에이전트에 제공하는 도구 목록 (task 도구 1개)
        self.tools = [task_tool]

    def _get_subagents(self) -> list[_SubagentSpec]:
        """서브에이전트 스펙을 실행 가능한 에이전트로 컴파일합니다.

        SubAgent는 create_agent()를 호출하여 LangGraph 그래프를 생성하고,
        CompiledSubAgent는 이미 완성된 runnable을 그대로 사용합니다.

        Returns:
            name, description, runnable을 포함하는 서브에이전트 스펙 리스트.

        Raises:
            ValueError: SubAgent에 model 또는 tools가 누락된 경우.
        """
        specs: list[_SubagentSpec] = []

        for spec in self._subagents:
            if "runnable" in spec:
                # CompiledSubAgent — 이미 완성된 그래프를 그대로 사용
                compiled = cast("CompiledSubAgent", spec)
                specs.append({"name": compiled["name"], "description": compiled["description"], "runnable": compiled["runnable"]})
                continue

            # SubAgent — 필수 필드 검증
            if "model" not in spec:
                msg = f"SubAgent '{spec['name']}' must specify 'model'"
                raise ValueError(msg)
            if "tools" not in spec:
                msg = f"SubAgent '{spec['name']}' must specify 'tools'"
                raise ValueError(msg)

            # 문자열 모델 이름을 BaseChatModel 인스턴스로 해석
            from deepagents._models import resolve_model  # noqa: PLC0415

            model = resolve_model(spec["model"])

            # 호출자가 제공한 미들웨어를 그대로 사용 (전체 스택 구성은 호출자 책임)
            middleware: list[AgentMiddleware] = list(spec.get("middleware", []))

            # interrupt_on 설정이 있으면 HIL 미들웨어 추가
            interrupt_on = spec.get("interrupt_on")
            if interrupt_on:
                middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

            # create_agent()로 LangGraph 에이전트 그래프 생성
            specs.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "runnable": create_agent(
                        model,
                        system_prompt=spec["system_prompt"],
                        tools=spec["tools"],
                        middleware=middleware,
                        name=spec["name"],
                    ),
                }
            )

        return specs

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """시스템 메시지에 서브에이전트 사용 지침을 주입합니다 (동기 버전).

        매 LLM 호출 전에 시스템 프롬프트에 task 도구 사용법과
        사용 가능한 서브에이전트 목록을 추가합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """시스템 메시지에 서브에이전트 사용 지침을 주입합니다 (비동기 버전).

        동기 버전과 동일한 로직이지만, 비동기 핸들러를 await합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 비동기 핸들러 함수.

        Returns:
            핸들러로부터 받은 모델 응답.
        """
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
