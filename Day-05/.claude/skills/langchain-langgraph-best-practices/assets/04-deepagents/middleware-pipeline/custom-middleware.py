"""
Custom Middleware: building reusable middleware components for DeepAgents.

Demonstrates:
- AgentMiddleware base class: state_schema, process_model_request hooks
- Custom middleware pattern: LocalContextMiddleware example
- State schema extension with PrivateStateAttr and reducers
- System prompt injection with XML tags and cache breakpoints
- Middleware stack composition and execution order
- PatchToolCallsMiddleware: fixing dangling tool calls at stream end
- Middleware receives runtime (backend, store, config access)
- Tool injection: middleware can add tools dynamically

Pattern:
- Middleware intercepts LLM calls before/after execution
- Each middleware extends agent state with custom fields
- Stack processes in order: early middleware sees raw state, late sees enriched
- System prompt built incrementally by each middleware layer

Key middleware stack (from graph.py):
1. TodoListMiddleware - task tracking state + tools
2. MemoryMiddleware - conversation memory injection
3. SkillsMiddleware - skill loading + tool exposure
4. FilesystemMiddleware - file operations
5. SubAgentMiddleware - spawning child agents
6. SummarizationMiddleware - conversation compaction
7. AnthropicPromptCachingMiddleware - cache control breakpoints
8. PatchToolCallsMiddleware - fix incomplete tool calls at stream end
9. User middleware (via create_deep_agent(middleware=[...]))

Example: LocalContextMiddleware gathers git/directory context once, injects into prompt.
"""

import asyncio
from typing import Any, Literal, TypedDict
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# from deepagents import create_deep_agent, AgentMiddleware
# from deepagents.state import AgentState, PrivateStateAttr
# from deepagents.runtime import AgentRuntime
# from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
model = FakeListChatModel(
    responses=[
        "I see the project context injected by middleware",
        "The stack order ensures context is available before processing",
    ]
)


# ==== Custom Middleware Class ====
# Pattern: LocalContextMiddleware-style implementation
# This middleware:
# - Gathers git branch, directory tree, file listing ONCE
# - Caches in state to avoid re-gathering
# - Injects into system prompt with <project_context> tags
# - Uses PrivateStateAttr to exclude internal fields from output

class LocalContextState(TypedDict):
    """State schema extension for local project context."""
    # PrivateStateAttr excludes from agent output
    local_context_gathered: bool  # Would be: Annotated[bool, PrivateStateAttr]
    git_branch: str  # Would be: Annotated[str, PrivateStateAttr]
    project_tree: str  # Would be: Annotated[str, PrivateStateAttr]
    file_count: int  # Would be: Annotated[int, PrivateStateAttr]


# class LocalContextMiddleware(AgentMiddleware):
#     """
#     Injects local project context (git, files, tree) into system prompt.
#
#     Gathers context once on first call, caches in state for subsequent calls.
#     Useful for giving agents awareness of current project structure.
#     """
#
#     @property
#     def state_schema(self) -> type[LocalContextState]:
#         """Extend agent state with project context fields."""
#         return LocalContextState
#
#     async def _gather_context(self, runtime: AgentRuntime) -> dict[str, Any]:
#         """Gather project context (called once)."""
#         # In real implementation:
#         # - Run `git branch --show-current` via backend.run_in_terminal
#         # - Run `tree -L 2` or `ls -R` for directory structure
#         # - Count files with glob patterns
#         # - Store in backend.store for persistence
#
#         return {
#             "git_branch": "main",
#             "project_tree": "src/\n  agents/\n  middleware/\ntests/",
#             "file_count": 42,
#         }
#
#     async def process_model_request(
#         self,
#         state: AgentState,
#         runtime: AgentRuntime,
#     ) -> AgentState:
#         """
#         Hook called BEFORE LLM invocation.
#
#         Middleware can:
#         - Modify state fields
#         - Inject system prompt context
#         - Add tools dynamically
#         - Transform messages
#         """
#         # Gather context on first call
#         if not state.get("local_context_gathered"):
#             context = await self._gather_context(runtime)
#             state["git_branch"] = context["git_branch"]
#             state["project_tree"] = context["project_tree"]
#             state["file_count"] = context["file_count"]
#             state["local_context_gathered"] = True
#
#         # Inject into system prompt
#         context_prompt = f"""
# <project_context>
# Current branch: {state['git_branch']}
# File count: {state['file_count']}
#
# Directory structure:
# {state['project_tree']}
# </project_context>
# """
#
#         # Prepend to system message (like MemoryMiddleware pattern)
#         messages = state.get("messages", [])
#         if messages and isinstance(messages[0], SystemMessage):
#             messages[0].content = context_prompt + messages[0].content
#         else:
#             messages.insert(0, SystemMessage(content=context_prompt))
#
#         state["messages"] = messages
#         return state


# ==== Middleware Stack Order ====
# Full stack from graph.py create_deep_agent():

MIDDLEWARE_STACK_DOCS = """
Middleware Stack Execution Order (early → late):

1. TodoListMiddleware
   - Extends state: current_todos (list), todo_operations (list)
   - Adds tools: create_todo, update_todo, list_todos
   - System prompt: <todo_list> current tasks </todo_list>

2. MemoryMiddleware
   - Extends state: memory_entries (list), last_summarized_turn (int)
   - System prompt: <agent_memory> past conversations </agent_memory>
   - Uses prompt caching for static memory blocks

3. SkillsMiddleware
   - Extends state: loaded_skills (dict), active_skill (str)
   - Adds tools: dynamically from SKILL.md tool definitions
   - System prompt: <agent_skills> available skills </agent_skills>
   - Progressive disclosure: loads SKILL.md on demand

4. FilesystemMiddleware
   - Adds tools: read_file, write_file, list_directory, search_files
   - No state extension (stateless tools)
   - Handles file operations with safety checks

5. SubAgentMiddleware
   - Extends state: subagent_results (list), active_subagents (list)
   - Adds tools: spawn_subagent, query_subagent
   - Manages child agent lifecycle

6. SummarizationMiddleware
   - Extends state: summarization_checkpoint (int)
   - Automatically compacts conversation history when >N turns
   - Preserves important context via summary

7. AnthropicPromptCachingMiddleware
   - No state extension
   - Injects cache_control breakpoints into system messages
   - Optimizes token usage for repeated prompt blocks

8. PatchToolCallsMiddleware
   - No state extension
   - Fixes dangling tool calls at stream end
   - See pattern below for details

9. User Custom Middleware (via create_deep_agent(middleware=[...]))
   - Your middleware runs LAST in the stack
   - Has access to all enriched state from earlier middleware
   - Can override or augment existing prompt sections

Stack flow:
- process_model_request() called top→bottom (1→9)
- Each middleware sees state enriched by previous middleware
- System prompt built incrementally
- LLM receives fully-processed state + prompt
"""


# ==== PatchToolCallsMiddleware Pattern ====
# Problem: Streaming LLM may end mid-tool-call, leaving partial JSON
# Solution: Detect dangling calls, append empty tool_calls=[] to complete message

PATCH_TOOL_CALLS_PATTERN = """
PatchToolCallsMiddleware: Fixing Incomplete Tool Calls

Problem:
- Streaming models may stop mid-tool-call due to token limit or other reasons
- Leaves AIMessage with partial tool_call JSON
- LangGraph crashes on invalid tool_call format

Example dangling message:
AIMessage(
    content="I'll use the search tool",
    tool_calls=[
        {"name": "search", "args": {"query": "
    ]  # Incomplete! Missing closing braces
)

Solution (from deepagents/middleware/patch_tool_calls.py):

class PatchToolCallsMiddleware(AgentMiddleware):
    async def process_model_request(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return state

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            # Check if tool_calls are malformed
            try:
                # Validate JSON structure
                json.dumps(last_msg.tool_calls)
            except (TypeError, ValueError):
                # Dangling tool call detected!
                # Fix: append message with empty tool_calls to signal completion
                messages.append(
                    AIMessage(
                        content="[Tool call interrupted]",
                        tool_calls=[]  # Empty = no tools to execute
                    )
                )
                state["messages"] = messages

        return state

Result:
- LangGraph sees completed message sequence
- No crashes on partial JSON
- Agent can continue or retry
"""


# ==== State Schema Extension Pattern ====
# Middleware extends agent state with custom fields

STATE_SCHEMA_PATTERN = """
State Schema Extension Pattern:

Base AgentState (from deepagents/state.py):
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: str
    parent_agent_id: Optional[str]
    depth: int
    # ... core fields

Middleware extends with state_schema property:

from typing import Annotated
from deepagents.state import PrivateStateAttr
import operator

class MyMiddlewareState(TypedDict):
    # Public field: visible in agent output
    custom_context: str

    # Private field: excluded from output (via PrivateStateAttr)
    internal_cache: Annotated[dict, PrivateStateAttr]

    # List field with reducer: appends instead of replaces
    operation_log: Annotated[List[str], operator.add]

class MyMiddleware(AgentMiddleware):
    @property
    def state_schema(self) -> type[MyMiddlewareState]:
        return MyMiddlewareState

    async def process_model_request(self, state, runtime):
        # Initialize custom fields if not present
        if "custom_context" not in state:
            state["custom_context"] = "initialized"

        # Append to list (reducer handles merging)
        state.setdefault("operation_log", []).append("processed")

        return state

Key concepts:
- PrivateStateAttr: excludes field from agent output JSON
- operator.add: reducer for list fields (appends instead of replaces)
- TypedDict: provides type hints for state fields
- Middleware must initialize fields it uses
"""


# ==== System Prompt Injection Pattern ====
# How middleware adds context to system prompt

SYSTEM_PROMPT_PATTERN = """
System Prompt Injection Pattern:

Middleware injects context by prepending to system message:

async def process_model_request(self, state, runtime):
    # Build context block
    context_block = f'''
<my_context>
Custom context here
Key: {state.get("custom_field")}
</my_context>
'''

    # Inject into system message
    messages = state.get("messages", [])
    if messages and isinstance(messages[0], SystemMessage):
        # Prepend to existing system message
        messages[0].content = context_block + messages[0].content
    else:
        # Insert new system message at start
        messages.insert(0, SystemMessage(content=context_block))

    state["messages"] = messages
    return state

Result system prompt (after multiple middleware):
<project_context>...</project_context>
<agent_memory>...</agent_memory>
<agent_skills>...</agent_skills>
<todo_list>...</todo_list>
<my_context>...</my_context>

Original system instructions here...

Benefits:
- XML tags: clear section boundaries
- Incremental: each middleware adds its section
- Cacheable: static sections get cache_control breakpoints
- Modular: middleware can be enabled/disabled independently
"""


# ==== Tool Injection Pattern ====
# Middleware can add tools dynamically

TOOL_INJECTION_PATTERN = """
Tool Injection Pattern:

Middleware adds tools to agent at runtime:

from langchain_core.tools import tool

class MyMiddleware(AgentMiddleware):
    async def process_model_request(self, state, runtime):
        # Define custom tool
        @tool
        def custom_tool(query: str) -> str:
            '''Custom tool description'''
            return f"Processed: {query}"

        # Add to agent's tool list
        # (In real implementation, tools passed via runtime.model.bind_tools())
        # Middleware typically stores tools in state for graph to bind

        return state

Examples from built-in middleware:

TodoListMiddleware:
- create_todo(title: str, description: str)
- update_todo(todo_id: str, status: Literal["pending", "done"])
- list_todos() -> List[Todo]

FilesystemMiddleware:
- read_file(path: str) -> str
- write_file(path: str, content: str)
- list_directory(path: str) -> List[str]
- search_files(pattern: str) -> List[str]

SubAgentMiddleware:
- spawn_subagent(agent_type: str, task: str) -> str
- query_subagent(subagent_id: str, query: str) -> str

Pattern:
- Tools access runtime (backend, store)
- Tools modify state (e.g., create_todo adds to state["current_todos"])
- Tool results flow back through message chain
"""


# ==== Middleware Runtime Access ====
# Runtime provides backend and store access

RUNTIME_ACCESS_PATTERN = """
Middleware Runtime Access:

AgentRuntime provides access to:

class AgentRuntime:
    backend: AgentBackend      # Terminal, file ops, tool execution
    store: BaseStore           # Persistent key-value storage
    config: RunnableConfig     # LangGraph config (thread_id, etc)
    checkpointer: BaseCheckpointer  # State persistence

Example usage in middleware:

async def process_model_request(self, state, runtime):
    # Run terminal command
    result = await runtime.backend.run_in_terminal(
        command=["git", "branch", "--show-current"],
        cwd="/project"
    )

    # Store persistent data
    await runtime.store.aput(
        namespace=["agent", state["current_agent"]],
        key="last_context",
        value={"branch": result.output}
    )

    # Retrieve stored data
    stored = await runtime.store.aget(
        namespace=["agent", state["current_agent"]],
        key="last_context"
    )

    # Access thread_id for scoped operations
    thread_id = runtime.config["configurable"]["thread_id"]

    return state

Common patterns:
- Backend: execute tools, file ops, terminal commands
- Store: persist data across turns (memory, cache, state)
- Config: thread_id for scoped storage, user_id for auth
- Checkpointer: save/load full agent state for resume
"""


# ==== Agent Setup (Commented) ====
# How to use custom middleware

# custom_middleware = LocalContextMiddleware()
#
# agent = create_deep_agent(
#     model=model,
#     system_prompt="You are a helpful assistant",
#     middleware=[custom_middleware],  # Add custom middleware to stack
#     checkpointer=InMemorySaver(),
# )
#
# # Middleware stack execution:
# # 1-8: Built-in middleware (todo, memory, skills, etc)
# # 9: custom_middleware (LocalContextMiddleware)
# #
# # On first call:
# # - LocalContextMiddleware gathers git/tree context
# # - Caches in state["local_context_gathered"] = True
# # - Injects <project_context> into system prompt
# #
# # On subsequent calls:
# # - Skips gathering (already cached)
# # - Re-injects prompt (prompt rebuilt each turn)
#
# config = {"configurable": {"thread_id": "demo-thread"}}
# result = await agent.ainvoke(
#     {"messages": [HumanMessage(content="Show me project structure")]},
#     config=config,
# )


# ==== Main ====
async def main():
    print("=== Custom Middleware Pattern ===\n")

    print("1. Middleware Base Class")
    print("-" * 50)
    print("AgentMiddleware provides hooks:")
    print("  - state_schema: extend agent state with custom fields")
    print("  - process_model_request(): called BEFORE LLM invocation")
    print("  - Receives runtime (backend, store, config access)")
    print()

    print("2. LocalContextMiddleware Example")
    print("-" * 50)
    print("Custom middleware that injects project context:")
    print("  class LocalContextMiddleware(AgentMiddleware):")
    print("      @property")
    print("      def state_schema(self) -> type[LocalContextState]:")
    print("          return LocalContextState  # git_branch, project_tree, file_count")
    print()
    print("      async def process_model_request(self, state, runtime):")
    print("          # Gather context once, cache in state")
    print("          if not state.get('local_context_gathered'):")
    print("              context = await self._gather_context(runtime)")
    print("              state['git_branch'] = context['git_branch']")
    print("              # ...")
    print()
    print("          # Inject into system prompt")
    print("          context_prompt = '<project_context>...</project_context>'")
    print("          messages[0].content = context_prompt + messages[0].content")
    print()

    print("3. Middleware Stack Order")
    print("-" * 50)
    print(MIDDLEWARE_STACK_DOCS)
    print()

    print("4. PatchToolCallsMiddleware Pattern")
    print("-" * 50)
    print(PATCH_TOOL_CALLS_PATTERN)
    print()

    print("5. State Schema Extension")
    print("-" * 50)
    print(STATE_SCHEMA_PATTERN)
    print()

    print("6. System Prompt Injection")
    print("-" * 50)
    print(SYSTEM_PROMPT_PATTERN)
    print()

    print("7. Tool Injection Pattern")
    print("-" * 50)
    print(TOOL_INJECTION_PATTERN)
    print()

    print("8. Runtime Access Pattern")
    print("-" * 50)
    print(RUNTIME_ACCESS_PATTERN)
    print()

    print("9. Usage Example (Commented)")
    print("-" * 50)
    print("# custom_middleware = LocalContextMiddleware()")
    print("# agent = create_deep_agent(")
    print("#     model=model,")
    print("#     middleware=[custom_middleware],  # Add to stack")
    print("# )")
    print("#")
    print("# result = await agent.ainvoke({")
    print("#     'messages': [HumanMessage(content='Show project')]")
    print("# })")
    print()
    print("Result:")
    print("  - Middleware gathers context once (cached)")
    print("  - Injects <project_context> into system prompt each turn")
    print("  - LLM sees enriched prompt with project awareness")
    print()

    print("\n=== Key Takeaways ===")
    print("- Middleware intercepts LLM calls for state/prompt enrichment")
    print("- Stack order matters: early middleware provides context for later")
    print("- State schema extension: add fields with PrivateStateAttr, reducers")
    print("- System prompt injection: XML tags for modular context blocks")
    print("- Tool injection: middleware adds tools dynamically")
    print("- Runtime access: backend (terminal), store (persistence), config")
    print("- PatchToolCallsMiddleware: fixes dangling tool calls at stream end")
    print("- Custom middleware runs LAST (after built-in stack)")


if __name__ == "__main__":
    asyncio.run(main())
