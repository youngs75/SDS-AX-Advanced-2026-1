"""
Non-Interactive Execution for Scripting and CI/CD Integration

Demonstrates:
- One-shot CLI execution (deepagents -n "task")
- Auto-approve mode for unattended operation
- Shell allow-list security configuration
- Streaming output collection from agent.astream()
- Session persistence with AsyncSqliteSaver
- Thread resume capability
- CLI backend with per-command timeout
- Exit code propagation patterns
"""

import asyncio
from typing import AsyncIterator, Tuple, Any, List, Optional
# from deepagents import create_deep_agent
# from deepagents.backends import CLIShellBackend
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# --- Model Configuration ---
# model = ChatOpenAI(model="gpt-4", temperature=0)
model = FakeListChatModel(responses=["Analysis complete", "Tests passed"])


# ==== One-Shot Execution Pattern ====

async def one_shot_execution(task: str) -> dict:
    """
    Pattern: Non-interactive single task execution.

    Key behaviors from deepagents_code/libs/cli/non_interactive.py:
    - Agent runs to completion without human interaction
    - Max 50 HITL iterations safety cap
    - Returns final result or error
    - Exit code 0 on success, 1 on failure
    """
    print(f"\n[One-Shot Mode] Executing task: {task}")

    # agent = create_deep_agent(
    #     llm=model,
    #     agent_name="scripting-agent",
    #     auto_approve=True,  # No human interaction
    #     max_iterations=50   # Safety limit
    # )

    # Simulated execution
    iteration_count = 0
    max_iterations = 50

    print(f"  → Running until completion (max {max_iterations} iterations)")
    print(f"  → auto_approve=True (all HITL interrupts approved)")

    # Simulate some iterations
    for i in range(3):
        iteration_count += 1
        print(f"  → Iteration {iteration_count}: Processing...")
        await asyncio.sleep(0.1)

    result = {
        "status": "completed",
        "iterations": iteration_count,
        "output": "Task finished successfully",
        "exit_code": 0
    }

    print(f"  ✓ Completed in {iteration_count} iterations")
    print(f"  ✓ Exit code: {result['exit_code']}")

    return result


# ==== Auto-Approve vs Shell Allow-List ====

def demonstrate_security_modes():
    """
    Pattern: Shell command execution security configurations.

    Key behaviors from deepagents_code/libs/cli/:
    - auto_approve=True → all HITL interrupts approved (dangerous)
    - shell_allow_list=None + auto_approve=True → any command allowed
    - shell_allow_list=["git", "npm"] → only these commands allowed
    - Command not in list → interrupt for human approval
    """
    print("\n[Security Modes] Command execution configurations:")

    # Mode 1: Full auto-approve (most dangerous)
    config_dangerous = {
        "auto_approve": True,
        "shell_allow_list": None  # No restrictions
    }
    print("\n  ⚠️  DANGEROUS MODE:")
    print(f"     {config_dangerous}")
    print("     → Agent can execute ANY shell command without approval")
    print("     → Use only in fully sandboxed environments")

    # Mode 2: Allow-list with auto-approve
    config_restricted = {
        "auto_approve": True,
        "shell_allow_list": ["git", "npm", "pytest"]
    }
    print("\n  ✓ RESTRICTED MODE:")
    print(f"     {config_restricted}")
    print("     → Only git/npm/pytest commands allowed")
    print("     → Other commands rejected without interrupt")

    # Mode 3: Allow-list without auto-approve
    config_interactive = {
        "auto_approve": False,
        "shell_allow_list": ["git", "npm", "pytest"]
    }
    print("\n  ✓ INTERACTIVE MODE:")
    print(f"     {config_interactive}")
    print("     → Allow-list commands auto-approved")
    print("     → Other commands trigger human approval interrupt")

    # Validation example (Pydantic in real implementation)
    def validate_command(command: str, allow_list: Optional[List[str]]) -> bool:
        if allow_list is None:
            return True  # No restrictions
        command_name = command.split()[0]
        return command_name in allow_list

    print("\n  Command Validation Examples:")
    test_commands = [
        "git status",
        "npm test",
        "rm -rf /",  # Dangerous
        "pytest tests/"
    ]

    for cmd in test_commands:
        allowed = validate_command(cmd, config_restricted["shell_allow_list"])
        status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
        print(f"     {status}: {cmd}")


# ==== Streaming Output Collection ====

async def streaming_output_collection(task: str) -> str:
    """
    Pattern: Collect output from agent.astream() for logging or display.

    Key behaviors from deepagents_code/libs/cli/non_interactive.py:
    - Stream chunk format: (namespace, stream_mode, data) — length 3 tuple
    - Filter main-agent events only (ignore subagent output)
    - AIMessage → text output
    - ToolCall → track for approval (if gated)
    - End event → collect final response
    """
    print(f"\n[Streaming Collection] Task: {task}")

    # agent = create_deep_agent(llm=model, agent_name="stream-agent")
    # config = {"configurable": {"thread_id": "stream-001"}}

    # Simulated stream events
    async def mock_astream() -> AsyncIterator[Tuple[str, str, Any]]:
        """Simulate agent.astream() output format."""
        events = [
            ("agent", "messages", AIMessage(content="Analyzing task...")),
            ("agent", "messages", AIMessage(content="Running tests...")),
            ("subagent", "messages", AIMessage(content="[subagent output]")),  # Filtered
            ("agent", "tool_calls", {"name": "shell", "args": {"command": "pytest"}}),
            ("agent", "messages", AIMessage(content="Tests passed!")),
            ("agent", "end", {"status": "completed"})
        ]
        for event in events:
            yield event
            await asyncio.sleep(0.1)

    collected_output = []
    tool_calls = []
    final_result = None

    print("  → Streaming events:")

    async for chunk in mock_astream():
        # Validate chunk format (length 3 tuple)
        if len(chunk) != 3:
            print(f"    ⚠️  Invalid chunk format: {chunk}")
            continue

        namespace, stream_mode, data = chunk

        # Filter main agent events only
        if namespace != "agent":
            print(f"    ⤷ Skipping {namespace} event")
            continue

        # Process by stream mode
        if stream_mode == "messages" and isinstance(data, AIMessage):
            text = data.content
            collected_output.append(text)
            print(f"    → {text}")

        elif stream_mode == "tool_calls":
            tool_calls.append(data)
            print(f"    → Tool call: {data['name']}")

        elif stream_mode == "end":
            final_result = data
            print(f"    ✓ End: {data['status']}")

    print(f"\n  Collected {len(collected_output)} messages, {len(tool_calls)} tool calls")

    return "\n".join(collected_output)


# ==== Session Persistence Pattern ====

async def session_persistence_demo():
    """
    Pattern: SQLite-based session storage for thread resume.

    Key behaviors from deepagents_code/libs/cli/sessions.py:
    - AsyncSqliteSaver: ~/.deepagents/sessions.db
    - Thread resume: deepagents -r [THREAD_ID]
    - Most recent thread: deepagents -r (no ID)
    - Thread listing: deepagents threads list
    - Metadata stored as JSON: agent_name, updated_at
    """
    print("\n[Session Persistence] Thread management:")

    # Setup checkpointer (real implementation)
    # db_path = Path.home() / ".deepagents" / "sessions.db"
    # checkpointer = AsyncSqliteSaver.from_conn_string(str(db_path))

    # agent = create_deep_agent(
    #     llm=model,
    #     agent_name="persistent-agent",
    #     checkpointer=checkpointer
    # )

    print("  → Database: ~/.deepagents/sessions.db")

    # Thread creation
    thread_id = "thread-abc-123"
    print(f"\n  Creating thread: {thread_id}")

    # Simulated metadata
    thread_metadata = {
        "agent_name": "persistent-agent",
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:35:00Z",
        "task": "Fix TypeScript errors"
    }

    print(f"  → Metadata: {thread_metadata}")

    # Resume scenarios
    print("\n  Resume Scenarios:")
    print(f"    deepagents -r {thread_id}  → Resume specific thread")
    print(f"    deepagents -r              → Resume most recent thread")
    print(f"    deepagents threads list    → List all threads")

    # Thread listing simulation
    print("\n  Available threads:")
    threads = [
        {"id": "thread-abc-123", "task": "Fix TypeScript errors", "updated": "5 min ago"},
        {"id": "thread-def-456", "task": "Add auth module", "updated": "2 hours ago"},
        {"id": "thread-ghi-789", "task": "Refactor API", "updated": "1 day ago"}
    ]

    for t in threads:
        print(f"    [{t['id']}] {t['task']} (updated {t['updated']})")

    print("\n  ✓ Session state persisted between runs")


# ==== CLI Backend Timeout Pattern ====

async def cli_backend_timeout_demo():
    """
    Pattern: Per-command timeout for shell execution.

    Key behaviors from deepagents_code/libs/cli/backends.py:
    - CLIShellBackend extends LocalShellBackend
    - execute(command, *, timeout=None) → subprocess with timeout
    - Monkey-patches FilesystemMiddleware at import time
    - Timeout prevents hanging on long-running commands
    """
    print("\n[CLI Backend] Per-command timeout pattern:")

    # backend = CLIShellBackend(timeout=30)  # Default 30s timeout

    commands = [
        {"cmd": "pytest tests/", "timeout": 60, "expected": "2-3 seconds"},
        {"cmd": "npm install", "timeout": 300, "expected": "30-60 seconds"},
        {"cmd": "git status", "timeout": 5, "expected": "<1 second"},
        {"cmd": "sleep 1000", "timeout": 10, "expected": "TIMEOUT"}
    ]

    print("  Command execution with timeouts:")

    for spec in commands:
        print(f"\n    Command: {spec['cmd']}")
        print(f"    Timeout: {spec['timeout']}s")
        print(f"    Expected: {spec['expected']}")

        # Simulated execution
        # result = await backend.execute(spec['cmd'], timeout=spec['timeout'])

        if spec['expected'] == "TIMEOUT":
            print(f"    ✗ Killed after {spec['timeout']}s timeout")
        else:
            print(f"    ✓ Completed in {spec['expected']}")

    print("\n  ✓ Timeouts prevent hanging on long commands")


# ==== Exit Code Propagation ====

def exit_code_patterns():
    """
    Pattern: Exit codes for CI/CD integration.

    Key behaviors from deepagents_code/libs/cli/:
    - Agent success → exit 0
    - Agent error → exit 1
    - Keyboard interrupt → exit 130
    - Exit codes used by CI/CD pipelines
    """
    print("\n[Exit Codes] CI/CD integration patterns:")

    scenarios = [
        {"result": "success", "exit_code": 0, "description": "Task completed successfully"},
        {"result": "error", "exit_code": 1, "description": "Agent encountered error"},
        {"result": "interrupted", "exit_code": 130, "description": "User keyboard interrupt"},
        {"result": "timeout", "exit_code": 1, "description": "Max iterations exceeded"}
    ]

    print("\n  Exit Code Mapping:")
    for s in scenarios:
        print(f"    {s['result']:12} → exit {s['exit_code']:3} | {s['description']}")

    print("\n  CI/CD Usage:")
    print("    #!/bin/bash")
    print("    deepagents -n \"Run all tests\"")
    print("    if [ $? -eq 0 ]; then")
    print("      echo \"Tests passed\"")
    print("      deploy_to_production")
    print("    else")
    print("      echo \"Tests failed\"")
    print("      exit 1")
    print("    fi")


# ==== Main Demo ====

async def main():
    """Demonstrate non-interactive execution patterns."""

    print("=" * 70)
    print("NON-INTERACTIVE EXECUTION FOR SCRIPTING AND CI/CD")
    print("=" * 70)

    # Pattern 1: One-shot execution
    result = await one_shot_execution("Fix all TypeScript errors")

    # Pattern 2: Security modes
    demonstrate_security_modes()

    # Pattern 3: Streaming output
    output = await streaming_output_collection("Run integration tests")

    # Pattern 4: Session persistence
    await session_persistence_demo()

    # Pattern 5: CLI backend timeout
    await cli_backend_timeout_demo()

    # Pattern 6: Exit codes
    exit_code_patterns()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. One-Shot Mode: -n flag runs task to completion, no interaction
    2. Auto-Approve: auto_approve=True skips all HITL interrupts
    3. Shell Allow-List: Restrict commands for security (git, npm, pytest)
    4. Streaming: agent.astream() yields (namespace, mode, data) tuples
    5. Persistence: AsyncSqliteSaver enables thread resume (-r flag)
    6. Timeout: CLIShellBackend.execute(cmd, timeout=N) prevents hangs
    7. Exit Codes: 0=success, 1=error, 130=interrupt (CI/CD friendly)

    CLI Examples:
    - deepagents -n "Fix errors"              → one-shot execution
    - deepagents -n "Test all" --auto-approve → no interrupts
    - deepagents --shell-allow-list "git,npm" → restrict commands
    - deepagents -r thread-123                → resume thread
    - deepagents threads list                 → show all threads
    """)


if __name__ == "__main__":
    asyncio.run(main())
