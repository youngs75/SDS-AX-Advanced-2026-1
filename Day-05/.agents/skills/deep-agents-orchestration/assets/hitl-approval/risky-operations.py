"""
Human-in-the-loop: interrupt-based approval for risky operations.

Demonstrates:
- interrupt_on configuration: tool name → True
- 3 decision types: approve, edit, reject
- Command(resume={"decisions": [...]}) pattern
- Batch interrupt handling
- Sub-agent interrupt propagation
- Checkpointer requirement
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.types import Command

# --- Model Configuration ---
model = FakeListChatModel(responses=[
    "I need to modify the config file. Let me request approval first.",
    "File updated successfully after approval.",
])


# ==== Risky Tools (require approval) ====

@tool
async def write_file(path: str, content: str) -> str:
    """Write content to a file. RISKY: modifies filesystem."""
    # Production: actual file write
    return f"Written {len(content)} bytes to {path}"

@tool
async def execute_command(command: str) -> str:
    """Execute a shell command. RISKY: system access."""
    # Production: subprocess execution
    return f"Executed: {command}"

@tool
async def delete_file(path: str) -> str:
    """Delete a file. RISKY: irreversible."""
    return f"Deleted: {path}"


# ==== Safe Tools (no approval needed) ====

@tool
async def read_file(path: str) -> str:
    """Read file contents. SAFE: read-only."""
    return f"Contents of {path}: ..."

@tool
async def list_files(directory: str) -> str:
    """List files in directory. SAFE: read-only."""
    return f"Files in {directory}: [main.py, config.yaml, README.md]"


# ==== Interrupt Configuration ====

# Map tool names to interrupt behavior
interrupt_config = {
    "write_file": True,       # Always require approval
    "execute_command": True,   # Always require approval
    "delete_file": True,       # Always require approval
    # read_file: not listed → no interruption
    # list_files: not listed → no interruption
}


# ==== Agent Setup ====

# agent = create_deep_agent(
#     model=model,
#     tools=[write_file, execute_command, delete_file, read_file, list_files],
#     interrupt_on=interrupt_config,
#     checkpointer=InMemorySaver(),  # REQUIRED for interrupts
# )


# ==== Decision Types ====

decision_types = {
    "approve": {
        "description": "Allow the tool call to proceed as-is",
        "resume_value": {"type": "approve"},
        "example": 'Command(resume={"decisions": [{"type": "approve"}]})',
    },
    "edit": {
        "description": "Modify tool arguments before proceeding",
        "resume_value": {"type": "edit", "args": {"path": "/safe/path.txt"}},
        "example": 'Command(resume={"decisions": [{"type": "edit", "args": {"path": "/safe/path.txt"}}]})',
    },
    "reject": {
        "description": "Deny the tool call and return error to agent",
        "resume_value": {"type": "reject", "reason": "Operation not permitted"},
        "example": 'Command(resume={"decisions": [{"type": "reject", "reason": "Not permitted"}]})',
    },
}


# ==== HITL Workflow Simulation ====

async def simulate_hitl_workflow():
    """Simulate a human-in-the-loop approval workflow."""

    print("--- Step 1: Agent requests risky operation ---")
    pending_call = {
        "tool": "write_file",
        "args": {"path": "/etc/config.yaml", "content": "new_setting: true"},
    }
    print(f"  Tool: {pending_call['tool']}")
    print(f"  Args: {pending_call['args']}")
    print(f"  Status: INTERRUPTED (waiting for human decision)")
    print()

    print("--- Step 2: Human reviews and decides ---")
    # Scenario A: Approve
    print("  Option A — Approve:")
    print(f"    {decision_types['approve']['example']}")
    print()

    # Scenario B: Edit (change path)
    print("  Option B — Edit (change path to safe location):")
    print(f"    {decision_types['edit']['example']}")
    print()

    # Scenario C: Reject
    print("  Option C — Reject:")
    print(f"    {decision_types['reject']['example']}")
    print()

    print("--- Step 3: Agent resumes after decision ---")
    print("  Approve → tool executes with original args")
    print("  Edit    → tool executes with modified args")
    print("  Reject  → agent receives error, can try alternative")
    print()

    # Batch interrupts
    print("--- Batch Interrupts ---")
    batch_calls = [
        {"tool": "write_file", "args": {"path": "/app/config.yaml"}},
        {"tool": "delete_file", "args": {"path": "/app/old_config.yaml"}},
    ]
    print("  Multiple tool calls interrupted simultaneously:")
    for i, call in enumerate(batch_calls):
        print(f"    [{i}] {call['tool']}({call['args']})")
    print()
    print("  Resume with decisions for EACH call:")
    print('    Command(resume={"decisions": [')
    print('      {"type": "approve"},           # approve write')
    print('      {"type": "reject", "reason": "keep old config"}  # reject delete')
    print("    ]})")


# ==== Main ====

async def main():
    print("=== Human-in-the-Loop Approval Pattern ===")
    print()

    # Show interrupt configuration
    print("--- Interrupt Configuration ---")
    all_tools = [write_file, execute_command, delete_file, read_file, list_files]
    for t in all_tools:
        status = "INTERRUPT" if t.name in interrupt_config else "AUTO-APPROVE"
        risk = "RISKY" if t.name in interrupt_config else "SAFE"
        print(f"  {t.name:20s} [{risk:5s}] → {status}")
    print()

    # Show decision types
    print("--- Decision Types ---")
    for dtype, info in decision_types.items():
        print(f"  {dtype:10s}: {info['description']}")
    print()

    # Run workflow simulation
    await simulate_hitl_workflow()

    print()
    print("--- Requirements ---")
    print("  1. checkpointer is REQUIRED (state must persist across interrupt)")
    print("  2. Sub-agent interrupts propagate to parent agent")
    print("  3. Access interrupt state via state['__interrupt__']")


if __name__ == "__main__":
    asyncio.run(main())
