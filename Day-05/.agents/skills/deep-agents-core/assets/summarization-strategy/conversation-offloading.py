"""
Conversation History Auto-Summarization and Context Offloading

Demonstrates:
- SummarizationMiddleware configuration with trigger/keep modes
- Automatic context window management at 85% threshold
- Conversation history offloading to persistent storage
- Tool argument truncation for token efficiency
- CompositeBackend routing for history files
- Model-specific trigger defaults
- Summarization flow: trigger → save → truncate → prepend context
"""

import asyncio
from datetime import datetime
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# In production, these would be real imports:
# from deepagents.core import create_deep_agent
# from deepagents.middleware.summarization import SummarizationMiddleware
# from deepagents.middleware.backend import CompositeBackend, FilesystemBackend


# --- Model Configuration ---
fake_model = FakeListChatModel(
    responses=[
        "Context window approaching limit. Triggering summarization...",
        "History saved. Continuing with reduced context.",
        "Working with offloaded conversation history.",
    ]
)


# ==== Summarization Configuration ====

def show_summarization_config():
    """Demonstrate SummarizationMiddleware configuration patterns."""
    print("=" * 60)
    print("SUMMARIZATION MIDDLEWARE CONFIGURATION")
    print("=" * 60)

    print("\n1. TRIGGER MODES (when to summarize):")
    print("-" * 60)

    trigger_modes = [
        {
            "name": "Fraction (Recommended for Claude)",
            "config": ("fraction", 0.85),
            "description": "Trigger at 85% of model's max_input_tokens",
            "example": "200k context → triggers at 170k tokens",
        },
        {
            "name": "Absolute Token Count",
            "config": ("token_count", 100000),
            "description": "Trigger at exact token count",
            "example": "Any model → triggers at 100k tokens",
        },
        {
            "name": "Message Count",
            "config": ("message_count", 50),
            "description": "Trigger after N messages",
            "example": "Any model → triggers after 50 messages",
        },
    ]

    for mode in trigger_modes:
        print(f"\n{mode['name']}:")
        print(f"  Config: {mode['config']}")
        print(f"  Description: {mode['description']}")
        print(f"  Example: {mode['example']}")

    print("\n\n2. KEEP MODES (what to retain after summarization):")
    print("-" * 60)

    keep_modes = [
        {
            "name": "Fraction (Recommended)",
            "config": ("fraction", 0.10),
            "description": "Keep last 10% of messages",
            "example": "100 messages → keep last 10",
        },
        {
            "name": "Token Count",
            "config": ("token_count", 10000),
            "description": "Keep last N tokens of messages",
            "example": "Keep messages totaling 10k tokens",
        },
        {
            "name": "Message Count",
            "config": ("message_count", 10),
            "description": "Keep last N messages",
            "example": "Keep last 10 messages exactly",
        },
    ]

    for mode in keep_modes:
        print(f"\n{mode['name']}:")
        print(f"  Config: {mode['config']}")
        print(f"  Description: {mode['description']}")
        print(f"  Example: {mode['example']}")

    print("\n\n3. MODEL-SPECIFIC DEFAULTS:")
    print("-" * 60)
    defaults = {
        "Claude (Anthropic)": {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.10),
            "reason": "Large context window benefits from fraction-based triggers",
        },
        "OpenAI GPT": {
            "trigger": ("token_count", 100000),
            "keep": ("message_count", 10),
            "reason": "Smaller context, use absolute thresholds",
        },
        "Other Models": {
            "trigger": ("message_count", 50),
            "keep": ("message_count", 10),
            "reason": "Safe defaults without token counting",
        },
    }

    for model, config in defaults.items():
        print(f"\n{model}:")
        print(f"  Trigger: {config['trigger']}")
        print(f"  Keep: {config['keep']}")
        print(f"  Reason: {config['reason']}")


# ==== Summarization Flow ====

async def demonstrate_summarization_flow():
    """Show the step-by-step summarization process."""
    print("\n\n" + "=" * 60)
    print("SUMMARIZATION FLOW")
    print("=" * 60)

    # Simulate conversation state
    thread_id = "conv_2026_02_10_abc123"
    current_messages = 100
    current_tokens = 170000
    max_tokens = 200000
    trigger_threshold = int(max_tokens * 0.85)

    print(f"\nCurrent State:")
    print(f"  Thread ID: {thread_id}")
    print(f"  Messages: {current_messages}")
    print(f"  Current tokens: {current_tokens:,}")
    print(f"  Max tokens: {max_tokens:,}")
    print(f"  Trigger threshold (85%): {trigger_threshold:,}")
    print(f"  Status: THRESHOLD EXCEEDED → Triggering summarization")

    print("\n\nStep-by-Step Summarization Process:")
    print("-" * 60)

    # Step 1: Check threshold
    print("\n✓ STEP 1: Check Threshold")
    print(f"  170,000 > 170,000 (85% of 200k) → TRUE")
    print(f"  Action: Proceed with summarization")

    # Step 2: Save full history
    print("\n✓ STEP 2: Save Full Conversation History")
    history_path = f"/conversation_history/{thread_id}.md"
    print(f"  Path: {history_path}")
    print(f"  Backend: FilesystemBackend via CompositeBackend route")
    print(f"  Format: Markdown with timestamps")
    print(f"  Mode: APPEND (preserves all previous summarizations)")

    # Step 3: Keep recent messages
    print("\n✓ STEP 3: Keep Recent Messages")
    keep_fraction = 0.10
    messages_to_keep = int(current_messages * keep_fraction)
    print(f"  Keep fraction: {keep_fraction} (10%)")
    print(f"  Messages to keep: {messages_to_keep}")
    print(f"  Messages removed: {current_messages - messages_to_keep}")

    # Step 4: Prepend summary context
    print("\n✓ STEP 4: Prepend Summary System Message")
    summary_text = (
        "Previous conversation was summarized and saved. "
        f"Full history available at {history_path}. "
        "Key context: User working on auth system refactoring, "
        "completed database migration, now implementing JWT tokens."
    )
    print(f"  Content: \"{summary_text}\"")
    print(f"  Type: SystemMessage")
    print(f"  Position: Prepended to remaining messages")

    # Step 5: Continue
    print("\n✓ STEP 5: Continue with Reduced Context")
    new_token_count = int(current_tokens * 0.10) + 200  # 10% of messages + summary
    print(f"  New token count: ~{new_token_count:,}")
    print(f"  Context freed: ~{current_tokens - new_token_count:,} tokens")
    print(f"  Status: Ready for new messages")


# ==== History Storage Format ====

def show_history_storage_format():
    """Demonstrate the conversation history file format."""
    print("\n\n" + "=" * 60)
    print("CONVERSATION HISTORY STORAGE FORMAT")
    print("=" * 60)

    print("\nFile: /conversation_history/conv_2026_02_10_abc123.md")
    print("-" * 60)

    history_content = """
# Conversation History: conv_2026_02_10_abc123

## Summarization Event 1
**Timestamp:** 2026-02-10 14:23:15 UTC
**Messages Summarized:** 1-80
**Trigger:** 170k tokens (85% of 200k max)

### Messages

**[2026-02-10 14:10:00] User:**
I need to refactor the authentication system to use JWT tokens instead of session cookies.

**[2026-02-10 14:10:15] Assistant:**
I'll help you refactor to JWT. Let me analyze the current auth implementation...

**[2026-02-10 14:12:30] User:**
First, let's migrate the database schema to store refresh tokens.

**[2026-02-10 14:12:45] Assistant:**
Creating migration for refresh_tokens table...

[... 76 more messages ...]

**[2026-02-10 14:23:10] User:**
Now implement the JWT token generation logic.

---

## Summarization Event 2
**Timestamp:** 2026-02-10 15:45:22 UTC
**Messages Summarized:** 81-160
**Trigger:** 168k tokens (85% of 200k max)

### Messages

**[2026-02-10 14:23:15] Assistant:**
Implementing JWT token generation with RS256 signing...

[... continues ...]
"""

    print(history_content)

    print("\nKey Features:")
    print("-" * 60)
    print("  • APPEND-ONLY: Each summarization adds a new section")
    print("  • TIMESTAMPED: Every event and message has timestamps")
    print("  • STRUCTURED: Clear sections with metadata")
    print("  • PERSISTENT: Survives entire conversation lifetime")
    print("  • SEARCHABLE: Can grep/search for specific topics")
    print("  • AUDITABLE: Complete record of all interactions")


# ==== Tool Argument Truncation ====

def show_tool_argument_truncation():
    """Demonstrate how tool arguments are truncated to save tokens."""
    print("\n\n" + "=" * 60)
    print("TOOL ARGUMENT TRUNCATION")
    print("=" * 60)

    print("\nProblem: Tool calls with large arguments waste tokens")
    print("-" * 60)
    print("  Example: write_file(path='app.py', content=<5000 tokens>)")
    print("  Issue: Old tool call args remain in context forever")
    print("  Impact: Wastes ~5000 tokens per write_file call")

    print("\n\nSolution: Truncate old tool call arguments")
    print("-" * 60)

    print("\nBEFORE TRUNCATION:")
    before = {
        "type": "tool_call",
        "name": "write_file",
        "args": {
            "path": "/src/auth.py",
            "content": "# Auth module\nclass AuthService:\n    def __init__(self):\n        ...\n" + "    # ... 200 more lines ...\n" * 50,
        },
        "result": "File written successfully",
    }
    print(f"  Tool: {before['name']}")
    print(f"  Args size: ~5000 tokens")
    print(f"  Args preview: {before['args']['content'][:100]}...")
    print(f"  Result: {before['result']}")

    print("\n\nAFTER TRUNCATION:")
    after = {
        "type": "tool_call",
        "name": "write_file",
        "args": "[truncated]",
        "result": "File written successfully",
    }
    print(f"  Tool: {after['name']}")
    print(f"  Args: {after['args']}")
    print(f"  Result: {after['result']}")
    print(f"  Tokens saved: ~5000")

    print("\n\nConfiguration:")
    print("-" * 60)
    print("  truncate_args_settings = {")
    print("      'enabled': True,")
    print("      'keep_recent': 10,  # Keep args for last 10 tool calls")
    print("      'exempt_tools': ['read_file'],  # Never truncate these")
    print("  }")

    print("\n\nWhen Tool Args Are Truncated:")
    print("-" * 60)
    print("  1. Tool call is older than 'keep_recent' threshold")
    print("  2. Tool name not in 'exempt_tools' list")
    print("  3. Summarization is triggered")
    print("  4. Args replaced with '[truncated]' string")
    print("  5. Tool name and result preserved for context")


# ==== CompositeBackend Integration ====

def show_composite_backend_setup():
    """Show how CompositeBackend routes conversation history."""
    print("\n\n" + "=" * 60)
    print("COMPOSITEBACKEND INTEGRATION")
    print("=" * 60)

    print("\nPurpose: Route different data types to appropriate storage")
    print("-" * 60)

    setup_code = """
from deepagents.middleware.backend import CompositeBackend, FilesystemBackend

# Main backend (e.g., Postgres, Redis, etc.)
main_backend = PostgresBackend(connection_string="...")

# Filesystem backends for large/persistent data
history_dir = "/var/deepagents/conversation_history"
results_dir = "/var/deepagents/large_tool_results"

# Composite backend with path-based routing
backend = CompositeBackend(
    default=main_backend,  # Default for all paths
    routes={
        "/conversation_history/": FilesystemBackend(root_dir=history_dir),
        "/large_tool_results/": FilesystemBackend(root_dir=results_dir),
    }
)
"""

    print("\nSetup Code:")
    print("-" * 60)
    print(setup_code)

    print("\nRouting Behavior:")
    print("-" * 60)

    routes = [
        {
            "path": "/conversation_history/thread_123.md",
            "backend": "FilesystemBackend (history_dir)",
            "reason": "Matches /conversation_history/ route",
        },
        {
            "path": "/large_tool_results/output_456.json",
            "backend": "FilesystemBackend (results_dir)",
            "reason": "Matches /large_tool_results/ route",
        },
        {
            "path": "/state/thread_123",
            "backend": "PostgresBackend (default)",
            "reason": "No matching route, uses default",
        },
    ]

    for route in routes:
        print(f"\nPath: {route['path']}")
        print(f"  → Backend: {route['backend']}")
        print(f"  → Reason: {route['reason']}")

    print("\n\nBenefits:")
    print("-" * 60)
    print("  • SEPARATION: Conversation history doesn't bloat main database")
    print("  • EFFICIENCY: File I/O for large append-only data")
    print("  • SCALABILITY: History files can be archived/compressed")
    print("  • FLEXIBILITY: Different backends for different data patterns")


# ==== Middleware Stack Position ====

def show_middleware_stack():
    """Show where SummarizationMiddleware fits in the stack."""
    print("\n\n" + "=" * 60)
    print("MIDDLEWARE STACK POSITION")
    print("=" * 60)

    print("\nCorrect Order (Top to Bottom):")
    print("-" * 60)

    stack = [
        {
            "name": "SubAgentMiddleware",
            "reason": "Must run first to expand subagent calls into messages",
        },
        {
            "name": "→ SummarizationMiddleware ←",
            "reason": "Must see all messages (including subagent results)",
        },
        {
            "name": "AnthropicPromptCachingMiddleware",
            "reason": "Must run after summarization to get correct cache breakpoints",
        },
        {
            "name": "Other Middleware",
            "reason": "Order depends on specific needs",
        },
    ]

    for i, layer in enumerate(stack, 1):
        print(f"\n{i}. {layer['name']}")
        print(f"   Reason: {layer['reason']}")

    print("\n\nWhy This Order Matters:")
    print("-" * 60)
    print("  • SubAgent first: Summarization needs to see expanded subagent messages")
    print("  • Caching last: Cache breakpoints must account for summarization")
    print("  • Wrong order: Subagent results might not be summarized, or caching breaks")


# ==== Main Demonstration ====

async def main():
    """Run all demonstration patterns."""
    print("\n" + "=" * 60)
    print("DEEPAGENTS: CONVERSATION HISTORY AUTO-SUMMARIZATION")
    print("=" * 60)

    # Show configuration options
    show_summarization_config()

    # Demonstrate the flow
    await demonstrate_summarization_flow()

    # Show history format
    show_history_storage_format()

    # Show tool arg truncation
    show_tool_argument_truncation()

    # Show backend setup
    show_composite_backend_setup()

    # Show middleware stack
    show_middleware_stack()

    print("\n\n" + "=" * 60)
    print("PRODUCTION USAGE EXAMPLE")
    print("=" * 60)

    production_code = """
from deepagents.core import create_deep_agent
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.middleware.backend import CompositeBackend, FilesystemBackend

# Setup backends
main_backend = PostgresBackend(connection_string="...")
history_backend = FilesystemBackend(root_dir="/var/deepagents/history")

backend = CompositeBackend(
    default=main_backend,
    routes={"/conversation_history/": history_backend}
)

# Create agent with summarization
agent = create_deep_agent(
    model="claude-opus-4-6",
    tools=[...],
    backend=backend,
    middleware=[
        SubAgentMiddleware(...),
        SummarizationMiddleware(
            trigger=("fraction", 0.85),  # At 85% of context window
            keep=("fraction", 0.10),     # Keep last 10% of messages
            truncate_args_settings={
                "enabled": True,
                "keep_recent": 10,       # Keep args for last 10 tool calls
            }
        ),
        AnthropicPromptCachingMiddleware(...),
    ]
)

# Run with automatic summarization
result = await agent.run(
    "Refactor the entire authentication system...",
    thread_id="long_conversation_123"
)

# History automatically saved to:
# /var/deepagents/history/long_conversation_123.md
"""

    print(production_code)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("  1. Auto-summarization prevents context overflow")
    print("  2. Full history persisted to filesystem for audit trail")
    print("  3. Tool argument truncation saves significant tokens")
    print("  4. CompositeBackend enables efficient storage routing")
    print("  5. Fraction-based triggers work best for large context models")
    print("  6. Middleware ordering is critical for correct behavior")


if __name__ == "__main__":
    asyncio.run(main())
