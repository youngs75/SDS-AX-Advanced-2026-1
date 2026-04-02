"""
Demonstrates: Full System Prompt Assembly Pipeline for create_deep_agent()

This asset shows how DeepAgents constructs the complete system prompt through a
5-layer assembly process, demonstrating the exact order and format of each layer.

Key concepts:
- 5-layer prompt assembly (base → custom → memory → skills → local context)
- AGENTS.md learning rules and memory injection patterns
- Skills metadata vs full disclosure strategy
- Local context injection (git, files, directory tree)
- Prompt optimization via cache control breakpoints
- Memory sources precedence (.global → .local)
- When agents should/shouldn't update AGENTS.md
- Ignore patterns for directory scanning
"""

import asyncio
from typing import List, Dict, Any

# from langchain_anthropic import ChatAnthropic
# from deepagents import create_deep_agent
# from deepagents.middleware import (
#     MemoryMiddleware,
#     SkillsMiddleware,
#     LocalContextMiddleware,
#     AnthropicPromptCachingMiddleware
# )

# For demonstration without actual API calls
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_models.fake import FakeListChatModel


# --- Model Configuration ---
# model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
model = FakeListChatModel(responses=[
    "I'll assemble the system prompt with all 5 layers.",
    "System prompt assembled successfully with memory, skills, and local context."
])


# ==== Layer 1: Base Instructions ====
BASE_INSTRUCTIONS = """
You are a professional coding assistant powered by Claude.

Core capabilities:
- Read, edit, and write files in the codebase
- Execute bash commands to run tests, builds, and scripts
- Search for files and patterns using glob and grep
- Browse web content for documentation and research
- Invoke specialized skills for complex workflows

Working principles:
- Always verify changes with fresh test/build output
- Keep diffs minimal and focused on the requested change
- Use appropriate tools for each task (prefer Edit over Write for existing files)
- Communicate clearly about what you're doing and why
"""


# ==== Layer 2: Custom Prompt ====
CUSTOM_PROMPT = """
You are a TypeScript expert specializing in React and Node.js development.

When implementing features:
- Use TypeScript strict mode
- Prefer functional components with hooks
- Write comprehensive JSDoc comments
- Follow ESLint and Prettier conventions
- Add unit tests for new functionality
"""


# ==== Layer 3: Memory (AGENTS.md) ====
AGENTS_MD_EXAMPLE = """
# Agent Memory for My Project

## Project Overview
This is a full-stack TypeScript application with React frontend and Express backend.
Uses PostgreSQL for persistence, Redis for caching.

## Architecture Decisions
- **Date**: 2024-01-15
  **Decision**: Use Prisma ORM instead of TypeORM
  **Reason**: Better TypeScript support and migration tooling
  **Impact**: All database models use Prisma schema

- **Date**: 2024-01-20
  **Decision**: Implement JWT-based authentication
  **Reason**: Stateless auth for easier horizontal scaling
  **Impact**: All API routes check Authorization header

## User Preferences
- Prefers async/await over .then() chains
- Likes explicit error handling over implicit
- Wants tests co-located with source files (*.test.ts)

## Common Patterns
- API error responses: `{ error: string, code: string, details?: any }`
- Database transactions: Always use Prisma.$transaction for multi-step operations
- Logging: Use Winston logger, never console.log in production code

## Known Issues
- Redis connection occasionally drops in development (harmless, auto-reconnects)
- Prisma migrations require manual review before production deploy

## Tool Usage Learnings
- When running tests, always use `npm test -- --coverage` for coverage reports
- Before database changes, run `npx prisma migrate dev --create-only` to review SQL
"""

MEMORY_LEARNING_RULES = """
AGENTS.md Learning Rules (from deepagents/middleware/memory.py):

**When to Update AGENTS.md:**
✓ User explicitly asks to remember something
✓ User describes their role, preferences, or workflows
✓ User provides feedback on agent's approach
✓ Establishing context for tool usage patterns
✓ Recording architectural decisions or constraints

**When NOT to Update:**
✗ Transient information (current task progress, temporary state)
✗ One-time requests without lasting relevance
✗ Simple Q&A without preference indication
✗ Casual conversation or small talk
✗ Information already captured in code/docs

**What to NEVER Store:**
⚠️ API keys, passwords, credentials, tokens
⚠️ Sensitive personal information
⚠️ Proprietary business data

**How Agent Updates Memory:**
1. Agent uses `edit_file` tool to modify AGENTS.md
2. Updates are appended or inserted in appropriate sections
3. Next session automatically loads the updated content
4. No manual user intervention required

**Memory Sources Precedence:**
1. ~/.deepagents/AGENTS.md (global, all projects)
2. ./.deepagents/AGENTS.md (local, this project)
Later sources override earlier ones if conflicting.
"""


# ==== Layer 4: Skills (Metadata vs Full Disclosure) ====
SKILLS_METADATA_EXAMPLE = """
Available Skills (Level 1: Metadata Only - Always in Prompt):

1. **autopilot**
   Full autonomous execution from idea to working code with planning and verification.

2. **ralph**
   Persistence mode that continues working until all tasks verified complete.

3. **ultrawork**
   Maximum parallel execution with multiple concurrent agents.

4. **plan**
   Interactive planning session with requirement gathering interview.

5. **tdd**
   Test-driven development workflow enforcing test-first implementation.

(~100 words total for all skill metadata)
"""

SKILL_FULL_DISCLOSURE_EXAMPLE = """
Level 2: Full SKILL.md Body - Loaded When Skill Triggers:

# autopilot Skill

## Purpose
Autonomous execution from high-level idea to working, tested code.

## Workflow
1. **Analyze Request**: Understand user's goal and constraints
2. **Create Plan**: Generate detailed implementation plan with verification steps
3. **Execute Plan**: Implement each step with continuous verification
4. **Test & Verify**: Run tests, check builds, validate functionality
5. **Iterate**: Fix issues until all checks pass
6. **Present Results**: Summary with evidence of completion

## When to Trigger
- User says "autopilot: <task>"
- Request starts with "build me" or "I want a"
- User wants hands-off completion

## Agent Behavior
- Spawns planner → executor → qa-tester pipeline
- Runs in background, updates user on progress
- Never stops until architect verification passes
- Automatically handles common errors and retries

## Configuration
- Max iterations: 10
- Verification required: yes
- Auto-commit: optional (ask user)

(<5k words for full skill documentation)
"""

SKILL_BUNDLED_RESOURCES_EXAMPLE = """
Level 3: Bundled Resources - Loaded as Needed:

autopilot/
  SKILL.md              # Documentation (loaded when triggered)
  resources/
    planner-template.md # Planning template (loaded if agent needs it)
    checklist.json      # Verification checklist (loaded if agent needs it)
    scripts/
      verify.sh         # Verification script (can run without reading)
      cleanup.sh        # Cleanup script (can run without reading)

Resources are unlimited in size. Scripts can be executed directly without
loading content into prompt, saving tokens.
"""


# ==== Layer 5: Local Context ====
LOCAL_CONTEXT_EXAMPLE = """
<local_context>
## Git Information
Branch: feature/user-authentication
Main branch: main

## Working Directory Files (first 20)
src/
  index.ts
  server.ts
  auth/
    jwt.ts
    middleware.ts
  models/
    User.ts
    Session.ts
  routes/
    auth.routes.ts
    user.routes.ts
  utils/
    logger.ts
    errors.ts
tests/
  auth/
    jwt.test.ts
    middleware.test.ts

## Directory Tree (max 3 levels)
.
├── src/
│   ├── auth/
│   │   ├── jwt.ts
│   │   └── middleware.ts
│   ├── models/
│   │   ├── User.ts
│   │   └── Session.ts
│   └── routes/
│       ├── auth.routes.ts
│       └── user.routes.ts
├── tests/
│   └── auth/
│       ├── jwt.test.ts
│       └── middleware.test.ts
├── package.json
├── tsconfig.json
└── README.md

## Ignore Patterns
.git, node_modules, .venv, __pycache__, dist, build, .next,
.cache, coverage, *.log, .DS_Store, .env*
</local_context>
"""


# ==== Prompt Assembly Process ====
def assemble_system_prompt() -> str:
    """
    Demonstrates the exact order of prompt assembly in create_deep_agent().

    Assembly order (from deepagents/graph.py + cli/agent.py):
    1. Base Instructions (SDK default)
    2. Custom Prompt (user-provided)
    3. Memory (AGENTS.md with cache control)
    4. Skills (metadata with cache control, full body on-demand)
    5. Local Context (git + files + tree)
    """

    layers = []

    # Layer 1: Base Instructions
    layers.append("# ==== LAYER 1: BASE INSTRUCTIONS ====")
    layers.append(BASE_INSTRUCTIONS.strip())
    layers.append("")

    # Layer 2: Custom Prompt
    layers.append("# ==== LAYER 2: CUSTOM PROMPT ====")
    layers.append(CUSTOM_PROMPT.strip())
    layers.append("")

    # Layer 3: Memory (with special tags)
    layers.append("# ==== LAYER 3: MEMORY (AGENTS.md) ====")
    layers.append("<agent_memory>")
    layers.append(AGENTS_MD_EXAMPLE.strip())
    layers.append("</agent_memory>")
    layers.append("")

    # Layer 4: Skills Metadata (cache control breakpoint here)
    layers.append("# ==== LAYER 4: SKILLS (METADATA) ====")
    layers.append(SKILLS_METADATA_EXAMPLE.strip())
    layers.append("")
    layers.append("# [CACHE_CONTROL BREAKPOINT]")
    layers.append("# Static content above is cached. Dynamic content below is not.")
    layers.append("")

    # Layer 5: Local Context
    layers.append("# ==== LAYER 5: LOCAL CONTEXT ====")
    layers.append(LOCAL_CONTEXT_EXAMPLE.strip())

    return "\n".join(layers)


def show_skill_loading_strategy():
    """Demonstrates the 3-level skill loading strategy."""

    print("\n" + "="*70)
    print("SKILL LOADING STRATEGY")
    print("="*70)

    print("\n📋 Level 1: Metadata Only (ALWAYS in prompt)")
    print("-" * 70)
    print("All skills show name + short description (~10 words each)")
    print("Total: ~100 words for all available skills")
    print("Purpose: Let agent know what's available without token cost")

    print("\n📖 Level 2: Full Documentation (ON TRIGGER)")
    print("-" * 70)
    print("When skill is invoked, load complete SKILL.md body")
    print("Typical size: <5k words per skill")
    print("Purpose: Provide complete context for skill execution")

    print("\n📦 Level 3: Bundled Resources (AS NEEDED)")
    print("-" * 70)
    print("Templates, checklists, scripts loaded individually")
    print("Scripts can run without reading (save tokens)")
    print("Unlimited size - only pay for what's used")

    print("\nExample Token Cost:")
    print("  10 skills × 10 words metadata = 100 words always")
    print("  1 triggered skill × 5000 words = 5000 words on-demand")
    print("  Total: 5100 words (vs 50,000 if all skills loaded)")


def show_memory_update_examples():
    """Shows examples of when agents should/shouldn't update AGENTS.md."""

    print("\n" + "="*70)
    print("AGENTS.MD UPDATE DECISION EXAMPLES")
    print("="*70)

    examples = [
        {
            "scenario": 'User: "I prefer using Zod for validation"',
            "should_update": True,
            "reason": "User preference - lasting relevance",
            "update": "Add to ## User Preferences section"
        },
        {
            "scenario": 'User: "Fix the login bug"',
            "should_update": False,
            "reason": "One-time task - no lasting learning",
            "update": "N/A"
        },
        {
            "scenario": 'User: "Always run linter before committing"',
            "should_update": True,
            "reason": "Workflow preference - affects future tasks",
            "update": "Add to ## Common Patterns or ## User Preferences"
        },
        {
            "scenario": 'Agent: Discovers tests require Docker to run',
            "should_update": True,
            "reason": "Tool usage pattern - important context",
            "update": "Add to ## Tool Usage Learnings"
        },
        {
            "scenario": 'User: "What does this function do?"',
            "should_update": False,
            "reason": "Simple Q&A - no preference indicated",
            "update": "N/A"
        },
        {
            "scenario": 'User: "Remember my API key is abc123"',
            "should_update": False,
            "reason": "NEVER store credentials",
            "update": "N/A - Warn user about security"
        }
    ]

    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. {ex['scenario']}")
        print(f"   Update? {'✓ YES' if ex['should_update'] else '✗ NO'}")
        print(f"   Reason: {ex['reason']}")
        print(f"   Action: {ex['update']}")


def show_cache_control_strategy():
    """Shows how prompt caching optimizes token costs."""

    print("\n" + "="*70)
    print("PROMPT CACHING STRATEGY")
    print("="*70)

    print("\n🔒 CACHED (Static - rarely changes):")
    print("  - Base Instructions")
    print("  - Custom Prompt")
    print("  - Skills Metadata")
    print("  → Cache these with 'cache_control' breakpoints")

    print("\n🔄 NOT CACHED (Dynamic - changes often):")
    print("  - Memory (AGENTS.md) - updates with learning")
    print("  - Local Context - changes per session/directory")
    print("  - Skill Full Docs - only loaded when triggered")

    print("\n💰 Token Cost Savings:")
    print("  Without caching: 10k static + 2k dynamic = 12k tokens/request")
    print("  With caching:    1k cached  + 2k dynamic = 3k tokens/request")
    print("  Savings: ~75% reduction after first request")


async def main():
    """
    Demonstrates the complete prompt assembly pipeline with explanations.
    """

    print("="*70)
    print("DEEPAGENTS SYSTEM PROMPT ASSEMBLY PIPELINE")
    print("="*70)

    print("\nThis demonstrates how create_deep_agent() constructs the full")
    print("system prompt through a 5-layer assembly process.\n")

    # Show the assembled prompt
    print("\n" + "="*70)
    print("ASSEMBLED SYSTEM PROMPT")
    print("="*70)
    assembled = assemble_system_prompt()
    print(assembled)

    # Explain each layer
    print("\n" + "="*70)
    print("LAYER BREAKDOWN")
    print("="*70)

    print("\n1️⃣  BASE INSTRUCTIONS")
    print("   - Default coding agent capabilities")
    print("   - Hardcoded in DeepAgents SDK")
    print("   - Provides core tool usage patterns")

    print("\n2️⃣  CUSTOM PROMPT")
    print("   - User-provided via create_deep_agent(system_prompt='...')")
    print("   - Project-specific expertise and conventions")
    print("   - Overrides base instructions if conflicting")

    print("\n3️⃣  MEMORY (AGENTS.md)")
    print("   - Loaded from ~/.deepagents/AGENTS.md + ./.deepagents/AGENTS.md")
    print("   - Wrapped in <agent_memory> tags")
    print("   - Agent can edit with edit_file tool to learn")
    print("   - Local overrides global if sections conflict")

    print("\n4️⃣  SKILLS (Metadata)")
    print("   - All skills show name + description (~100 words total)")
    print("   - Full SKILL.md loaded only when skill triggers")
    print("   - Cache control breakpoint after this layer")

    print("\n5️⃣  LOCAL CONTEXT")
    print("   - Git branch information")
    print("   - First 20 files in working directory")
    print("   - Directory tree (max 3 levels)")
    print("   - Ignores: .git, node_modules, .venv, etc.")

    # Show skill loading strategy
    show_skill_loading_strategy()

    # Show memory update rules
    show_memory_update_examples()

    # Show caching strategy
    show_cache_control_strategy()

    # Show memory learning rules
    print("\n" + "="*70)
    print("MEMORY LEARNING RULES")
    print("="*70)
    print(MEMORY_LEARNING_RULES.strip())

    # Final summary
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. **5-Layer Assembly**: Each layer serves a specific purpose
   - Base → Custom → Memory → Skills → Local Context

2. **Memory Learning**: AGENTS.md accumulates knowledge
   - Agent edits it via edit_file tool
   - Next session loads updates automatically
   - Global + Local sources, local overrides global

3. **Smart Skill Loading**: 3-level disclosure strategy
   - Metadata always visible (~100 words)
   - Full docs on trigger (<5k words)
   - Resources as needed (unlimited)

4. **Prompt Caching**: Static parts cached, dynamic parts fresh
   - Reduces token cost by ~75% after first request
   - Cache static: base, custom, skills metadata
   - Don't cache dynamic: memory, local context

5. **Local Context**: Automatic environment awareness
   - Git branch, file listing, directory tree
   - Refreshed every session
   - Respects ignore patterns

6. **Learning Discipline**: Clear rules for when to update memory
   - DO: preferences, patterns, decisions, tool learnings
   - DON'T: transient state, one-time tasks, Q&A
   - NEVER: credentials, secrets, sensitive data
""")


if __name__ == "__main__":
    asyncio.run(main())
