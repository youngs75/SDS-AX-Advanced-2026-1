"""
Skills: progressive disclosure with SKILL.md on-demand loading.

Demonstrates:
- SKILL.md structure: frontmatter + body
- Skill vs Tool distinction
- Progressive disclosure: load on demand, not all at once
- 3 loading backends: StateBackend, StoreBackend, FilesystemBackend
- Skill inheritance to sub-agents
- Custom skill creation pattern
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
model = FakeListChatModel(
    responses=[
        "I'll load the Python best practices skill to help with this task.",
        "Based on the loaded skill, here are the recommendations...",
    ]
)


# ==== SKILL.md Structure ====

# A skill is a SKILL.md file with:
# 1. Frontmatter: name, description (for discovery)
# 2. Body: instructions, references, asset paths (for execution)

example_skill_md = """---
name: python-best-practices
description: Python coding standards, type hints, async patterns, and testing conventions.
---

# Python Best Practices

## Rules

1. Use type hints for all function signatures.
2. Prefer `async def` for I/O-bound operations.
3. Use `pytest` for testing with `pytest-asyncio` for async tests.

## Reference Documents

- `references/typing-guide.md`
- `references/async-patterns.md`

## Asset Templates

- `assets/test-template.py` — pytest boilerplate
- `assets/async-service.py` — async service pattern
"""


# ==== Skill vs Tool ====

skill_vs_tool = {
    "Skill": {
        "what": "Context bundle (SKILL.md) loaded into agent's system prompt",
        "when": "Domain knowledge, coding standards, workflow guides",
        "how": "Progressive disclosure: frontmatter first, body on demand",
        "cost": "Token cost = loaded content size",
        "example": "python-best-practices, react-patterns, api-design",
    },
    "Tool": {
        "what": "Executable function (@tool) the agent can call",
        "when": "Actions: search, file ops, API calls, computations",
        "how": "Always available once registered",
        "cost": "Token cost = schema only (small)",
        "example": "web_search, write_file, analyze_data",
    },
}


# ==== Loading Backends ====

loading_backends = {
    "StateBackend": {
        "storage": "Virtual filesystem in graph state",
        "persistence": "Ephemeral (thread-scoped)",
        "use_case": "Skills bundled with agent, no disk needed",
    },
    "StoreBackend": {
        "storage": "Cross-thread key-value store",
        "persistence": "Permanent (cross-thread)",
        "use_case": "Shared skills across agents/sessions",
    },
    "FilesystemBackend": {
        "storage": "Local disk files",
        "persistence": "Permanent (disk-based)",
        "use_case": "Project-specific skills in repository",
    },
}


# ==== Custom Skill Creation ====

custom_skill_template = """---
name: {name}
description: {description}
---

# {title}

## Rules

1. First rule...
2. Second rule...

## Reference Documents

- `references/doc1.md`

## Asset Templates

- `assets/template.py`
"""


# ==== Agent Setup ====

# Skills are passed as list of paths or skill objects
# agent = create_deep_agent(
#     model=model,
#     skills=[
#         "skills/python-best-practices/SKILL.md",
#         "skills/api-design/SKILL.md",
#     ],
#     checkpointer=InMemorySaver(),
# )

# Skills inherit to sub-agents automatically:
# agent = create_deep_agent(
#     model=model,
#     skills=["skills/python-best-practices/SKILL.md"],
#     subagents=[{
#         "name": "coder",
#         "description": "Writes Python code",
#         # Skills from parent are automatically available
#         # "skills": ["additional/skill.md"],  # Add extra skills
#     }],
# )


# ==== Main ====


async def main():
    print("=== Progressive Disclosure Skills Pattern ===")
    print()

    # SKILL.md structure
    print("--- SKILL.md Structure ---")
    print(example_skill_md)

    # Skill vs Tool
    print("--- Skill vs Tool ---")
    for category, details in skill_vs_tool.items():
        print(f"  {category}:")
        for key, value in details.items():
            print(f"    {key:10s}: {value}")
        print()

    # Progressive disclosure flow
    print("--- Progressive Disclosure Flow ---")
    print("  1. Discovery:  Agent sees skill name + description (frontmatter)")
    print("  2. Decision:   Agent decides if skill is relevant to current task")
    print("  3. Loading:    Agent loads full SKILL.md body into context")
    print("  4. Reference:  Agent follows links to reference docs as needed")
    print("  5. Assets:     Agent uses asset templates for code generation")
    print("  Token savings: Only pay for skills actually used!")
    print()

    # Loading backends
    print("--- Loading Backends ---")
    for name, info in loading_backends.items():
        print(f"  {name}:")
        for key, value in info.items():
            print(f"    {key:15s}: {value}")
        print()

    # Inheritance
    print("--- Skill Inheritance ---")
    print("  Parent skills automatically available to all sub-agents")
    print("  Sub-agents can add extra skills (additive only)")
    print("  No way to remove parent skills from sub-agent")
    print()

    # Custom skill creation
    print("--- Custom Skill Template ---")
    filled = custom_skill_template.format(
        name="my-domain-skill",
        description="Domain-specific guidance for my project",
        title="My Domain Skill",
    )
    print(filled)


if __name__ == "__main__":
    asyncio.run(main())
