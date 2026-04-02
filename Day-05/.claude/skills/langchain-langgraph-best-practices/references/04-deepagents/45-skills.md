# Skills

## Read This When

- Need to understand SKILL.md structure and frontmatter
- Implementing progressive disclosure for domain knowledge
- Choosing between State/Store/Filesystem backends for skills
- Designing custom skills for agent workflows
- Debugging why skills aren't loading or being used

## Skip This When

- Not using skill-based context injection
- All knowledge can be embedded in system prompt
- Working with tool-only agents (no reference documentation)

## Official References

1. https://docs.langchain.com/oss/python/deepagents/skills - Why: Skill structure, loading mechanisms, and progressive disclosure patterns.
2. https://docs.langchain.com/oss/python/deepagents/harness - Why: Skill integration with harness and backend routing.

## Core Guidance

### 1. What is a Skill?

A structured bundle of domain knowledge that loads into the agent's context on demand. Unlike tools (executable functions), skills provide reference information, guidelines, and templates.

**Key Principle**: Skills are TEXT that provides knowledge. Tools are FUNCTIONS that perform actions.

### 2. Skill vs Tool

| Aspect | Skill | Tool |
|--------|-------|------|
| Nature | Context bundle (text) | Executable function |
| When loaded | On demand (progressive disclosure) | Always available |
| Token cost | Loaded when needed | Schema always in context |
| Purpose | Domain knowledge, guidelines, templates | Actions, API calls, computations |
| Example | API design guidelines, code patterns | Execute code, call API, search |

**Use Skills For**:
- Reference documentation
- Best practices and patterns
- Decision frameworks
- Templates and examples

**Use Tools For**:
- Executing code
- Calling APIs
- File operations
- Computations

### 3. SKILL.md Structure

```markdown
---
name: my-skill
description: Brief description for LLM selection
---

# Skill Title

## Rules
1. Guideline one
2. Guideline two

## Reference Documents Depth Router
### Depth 1: Category
- `file1.md`
- `file2.md`

### Depth 2: Detailed Topics
- `detailed/topic1.md`
- `detailed/topic2.md`

## Asset Usage Map
| Asset path | Use when | How to apply | Expected output |
|-----------|----------|--------------|-----------------|
| `template.txt` | Creating new component | Copy and fill placeholders | Functional component |
| `checklist.md` | Before deployment | Review each item | Deployment readiness |
```

**Required Frontmatter**:
- `name`: Unique skill identifier
- `description`: Used by LLM to decide when to load skill

**Body Sections**:
- **Rules**: Core guidelines (always visible)
- **Reference Documents Depth Router**: Organized by specificity level
- **Asset Usage Map**: Templates and examples with usage instructions

### 4. Progressive Disclosure

Skills are NOT loaded all at once:

| Stage | What's Loaded | Token Cost |
|-------|--------------|-----------|
| Discovery | Skill name + description | Minimal (~20 tokens) |
| Initial Load | SKILL.md router only | Low (~500 tokens) |
| Depth 1 | Category-level files | Medium (~1000 tokens each) |
| Depth 2 | Detailed topic files | High (~2000+ tokens each) |

**Best Practice**:
- Keep SKILL.md router under 100 lines
- Use Depth Router to point to detailed files
- Load only one reference file at a time
- Return to router before opening another file

**Example Flow**:
```
1. Agent sees skill description → "I need API design guidance"
2. Loads SKILL.md → Sees router with categories
3. Loads Depth 1: "REST API Patterns" → Gets overview
4. Loads Depth 2: "Pagination Strategies" → Gets detailed examples
```

### 5. Three Loading Backends

| Backend | Storage | Lifetime | Use Case |
|---------|---------|----------|----------|
| StateBackend | Virtual filesystem (LangGraph state) | Single thread | Ephemeral, per-conversation skills |
| StoreBackend | LangGraph Store (permanent) | Persistent | Shared skills across threads |
| FilesystemBackend | Local disk | Session-based | Development, large skill sets |

**StateBackend Example**:
```python
# Skills stored in state["files"]
agent = create_deep_agent(
    model=model,
    backend=StateBackend(),  # Default
)
# Skills vanish when thread ends
```

**StoreBackend Example**:
```python
# Skills stored in LangGraph Store
agent = create_deep_agent(
    model=model,
    backend=StoreBackend(namespace=("skills",)),
)
# Skills persist across threads and sessions
```

**FilesystemBackend Example**:
```python
# Skills loaded from disk
agent = create_deep_agent(
    model=model,
    backend=FilesystemBackend(
        root_dir="./skills",
        virtual_mode=True,  # Mounted at /skills/ in virtual filesystem
    ),
)
```

### 6. Skill Inheritance

Parent agent skills are automatically available to sub-agents through the shared backend.

```python
# Parent agent with skills
agent = create_deep_agent(
    model=model,
    backend=StoreBackend(namespace=("skills",)),
    subagents=[
        {
            "name": "researcher",
            "description": "Research specialist",
            "system_prompt": "You can access skills from /skills/",
        }
    ],
)

# Sub-agent can read same skill files through shared backend
```

**Why This Matters**:
- No need to duplicate skill configuration in sub-agents
- Consistent knowledge across agent hierarchy
- Sub-agents can access specialized skills from parent

### 7. Custom Skill Creation

**Option 1: Composite Backend with Filesystem**
```python
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend

agent = create_deep_agent(
    model=model,
    backend=CompositeBackend(
        default=StateBackend,
        routes={
            "/skills/": FilesystemBackend(
                root_dir="./skills",
                virtual_mode=True,
            ),
        }
    ),
    system_prompt="Load skills from /skills/ when you need domain knowledge.",
)
```

**Option 2: Pre-load to State**
```python
# Load skills into state at initialization
initial_state = {
    "files": {
        "/skills/api-design/SKILL.md": skill_content,
        "/skills/api-design/rest-patterns.md": patterns_content,
    }
}

agent = create_deep_agent(
    model=model,
    backend=StateBackend(),
)

result = agent.invoke({"messages": [...], **initial_state})
```

### 8. Token-Efficient Patterns

**Pattern 1: Router-First Navigation**
```markdown
# SKILL.md (100 lines)
## Reference Documents Depth Router
- `overview.md` - Start here
- `patterns/rest.md` - REST API patterns
- `patterns/graphql.md` - GraphQL patterns

# Agent reads SKILL.md → Decides which file to load next
```

**Pattern 2: Just-In-Time Loading**
```python
# Don't load all skills upfront
system_prompt = """
Available skills:
- /skills/api-design/SKILL.md
- /skills/security/SKILL.md

Load a skill when you need domain knowledge.
"""
# Agent loads skills only when task requires them
```

**Pattern 3: Skill Summaries**
```markdown
# SKILL.md
## Quick Reference
- REST: Use for resource-oriented APIs
- GraphQL: Use for flexible query requirements
- gRPC: Use for high-performance microservices

## Full Documentation
See Reference Documents Depth Router below...
```

### 9. Skill Design Guidelines

| Guideline | Reason |
|-----------|--------|
| Keep SKILL.md under 100 lines | Minimize initial token cost |
| Use clear category names | Help agent find right file quickly |
| Include usage examples in Asset Map | Show how to apply knowledge |
| Organize by specificity (Depth 1 → 2) | Enable progressive disclosure |
| Write actionable rules | Make guidelines executable |

**Good Skill Structure**:
```
/skills/
  api-design/
    SKILL.md              # Router + core rules (100 lines)
    overview.md           # Depth 1: General patterns (500 lines)
    rest/
      pagination.md       # Depth 2: Specific technique (200 lines)
      filtering.md        # Depth 2: Specific technique (200 lines)
```

**Bad Skill Structure**:
```
/skills/
  everything.md          # 5000 lines, always loaded
```

## Quick Checklist

- [ ] Is SKILL.md frontmatter (name, description) set for discovery?
- [ ] Is progressive disclosure used (not loading everything at once)?
- [ ] Is the correct loading backend chosen (State/Store/Filesystem)?
- [ ] Are skills organized by depth (router → detail files)?
- [ ] Is token budget considered in skill design?
- [ ] Are skill descriptions clear for LLM selection?
- [ ] Is Asset Usage Map included with templates/examples?
- [ ] Do sub-agents have access to parent skills via shared backend?

## Next File

`50-context-engineering.md` - System prompts, middleware, and context optimization strategies.
