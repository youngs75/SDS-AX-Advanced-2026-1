# LangChain Core Primitives

## Read This When

- You need to decide what belongs in low-level reusable contracts.
- You are designing shared abstractions used by multiple agents.
- You need to understand the boundary between core contracts and framework conveniences.

## Skip This When

- The task is mainly app-level agent wiring or runtime orchestration.

## Official References

1. https://docs.langchain.com/oss/python/concepts/products
   - Why: confirms layer boundaries.
2. https://reference.langchain.com/python/langchain_core/
   - Why: canonical low-level API surface.

## Core Guidance

1. **`langchain_core` is the reusable contract layer** — provider-agnostic, framework-agnostic.
   - Core defines interfaces and protocols used by all layers above.
   - No business logic, no provider lock-in, no application-specific policy.

2. **Modules in langchain_core**:

| Module | Contains | Example |
|--------|----------|---------|
| `messages` | Message types and content blocks | `HumanMessage`, `AIMessage`, `ToolMessage` |
| `runnables` | Execution interface and composition | `Runnable`, `RunnableConfig`, `RunnableLambda` |
| `tools` | Tool base class and decorators | `BaseTool`, `@tool`, `ToolCall` |
| `language_models` | Model abstractions | `BaseChatModel`, `FakeListChatModel` |
| `callbacks` | Lifecycle hooks | `CallbackHandler`, tracing |
| `output_parsers` | Output transformation | `StrOutputParser`, `JsonOutputParser` |
| `embeddings` | Embedding abstractions | `Embeddings` base class |
| `rate_limiters` | Rate control | `InMemoryRateLimiter` |

3. **Layer boundary rules**:
   - **Core** (`langchain_core`): reusable contracts, no business logic, no provider lock-in
   - **Framework** (`langchain`): `create_agent`, middleware, tools, provider packages
   - **Runtime** (`langgraph`): graphs, persistence, interrupts, streaming
   - **Harness** (`deepagents`): planning, delegation, context engineering

4. **When to import from core directly**:

```python
# Import from langchain_core when:
# - Building reusable primitives/libraries
# - Need types without full framework dependency
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import FakeListChatModel

# Import from langchain when:
# - Building application code
# - Need framework-level convenience
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
```

5. **Keep product/business routing policy out of this layer.**
   - Core should be usable by any agent system, not just yours.
   - Product-specific routing, prompt templates, and business logic belong in framework or harness layers.

## Asset Examples

| Module | Asset Path |
|--------|-----------|
| Prompt engineering (XML sections, template variables, few-shot) | `assets/01-langchain-core/prompt-patterns/` |
| Messages (5 types, content blocks, multimodal, tool_call_id) | `assets/01-langchain-core/message-patterns/` |
| Tools (@tool, BaseTool, StructuredTool, InjectedToolArg) | `assets/01-langchain-core/tool-definition/` |
| Runnables (protocol, LCEL pipe, RunnableLambda, RunnableParallel) | `assets/01-langchain-core/runnable-patterns/` |

## Quick Checklist

- [ ] Is this reusable across multiple agents? → core candidate
- [ ] Does this encode business logic? → move to framework layer
- [ ] Does this lock to one provider? → move to framework layer
- [ ] Are core imports used for library code, langchain imports for app code?

## Next File

- Message + tool contracts: `20-message-tool-schema.md`
