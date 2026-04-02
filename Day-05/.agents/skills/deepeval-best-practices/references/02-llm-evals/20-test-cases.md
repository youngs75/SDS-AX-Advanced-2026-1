# DeepEval Test Cases

## Read This When
- Need parameter details for `LLMTestCase`, `ConversationalTestCase`, `ArenaTestCase`, `Turn`, or `ToolCall`
- Working with multimodal inputs/outputs using `MLLMImage`
- Setting up side-by-side model comparisons with `ArenaTestCase` and `Contestant`

## Skip This When
- Need to understand how to run evaluations (`evaluate()`, `assert_test()`) → `references/02-llm-evals/10-evaluation-fundamentals.md`
- Need dataset management, golden creation, or synthetic data generation → `references/02-llm-evals/30-datasets-and-goldens.md`

---

## Overview

DeepEval provides three test case types:
- `LLMTestCase` — single-turn interaction (one input, one output)
- `ConversationalTestCase` — multi-turn conversation sequence
- `ArenaTestCase` — side-by-side comparison of multiple LLM versions

All test case types are importable from `deepeval.test_case`.

---

## LLMTestCase

Represents a single atomic interaction with an LLM application.

```python
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You're eligible for a 30 day refund at no extra cost.",
    context=["All customers are eligible for a 30 day full refund at no extra cost."],
    retrieval_context=["Only shoes can be refunded."],
    tools_called=[ToolCall(name="PolicySearch", output="30 day refund policy")],
    expected_tools=[ToolCall(name="PolicySearch")],
    token_cost=0.0023,
    completion_time=1.4,
    name="refund-query-001",
    tags=["refund", "shoes"]
)
```

### LLMTestCase Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `str` or multimodal | Yes | The user's query/input to the LLM app. Should not include prompt templates — only what the user provides. |
| `actual_output` | `str` or multimodal | Practically required | What the LLM app actually output for the given input. Marked optional in the class signature but needed by almost all metrics. |
| `expected_output` | `str` | Optional | The ideal/golden output. Does not require exact match due to non-deterministic LLM behavior. Used by metrics like `AnswerCorrectnessMetric`. |
| `context` | `List[str]` | Optional | Supplementary golden-truth data (static, not dynamically retrieved). Represents ground-truth knowledge. Used by `FaithfulnessMetric`, `HallucinationMetric`. |
| `retrieval_context` | `List[str]` | Optional | Text chunks actually retrieved by the RAG pipeline at runtime. Used by `ContextualPrecisionMetric`, `ContextualRecallMetric`, `FaithfulnessMetric`. |
| `tools_called` | `List[ToolCall]` | Optional | Tools actually invoked by the LLM agent during this interaction. Used by `ToolCorrectnessMetric`. |
| `expected_tools` | `List[ToolCall]` | Optional | Tools that ideally should have been used. Used by `ToolCorrectnessMetric`. |
| `token_cost` | `float` | Optional | Cost of the LLM interaction in tokens/dollars. Useful for custom cost-based metrics. |
| `completion_time` | `float` | Optional | Duration of the interaction in seconds. Useful for latency-based custom metrics. |
| `name` | `str` | Optional | Unique identifier for searching/filtering this test case in Confident AI. |
| `tags` | `List[str]` | Optional | Categorization labels for grouping test cases in Confident AI. |

### MCP Parameters (for MCP evaluations)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mcp_servers` | `List[MCPServer]` | Optional | MCP server instances available to the agent |
| `mcp_tools_called` | `List[MCPToolCall]` | Optional | MCP tools invoked during the interaction |
| `mcp_resources_called` | `List[MCPResourceCall]` | Optional | MCP resources read during the interaction |
| `mcp_prompts_called` | `List[MCPPromptCall]` | Optional | MCP prompts retrieved during the interaction |

### LLM Interaction Scope

`input` and `actual_output` can be scoped at different levels:
- **Agent level** — the full end-to-end process
- **RAG pipeline level** — retriever + LLM generation together
- **Individual component level** — just the retriever or just the LLM

---

## ToolCall Data Model

Represents a single tool invocation by an LLM agent.

```python
from deepeval.test_case import ToolCall

tool_call = ToolCall(
    name="WebSearch",                        # required
    description="Searches the internet",    # optional
    reasoning="Need current information",    # optional
    output={"results": ["..."]},             # optional
    input_parameters={"query": "France capital"}  # optional
)
```

### ToolCall Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the tool |
| `description` | `str` | Optional | Description of what the tool does |
| `reasoning` | `str` | Optional | The agent's reasoning for calling this tool |
| `output` | `Any` | Optional | The value returned by the tool |
| `input_parameters` | `Dict[str, Any]` | Optional | Arguments passed to the tool |

---

## MLLMImage (Multimodal Support)

Allows text inputs and outputs to include images.

```python
from deepeval.test_case import LLMTestCase, MLLMImage

# Local file
shoes = MLLMImage(url='./shoes.png', local=True)

# Online image
banner = MLLMImage(url='https://example.com/image.png')

# Base64
encoded = MLLMImage(dataBase64="iVBORw0K...", mimeType="image/png")

test_case = LLMTestCase(
    input=f"What color are these shoes? {shoes}",
    actual_output="They are red."
)
```

### MLLMImage Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | `str` | Conditionally required | File path or URL of the image. Required if `dataBase64` not provided. |
| `dataBase64` | `str` | Conditionally required | Base64-encoded image data. Required if `url` not provided. |
| `mimeType` | `str` | Required with dataBase64 | MIME type of the image (e.g., `"image/png"`) |
| `local` | `bool` | Optional | `True` for local files, `False` (default) for online images |
| `filename` | `str` | Optional | Custom filename override |

Images are converted to deepeval slugs (`[DEEPEVAL:IMAGE:uuid]`) internally. Use `convert_to_multi_modal_array()` to inspect:

```python
from deepeval.utils import convert_to_multi_modal_array
print(convert_to_multi_modal_array(test_case.input))
# ["What color are these shoes? ", [DEEPEVAL:IMAGE:abc123]]
```

---

## ConversationalTestCase

Represents a sequence of multi-turn LLM interactions (for conversational chatbots).

```python
from deepeval.test_case import ConversationalTestCase, Turn

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I want a refund."),
        Turn(role="assistant", content="I can help with that. When did you purchase?"),
        Turn(role="user", content="Last week."),
        Turn(role="assistant", content="Initiating refund process now."),
    ],
    scenario="Frustrated customer requesting a refund.",
    expected_outcome="The AI successfully initiates a refund.",
    user_description="A customer who purchased shoes that don't fit.",
    chatbot_role="A friendly customer service representative.",
    context=["All purchases within 30 days qualify for full refund."],
    name="refund-conversation-001",
    tags=["refund", "customer-service"]
)
```

### ConversationalTestCase Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `turns` | `List[Turn]` | Yes | Ordered list of conversation turns (user and assistant messages) |
| `scenario` | `str` | Optional | Description of the conversation context/situation |
| `expected_outcome` | `str` | Optional | What the conversation should ultimately achieve |
| `user_description` | `str` | Optional | Description of the user's persona/background |
| `chatbot_role` | `str` | Optional | Role definition for the chatbot; used by `RoleAdherenceMetric` |
| `context` | `List[str]` | Optional | Static supplementary ground-truth data available to the chatbot |
| `name` | `str` | Optional | Identifier for Confident AI searching/filtering |
| `tags` | `List[str]` | Optional | Categorization labels for Confident AI |

Note: Component-level evaluation does NOT apply to `ConversationalTestCase`. Multi-turn use cases should be evaluated end-to-end only.

### MCP Parameters for ConversationalTestCase

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mcp_servers` | `List[MCPServer]` | Optional | MCP server instances for the conversation |

---

## Turn Data Model

A single message within a `ConversationalTestCase`.

```python
from deepeval.test_case import Turn, ToolCall

# User turn
user_turn = Turn(role="user", content="What's my order status?")

# Assistant turn with retrieval and tool usage
assistant_turn = Turn(
    role="assistant",
    content="Your order ships tomorrow.",
    retrieval_context=["Order #123: ships 2026-02-21"],
    tools_called=[ToolCall(name="OrderLookup", output={"status": "shipping"})],
    user_id="user-456"
)
```

### Turn Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `role` | `Literal["user", "assistant"]` | Yes | Who is speaking in this turn |
| `content` | `str` or multimodal | Yes | The text content of the turn |
| `retrieval_context` | `List[str]` | Optional | RAG chunks retrieved for this specific assistant turn. Only supply when `role="assistant"`. |
| `tools_called` | `List[ToolCall]` | Optional | Tools the assistant invoked for this turn. Only supply when `role="assistant"`. |
| `user_id` | `str` | Optional | Identifier for the user in this turn (multi-user conversations) |

**MCP parameters for Turn:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mcp_tools_called` | `List[MCPToolCall]` | Optional | MCP tools called in this assistant turn |
| `mcp_resources_called` | `List[MCPResourceCall]` | Optional | MCP resources read in this assistant turn |
| `mcp_prompts_called` | `List[MCPPromptCall]` | Optional | MCP prompts retrieved in this assistant turn |

### Multimodal Turns

```python
from deepeval.test_case import ConversationalTestCase, Turn, MLLMImage

image = MLLMImage(url='./product.png', local=True)

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content=f"Is this the right product? {image}"),
        Turn(role="assistant", content="Yes, that's the correct item."),
    ],
    scenario=f"User verifying product using photo {image}",
    expected_outcome="Assistant confirms the product identity."
)
```

---

## ArenaTestCase

Compares multiple LLM versions side-by-side to determine which performs better.

```python
from deepeval.test_case import ArenaTestCase, LLMTestCase, Contestant
from deepeval.metrics import ArenaGEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import compare

test_case = ArenaTestCase(contestants=[
    Contestant(
        name="GPT-4.1",
        hyperparameters={"model": "gpt-4.1", "temperature": 0.7},
        test_case=LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris",
        ),
    ),
    Contestant(
        name="Claude-4",
        hyperparameters={"model": "claude-4"},
        test_case=LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
        ),
    ),
    Contestant(
        name="Gemini-2.5",
        hyperparameters={"model": "gemini-2.5-flash"},
        test_case=LLMTestCase(
            input="What is the capital of France?",
            actual_output="Absolutely! The capital of France is Paris.",
        ),
    ),
])

arena_metric = ArenaGEval(
    name="Conciseness",
    criteria="Choose the more concise and accurate response based on input and actual output.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
)

compare(test_cases=[test_case], metric=arena_metric)
```

### ArenaTestCase Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contestants` | `List[Contestant]` | Yes | List of 2+ LLM versions to compare. All `input` and `expected_output` values MUST match across contestants. |

### Contestant Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Human-readable identifier for this LLM version |
| `test_case` | `LLMTestCase` | Yes | The single-turn test case for this contestant |
| `hyperparameters` | `dict` | Optional | Model configuration metadata (can include `Prompt` objects) |

**Critical constraint:** All contestants must share identical `input` and `expected_output` values. DeepEval validates this automatically and raises errors on mismatch.

**Evaluation behavior:**
- Contestant names are masked (ensuring unbiased judging)
- Contestant order is randomized
- A single "winner" is selected from contestants

### Arena with Prompt Comparison

```python
from deepeval.prompt import Prompt

prompt_v1 = Prompt(alias="Prompt V1", text_template="You are concise.")
prompt_v2 = Prompt(alias="Prompt V2", text_template="You are verbose and detailed.")

test_case = ArenaTestCase(contestants=[
    Contestant(
        name="Concise Prompt",
        hyperparameters={"prompt": prompt_v1},
        test_case=LLMTestCase(input="...", actual_output="...")
    ),
    Contestant(
        name="Verbose Prompt",
        hyperparameters={"prompt": prompt_v2},
        test_case=LLMTestCase(input="...", actual_output="...")
    ),
])
```

### Arena with Images

```python
from deepeval.test_case import ArenaTestCase, LLMTestCase, Contestant, MLLMImage

shoes = MLLMImage(url='./shoes.png', local=True)

test_case = ArenaTestCase(contestants=[
    Contestant(
        name="GPT-4",
        test_case=LLMTestCase(
            input=f"What's in this image? {shoes}",
            actual_output="That's a red shoe"
        ),
    ),
    Contestant(
        name="Claude-4",
        test_case=LLMTestCase(
            input=f"What's in this image? {shoes}",
            actual_output="The image shows a pair of red shoes"
        ),
    )
])
```

ArenaTestCase currently supports **single-turn, text-based and multimodal** comparisons only. ConversationalTestCase support is planned.

---

## LLMTestCaseParams Enum

Used in metric definitions to specify which test case fields a metric evaluates:

```python
from deepeval.test_case import LLMTestCaseParams

# Available values:
LLMTestCaseParams.INPUT
LLMTestCaseParams.ACTUAL_OUTPUT
LLMTestCaseParams.EXPECTED_OUTPUT
LLMTestCaseParams.CONTEXT
LLMTestCaseParams.RETRIEVAL_CONTEXT
LLMTestCaseParams.TOOLS_CALLED
LLMTestCaseParams.EXPECTED_TOOLS
```

---

## Evaluation Approaches by Test Case Type

| Evaluation Type | LLMTestCase | ConversationalTestCase | ArenaTestCase |
|-----------------|-------------|------------------------|---------------|
| End-to-end | Yes | Yes | via `compare()` |
| Component-level | Yes | No | No |
| Standalone metric | Yes | Yes | No |
| CI/CD (assert_test) | Yes | Yes | No |

---

## Related Reference Files

- `10-evaluation-fundamentals.md` - evaluate() and assert_test() functions
- `30-datasets-and-goldens.md` - Golden and EvaluationDataset
- `40-tracing-and-observability.md` - @observe for component-level evals
- `60-mcp-and-component-evals.md` - MCP evaluation details
