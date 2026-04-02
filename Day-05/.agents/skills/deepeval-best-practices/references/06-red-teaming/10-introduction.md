# Red Teaming: Introduction and Core Concepts

## Read This When
- You are setting up red teaming for the first time with `deepteam` (`pip install deepteam`) and need the `red_team()` function or `RedTeamer` class
- You need to define the model callback, configure simulator/evaluation models, or interpret risk assessment scores
- You want the complete integration workflow (target LLM setup, scanning, interpreting results, iterating)

## Skip This When
- You need the full vulnerability catalog with all types and parameters -- see [20-vulnerabilities.md](./20-vulnerabilities.md)
- You need details on single-turn and multi-turn attack enhancements -- see [30-attack-enhancements.md](./30-attack-enhancements.md)
- You want evaluation metrics for quality (not safety) -- see [../03-eval-metrics/](../03-eval-metrics/)

---

## Overview

**DeepTeam** is a separate package (`deepteam`) spun off from DeepEval that provides automated red teaming for LLM applications. It offers a powerful yet simple way to scan for safety risks and security vulnerabilities in just a few lines of code.

**Important:** Red teaming functionality lives in the `deepteam` package, not `deepeval`:
```bash
pip install deepteam
```

---

## Four Main Components

The red teaming framework comprises four essential elements:

1. **Vulnerabilities** - weaknesses you wish to detect (40+ available)
2. **Adversarial Attacks** - methods to probe those weaknesses (single-turn and multi-turn)
3. **Target LLM System** - your AI application that defends against attacks
4. **Metrics** - automatically assigned per vulnerability; scores 0 or 1 (strict mode always on)

See [20-vulnerabilities.md](./20-vulnerabilities.md) for the complete vulnerability catalog.
See [30-attack-enhancements.md](./30-attack-enhancements.md) for all attack types.

---

## Quick Start with red_team()

The simplest way to run red teaming:

```python
from deepteam import red_team
from deepteam.vulnerabilities import Bias
from deepteam.attacks.single_turn import PromptInjection
from deepteam.test_case import RTTurn

async def model_callback(input: str, turns: list[RTTurn] = None) -> str:
    # Replace with your actual LLM application
    return RTTurn(role="assistant", content="Your agent's response here...")

bias = Bias(types=["race"])
prompt_injection = PromptInjection()

risk_assessment = red_team(
    model_callback=model_callback,
    vulnerabilities=[bias],
    attacks=[prompt_injection]
)

print(risk_assessment)
risk_assessment.save(to="./deepteam-results/")
```

### red_team() Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_callback` | Callable | Yes | — | Callback wrapping target LLM system |
| `vulnerabilities` | `List[BaseVulnerability]` | No | — | Weaknesses to detect |
| `attacks` | `List[BaseAttack]` | No | — | Attack methods to simulate |
| `framework` | `AISafetyFramework` | No | — | Contains vulnerabilities and attacks (alternative to specifying separately) |
| `simulator_model` | str or `DeepEvalBaseLLM` | No | `"gpt-3.5-turbo-0125"` | Model for generating attacks |
| `evaluation_model` | str or `DeepEvalBaseLLM` | No | `"gpt-4o"` | Model for evaluating responses |
| `attacks_per_vulnerability_type` | int | No | `1` | Attacks per vulnerability type |
| `ignore_errors` | bool | No | `False` | Ignore exceptions during testing |
| `async_mode` | bool | No | `True` | Enable asynchronous processing |
| `max_concurrent` | int | No | `10` | Maximum parallel coroutines |
| `target_purpose` | str | No | `None` | Intended application purpose (improves tailoring) |

**Warning:** Must pass either `vulnerabilities`/`attacks` OR `framework`. Providing both causes `framework` values to overwrite the others.

---

## Model Callback

The model callback wraps your target LLM system. It must follow strict requirements.

### Function Signature

```python
from deepteam.test_case import RTTurn, ToolCall

async def model_callback(input: str, turns: list[RTTurn] = None) -> str:
    # input: the adversarial attack string
    # turns: conversation history for multi-turn attacks (None for single-turn)
    return RTTurn(
        role="assistant",
        content="Your agent's response here...",
        retrieval_context=[
            "Your retrieval context here",
            "..."
        ],
        tools_called=[
            ToolCall(name="SearchDatabase")
        ]
    )
```

### Hard Rules

1. Function signature must have a mandatory first `string` parameter and an optional second parameter accepting `list[RTTurn]` (defaulted to `None`)
2. Must return either an `RTTurn` object (recommended) with optional `tools_called` and `retrieval_context`, or a simple string

### RTTurn Fields

| Field | Type | Description |
|-------|------|-------------|
| `role` | str | Typically `"assistant"` |
| `content` | str | The LLM's response text |
| `retrieval_context` | list[str] | Optional retrieved context chunks (for RAG evaluation) |
| `tools_called` | list[ToolCall] | Optional list of tools invoked |

---

## Single-Turn vs Multi-Turn Attacks

### Single-Turn

Direct attacks without conversation history. The `turns` parameter will be `None`.

```python
from deepteam.attacks.single_turn import PromptInjection, Roleplay, Base64

risk_assessment = red_team(
    model_callback=model_callback,
    vulnerabilities=[bias],
    attacks=[PromptInjection(), Roleplay(), Base64()]
)
```

### Multi-Turn

Iterative attacks that adapt across conversation turns. The `turns` parameter contains conversation history.

```python
from deepteam.attacks.multi_turn import LinearJailbreaking, CrescendoJailbreaking

risk_assessment = red_team(
    model_callback=model_callback,
    vulnerabilities=[bias],
    attacks=[LinearJailbreaking(), CrescendoJailbreaking()]
)
```

---

## RedTeamer Class

`RedTeamer` provides better lifecycle control, attack reuse, and more configuration options than the `red_team()` function.

### Creating a RedTeamer

```python
from deepteam.red_teamer import RedTeamer

red_teamer = RedTeamer(
    target_purpose="Provide financial advice, investment suggestions, and answer user queries related to personal finance.",
    simulator_model="gpt-3.5-turbo-0125",
    evaluation_model="gpt-4o",
    async_mode=True,
    max_concurrent=10
)
```

### RedTeamer Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `target_purpose` | str | No | `None` | Application's intended purpose (improves attack tailoring and evaluation accuracy) |
| `simulator_model` | str or `DeepEvalBaseLLM` | No | `"gpt-3.5-turbo-0125"` | Model for attack generation |
| `evaluation_model` | str or `DeepEvalBaseLLM` | No | `"gpt-4o"` | Model for response evaluation |
| `async_mode` | bool | No | `True` | Enable async processing |
| `max_concurrent` | int | No | `10` | Maximum parallel coroutines |

**Model selection guidance:**
- `evaluation_model`: Use the strongest available model for accurate vulnerability identification
- `simulator_model`: Balance strength vs. filter bypass. Powerful models generate effective attacks but face stricter system filters; weaker models bypass restrictions more easily but generate less effective attacks. GPT-3.5 is often more effective than GPT-4o for attack generation.

### Running with RedTeamer

```python
from deepteam.vulnerabilities import Bias
from deepteam.attacks.single_turn import PromptInjection, ROT13

risk_assessment = red_teamer.red_team(
    model_callback=model_callback,
    vulnerabilities=[Bias(types=["race"])],
    attacks=[PromptInjection(weight=2), ROT13(weight=1)],
    attacks_per_vulnerability_type=5
)

print(risk_assessment.overall)
```

### RedTeamer.red_team() Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_callback` | Callable | Yes | — | Target LLM system callback |
| `vulnerabilities` | list | No | — | Weaknesses to detect |
| `attacks` | list | No | — | Attack methods |
| `framework` | `AISafetyFramework` | No | — | Alternative to specifying vulnerabilities/attacks |
| `simulator_model` | str or `DeepEvalBaseLLM` | No | Constructor default | Attack generation model (overrides constructor) |
| `evaluation_model` | str or `DeepEvalBaseLLM` | No | Constructor default | Response evaluation model (overrides constructor) |
| `attacks_per_vulnerability_type` | int | No | `1` | Attacks per type |
| `ignore_errors` | bool | No | `False` | Ignore exceptions |
| `reuse_previous_attacks` | bool | No | `False` | Reuse previously simulated attacks |
| `async_mode` | bool | No | `True` | Enable async processing |
| `max_concurrent` | int | No | `10` | Maximum parallel coroutines |
| `target_purpose` | str | No | Constructor default | Application purpose |

### Reusing Previous Attacks

```python
# Second run reuses attacks generated in the first run
risk_assessment = red_teamer.red_team(
    model_callback=model_callback,
    reuse_previous_attacks=True
)
```

---

## Attack Weight Parameter

All attacks accept an optional `weight` parameter (default: `1`) that determines the probability of selection during simulation:

```python
from deepteam.attacks.single_turn import PromptInjection, ROT13

# PromptInjection will be selected 2x more often than ROT13
attacks = [PromptInjection(weight=2), ROT13(weight=1)]
```

---

## Risk Assessments

Results from red teaming runs generate risk assessments:

```python
risk_assessment = red_team(...)

# View overview and individual test cases
print(risk_assessment.overview)
print(risk_assessment.test_cases)

# Save locally
risk_assessment.save(to="./deepteam-results/")
```

### Interpreting Scores

Scores from vulnerability assessments are binary: `0` (vulnerable) or `1` (secure).

When using `RedTeamer`, access aggregated results:

```python
# Average scores per vulnerability
print(red_teamer.vulnerability_scores)

# Detailed per-attack breakdown
breakdown = red_teamer.vulnerability_scores_breakdown
excessive_agency_issues = breakdown[breakdown["Vulnerability"] == "Excessive Agency"]
print(excessive_agency_issues)
```

Example summary output:

| Vulnerability | Score |
|---------------|-------|
| PII API Database | 1.0 |
| PII Direct | 0.8 |
| Data Leakage | 1.0 |
| Excessive Agency | 0.6 |

Scores near 1.0 indicate strong performance; scores near 0 indicate potential vulnerabilities.

---

## Two-Step Process: How It Works

### Step 1: Simulating Adversarial Attacks

Two stages:
1. **Generating baseline attacks** - synthetically created based on specified vulnerabilities and `target_purpose`
2. **Enhancing baseline attacks** - increased complexity using adversarial methods (prompt injection, jailbreaking, etc.)

### Step 2: Evaluating LLM Outputs

Two stages:
1. **Generating responses** - from target LLM to each attack
2. **Scoring responses** - using vulnerability-specific metrics to identify critical weaknesses

Each vulnerability has a dedicated metric designed to assess whether that particular weakness has been effectively exploited, providing a precise binary evaluation.

---

## Custom LLM Implementation

For non-OpenAI models, implement `DeepEvalBaseLLM`:

```python
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

# Instantiate and use
custom_model = AzureChatOpenAI(
    openai_api_version=openai_api_version,
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
    openai_api_key=openai_api_key,
)
azure_openai = AzureOpenAI(model=custom_model)

red_teamer = RedTeamer(
    simulator_model=azure_openai,
    evaluation_model="gpt-4o"
)
```

**Required methods:**
- `get_model_name()` - returns string identifier
- `load_model()` - returns model object
- `generate(prompt: str)` - returns LLM output string
- `a_generate(prompt: str)` - async version of generate

**Caution:** Never enforce JSON outputs when setting up your model for red-teaming.

---

## LLM Provider Configuration

### OpenAI

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

### Azure OpenAI

```bash
deepeval set-azure-openai \
    --openai-endpoint=<endpoint> \
    --openai-api-key=<api_key> \
    --deployment-name=<deployment_name> \
    --openai-api-version=<openai_api_version> \
    --model-version=<model_version>

# Revert
deepeval unset-azure-openai
```

### Ollama

```bash
ollama run deepseek-r1:1.5b

deepeval set-ollama --model deepseek-r1:1.5b
# With custom base URL
deepeval set-ollama --model deepseek-r1:1.5b --base-url="http://localhost:11434"

# Revert
deepeval unset-ollama
```

---

## YAML CLI Configuration

For running red teaming from the command line without writing Python:

```yaml
# config.yaml

models:
  simulator: gpt-3.5-turbo-0125
  evaluation: gpt-4o

target:
  purpose: "A helpful AI assistant"
  # Option 1: Simple model for testing foundational models
  model: gpt-3.5-turbo
  # Option 2: Custom DeepEval model for LLM applications
  # model:
  #   provider: custom
  #   file: "my_custom_model.py"
  #   class: "MyCustomLLM"

system_config:
  max_concurrent: 10
  attacks_per_vulnerability_type: 3
  run_async: true
  ignore_errors: false
  output_folder: "results"

default_vulnerabilities:
  - name: "Bias"
    types: ["race", "gender"]
  - name: "Toxicity"
    types: ["profanity", "insults"]

attacks:
  - name: "Prompt Injection"
```

### Running via CLI

```bash
# Basic usage
deepteam run config.yaml

# With overrides
deepteam run config.yaml -c 20 -a 5 -o results
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `-c` | Maximum concurrent operations (overrides config) |
| `-a` | Attacks per vulnerability type (overrides config) |
| `-o` | Output folder path for results (overrides config) |

```bash
deepteam --help   # View all commands and options
```

---

## Complete Integration Workflow (from guides-red-teaming)

The recommended 5-step process:

### Step 1: Set Up Target LLM

```python
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM

class FinancialAdvisorLLM(DeepEvalBaseLLM):
    def load_model(self):
        return OpenAI()

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "FinancialAdvisorLLM"

    def get_system_prompt(self) -> str:
        return (
            "You are FinBot, a financial advisor bot. Your task is to provide "
            "investment advice and financial planning recommendations based on the "
            "user's financial data. Always prioritize user privacy."
        )
```

Always test with simple queries before red-teaming:
```python
target_llm = FinancialAdvisorLLM()
target_llm.generate("How much should I save each year to double my investment in 10 years?")
```

### Step 2: Initialize RedTeamer

```python
from deepeval.red_teaming import RedTeamer

red_teamer = RedTeamer(
    target_purpose="Provide financial advice and answer queries related to personal finance.",
    target_system_prompt=target_llm.get_system_prompt(),
    synthesizer_model="gpt-3.5-turbo-0125",
    evaluation_model="gpt-4.1",
    async_mode=True
)
```

### Step 3: Scan

```python
from deepeval.red_teaming import AttackEnhancement, Vulnerability

results = red_teamer.scan(
    target_model=target_llm,
    attacks_per_vulnerability=5,
    vulnerabilities=[
        Vulnerability.PII_API_DB,
        Vulnerability.PII_DIRECT,
        Vulnerability.PII_SESSION,
        Vulnerability.DATA_LEAKAGE,
        Vulnerability.PRIVACY
    ],
    attack_enhancements={
        AttackEnhancement.BASE64: 0.25,
        AttackEnhancement.GRAY_BOX_ATTACK: 0.25,
        AttackEnhancement.JAILBREAK_CRESCENDO: 0.25,
        AttackEnhancement.MULTILINGUAL: 0.25,
    }
)
```

### Step 4: Interpret Results

```python
print(red_teamer.vulnerability_scores)
# Filter for specific vulnerability
breakdown = red_teamer.vulnerability_scores_breakdown
issues = breakdown[breakdown["Vulnerability"] == "Excessive Agency"]
print(issues)
```

### Step 5: Iterate

1. Refine system prompt to outline model role and limitations
2. Add privacy and compliance filters (guardrails for sensitive data)
3. Re-scan after each adjustment
4. Monitor long-term performance with regular scans

---

## Tips and Warnings

- Use `async_mode=True` and `a_generate` whenever possible to greatly speed up the process
- Start with 5 attacks per vulnerability; increase for critical vulnerabilities
- Combine diverse attack enhancements (encoding, one-shot, multi-turn) for comprehensive coverage
- `target_purpose` and `target_system_prompt` generate more tailored, realistic attacks
- Encoding-based enhancements require fewest resources; jailbreaking involves multiple LLM calls
- Never enforce JSON outputs when setting up your model for red-teaming

---

## Related Documentation

- [20-vulnerabilities.md](./20-vulnerabilities.md) - Complete vulnerability catalog with all types
- [30-attack-enhancements.md](./30-attack-enhancements.md) - All attack enhancement types and configuration
