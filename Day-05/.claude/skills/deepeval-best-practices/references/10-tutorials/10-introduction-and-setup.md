# Tutorial: Introduction and Setup

## Read This When
- Want to install DeepEval, write your first evaluation test, and run it with `deepeval test run`
- Need to set up Confident AI integration (login, API key configuration, dashboard access)
- Starting the tutorial series and need an overview of what each tutorial covers

## Skip This When
- Already have DeepEval installed and configured -- jump to a specific tutorial: `references/10-tutorials/20-medical-chatbot.md`, `references/10-tutorials/30-rag-qa-agent.md`, or `references/10-tutorials/40-summarization-agent.md`
- Need detailed installation and setup reference -- see `references/01-getting-started/10-installation-and-setup.md`

---

## Introduction

**DeepEval** is a powerful open-source LLM evaluation framework. The tutorials demonstrate how to enhance LLM applications progressively, covering everything from initial development through post-production deployment.

### Tutorials Available

1. **Start Here: Install & Run Your First Eval** - Getting started with DeepEval installation and first evaluation
2. **Meeting Summarizer** - Development and evaluation of a summarization agent
3. **RAG QA Agent** - Evaluating retrieval-augmented generation pipelines for accuracy, relevance, and completeness
4. **Medical Chatbot** - Testing healthcare-focused LLM chatbots for hallucinations and safety

### What You'll Learn

#### Development Evals
- Select evaluation metrics aligned with your task
- Use `deepeval` to measure and track LLM performance
- Interpret results to tune prompts, models, and hyperparameters
- Scale evaluations across diverse inputs and edge cases

#### Production Evals
- Continuously evaluate LLM performance in production
- Run A/B tests on different models or configurations using real data
- Feed production insights back into development workflows

### Key Terminologies

- **Hyperparameters**: Configuration values shaping your LLM application (system prompts, user prompts, model choice, temperature, chunk size for RAG)
- **System Prompt**: Defines overall LLM behavior across all interactions
- **Generation Model**: The LLM being evaluated
- **Evaluation Model**: A separate LLM used to score or assess generation model outputs

### What DeepEval Offers

Supports evaluation metrics for:
- RAG applications (Retrieval-Augmented Generation)
- Conversational applications
- Agentic applications

### Target Audience

Designed for developers shipping LLM features, researchers testing prompt variations, and teams optimizing LLM outputs at scale.

> **Important Note:** LLM evaluation isn't a one-time step — it's a continuous loop.

---

## Set Up DeepEval

### Installing DeepEval

**DeepEval** is a powerful LLM evaluation framework. Start by installing it using pip:

```bash
pip install -U deepeval
```

### Write Your First Test

Evaluate LLM output correctness using `GEval`, an LLM-as-a-judge metric.

> **Important:** Test files must use the `test_` prefix (e.g., `test_app.py`) for DeepEval recognition.

#### Code Example: test_app.py

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5)

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    # Replace this with the actual output from your LLM application
    actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
    expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs.")

evaluate([test_case], [correctness_metric])
```

#### Running Your Evaluation

Execute the following terminal command:

```bash
deepeval test run test_app.py
```

#### API Key Setup

LLM-as-a-judge metrics rely on an evaluation model. By default, DeepEval uses OpenAI's models, so set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

For custom LLM usage, refer to the custom evaluation models documentation.

### Setting Up Confident AI

Connect DeepEval to Confident AI — a cloud platform offering dashboards, logging, collaboration, and more. **Free to start.**

#### Sign Up

[Sign up here](https://www.confident-ai.com) or run:

```bash
deepeval login
```

#### API Key Configuration

Navigate to your Settings page and copy your Confident AI API Key from the Project API Key box.

#### Login Methods

**Python approach** (main.py):

```python
deepeval.login("your-confident-api-key")
```

**CLI approach:**

```bash
deepeval login --confident-api-key "your-confident-api-key"
```

#### Login Persistence

The `deepeval login` command persists keys to `.env.local` by default. For custom paths:

```bash
deepeval login --confident-api-key "ck_..." --save dotenv:.env.custom
```

Keys are saved as `api_key` and `CONFIDENT_API_KEY`. Secrets are never stored in JSON keystores.

#### Logout / Key Rotation

Clear credentials using:

```bash
# Removes from .env.local (default)
deepeval logout

# Or specify custom target
deepeval logout --save dotenv:.myconf.env
```

### Related Resources

- [GitHub Repository](https://github.com/confident-ai/deepeval)
- [Discord Community](https://discord.gg/a3K9c8GRGt)
- [LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM-as-a-Judge Explanation](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
