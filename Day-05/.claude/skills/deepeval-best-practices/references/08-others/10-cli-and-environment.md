# CLI and Environment Variables

## Read This When
- You need to configure DeepEval CLI commands (`deepeval login`, `deepeval test run`, `deepeval set-<provider>`)
- You are setting up LLM provider connections (OpenAI, Azure, Anthropic, Ollama, Gemini, LiteLLM, etc.) via CLI or environment variables
- You need to tune timeout, retry, concurrency, or debug environment variables for your evaluation pipeline

## Skip This When
- You want to simulate multi-turn conversations with a chatbot -- see [20-conversation-simulator.md](./20-conversation-simulator.md)
- You need data privacy details, troubleshooting guides, or miscellaneous settings -- see [30-data-privacy-and-misc.md](./30-data-privacy-and-misc.md)
- You need to implement a custom LLM wrapper class (`DeepEvalBaseLLM`) -- see [../07-benchmarks/10-overview.md](../07-benchmarks/10-overview.md)

---

This file is a comprehensive reference for all DeepEval CLI commands and environment variables.

---

## CLI Overview

DeepEval provides a command-line interface for authentication, test execution, model configuration, and settings management.

```bash
# Install / upgrade
pip install -U deepeval

# View all available commands
deepeval --help

# View help for a specific command
deepeval <command> --help
```

**Settings storage:** DeepEval reads settings from dotenv files in the current working directory (or `ENV_DIR_PATH`). It also uses a legacy JSON keystore at `.deepeval/.deepeval` for non-secret keys (fallback only).

**Dotenv precedence (lowest to highest):** `.env` → `.env.<APP_ENV>` → `.env.local`

Process environment variables always take highest precedence over dotenv files.

---

## Authentication Commands

### `deepeval login`

Log in to Confident AI (DeepEval's cloud platform). Once authenticated, test runs automatically upload to the cloud.

```bash
# Interactive login (prompts for key)
deepeval login

# Non-interactive with explicit key
deepeval login --confident-api-key "ck_..."

# Save to custom dotenv file
deepeval login --confident-api-key "ck_..." --save dotenv:.env.custom
```

Keys are saved as `CONFIDENT_API_KEY`. Secrets are never stored in the JSON keystore.

### `deepeval logout`

Remove Confident AI credentials from local persistence.

```bash
# Remove from .env.local (default)
deepeval logout

# Remove from custom path
deepeval logout --save dotenv:.myconf.env
```

### `deepeval view`

Open the latest test run on Confident AI in your browser. Uploads cached run artifacts if needed.

```bash
deepeval view
```

---

## Test Execution Commands

### `deepeval test run`

Run evaluations through pytest integration. Test files must use the `test_` prefix.

```bash
deepeval test run test_app.py

# Run specific test function
deepeval test run test_app.py::test_correctness

# Run with verbose output
deepeval test run test_app.py -v

# Run with parallel execution
deepeval test run test_app.py -n auto
```

---

## Settings Commands

### `deepeval settings`

Read and write individual settings keys.

```bash
# List all settings for a provider
deepeval settings -l anthropic
deepeval settings -l openai
deepeval settings -l azure

# Set a specific key
deepeval settings --set KEY=VALUE
```

---

## Model Provider Commands

All provider commands come in pairs: `set-<provider>` and `unset-<provider>`.

**When you set a provider:**
- Sets `USE_<PROVIDER>_MODEL = True` for the chosen provider
- Turns all other `USE_*` flags off (only one LLM provider active at a time)

**When you unset a provider:**
- Disables only that provider's `USE_*` flag
- Leaves all others untouched

**Persistence:** Add `--save=dotenv[:path]` to write settings to a dotenv file (default: `.env.local`). Use `--quiet` / `-q` to suppress output.

```bash
# Set a default save target globally
export DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local
```

> Token costs are expressed in **USD per token**. For published pricing in $/MTok, divide by 1,000,000.
> Example: $3/MTok = 0.000003 USD per token.

### LLM Provider Commands

| Provider | Set Command | Unset Command |
|----------|-------------|---------------|
| OpenAI | `set-openai` | `unset-openai` |
| Azure OpenAI | `set-azure-openai` | `unset-azure-openai` |
| Anthropic | `set-anthropic` | `unset-anthropic` |
| AWS Bedrock | `set-bedrock` | `unset-bedrock` |
| Ollama (local) | `set-ollama` | `unset-ollama` |
| Local HTTP model | `set-local-model` | `unset-local-model` |
| Grok | `set-grok` | `unset-grok` |
| Moonshot (Kimi) | `set-moonshot` | `unset-moonshot` |
| DeepSeek | `set-deepseek` | `unset-deepseek` |
| Gemini | `set-gemini` | `unset-gemini` |
| LiteLLM | `set-litellm` | `unset-litellm` |
| OpenRouter | `set-openrouter` | `unset-openrouter` |
| Portkey | `set-portkey` | `unset-portkey` |

### Embedding Provider Commands

| Provider | Set Command | Unset Command |
|----------|-------------|---------------|
| Azure OpenAI | `set-azure-openai-embedding` | `unset-azure-openai-embedding` |
| Local (HTTP) | `set-local-embeddings` | `unset-local-embeddings` |
| Ollama | `set-ollama-embeddings` | `unset-ollama-embeddings` |

---

## Provider-Specific CLI Flags

### `deepeval set-openai`

```bash
deepeval set-openai \
    --model=gpt-4.1 \
    --cost-per-input-token=0.000002 \
    --cost-per-output-token=0.000008 \
    --save=dotenv
```

### `deepeval set-anthropic`

```bash
deepeval set-anthropic \
    -m claude-3-7-sonnet-latest \
    -i 0.000003 \
    -o 0.000015 \
    --save=dotenv
```

### `deepeval set-azure-openai`

```bash
deepeval set-azure-openai \
    --base-url="https://your-resource.azure.openai.com/" \
    --model-name="gpt-4.1" \
    --deployment-name="your-deployment" \
    --api-version="2025-01-01-preview" \
    --save=dotenv
```

### `deepeval set-azure-openai-embedding`

```bash
deepeval set-azure-openai-embedding \
    --deployment-name="your-embedding-deployment" \
    --save=dotenv
```

### `deepeval set-gemini`

```bash
deepeval set-gemini \
    --model="gemini-2.5-flash" \
    --save=dotenv
```

### `deepeval set-grok`

```bash
deepeval set-grok \
    --model grok-4.1 \
    --temperature=0 \
    --save=dotenv
```

### `deepeval set-ollama`

```bash
# Start model first: ollama run deepseek-r1:1.5b
deepeval set-ollama --model=deepseek-r1:1.5b

# With custom port/URL
deepeval set-ollama \
    --model=deepseek-r1:1.5b \
    --base-url="http://localhost:11434" \
    --save=dotenv
```

### `deepeval set-ollama-embeddings`

```bash
deepeval set-ollama-embeddings \
    --model=nomic-embed-text \
    --save=dotenv
```

### `deepeval set-local-model` (vLLM, LM Studio, any OpenAI-compatible API)

```bash
# vLLM (default: http://localhost:8000/v1/)
deepeval set-local-model \
    --model=<model_name> \
    --base-url="http://localhost:8000/v1/" \
    --save=dotenv

# LM Studio (default: http://localhost:1234/v1/)
deepeval set-local-model \
    --model=<model_name> \
    --base-url="http://localhost:1234/v1/" \
    --save=dotenv
```

### `deepeval set-local-embeddings`

```bash
deepeval set-local-embeddings \
    --model=<embedding_model_name> \
    --base-url="http://localhost:1234/v1/" \
    --save=dotenv
```

### `deepeval set-litellm`

```bash
# With provider prefix in model name
deepeval set-litellm --model=openai/gpt-3.5-turbo
deepeval set-litellm --model=anthropic/claude-3-opus
deepeval set-litellm --model=google/gemini-pro

# With custom API base
deepeval set-litellm \
    --model=openai/gpt-3.5-turbo \
    --base-url="https://your-custom-endpoint.com" \
    --save=dotenv
```

### `deepeval set-openrouter`

```bash
deepeval set-openrouter \
    --model "openai/gpt-4.1" \
    --base-url "https://openrouter.ai/api/v1" \
    --temperature=0 \
    --prompt-api-key \
    --save=dotenv
```

---

## Debug Controls

```bash
# Enable structured logs, gRPC wire tracing, and Confident tracing
deepeval set-debug \
    --log-level DEBUG \
    --debug-async \
    --retry-before-level INFO \
    --retry-after-level ERROR \
    --grpc \
    --grpc-verbosity DEBUG \
    --grpc-trace list_tracers \
    --trace-verbose \
    --trace-env staging \
    --trace-flush \
    --save=dotenv

# Restore defaults
deepeval unset-debug --save=dotenv
```

---

## Environment Variables Reference

### General Settings

| Variable | Values | Effect |
|----------|--------|--------|
| `CONFIDENT_API_KEY` | string / unset | Logs into Confident AI; enables tracing and cloud result uploads |
| `DEEPEVAL_DISABLE_DOTENV` | `1` / `0` / unset | Disables dotenv autoload at import; useful in pytest/CI |
| `ENV_DIR_PATH` | path / unset | Directory containing `.env` files (defaults to CWD) |
| `APP_ENV` | string / unset | Loads `.env.{APP_ENV}` between `.env` and `.env.local` |
| `DEEPEVAL_DISABLE_LEGACY_KEYFILE` | `1` / `0` / unset | Disables reading legacy `.deepeval/.deepeval` keystore |
| `DEEPEVAL_DEFAULT_SAVE` | `dotenv[:path]` / unset | Default persistence target for `deepeval set-* --save` |
| `DEEPEVAL_FILE_SYSTEM` | `READ_ONLY` / unset | Restricts file writes in constrained environments |
| `DEEPEVAL_RESULTS_FOLDER` | path / unset | Exports timestamped JSON of latest test run to folder |
| `DEEPEVAL_IDENTIFIER` | string / unset | Default identifier for runs |

### Display / Truncation

| Variable | Values | Effect |
|----------|--------|--------|
| `DEEPEVAL_MAXLEN_TINY` | int | Max length for "tiny" shorteners (default: 40) |
| `DEEPEVAL_MAXLEN_SHORT` | int | Max length for "short" shorteners (default: 60) |
| `DEEPEVAL_MAXLEN_MEDIUM` | int | Max length for "medium" shorteners (default: 120) |
| `DEEPEVAL_MAXLEN_LONG` | int | Max length for "long" shorteners (default: 240) |
| `DEEPEVAL_SHORTEN_DEFAULT_MAXLEN` | int / unset | Overrides default max length for `shorten()` |
| `DEEPEVAL_SHORTEN_SUFFIX` | string | Suffix used by `shorten()` (default: `...`) |
| `DEEPEVAL_VERBOSE_MODE` | `1` / `0` / unset | Enables verbose mode globally |
| `DEEPEVAL_LOG_STACK_TRACES` | `1` / `0` / unset | Logs stack traces for errors |

### Retry / Backoff Tuning

| Variable | Type | Default | Notes |
|----------|------|---------|-------|
| `DEEPEVAL_RETRY_MAX_ATTEMPTS` | int | `2` | Total attempts (1 retry) |
| `DEEPEVAL_RETRY_INITIAL_SECONDS` | float | `1.0` | Initial backoff duration |
| `DEEPEVAL_RETRY_EXP_BASE` | float | `2.0` | Exponential base (must be >= 1) |
| `DEEPEVAL_RETRY_JITTER` | float | `2.0` | Random jitter added per retry |
| `DEEPEVAL_RETRY_CAP_SECONDS` | float | `5.0` | Maximum sleep between retries |
| `DEEPEVAL_SDK_RETRY_PROVIDERS` | list / unset | -- | Provider slugs for SDK-managed retry delegation |
| `DEEPEVAL_RETRY_BEFORE_LOG_LEVEL` | int / unset | INFO | Log level for "before retry" messages |
| `DEEPEVAL_RETRY_AFTER_LOG_LEVEL` | int / unset | ERROR | Log level for "after retry" messages |

### Timeouts / Concurrency

| Variable | Values | Effect |
|----------|--------|--------|
| `DEEPEVAL_MAX_CONCURRENT_DOC_PROCESSING` | int | Max concurrent document tasks (default: 2) |
| `DEEPEVAL_TIMEOUT_THREAD_LIMIT` | int | Max threads for timeout machinery (default: 128) |
| `DEEPEVAL_TIMEOUT_SEMAPHORE_WARN_AFTER_SECONDS` | float | Warning threshold (default: 5.0) |
| `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` | float / unset | Per-attempt timeout for provider calls (preferred) |
| `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE` | float / unset | Total time budget per task including retries (preferred) |
| `DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE` | float / unset | Extra buffer time for async gather/cleanup |
| `DEEPEVAL_DISABLE_TIMEOUTS` | `1` / `0` / unset | Disables all DeepEval timeout machinery |

### Telemetry / Debug

| Variable | Values | Effect |
|----------|--------|--------|
| `DEEPEVAL_DEBUG_ASYNC` | `1` / `0` / unset | Enables extra async debugging |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | `1` / `0` / unset | Opts out of Sentry telemetry collection |
| `DEEPEVAL_UPDATE_WARNING_OPT_IN` | `1` / `0` / unset | Opts into update version warnings |
| `DEEPEVAL_GRPC_LOGGING` | `1` / `0` / unset | Enables extra gRPC logging |
| `LOG_LEVEL` | `DEBUG`, `INFO`, etc. | Sets global log level used by DeepEval |
| `ERROR_REPORTING` | `1` / unset | Enables error tracking (requires explicit consent) |

### Model Settings — OpenAI

| Variable | Values | Effect |
|----------|--------|--------|
| `OPENAI_API_KEY` | string / unset | OpenAI API key |
| `USE_OPENAI_MODEL` | `1` / `0` / unset | Prefer OpenAI as default LLM evaluator |
| `OPENAI_MODEL_NAME` | string / unset | Default model name (falls back to `gpt-4.1`) |
| `OPENAI_COST_PER_INPUT_TOKEN` | float / unset | Input token cost for reporting |
| `OPENAI_COST_PER_OUTPUT_TOKEN` | float / unset | Output token cost for reporting |

### Model Settings — Anthropic

| Variable | Values | Effect |
|----------|--------|--------|
| `ANTHROPIC_API_KEY` | string / unset | Anthropic API key |
| `USE_ANTHROPIC_MODEL` | `1` / `0` / unset | Prefer Anthropic as default LLM evaluator |
| `ANTHROPIC_MODEL_NAME` | string / unset | Default model name |
| `ANTHROPIC_COST_PER_INPUT_TOKEN` | float / unset | Input token cost |
| `ANTHROPIC_COST_PER_OUTPUT_TOKEN` | float / unset | Output token cost |

### Model Settings — Azure OpenAI

| Variable | Values | Effect |
|----------|--------|--------|
| `AZURE_OPENAI_API_KEY` | string / unset | Azure OpenAI API key |
| `AZURE_OPENAI_AD_TOKEN` | string / unset | Azure AD token (alternative to API key) |
| `USE_AZURE_OPENAI` | `1` / `0` / unset | Prefer Azure OpenAI as default |
| `AZURE_OPENAI_ENDPOINT` | string / unset | Endpoint URL |
| `OPENAI_API_VERSION` | string / unset | API version (e.g., `2025-01-01-preview`) |
| `AZURE_DEPLOYMENT_NAME` | string / unset | Deployment name |
| `AZURE_MODEL_NAME` | string / unset | Model name |
| `AZURE_MODEL_VERSION` | string / unset | Model version |

### Model Settings — AWS / Amazon Bedrock

| Variable | Values | Effect |
|----------|--------|--------|
| `AWS_ACCESS_KEY_ID` | string / unset | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | string / unset | AWS secret key |
| `USE_AWS_BEDROCK_MODEL` | `1` / `0` / unset | Prefer Bedrock as default LLM |
| `AWS_BEDROCK_MODEL_NAME` | string / unset | Bedrock model ID |
| `AWS_BEDROCK_REGION` | string / unset | AWS region (e.g., `us-east-1`) |
| `AWS_BEDROCK_COST_PER_INPUT_TOKEN` | float / unset | Input token cost |
| `AWS_BEDROCK_COST_PER_OUTPUT_TOKEN` | float / unset | Output token cost |

### Model Settings — Gemini

| Variable | Values | Effect |
|----------|--------|--------|
| `GOOGLE_API_KEY` | string / unset | Google API key |
| `USE_GEMINI_MODEL` | `1` / `0` / unset | Prefer Gemini as default LLM |
| `GEMINI_MODEL_NAME` | string / unset | Default Gemini model |

### Model Settings — Grok

| Variable | Values | Effect |
|----------|--------|--------|
| `GROK_API_KEY` | string / unset | xAI API key |
| `USE_GROK_MODEL` | `1` / `0` / unset | Prefer Grok as default LLM |
| `GROK_MODEL_NAME` | string / unset | Default Grok model |

### Model Settings — Ollama / Local

| Variable | Values | Effect |
|----------|--------|--------|
| `USE_OLLAMA_MODEL` | `1` / `0` / unset | Prefer Ollama as default LLM |
| `OLLAMA_MODEL_NAME` | string / unset | Default Ollama model |
| `LOCAL_MODEL_BASE_URL` | string / unset | Base URL for Ollama/local model (default: `http://localhost:11434`) |
| `USE_LOCAL_MODEL` | `1` / `0` / unset | Prefer local HTTP model as default |
| `LOCAL_MODEL_API_KEY` | string / unset | Optional API key for local model |
| `LOCAL_MODEL_NAME` | string / unset | Local model name |

### Model Settings — LiteLLM / OpenRouter

| Variable | Values | Effect |
|----------|--------|--------|
| `USE_LITELLM_MODEL` | `1` / `0` / unset | Prefer LiteLLM as default |
| `LITELLM_MODEL_NAME` | string / unset | LiteLLM model (e.g., `openai/gpt-3.5-turbo`) |
| `LITELLM_API_BASE` | string / unset | Custom endpoint URL |
| `OPENROUTER_API_KEY` | string / unset | OpenRouter API key |
| `OPENROUTER_MODEL_NAME` | string / unset | OpenRouter model |
| `OPENROUTER_BASE_URL` | string / unset | OpenRouter base URL |

### Embeddings Settings

| Variable | Values | Effect |
|----------|--------|--------|
| `USE_AZURE_OPENAI_EMBEDDING` | `1` / `0` / unset | Prefer Azure OpenAI embeddings |
| `AZURE_EMBEDDING_DEPLOYMENT_NAME` | string / unset | Azure embedding deployment name |
| `USE_LOCAL_EMBEDDINGS` | `1` / `0` / unset | Prefer local HTTP embeddings |
| `LOCAL_EMBEDDING_API_KEY` | string / unset | Optional API key for local embeddings |
| `LOCAL_EMBEDDING_MODEL_NAME` | string / unset | Local embedding model name |
| `LOCAL_EMBEDDING_BASE_URL` | string / unset | Base URL for local embedding endpoint |
| `USE_OLLAMA_EMBEDDINGS` | `1` / `0` / unset | Prefer Ollama embeddings |
| `OLLAMA_EMBEDDING_MODEL_NAME` | string / unset | Ollama embedding model |

### Tracing

| Variable | Values | Effect |
|----------|--------|--------|
| `CONFIDENT_TRACE_FLUSH` | `1` / unset | Ensures traces are flushed before process exit (important for component-level evals) |

---

## Boolean Flag Semantics

Boolean variables use case-insensitive parsing:

**Truthy tokens:** `1`, `true`, `t`, `yes`, `y`, `on`, `enable`, `enabled`

**Falsy tokens:** `0`, `false`, `f`, `no`, `n`, `off`, `disable`, `disabled`

---

## Common CLI Patterns

### Setting up for local development (no cloud)

```bash
# .env.local
OPENAI_API_KEY=sk-...
DEEPEVAL_RESULTS_FOLDER=./eval-results
```

### Setting up with Confident AI cloud

```bash
deepeval login --confident-api-key "ck_..."
export OPENAI_API_KEY=sk-...
```

### Disabling dotenv in CI/CD

```bash
DEEPEVAL_DISABLE_DOTENV=1 pytest test_app.py
```

### Switching to Azure OpenAI for evaluation

```bash
deepeval set-azure-openai \
    --base-url="https://my-resource.azure.openai.com/" \
    --model-name="gpt-4o" \
    --deployment-name="gpt-4o-eval" \
    --api-version="2025-01-01-preview" \
    --save=dotenv
```

### Switching to a local Ollama model

```bash
# 1. Start model
ollama run llama3.1

# 2. Configure
deepeval set-ollama --model=llama3.1 --save=dotenv

# 3. Verify
deepeval settings -l ollama
```

### Debugging timeouts

```bash
export LOG_LEVEL=DEBUG
export DEEPEVAL_VERBOSE_MODE=1
export DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600
export DEEPEVAL_RETRY_MAX_ATTEMPTS=2
```

### Runtime settings reset (after env var changes)

```python
from deepeval.config.settings import reset_settings
reset_settings(reload_dotenv=True)
```

### Persist settings changes from code

```python
from deepeval.config.settings import get_settings
settings = get_settings()
with settings.edit(save="dotenv"):
    settings.DEEPEVAL_VERBOSE_MODE = True
```

---

## Related Reference Files

- `20-conversation-simulator.md` — ConversationSimulator class reference
- `30-data-privacy-and-misc.md` — Data privacy, troubleshooting, and miscellaneous features
- `../01-getting-started/30-custom-models-and-embeddings.md` — Custom model and embedding configuration
- `../01-getting-started/40-integrations.md` — Model provider integration details
