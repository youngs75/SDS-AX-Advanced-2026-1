# Data Privacy, Troubleshooting, and Miscellaneous

## Read This When
- You need to understand what data DeepEval transmits, how to opt out of telemetry, or Confident AI's data storage policies
- You are troubleshooting SSL errors, timeout issues, dotenv loading problems, settings cache issues, or JSON validation errors from custom LLMs
- You need miscellaneous configuration (saving results as JSON locally, read-only filesystem mode, async debugging, trace flushing)

## Skip This When
- You need CLI commands or LLM provider setup -- see [10-cli-and-environment.md](./10-cli-and-environment.md)
- You want to simulate conversations for chatbot evaluation -- see [20-conversation-simulator.md](./20-conversation-simulator.md)
- You need to evaluate RAG or agent applications -- see [../09-guides/10-rag-evaluation.md](../09-guides/10-rag-evaluation.md) or [../09-guides/20-agent-evaluation.md](../09-guides/20-agent-evaluation.md)

---

## Data Privacy

### Privacy When Using DeepEval (Open-Source)

#### Telemetry Collection

By default, DeepEval uses Sentry to collect minimal telemetry data:
- Evaluation counts
- Metric types used

Personally identifiable information (PII) is explicitly excluded from all telemetry tracking.

**Opt out of telemetry:**
```bash
export DEEPEVAL_TELEMETRY_OPT_OUT=1
```

Or in `.env.local`:
```
DEEPEVAL_TELEMETRY_OPT_OUT=1
```

#### Error Reporting

Error tracking only occurs with explicit user consent. It does not collect user or company data.

**Enable error reporting:**
```bash
export ERROR_REPORTING=1
```

### Privacy When Using Confident AI (Cloud)

**Data storage:** Data sent to Confident AI is securely stored in AWS private cloud databases. Your organization is the sole entity that can access the data you store (unless on a VIP plan with different arrangements).

**Compliance:** Organizations with compliance requirements can upgrade membership for enhanced security features.

**Emergency contact:** If you suspect accidental transmission of sensitive information, immediately contact:
```
support@confident-ai.com
```
Request data deletion as soon as possible.

### What Data Leaves Your Machine

| Scenario | Data Transmitted |
|----------|-----------------|
| Using DeepEval locally (no login) | Telemetry only (metric types, counts) — no LLM content |
| Using `deepeval login` + running evals | Test case inputs/outputs, metric scores, test run metadata sent to Confident AI |
| Using `@observe` tracing | Trace spans, LLM inputs/outputs sent to Confident AI |
| Error reporting enabled | Error tracebacks (no user/company data) |

---

## Troubleshooting

### TLS / SSL Certificate Errors

**Symptom:** Uploading results to Confident AI fails with:
```
SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

This is typically a certificate verification failure in the local environment (not within DeepEval itself).

**Step 1: Check with `curl`**
```bash
curl -v https://api.confident-ai.com/
```
If `curl` reports an SSL/certificate error, the issue is at the system level.

**Step 2: Check with Python `requests`**
```bash
unset REQUESTS_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR
python -m pip install -U certifi
python - << 'PY'
import requests
r = requests.get("https://api.confident-ai.com")
print(r.status_code)
PY
```
If this fails with a certificate error, you have a system-level SSL configuration issue.

**Step 3: Re-run DeepEval**
If the Python snippet succeeds, re-run your `deepeval` evaluation from the same terminal session and check whether the upload still fails.

---

### Configuring Logging

DeepEval uses the standard Python `logging` module. To see logs, configure logging output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Environment flags for debugging:**

| Variable | Effect |
|----------|--------|
| `LOG_LEVEL` | Sets global log level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DEEPEVAL_VERBOSE_MODE` | Enables additional warnings and diagnostics |
| `DEEPEVAL_LOG_STACK_TRACES` | Includes stack traces in retry log messages |
| `DEEPEVAL_RETRY_BEFORE_LOG_LEVEL` | Log level for "before retry sleep" messages (read at call-time) |
| `DEEPEVAL_RETRY_AFTER_LOG_LEVEL` | Log level for "after retry attempt" messages (read at call-time) |

**Quick debug setup:**
```bash
export LOG_LEVEL=DEBUG
export DEEPEVAL_VERBOSE_MODE=1
```

---

### Timeout Issues

**Symptom:** Evaluations frequently time out or appear to hang.

DeepEval uses:
- An outer time budget per task (metric + test case), including retries
- An optional per-attempt timeout for individual provider calls

**Key timeout environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE` | unset | Total time budget per task (includes all retries) |
| `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` | unset | Per-attempt timeout for each provider call |
| `DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE` | unset | Extra buffer for async gather/cleanup |
| `DEEPEVAL_RETRY_MAX_ATTEMPTS` | `2` | Total attempts (first try + retries) |
| `DEEPEVAL_RETRY_INITIAL_SECONDS` | `1.0` | Initial backoff duration |
| `DEEPEVAL_RETRY_EXP_BASE` | `2.0` | Exponential backoff base |
| `DEEPEVAL_RETRY_JITTER` | `2.0` | Random jitter per retry |
| `DEEPEVAL_RETRY_CAP_SECONDS` | `5.0` | Max sleep between retries |
| `DEEPEVAL_SDK_RETRY_PROVIDERS` | unset | Provider slugs for SDK-managed retry delegation |
| `DEEPEVAL_DISABLE_TIMEOUTS` | unset | Disables all DeepEval timeout machinery |

**Recommended starting point for high-latency or rate-limited environments:**
```bash
export LOG_LEVEL=DEBUG
export DEEPEVAL_VERBOSE_MODE=1
export DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600
export DEEPEVAL_RETRY_MAX_ATTEMPTS=2
```

**Tip:** Increasing the outer budget (`DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE`) is usually the safest first step for timeout issues.

---

### Dotenv Loading Issues

**Symptom:** In pytest, a project `.env` is loaded that you didn't intend to load; or environment variable changes at runtime are not picked up.

**Root cause:** DeepEval loads dotenv files at import time (`import deepeval`). Dotenv never overrides existing process environment variables.

**Load order (lowest to highest priority):**
`.env` → `.env.{APP_ENV}` → `.env.local`

**Controls:**
```bash
# Skip dotenv loading entirely
DEEPEVAL_DISABLE_DOTENV=1 pytest -q

# Use a specific directory for dotenv files
ENV_DIR_PATH=/path/to/project pytest -q

# Load an environment-specific dotenv file
APP_ENV=production pytest -q
```

**Tip:** Set `DEEPEVAL_DISABLE_DOTENV=1` **before** anything imports `deepeval`:
```bash
export DEEPEVAL_DISABLE_DOTENV=1
pytest test_app.py
```

---

### Settings Cache Issues

**Symptom:** You changed an environment variable at runtime but DeepEval is not picking up the change.

**Solution:** Reset the settings cache:
```python
from deepeval.config.settings import reset_settings
reset_settings(reload_dotenv=True)
```

**Persisting settings changes from code:**
```python
from deepeval.config.settings import get_settings

settings = get_settings()
with settings.edit(save="dotenv"):
    settings.DEEPEVAL_VERBOSE_MODE = True
```

Note: Computed fields (derived timeout settings) are not persisted.

---

### JSON Validation Errors from Custom LLMs

**Symptom:**
```
ValueError: Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.
```

**Root cause:** Smaller or open-source LLMs often fail to generate valid JSON for metric evaluation.

**Solutions:**

1. **Use a more capable model** (GPT-4o, Claude 3 Sonnet, etc.)

2. **Implement JSON confinement** by modifying the `generate()` signature to accept a `BaseModel` schema:
   ```python
   from pydantic import BaseModel
   from deepeval.models import DeepEvalBaseLLM

   class MyLLM(DeepEvalBaseLLM):
       def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
           # use lm-format-enforcer or instructor to enforce schema
           pass

       async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
           return self.generate(prompt, schema)
   ```

3. **Use `lm-format-enforcer`** for HuggingFace transformers models
4. **Use `instructor`** for API-based models (Claude, Gemini, OpenAI)

See `../01-getting-started/30-custom-models-and-embeddings.md` for full JSON confinement examples.

---

### Reporting Issues

When opening a GitHub issue at https://github.com/confident-ai/deepeval, include:

- `deepeval` version (`pip show deepeval`)
- OS + Python version
- A minimal reproduction script
- Full traceback
- Logs with `LOG_LEVEL=DEBUG`
- Any non-default timeout/retry environment variables

**Always redact API keys and secrets before sharing.**

Community support is also available at: https://discord.gg/a3K9c8GRGt

---

## Miscellaneous

### Update Warning Opt-In

DeepEval can notify you when a new version is available. This is **highly recommended** to stay aware of changes and improvements.

**Enable update warnings:**
```bash
export DEEPEVAL_UPDATE_WARNING_OPT_IN=1
```

Or in `.env.local`:
```
DEEPEVAL_UPDATE_WARNING_OPT_IN=1
```

---

### Saving Test Results Locally as JSON

Export evaluation results to a local folder as timestamped JSON files:

```bash
# Linux/Mac
export DEEPEVAL_RESULTS_FOLDER="./eval-results"

# Windows
set DEEPEVAL_RESULTS_FOLDER=.\eval-results
```

After running evaluations, a timestamped JSON file will appear in the specified folder.

---

### Read-Only File System Environments

For containerized or constrained environments where file writes are restricted:

```bash
export DEEPEVAL_FILE_SYSTEM=READ_ONLY
```

---

### Disabling the Legacy JSON Keystore

DeepEval uses a legacy JSON keystore at `.deepeval/.deepeval` for non-secret keys (as a fallback). To disable reading from it:

```bash
export DEEPEVAL_DISABLE_LEGACY_KEYFILE=1
```

---

### Async Debugging

For debugging async evaluation issues:

```bash
export DEEPEVAL_DEBUG_ASYNC=1
export DEEPEVAL_GRPC_LOGGING=1
```

---

### Custom Run Identifier

Tag test runs with a custom identifier for easier filtering in Confident AI:

```bash
export DEEPEVAL_IDENTIFIER="my-experiment-v2"
```

---

### Tracing Flush for Component-Level Evals

When running component-level evaluations (using `@observe`), ensure traces are not lost when the process exits:

```bash
export CONFIDENT_TRACE_FLUSH=1
```

This is especially important when running the evaluation script directly (e.g., `python main.py`) rather than through pytest.

---

## Quick Reference: Most Useful Environment Variables

```bash
# --- Authentication ---
CONFIDENT_API_KEY="ck_..."           # Connect to Confident AI

# --- OpenAI (default evaluator) ---
OPENAI_API_KEY="sk-..."

# --- Results ---
DEEPEVAL_RESULTS_FOLDER="./results"  # Save results as JSON locally

# --- Debugging ---
LOG_LEVEL=DEBUG
DEEPEVAL_VERBOSE_MODE=1
DEEPEVAL_LOG_STACK_TRACES=1

# --- Timeouts (increase for slow/rate-limited providers) ---
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600

# --- CI/CD (prevent loading local .env files) ---
DEEPEVAL_DISABLE_DOTENV=1

# --- Privacy ---
DEEPEVAL_TELEMETRY_OPT_OUT=1

# --- Updates ---
DEEPEVAL_UPDATE_WARNING_OPT_IN=1

# --- Tracing ---
CONFIDENT_TRACE_FLUSH=1
```

---

## Related Reference Files

- `10-cli-and-environment.md` — Complete CLI command reference and all environment variables
- `20-conversation-simulator.md` — ConversationSimulator class reference
- `../01-getting-started/30-custom-models-and-embeddings.md` — Custom model JSON confinement solutions
