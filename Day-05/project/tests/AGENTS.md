<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# tests

## Purpose
A pytest test suite that verifies functionality/regression for Loop 1/2/3 and LLM/Langfuse bridges.

## Key Files
| File | Description |
|------|-------------|
| `conftest.py` | Shared fixtures and test environment configuration |
| `test_synthesizer.py` | Loop1 synthetic generation tests |
| `test_loop1_synthesizer_generation.py` | Loop1 generation flow tests |
| `test_openrouter_model.py` | LLM client/model layer tests |
| `test_custom_metrics.py` | Loop2 custom metric tests |
| `test_langfuse_sampling.py` | Langfuse sampling / failure extraction tests |
| `test_prompt_optimizer.py` | Step8 optimization logic tests |
| `test_calibration_cases.py` | Calibration case loading / validation |
| `test_golden_sampling.py` | Golden sampling / delivery path tests |
| `test_remediation_agent.py` | Step7 remediation tests |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Tests dependent on external APIs should be fixed with mocking/monkeypatch to avoid network calls.
- When adding new features, test both happy-path and edge cases (empty inputs, thresholds, missing fields).
- Maintain scenario-based test naming (`test_xxx_when_yyy`).

### Testing Requirements
- Run tests directly related to changed files first, then expand scope as needed.
- Basic smoke:
  - `.venv/bin/pytest tests/test_golden_sampling.py tests/test_langfuse_sampling.py tests/test_prompt_optimizer.py tests/test_custom_metrics.py -q`

### Common Patterns
- Deterministic sampling comparison (`sample_seed`) verification
- Isolate path/environment-dependent values using fixtures and tmp_path

## Dependencies

### Internal
- All `src/*` modules

### External
- `pytest`
- `pytest-asyncio`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
