# AgentOps Project

## Purpose
The Day3 project is a hands-on codebase that implements a `Dataset → Evaluation → Improve` closed loop. It separates Loop 1 (dataset construction), Loop 2 (evaluation/monitoring/prompt optimization), and Loop 3 (remediation suggestions) into step-based scripts and modules, providing a reproducible operational workflow.

## Key Files
| File | Description |
|------|-------------|
| `README.md` | Overall execution overview and quick start guide |
| `pyproject.toml` | Python dependencies / pytest configuration |
| `.env.example` | OpenRouter/Langfuse environment variable template |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `src/` | Core library code (see `src/AGENTS.md`) |
| `scripts/` | Step-by-step CLI execution entry points (see `scripts/AGENTS.md`) |
| `docs/` | Operations/analysis/execution documentation (see `docs/AGENTS.md`) |
| `tests/` | Unit/integration tests (see `tests/AGENTS.md`) |
| `data/` | Sample inputs and execution artifact storage (see `data/AGENTS.md`) |
| `eval/` | DeepEval pytest CI gate (see `eval/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Use `scripts/run_pipeline.py` as the default execution entry point; only run individual `scripts/0X_*.py` files directly when debugging a single step.
- At the root level, only handle configuration/documentation/execution flow changes; business logic modifications must be done in `src/`.
- Treat `.venv/`, `.pytest_cache/`, `__pycache__/`, and `*.egg-info/` as generated artifacts and do not modify them.

### Testing Requirements
- Start by running the minimum tests relevant to the scope of changes.
- Standard verification order:
  1. `.venv/bin/python scripts/run_pipeline.py --help`
  2. Run `pytest` targeted tests related to the modified module
  3. If needed, run `.venv/bin/python -m compileall src scripts tests`

### Common Patterns
- Step 5: Golden offline evaluation (with optional sampling)
- Step 6: Langfuse score-based monitoring / failure extraction
- Step 8: Calibration + Langfuse failure hints for evaluation prompt optimization

## Dependencies

### Internal
- All modules use shared settings via `get_settings()` from `src/settings.py`.
- `scripts/` only calls functions from `src/` and does not duplicate domain logic.

### External
- `deepeval` (metrics / synthetic data / pytest evaluation)
- `langfuse` (monitoring / score storage / retrieval)
- `langchain`, `langgraph`, `deepagents` (Loop 3 remediation agents)
- `openai` + OpenRouter endpoint (LLM calls)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
