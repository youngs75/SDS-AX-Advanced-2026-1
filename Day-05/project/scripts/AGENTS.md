<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# scripts

## Purpose
Provides step-by-step execution entry points. Each script calls `src/` modules to execute or orchestrate Loop 1/2/3 workflows.

## Key Files
| File | Description |
|------|-------------|
| `run_pipeline.py` | Step 1–8 unified execution orchestrator |
| `run_golden_openai.py` | OpenAI direct convenience runner for Loop1 |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Scripts should only handle argument parsing/output formatting; delegate business logic to `src/`.
- Keep the number of entrypoints small; prefer `run_golden_openai.py` or `run_pipeline.py` over adding step-specific wrappers.
- When adding new options, also reflect them in `run_pipeline.py` to maintain the unified execution path.
- Document execution examples based on `.venv/bin/python`.

### Testing Requirements
- Minimum verification for CLI changes:
  - `.venv/bin/python scripts/run_pipeline.py --help`
  - `.venv/bin/python scripts/run_golden_openai.py --help`
  - Related Step unit tests

### Common Patterns
- `sys.path.insert(0, project_root)` pattern for package imports
- Summary output + artifact path guidance upon Step completion
- Prefer fewer, clearer entrypoints over overlapping one-off wrappers

## Dependencies

### Internal
- `src/loop1_dataset/*`
- `src/loop2_evaluation/*`
- `src/loop3_remediation/*`

### External
- `argparse`, `asyncio` (standard library)
- Loop-specific external packages are handled by `src/` modules

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
