<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-20 23:46:10 +0900 | Updated: 2026-02-20 23:46:10 +0900 -->

# eval

## Purpose
Stores CI gate tests using DeepEval's pytest integration. This area automatically runs quality gates based on the Golden Dataset.

## Key Files
| File | Description |
|------|-------------|
| `test_agent_eval.py` | RAG quality / response completeness CI gate tests |
| `__init__.py` | Module initialization |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Tests in this directory serve as actual evaluation gates; document team-agreed criteria when changing thresholds.
- Maintain skip behavior when Golden Dataset is absent to ensure CI stability.
- When changing metric configurations, update alongside `src/loop2_evaluation/rag_metrics.py` and `custom_metrics.py`.

### Testing Requirements
- `.venv/bin/pytest eval/test_agent_eval.py -q` or `deepeval test run eval/test_agent_eval.py`

### Common Patterns
- `LLMTestCase` conversion is synchronized with Golden JSON fields
- Test parameterization (`pytest.mark.parametrize`) for case expansion

## Dependencies

### Internal
- `src/loop2_evaluation/*`
- `src/llm/deepeval_model.py`

### External
- `deepeval`
- `pytest`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
