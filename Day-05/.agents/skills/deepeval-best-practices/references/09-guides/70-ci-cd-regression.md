# CI/CD Regression Testing — Reference Guide

Source: guides-regression-testing-in-cicd.md
URL: https://deepeval.com/guides/guides-regression-testing-in-cicd

## Read This When
- Need to set up LLM regression testing in a CI/CD pipeline (GitHub Actions, Travis CI, CircleCI)
- Want a minimal pytest + DeepEval test file template with `assert_test` and `EvaluationDataset`
- Looking for the YAML workflow configuration to run `deepeval test run` on push/pull requests

## Skip This When
- Need comprehensive CI/CD configuration options, flags, and environment variables -- see `references/02-llm-evals/50-ci-cd-and-configs.md`
- Want to track hyperparameters across CI/CD runs and compare results -- see `references/09-guides/80-observability-and-optimization.md`
- Looking for a full tutorial with CI/CD integration as part of a larger project -- see `references/10-tutorials/20-rag-qa-agent.md` or `references/10-tutorials/30-summarization-agent.md`

---

# Regression Testing LLM Systems in CI/CD

## Overview

"Regression testing ensures your LLM systems doesn't degrade in performance over time, and there is no better place to do it than in CI/CD environments."

## Creating Your Test File

The framework treats evaluation dataset rows as unit test cases. Here's the complete code example:

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="...", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")

dataset = EvaluationDataset(
    test_cases=[first_test_case, second_test_case]
)

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_example(test_case: LLMTestCase):
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])
```

**Command to verify functionality:**

```bash
deepeval test run test_file.py
```

## Setting Up Your YAML File

For GitHub Actions integration, create `.github/workflows/regression.yml`:

```yaml
name: LLM Regression Test
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      - name: Install Dependencies
        run: poetry install --no-root
      - name: Run DeepEval Unit Tests
        run: poetry run deepeval test run test_file.py
```

## Important Notes

**Tip:** The guide notes that hardcoded test cases are "for demonstration purposes only" and recommends using one of three dataset loading approaches for scalability.

**Note:** While GitHub Actions is demonstrated, similar implementations work with "Travis CI or CircleCI." Additionally, you may need environment variables like `OPENAI_API_KEY` and `CONFIDENT_API_KEY`.

## Related Resources

- [AI Agent Evaluation](/guides/guides-ai-agent-evaluation)
- [RAG Evaluation](/guides/guides-rag-evaluation)
- [Building Custom Metrics](/guides/guides-building-custom-metrics)
- [LLM Observability & Monitoring](/guides/guides-llm-observability)
