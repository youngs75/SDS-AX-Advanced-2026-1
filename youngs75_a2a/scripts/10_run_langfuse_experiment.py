#!/usr/bin/env python3
"""Langfuse Dataset 기반 평가 실험을 실행한다.

실행:
    python -m youngs75_a2a.scripts.10_run_langfuse_experiment
    python -m youngs75_a2a.scripts.10_run_langfuse_experiment --run-name "v1-baseline"
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# youngs75_a2a/.env를 명시적으로 로드
_pkg_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_pkg_env, override=True)

from langfuse import Langfuse
from youngs75_a2a.eval_pipeline.settings import get_settings
from youngs75_a2a.eval_pipeline.observability.langfuse import enabled

_settings = get_settings()


def run_experiment(dataset_name: str, run_name: str | None) -> None:
    if not enabled():
        print("❌ Langfuse가 비활성화 상태입니다.")
        print(f"   LANGFUSE_HOST={_settings.langfuse_host}")
        print(f"   LANGFUSE_PUBLIC_KEY={_settings.langfuse_public_key[:10] if _settings.langfuse_public_key else 'EMPTY'}...")
        return

    # 에이전트 import (lazy — .env 로드 후)
    from youngs75_a2a.eval_pipeline.my_agent import run_coding_agent
    from deepeval.test_case import LLMTestCase
    from youngs75_a2a.eval_pipeline.loop2_evaluation.custom_metrics import (
        create_response_completeness_metric,
    )

    lf = Langfuse(
        public_key=_settings.langfuse_public_key,
        secret_key=_settings.langfuse_secret_key,
        host=_settings.langfuse_host,
    )

    dataset = lf.get_dataset(dataset_name)

    if not run_name:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"coding-eval-{ts}"

    print(f"📂 데이터셋: {dataset_name} ({len(dataset.items)}개 항목)")
    print(f"🚀 실험: {run_name}")
    print(f"   모델: {os.getenv('MODEL_NAME', 'gpt-5.4')}")
    print()

    results = []
    for i, item in enumerate(dataset.items):
        query = item.input.get("query", "") if isinstance(item.input, dict) else str(item.input)
        expected = item.expected_output.get("response", "") if isinstance(item.expected_output, dict) else str(item.expected_output)

        print(f"[{i + 1}/{len(dataset.items)}] {item.id}: {query[:60]}...")

        # 1. Agent 실행
        try:
            actual_output = run_coding_agent(query)
            print(f"   ✅ 에이전트 응답 ({len(actual_output)}자)")
        except Exception as e:
            print(f"   ❌ 에이전트 실패: {e}")
            results.append({"item_id": item.id, "score": 0.0, "error": str(e)})
            continue

        # 2. DeepEval 평가
        try:
            tc = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected)
            metric = create_response_completeness_metric(threshold=0.5)
            metric.measure(tc)
            score = metric.score
            reason = getattr(metric, "reason", "")
            print(f"   📊 response_completeness: {score:.2f}")
        except Exception as e:
            score = 0.0
            reason = str(e)
            print(f"   ⚠️ 평가 실패: {e}")

        # 3. Langfuse에 score 기록
        try:
            trace = lf.trace(
                name=f"coding-eval-{item.id}",
                input={"query": query},
                output={"response": actual_output[:1000]},
                tags=["coding-eval", run_name],
                metadata={"dataset_item_id": item.id, "run_name": run_name},
            )
            lf.score(
                trace_id=trace.id,
                name="response_completeness",
                value=score,
                comment=reason[:200] if reason else None,
            )
        except Exception as e:
            print(f"   ⚠️ Langfuse 기록 실패: {e}")

        results.append({"item_id": item.id, "score": score})

    lf.flush()

    scores = [r["score"] for r in results if "error" not in r]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n🎉 완료: {len(scores)}/{len(results)} 성공, 평균 score: {avg:.2f}")
    print(f"   확인: {_settings.langfuse_host}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="coding-assistant-golden")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_name, args.run_name)


if __name__ == "__main__":
    main()
