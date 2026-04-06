#!/usr/bin/env python3
"""OpenAI direct 기반 Golden Dataset 생성 전용 스크립트.

이 파일은 "Golden Dataset을 만들고 싶을 때" 사용하는 가장 단순한 진입점입니다.

언제 이 파일을 쓰면 되는가?
    - Loop 1 결과물인 `data/golden/golden_dataset.json`만 빠르게 만들고 싶을 때
    - OpenAI direct 경로로 생성이 실제로 되는지 확인하고 싶을 때
    - `run_pipeline.py`의 많은 옵션 없이, Golden 생성만 간단히 돌리고 싶을 때

언제 이 파일을 쓰지 않는가?
    - Step 5 평가, Step 6 Langfuse 모니터링, Step 8 프롬프트 최적화까지
      이어서 실행하고 싶을 때는 `run_pipeline.py`를 사용합니다.

이 스크립트가 실제로 하는 일:
    1. LLM provider를 OpenAI direct로 고정합니다.
    2. 기본 모델을 `gpt-5.4-mini`로 설정합니다.
    3. `src.loop1_dataset.golden_builder.build_golden_dataset()`를 호출합니다.
    4. 결과로 `data/synthetic/synthetic_dataset.json`과
       `data/golden/golden_dataset.json`을 생성/갱신합니다.

중요한 점:
    - 이 파일은 "Golden 생성용 편의 러너"이지, Golden 생성 로직 본체가 아닙니다.
    - 실제 생성/검증 로직은 `src/loop1_dataset/` 아래에 있습니다.
    - 즉, 이 파일은 입력 인자를 정리하고 실행 모드를 고정하는 얇은 실행 껍데기입니다.

예시:
    .venv/bin/python scripts/run_golden_openai.py
    .venv/bin/python scripts/run_golden_openai.py --num-goldens 4
    .venv/bin/python scripts/run_golden_openai.py --reviewed-csv data/review/tmp_reviewed.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# scripts/ 아래에서 실행해도 src/ 패키지를 바로 import할 수 있도록
# 프로젝트 루트를 sys.path 앞쪽에 넣습니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 주의:
# settings.get_settings()는 처음 호출될 때 환경변수를 읽어 캐시합니다.
# 그래서 src.settings 등을 import하기 전에 provider/model 관련 환경변수를
# 먼저 세팅해야 우리가 원하는 값이 실제 실행에 반영됩니다.
DEFAULT_MODEL_NAME = "gpt-5.4-mini"


def _parse_args() -> argparse.Namespace:
    """CLI 인자를 읽습니다.

    초보자가 바로 사용할 수 있게:
    - 기본값은 OpenAI + gpt-5.4-mini
    - 기본 실행은 skip-review=True
    로 두었습니다.
    """
    parser = argparse.ArgumentParser(
        description="Run Golden Builder with OpenAI direct defaults",
    )
    parser.add_argument(
        "--num-goldens",
        type=int,
        default=2,
        help="최종적으로 확정할 Golden 개수 (기본: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"OpenAI model name (기본: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="기본 corpus 대신 사용할 디렉토리 경로",
    )
    parser.add_argument(
        "--reviewed-csv",
        type=str,
        default=None,
        help="리뷰가 끝난 CSV를 직접 지정할 때 사용",
    )
    parser.add_argument(
        "--agent-module",
        type=str,
        default="src.my_agent",
        help="expected_tools를 생성할 agent 모듈 경로 (기본: src.my_agent)",
    )
    parser.add_argument(
        "--with-review",
        action="store_true",
        help="자동 검증 후보 CSV도 같이 만들고 싶을 때 사용",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # 이 스크립트의 핵심 목적은 "OpenAI direct 경로를 쉽게 실행하는 것"입니다.
    # 따라서 provider/model 기본값을 여기서 명시적으로 고정합니다.
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_MODEL_NAME"] = args.model

    from src.loop1_dataset.golden_builder import build_golden_dataset
    from src.settings import get_settings

    settings = get_settings()
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None
    reviewed_csv_path = Path(args.reviewed_csv) if args.reviewed_csv else None
    skip_review = not args.with_review

    print("[OpenAI Golden Runner] configuration")
    print(f"  provider: {settings.llm_provider}")
    print(f"  model: {settings.openai_model_name}")
    print(f"  openai key configured: {bool(settings.openai_api_key)}")
    print(f"  num_goldens: {args.num_goldens}")
    print(f"  skip_review: {skip_review}")
    if corpus_dir is not None:
        print(f"  corpus_dir: {corpus_dir}")
    if reviewed_csv_path is not None:
        print(f"  reviewed_csv: {reviewed_csv_path}")
    print(f"  agent_module: {args.agent_module}")

    # business logic는 src/ 쪽 함수가 담당합니다.
    # 이 스크립트는 "어떤 모드로 실행할지"만 정해서 넘깁니다.
    items = build_golden_dataset(
        corpus_dir=corpus_dir,
        num_goldens=args.num_goldens,
        skip_review=skip_review,
        reviewed_csv_path=reviewed_csv_path,
        agent_module=args.agent_module,
    )

    print("[OpenAI Golden Runner] done")
    print(f"  golden items: {len(items)}")
    print(f"  synthetic path: {settings.data_dir / 'synthetic' / 'synthetic_dataset.json'}")
    print(f"  golden path: {settings.data_dir / 'golden' / 'golden_dataset.json'}")
    if not skip_review:
        print(f"  review csv path: {settings.data_dir / 'review' / 'review_dataset.csv'}")


if __name__ == "__main__":
    main()
