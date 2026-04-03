"""합성 데이터셋 생성 모듈 (Loop 1 - Step 1).

DeepEval의 Synthesizer를 사용하여 소스 문서(corpus)에서
질문-답변 쌍(Golden)을 자동 생성합니다.

작동 흐름:
    1. corpus 디렉토리에서 실제 도메인 문서만 로드
    2. DeepEval Synthesizer에 문서를 컨텍스트로 전달
    3. LLM이 각 컨텍스트에서 질문-답변 쌍을 생성
    4. 결과를 JSON 파일로 저장

사용 예시:
    from src.loop1_dataset.synthesizer import generate_synthetic_dataset
    items = generate_synthetic_dataset(num_goldens=10)
    # → data/synthetic/synthetic_dataset.json 생성

핵심 개념:
    - Synthetic Dataset: LLM이 자동 생성한 평가용 데이터 후보
    - Golden: 자동 검증/수동 검토를 통과해 확정된 데이터
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from uuid import uuid4

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer

from src.llm.deepeval_model import get_deepeval_model
from src.settings import get_settings

_META_FILENAMES = {
    "agents.md",
    "readme.md",
    "changelog.md",
    "license",
    "license.md",
}

# DeepEval Synthesizer는 별도의 system prompt를 직접 받지 않기 때문에,
# "문서 본문 앞에 생성 규칙을 덧붙인 컨텍스트"를 만들어 전달합니다.
# 이렇게 하면 모델이 실제 문서 내용만 사용해서 질문을 만들고,
# 파일 구조/운영 규칙 같은 메타 정보로 새지 않도록 제약할 수 있습니다.
_SYNTHESIS_GUARDRAILS = """\
<generation_rules>
You are generating evaluation QA pairs from a single domain document.

Follow these rules strictly:
1. Generate questions only about the business/domain facts in the document.
2. Do not ask about file names, repository structure, prompts, testing workflow, manifests, validators, or agent instructions.
3. Do not invent facts that are not supported by the document.
4. Prefer short, answerable questions grounded in the provided document text.
5. If the document is too short, still stay on the same topic instead of switching topics.
</generation_rules>
"""


def _is_corpus_document(path: Path) -> bool:
    """합성 데이터 생성에 사용할 실제 도메인 문서인지 판별합니다."""
    name = path.name.lower()
    if name in _META_FILENAMES:
        return False
    if name.startswith("."):
        return False
    return path.suffix.lower() in {".md", ".txt"}


def _extract_topic_label(path: Path, content: str) -> str:
    """문서 경로/제목을 바탕으로 주제 라벨을 추출합니다."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()

    stem = re.sub(r"^\d+[_-]*", "", path.stem)
    normalized = stem.replace("_", " ").replace("-", " ").strip()
    return normalized or path.stem


def _build_synthesis_context(*, topic_label: str, content: str) -> str:
    """DeepEval Synthesizer에 넘길 최종 컨텍스트 문자열을 만듭니다.

    저장용 context에는 원문 문서를 넣고,
    "생성 시에만" guardrail이 붙은 컨텍스트를 사용합니다.
    그래야 Golden Dataset에는 실제 근거 문서만 남고,
    모델 입력에서는 주제 이탈을 막을 수 있습니다.
    """
    return (
        f"{_SYNTHESIS_GUARDRAILS}\n"
        f"<document_topic>\n{topic_label}\n</document_topic>\n\n"
        f"<document>\n{content}\n</document>\n"
    )


def _load_corpus_documents(
    corpus_dir: Path,
) -> tuple[list[list[str]], list[str], list[str], dict[str, list[str]]]:
    """corpus 디렉토리에서 실제 생성 대상 문서를 로드합니다.

    빈 파일과 메타 문서는 제외합니다.
    """
    # contexts:
    #   DeepEval Synthesizer에 실제로 전달할 입력.
    #   여기에는 guardrail 프롬프트가 붙어 있습니다.
    #
    # raw_context_by_source:
    #   최종 Golden JSON에 저장할 "원본 문서"입니다.
    #   생성용 guardrail 문자열까지 저장하면 나중에 사람이 볼 때 불필요하게 복잡해집니다.
    contexts: list[list[str]] = []
    source_files: list[str] = []
    topic_labels: list[str] = []
    raw_context_by_source: dict[str, list[str]] = {}

    if not corpus_dir.exists():
        return contexts, source_files, topic_labels, raw_context_by_source

    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file() or not _is_corpus_document(path):
            continue

        content = path.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            continue

        source_file = str(path)
        topic_label = _extract_topic_label(path, content)
        synthesis_context = _build_synthesis_context(
            topic_label=topic_label,
            content=content,
        )

        # DeepEval 입력은 "주제 제약 + 원문 문서"를 합친 문자열을 사용합니다.
        contexts.append([synthesis_context])
        source_files.append(source_file)
        topic_labels.append(topic_label)
        raw_context_by_source[source_file] = [content]

    return contexts, source_files, topic_labels, raw_context_by_source


def _resolve_max_goldens_per_context(
    *,
    requested_total: int,
    context_count: int,
    max_goldens_per_context: int | None,
) -> int:
    """목표 개수에 맞게 context당 생성 개수를 계산합니다."""
    if context_count <= 0:
        return 1
    if max_goldens_per_context is not None and max_goldens_per_context > 0:
        return max_goldens_per_context

    return max(1, math.ceil(max(1, requested_total) / context_count))


def generate_synthetic_dataset(
    *,
    corpus_dir: Path | None = None,
    output_path: Path | None = None,
    num_goldens: int = 10,
    max_goldens_per_context: int | None = None,
) -> list[dict]:
    """DeepEval Synthesizer를 사용하여 합성 데이터셋을 생성합니다.

    Args:
        corpus_dir: 소스 문서 디렉토리 (기본: settings.local_corpus_dir)
        output_path: 출력 JSON 경로 (기본: data/synthetic/synthetic_dataset.json)
        num_goldens: 생성할 총 golden 수
        max_goldens_per_context: 하나의 컨텍스트에서 생성할 최대 golden 수.
            None이면 목표 개수와 문서 수를 기준으로 자동 계산
    """
    settings = get_settings()
    corpus_dir = corpus_dir or settings.local_corpus_dir
    output_path = output_path or (settings.data_dir / "synthetic" / "synthetic_dataset.json")

    # 1. provider에 맞는 DeepEval adapter를 준비합니다.
    model = get_deepeval_model()
    # 2. corpus 문서를 읽을 때, 생성용 컨텍스트와 저장용 원문 컨텍스트를 함께 준비합니다.
    contexts, source_files, topic_labels, raw_context_by_source = _load_corpus_documents(corpus_dir)

    if not contexts:
        raise ValueError(f"생성에 사용할 corpus 문서가 없습니다: {corpus_dir}")

    # 문서 수가 적더라도 목표 개수에 최대한 맞게 후보를 만들 수 있도록
    # context당 생성 개수를 자동 계산합니다.
    resolved_max_goldens_per_context = _resolve_max_goldens_per_context(
        requested_total=num_goldens,
        context_count=len(contexts),
        max_goldens_per_context=max_goldens_per_context,
    )

    # DeepEval Synthesizer가 실제 후보 질문/답변을 생성하는 단계입니다.
    synthesizer = Synthesizer(model=model)
    generated_goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        source_files=source_files,
        max_goldens_per_context=resolved_max_goldens_per_context,
    )
    if num_goldens > 0:
        generated_goldens = generated_goldens[:num_goldens]

    # DeepEval이 반환한 결과를 더 다루기 쉬운 형태로 감쌉니다.
    dataset = EvaluationDataset(goldens=generated_goldens)
    topic_by_source = dict(zip(source_files, topic_labels, strict=False))

    items: list[dict] = []
    for golden in dataset.goldens:
        source_file = golden.source_file if hasattr(golden, "source_file") else ""
        item = {
            "id": uuid4().hex[:12],
            "input": golden.input if hasattr(golden, "input") else "",
            "expected_output": golden.expected_output if hasattr(golden, "expected_output") else "",
            # 저장 시에는 guardrail이 섞인 생성용 컨텍스트가 아니라,
            # 실제 근거 문서 원문만 남깁니다.
            "context": raw_context_by_source.get(source_file, []),
            "source_file": source_file,
            "topic": topic_by_source.get(source_file, ""),
            "synthetic_input_quality": golden.synthetic_input_quality
            if hasattr(golden, "synthetic_input_quality")
            else 0.0,
        }
        items.append(item)

    # Step 4와 리뷰 단계에서 그대로 재사용할 수 있도록 JSON으로 저장합니다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return items
