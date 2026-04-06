from __future__ import annotations

from unittest.mock import MagicMock

from deepeval.dataset.golden import Golden


def test_generate_synthetic_dataset_when_meta_files_exist_then_excludes_them(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "00_sla.md").write_text("# SLA\n가용성 지연 오류율", encoding="utf-8")
    (corpus_dir / "AGENTS.md").write_text("# corpus\nstep1 instructions", encoding="utf-8")

    output_path = tmp_path / "synthetic.json"

    class FakeSynthesizer:
        def __init__(self, model):
            self.model = model
            self.called_kwargs = None

        def generate_goldens_from_contexts(self, **kwargs):
            self.called_kwargs = kwargs
            return [
                Golden(
                    input="SLA란 무엇인가요?",
                    expected_output="SLA는 서비스 수준 합의입니다.",
                    context=["가용성 지연 오류율"],
                    source_file=kwargs["source_files"][0],
                    synthetic_input_quality=0.9,
                ),
            ]

    fake_synth = FakeSynthesizer(model=MagicMock())
    monkeypatch.setattr(
        "src.loop1_dataset.synthesizer.get_deepeval_model",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "src.loop1_dataset.synthesizer.Synthesizer",
        lambda model: fake_synth,
    )

    from src.loop1_dataset.synthesizer import generate_synthetic_dataset

    items = generate_synthetic_dataset(
        corpus_dir=corpus_dir,
        output_path=output_path,
        num_goldens=3,
    )

    assert len(items) == 1
    assert output_path.exists()
    assert fake_synth.called_kwargs is not None
    assert fake_synth.called_kwargs["source_files"] == [str(corpus_dir / "00_sla.md")]
    assert fake_synth.called_kwargs["max_goldens_per_context"] == 3
    generation_context = fake_synth.called_kwargs["contexts"][0][0]
    assert "<generation_rules>" in generation_context
    assert "Do not ask about file names" in generation_context
    assert "<document_topic>\nSLA\n</document_topic>" in generation_context
    assert "# SLA\n가용성 지연 오류율" in generation_context
    assert items[0]["context"] == ["# SLA\n가용성 지연 오류율"]
    assert items[0]["topic"] == "SLA"
