"""Unit tests for eval/metrics.py — sanity-check BLEU and chrF on known inputs."""

import pytest
from src.eval.metrics import compute_metrics


def test_perfect_translation():
    refs = ["The cat sat on the mat.", "Hello world."]
    hyps = ["The cat sat on the mat.", "Hello world."]
    result = compute_metrics(hyps, refs)
    assert result.bleu == pytest.approx(100.0, abs=1.0)
    assert result.chrf == pytest.approx(100.0, abs=1.0)


def test_empty_hypothesis_gives_low_scores():
    refs = ["The cat sat on the mat."]
    hyps = [""]
    result = compute_metrics(hyps, refs)
    assert result.bleu == pytest.approx(0.0, abs=1.0)
    assert result.chrf < 10.0


def test_partial_overlap_between_zero_and_perfect():
    refs = ["The cat sat on the mat."]
    hyps = ["The cat sat."]
    result = compute_metrics(hyps, refs)
    assert 0.0 < result.bleu < 100.0
    assert 0.0 < result.chrf < 100.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Hypothesis count"):
        compute_metrics(["one sentence"], ["ref1", "ref2"])


def test_sacrebleu_signature_is_nonempty():
    result = compute_metrics(["hello"], ["hello"])
    assert isinstance(result.sacrebleu_signature, str)
    assert len(result.sacrebleu_signature) > 0
