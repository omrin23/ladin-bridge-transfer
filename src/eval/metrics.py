"""
BLEU and chrF computation via sacrebleu.

sacrebleu is used throughout for consistency and to enable reproducibility
via its signature string. chrF is the primary metric for morphologically
rich, low-resource targets; BLEU is reported for comparability.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from sacrebleu.metrics import BLEU, CHRF

logger = logging.getLogger(__name__)


class MetricResult(NamedTuple):
    bleu: float
    chrf: float
    sacrebleu_signature: str


def compute_metrics(
    hypotheses: list[str],
    references: list[str],
) -> MetricResult:
    """
    Compute corpus-level BLEU and chrF with sacrebleu defaults.

    Both metrics operate at the corpus level (not average of sentence scores).
    sacrebleu handles tokenization internally; we do not pre-tokenize.

    Returns a MetricResult with bleu, chrf (both as floats), and the
    sacrebleu signature string for reproducibility logging.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Hypothesis count ({len(hypotheses)}) != reference count ({len(references)})"
        )

    bleu_metric = BLEU()
    chrf_metric = CHRF()

    bleu_result = bleu_metric.corpus_score(hypotheses, [references])
    chrf_result = chrf_metric.corpus_score(hypotheses, [references])

    bleu_score = round(bleu_result.score, 4)
    chrf_score = round(chrf_result.score, 4)

    # Signature captures tokenization and metric parameters for reproducibility
    signature = bleu_metric.get_signature().format()

    logger.info("BLEU=%.2f  chrF=%.2f  [%s]", bleu_score, chrf_score, signature)
    return MetricResult(bleu=bleu_score, chrf=chrf_score, sacrebleu_signature=signature)
