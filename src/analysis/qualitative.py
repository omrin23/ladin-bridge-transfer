"""
Qualitative analysis: sample 25 FLORES+ devtest items per condition and
dump (source, reference, hypothesis) to CSV for manual error typing.

Error categories (to be filled by hand):
  lexical | morphological/agreement | omission | hallucination | other
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from src.config import SEED, LADIN, ITA

logger = logging.getLogger(__name__)

SAMPLE_SIZE: int = 25
ERROR_CATEGORIES: tuple[str, ...] = (
    "lexical",
    "morphological/agreement",
    "omission",
    "hallucination",
    "other",
)


def sample_qualitative(
    flores_sentences: dict[str, list[str]],
    hypotheses: dict[str, list[str]],
    condition: str,
    results_dir: str | Path,
    src_lang: str = LADIN,
    tgt_lang: str = ITA,
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> Path:
    """
    Sample n devtest items for a condition and write a CSV with blank error_type.

    hypotheses: {pair_key: [hyp, ...]} — e.g. {"lld_Latn->ita_Latn": [...]}
    Returns the path to the written CSV.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pair_key = f"{src_lang}->{tgt_lang}"
    if pair_key not in hypotheses:
        raise KeyError(
            f"Hypotheses dict missing key '{pair_key}'. "
            f"Available: {list(hypotheses.keys())}"
        )

    src_sents = flores_sentences[src_lang]
    ref_sents = flores_sentences[tgt_lang]
    hyp_sents = hypotheses[pair_key]

    n_total = len(src_sents)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n_total), min(n, n_total)))

    rows = []
    for idx in indices:
        rows.append({
            "index": idx,
            "source": src_sents[idx],
            "reference": ref_sents[idx],
            "hypothesis": hyp_sents[idx],
            "error_type": "",  # filled manually
            "notes": "",
        })

    df = pd.DataFrame(rows)
    out_path = results_dir / f"qualitative_{condition}_{src_lang[:3]}_{tgt_lang[:3]}.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "Qualitative sample (%d items, seed=%d) written to %s",
        len(rows), seed, out_path,
    )
    return out_path


def run_qualitative_all(
    flores_sentences: dict[str, list[str]],
    all_hypotheses: dict[str, dict[str, list[str]]],
    results_dir: str | Path,
    seed: int = SEED,
) -> list[Path]:
    """
    Run qualitative sampling for all conditions.

    all_hypotheses: {condition: {pair_key: [hyp, ...]}}
    Returns list of written CSV paths.
    """
    paths = []
    for condition, hypotheses in all_hypotheses.items():
        try:
            p = sample_qualitative(
                flores_sentences=flores_sentences,
                hypotheses=hypotheses,
                condition=condition,
                results_dir=results_dir,
                seed=seed,
            )
            paths.append(p)
        except (KeyError, Exception) as e:
            logger.warning("Qualitative sampling failed for %s: %s", condition, e)
    return paths
