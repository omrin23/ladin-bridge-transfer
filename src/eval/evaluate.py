"""
Run evaluation over all translation pairs for a given model condition,
and write the results JSON following the exact schema in 01_CODING_INSTRUCTIONS.md.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import (
    MODEL_NAME, LADIN, ITA,
    BEAM_SIZE, BATCH_SIZE, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LEARNING_RATE,
    EPOCHS_BRIDGE, EPOCHS_LADIN, SEED,
    SUBSET_SIZE,
)
from src.eval.metrics import compute_metrics
from src.eval.translate import translate_batch
from src.languages import all_eval_pairs

logger = logging.getLogger(__name__)


def evaluate_condition(
    model,
    tokenizer,
    flores_sentences: dict[str, list[str]],
    condition: str,
    data_setting: str,
    bridge_language: str | None,
    results_dir: str | Path,
    ladin_pairs_count: int = SUBSET_SIZE,
    bridge_pairs_count: int = SUBSET_SIZE,
    synthetic_pairs_count: int = 0,
    ladin_added_as_new_code: bool = True,
    ladin_seed_source: str = ITA,
    extra_hyperparams: dict | None = None,
) -> dict:
    """
    Evaluate all translation pairs on FLORES+ devtest and write a results JSON.

    Writes results/{condition}__{data_setting}.json.
    Returns the results dict.

    Parameters
    ----------
    condition       : one of zero_shot | direct | bridge_fr | bridge_es | bridge_pt
    data_setting    : adita_only | adita_plus_synth
    bridge_language : NLLB code for the bridge (None for zero_shot/direct)
    results_dir     : directory to write the JSON file
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_pairs = all_eval_pairs()
    scores: dict[str, dict[str, float]] = {}
    sacrebleu_sig: str = ""

    for pair in eval_pairs:
        if pair.src not in flores_sentences or pair.tgt not in flores_sentences:
            logger.warning("Skipping pair %s — language not in FLORES+ data.", pair)
            continue

        src_sents = flores_sentences[pair.src]
        ref_sents = flores_sentences[pair.tgt]

        hyps = translate_batch(
            model=model,
            tokenizer=tokenizer,
            sources=src_sents,
            src_lang=pair.src,
            tgt_lang=pair.tgt,
            batch_size=BATCH_SIZE,
            beam_size=BEAM_SIZE,
            max_seq_len=MAX_SEQ_LEN,
        )

        result = compute_metrics(hyps, ref_sents)
        scores[pair.key()] = {"bleu": result.bleu, "chrf": result.chrf}
        sacrebleu_sig = result.sacrebleu_signature  # same for all pairs

    hparams = {
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr": LEARNING_RATE,
        "epochs_bridge": EPOCHS_BRIDGE,
        "epochs_ladin": EPOCHS_LADIN,
        "batch_size": BATCH_SIZE,
        "beam_size": BEAM_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
    }
    if extra_hyperparams:
        hparams.update(extra_hyperparams)

    results = {
        "condition": condition,
        "data_setting": data_setting,
        "bridge_language": bridge_language,
        "base_model": MODEL_NAME,
        "ladin_added_as_new_code": ladin_added_as_new_code,
        "ladin_seed_source": ladin_seed_source,
        "seed": SEED,
        "hyperparameters": hparams,
        "data_sizes": {
            "ladin_pairs": ladin_pairs_count,
            "bridge_pairs": bridge_pairs_count,
            "synthetic_pairs": synthetic_pairs_count,
        },
        "sacrebleu_signature": sacrebleu_sig,
        "scores": scores,
        "eval_set": "flores_plus_devtest",
        "n_eval_sentences": len(next(iter(flores_sentences.values()))),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = results_dir / f"{condition}__{data_setting}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", out_path)

    return results
