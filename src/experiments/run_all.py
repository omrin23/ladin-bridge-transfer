"""
Orchestrate the full experiment matrix:

  1. Zero-shot baseline (no fine-tuning, Italian proxy for Ladin)
  2. Direct condition (LoRA on ADIta_Lad only)
  3. Bridge-FR, Bridge-ES, Bridge-PT (two-stage LoRA)
  4. Similarity analysis (subword Jaccard + encoder cosine distance)
  5. Ablation (Direct + best bridge, ADIta_Lad only vs. ADIta_Lad + SDLad-Ita)

Results are saved to Drive after every stage; a session disconnect cannot
lose completed work — re-run skips conditions where the JSON already exists.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

from src.config import (
    MODEL_NAME, LADIN, ITA, ENG, FRA, SPA, POR,
    SEED, SUBSET_SIZE, EPOCHS_BRIDGE, EPOCHS_LADIN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    LEARNING_RATE, BATCH_SIZE, GRAD_ACCUM,
    LADIN_HF_REPO, FLORES_HF_REPO,
)
from src.seeding import set_all_seeds
from src.model.build import prepare_model_and_tokenizer
from src.data.loaders import (
    load_adita_lad, load_sdlad_ita, load_opus_bridge,
    load_flores_devtest, assert_no_flores_leakage,
)
from src.train.pipeline import run_direct, run_bridge
from src.eval.evaluate import evaluate_condition
from src.analysis.similarity import run_similarity_analysis
from src.analysis.qualitative import run_qualitative_all
from src.analysis.aggregate import aggregate

logger = logging.getLogger(__name__)

BRIDGE_LANGS: dict[str, str] = {
    "bridge_fr": FRA,  # OPUS Books fr-it
    "bridge_es": SPA,  # OPUS Books es-it
    "bridge_pt": POR,  # OPUS Books it-pt
}


def _result_exists(results_dir: Path, condition: str, data_setting: str) -> bool:
    """Skip conditions whose result JSON already exists (resume after disconnect)."""
    p = results_dir / f"{condition}__{data_setting}.json"
    if p.exists():
        logger.info("Skipping %s/%s — result already exists.", condition, data_setting)
        return True
    return False


def _make_run_config(
    condition: str,
    data_setting: str,
    bridge_language: str | None,
    ladin_added: bool = True,
) -> dict:
    """Build the metadata dict logged alongside every adapter checkpoint."""
    return {
        "condition": condition,
        "data_setting": data_setting,
        "bridge_language": bridge_language,
        "base_model": MODEL_NAME,
        "ladin_added_as_new_code": ladin_added,
        "ladin_seed_source": ITA,
        "seed": SEED,
        "hyperparameters": {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lr": LEARNING_RATE,
            "epochs_bridge": EPOCHS_BRIDGE,
            "epochs_ladin": EPOCHS_LADIN,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
        },
    }


def run_experiment(
    drive_root: str | Path,
    run_zero_shot: bool = True,
    run_direct_cond: bool = True,
    run_bridges: bool = True,
    run_similarity: bool = True,
    run_ablation: bool = True,
    cache_dir: str | None = None,
) -> None:
    """
    Run the complete experiment matrix.

    drive_root: Google Drive folder where checkpoints and results are written.
    Set individual flags to False to skip completed sections.
    """
    drive_root = Path(drive_root)
    results_dir = drive_root / "results"
    checkpoints_dir = drive_root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(SEED)
    logger.info("=" * 60)
    logger.info("Ladin bridge-transfer experiment | seed=%d", SEED)
    logger.info("Drive root: %s", drive_root)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load FLORES+ devtest once (shared across all conditions)
    # ------------------------------------------------------------------
    logger.info("Loading FLORES+ devtest...")
    flores = load_flores_devtest(cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Load ADIta_Lad once
    # ------------------------------------------------------------------
    logger.info("Loading ADIta_Lad...")
    # Base model + tokenizer for preprocessing (no LoRA yet, just tokenizer)
    from transformers import NllbTokenizer
    _tok_for_prep = NllbTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    _tok_for_prep.add_special_tokens({"additional_special_tokens": [LADIN]})

    ladin_pairs = load_adita_lad(tokenizer=_tok_for_prep, cache_dir=cache_dir)
    assert_no_flores_leakage(ladin_pairs, flores)
    logger.info("ADIta_Lad: %d pairs after preprocessing", len(ladin_pairs))

    # ------------------------------------------------------------------
    # Zero-shot baseline (Italian proxy — no fine-tuning)
    # ------------------------------------------------------------------
    if run_zero_shot and not _result_exists(results_dir, "zero_shot", "adita_only"):
        logger.info("--- Zero-shot evaluation ---")
        model_zs, tok_zs = prepare_model_and_tokenizer(cache_dir=cache_dir if cache_dir else None)
        # Zero-shot uses Italian as proxy for Ladin (ladin_added_as_new_code=True
        # but the embedding is just the ita seed — no fine-tuning has happened).
        evaluate_condition(
            model=model_zs,
            tokenizer=tok_zs,
            flores_sentences=flores,
            condition="zero_shot",
            data_setting="adita_only",
            bridge_language=None,
            results_dir=results_dir,
            ladin_pairs_count=0,
            bridge_pairs_count=0,
        )
        del model_zs, tok_zs
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Direct condition
    # ------------------------------------------------------------------
    if run_direct_cond and not _result_exists(results_dir, "direct", "adita_only"):
        logger.info("--- Direct condition ---")
        model_d, tok_d = prepare_model_and_tokenizer(cache_dir=cache_dir if cache_dir else None)
        run_config = _make_run_config("direct", "adita_only", None)
        run_direct(
            model=model_d,
            tokenizer=tok_d,
            ladin_pairs=ladin_pairs,
            output_dir=checkpoints_dir / "direct",
            run_config=run_config,
        )
        evaluate_condition(
            model=model_d,
            tokenizer=tok_d,
            flores_sentences=flores,
            condition="direct",
            data_setting="adita_only",
            bridge_language=None,
            results_dir=results_dir,
            ladin_pairs_count=len(ladin_pairs),
        )
        del model_d, tok_d
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Bridge conditions
    # ------------------------------------------------------------------
    if run_bridges:
        for condition_name, bridge_lang in BRIDGE_LANGS.items():
            if _result_exists(results_dir, condition_name, "adita_only"):
                continue
            logger.info("--- %s ---", condition_name)

            bridge_pairs = load_opus_bridge(
                bridge_lang=bridge_lang,
                tokenizer=_tok_for_prep,
                cache_dir=cache_dir,
            )
            assert_no_flores_leakage(bridge_pairs, flores)

            model_b, tok_b = prepare_model_and_tokenizer(cache_dir=cache_dir if cache_dir else None)
            run_config = _make_run_config(condition_name, "adita_only", bridge_lang)
            run_bridge(
                model=model_b,
                tokenizer=tok_b,
                bridge_pairs=bridge_pairs,
                ladin_pairs=ladin_pairs,
                bridge_lang=bridge_lang,
                output_dir=checkpoints_dir / condition_name,
                run_config=run_config,
            )
            evaluate_condition(
                model=model_b,
                tokenizer=tok_b,
                flores_sentences=flores,
                condition=condition_name,
                data_setting="adita_only",
                bridge_language=bridge_lang,
                results_dir=results_dir,
                ladin_pairs_count=len(ladin_pairs),
                bridge_pairs_count=len(bridge_pairs),
            )
            del model_b, tok_b
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Similarity analysis
    # ------------------------------------------------------------------
    if run_similarity and not (results_dir / "similarity.json").exists():
        logger.info("--- Similarity analysis ---")
        model_sim, tok_sim = prepare_model_and_tokenizer(cache_dir=cache_dir if cache_dir else None)
        run_similarity_analysis(
            model=model_sim,
            tokenizer=tok_sim,
            flores_sentences=flores,
            results_dir=results_dir,
        )
        del model_sim, tok_sim
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Ablation (Direct + best bridge, both data_settings)
    # ------------------------------------------------------------------
    if run_ablation:
        _run_ablation(
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            flores=flores,
            ladin_pairs=ladin_pairs,
            tok_for_prep=_tok_for_prep,
            cache_dir=cache_dir,
        )

    # ------------------------------------------------------------------
    # Aggregate all results into master CSVs
    # ------------------------------------------------------------------
    aggregate(results_dir)
    logger.info("=== All experiments complete. Results in %s ===", results_dir)


def _run_ablation(
    results_dir: Path,
    checkpoints_dir: Path,
    flores: dict,
    ladin_pairs: list,
    tok_for_prep,
    cache_dir: str | None,
) -> None:
    """Run the ablation: ADIta_Lad+SDLad-Ita for Direct and best bridge condition."""
    from src.data.loaders import load_sdlad_ita

    try:
        synth_pairs = load_sdlad_ita(tokenizer=tok_for_prep, cache_dir=cache_dir)
        assert_no_flores_leakage(synth_pairs, flores)
    except ValueError as e:
        logger.warning("Ablation skipped: %s", e)
        return

    combined_pairs = ladin_pairs + synth_pairs
    logger.info("Ablation combined pairs: %d (ladin) + %d (synth) = %d", len(ladin_pairs), len(synth_pairs), len(combined_pairs))

    # Determine best bridge from existing results
    best_bridge_condition = _find_best_bridge(results_dir)
    ablation_conditions = ["direct"]
    if best_bridge_condition:
        ablation_conditions.append(best_bridge_condition)

    for condition_name in ablation_conditions:
        data_setting = "adita_plus_synth"
        if _result_exists(results_dir, condition_name, data_setting):
            continue

        logger.info("--- Ablation: %s | %s ---", condition_name, data_setting)
        model_a, tok_a = prepare_model_and_tokenizer(cache_dir=cache_dir if cache_dir else None)
        run_config = _make_run_config(condition_name, data_setting, BRIDGE_LANGS.get(condition_name))

        if condition_name == "direct":
            run_direct(
                model=model_a,
                tokenizer=tok_a,
                ladin_pairs=combined_pairs,
                output_dir=checkpoints_dir / f"{condition_name}_ablation",
                run_config=run_config,
            )
        else:
            bridge_lang = BRIDGE_LANGS[condition_name]
            bridge_pairs = load_opus_bridge(bridge_lang=bridge_lang, tokenizer=tok_for_prep, cache_dir=cache_dir)
            run_bridge(
                model=model_a,
                tokenizer=tok_a,
                bridge_pairs=bridge_pairs,
                ladin_pairs=combined_pairs,
                bridge_lang=bridge_lang,
                output_dir=checkpoints_dir / f"{condition_name}_ablation",
                run_config=run_config,
            )

        evaluate_condition(
            model=model_a,
            tokenizer=tok_a,
            flores_sentences=flores,
            condition=condition_name,
            data_setting=data_setting,
            bridge_language=BRIDGE_LANGS.get(condition_name),
            results_dir=results_dir,
            ladin_pairs_count=len(combined_pairs),
            synthetic_pairs_count=len(synth_pairs),
        )
        del model_a, tok_a
        torch.cuda.empty_cache()


def _find_best_bridge(results_dir: Path) -> str | None:
    """Return the bridge condition with the highest chrF on lld_Latn→ita_Latn."""
    best_chrf = -1.0
    best_cond = None
    for cond in ["bridge_fr", "bridge_es", "bridge_pt"]:
        p = results_dir / f"{cond}__adita_only.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        chrf = data.get("scores", {}).get("lld_Latn->ita_Latn", {}).get("chrf", -1.0)
        if chrf > best_chrf:
            best_chrf = chrf
            best_cond = cond
    if best_cond:
        logger.info("Best bridge condition: %s (chrF=%.2f)", best_cond, best_chrf)
    return best_cond
