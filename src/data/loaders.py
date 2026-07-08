"""
Dataset loaders for all corpora used in the project:
  - ADIta_Lad  (sfrontull/lld_valbadia-ita) — authentic Ladin↔Italian
  - SDLad-Ita  (synthetic, ablation only)
  - FLORES+    (openlanguagedata/flores_plus) — multi-parallel eval
  - OPUS Books (Helsinki-NLP/opus_books) — romance bridge pairs (fr-it, es-it, it-pt)
"""

from __future__ import annotations

import logging
from typing import Optional

from datasets import load_dataset, DatasetDict, Dataset

from src.config import (
    LADIN, ITA, FRA, SPA, POR,
    LADIN_HF_REPO, FLORES_HF_REPO, SYNTH_HF_REPO,
    OPUS_CONFIGS, OPUS_LANG_KEYS,
    SEED, SUBSET_SIZE,
)
from src.data.preprocess import preprocess_pairs
from src.data.subsample import deterministic_subsample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_translation_pairs(
    dataset: Dataset,
    src_key: str,
    tgt_key: str,
) -> list[tuple[str, str]]:
    """
    Extract (src, tgt) string pairs from a HF Dataset.

    Handles two common column layouts:
    1. A `translation` dict column: example["translation"][key]
    2. Flat columns named by the key directly: example[key]
    """
    if "translation" in dataset.column_names:
        return [
            (row["translation"][src_key], row["translation"][tgt_key])
            for row in dataset
        ]
    if src_key in dataset.column_names and tgt_key in dataset.column_names:
        return [(row[src_key], row[tgt_key]) for row in dataset]
    raise ValueError(
        f"Cannot find columns for keys '{src_key}' and '{tgt_key}'. "
        f"Available columns: {dataset.column_names}"
    )


# ---------------------------------------------------------------------------
# ADIta_Lad — authentic Italian–Ladin parallel corpus (Val Badia variety)
# ---------------------------------------------------------------------------

def load_adita_lad(
    tokenizer=None,
    hf_repo: str = LADIN_HF_REPO,
    cache_dir: str | None = None,
) -> list[tuple[str, str]]:
    """
    Load ADIta_Lad from HuggingFace.

    Returns cleaned (ita_Latn, lld_Latn) pairs in the canonical order
    (Italian as src, Ladin as tgt); callers that need the reverse direction
    should swap themselves.

    The dataset is sfrontull/lld_valbadia-ita — Val Badia variety, 18,139 pairs.
    Column layout is detected automatically (translation dict or flat columns).
    """
    logger.info("Loading ADIta_Lad from %s", hf_repo)
    ds = load_dataset(hf_repo, split="train", cache_dir=cache_dir)

    # Try common key variants used by sfrontull datasets
    for src_key, tgt_key in [
        ("ita_Latn", "lld_Latn"),
        ("ita", "lld"),
        ("it", "lld"),
    ]:
        try:
            raw = _extract_translation_pairs(ds, src_key, tgt_key)
            logger.info(
                "ADIta_Lad: %d raw pairs loaded (keys: '%s', '%s')",
                len(raw), src_key, tgt_key,
            )
            return preprocess_pairs(raw, tokenizer=tokenizer)
        except (ValueError, KeyError):
            continue

    # Auto-detect keys from the actual translation dict (handles dataset schema drift)
    if "translation" in ds.column_names:
        sample_keys = list(ds[0]["translation"].keys())
        logger.info("ADIta_Lad: auto-detecting keys from translation dict: %s", sample_keys)
        ita_key = next((k for k in sample_keys if "ita" in k.lower() or k == "it"), None)
        lad_key = next((k for k in sample_keys if "lld" in k.lower() or "lad" in k.lower()), None)
        if ita_key and lad_key:
            raw = _extract_translation_pairs(ds, ita_key, lad_key)
            logger.info(
                "ADIta_Lad: %d raw pairs loaded (auto-detected keys: '%s', '%s')",
                len(raw), ita_key, lad_key,
            )
            return preprocess_pairs(raw, tokenizer=tokenizer)

    raise RuntimeError(
        f"Could not parse ADIta_Lad columns from {hf_repo}. "
        f"Columns found: {ds.column_names}. Inspect the dataset and update key variants."
    )


# ---------------------------------------------------------------------------
# SDLad-Ita — synthetic Ladin–Italian (ablation only)
# ---------------------------------------------------------------------------

def load_sdlad_ita(
    tokenizer=None,
    hf_repo: str | None = SYNTH_HF_REPO,
    cache_dir: str | None = None,
) -> list[tuple[str, str]]:
    """
    Load the synthetic SDLad-Ita dataset (ablation only).

    hf_repo must be confirmed before calling this; see SYNTH_HF_REPO in config.py.
    """
    if hf_repo is None:
        raise ValueError(
            "SYNTH_HF_REPO is not set. Locate the SDLad-Ita HF dataset repo "
            "(same paper as ADIta_Lad: arXiv 2509.03962) and set it in config.py."
        )
    logger.info("Loading SDLad-Ita (synthetic) from %s", hf_repo)
    ds = load_dataset(hf_repo, split="train", cache_dir=cache_dir)

    for src_key, tgt_key in [
        ("lld_Latn", "ita_Latn"),
        ("lld", "ita"),
        ("lld", "it"),
    ]:
        try:
            raw = _extract_translation_pairs(ds, src_key, tgt_key)
            logger.info("SDLad-Ita: %d raw pairs (keys: '%s', '%s')", len(raw), src_key, tgt_key)
            return preprocess_pairs(raw, tokenizer=tokenizer)
        except (ValueError, KeyError):
            continue

    raise RuntimeError(
        f"Could not parse SDLad-Ita columns from {hf_repo}. "
        f"Columns: {ds.column_names}."
    )


# ---------------------------------------------------------------------------
# FLORES+ devtest — multi-parallel evaluation set
# ---------------------------------------------------------------------------

def load_flores_devtest(
    langs: list[str] | None = None,
    hf_repo: str = FLORES_HF_REPO,
    cache_dir: str | None = None,
) -> dict[str, list[str]]:
    """
    Load FLORES+ devtest sentences for each language.

    Returns {nllb_lang_code: [sentence, ...]} with 1,012 entries per language,
    aligned across all languages (same index = same source sentence).

    Uses the Val Badia Ladin variety (lld_Latn) to match ADIta_Lad.
    Requires HF login + accepted FLORES+ license (CC-BY-SA, gated).
    """
    if langs is None:
        langs = [LADIN, ITA, FRA, SPA, POR]

    result: dict[str, list[str]] = {}
    for lang in langs:
        logger.info("Loading FLORES+ devtest: %s", lang)
        ds = load_dataset(hf_repo, lang, split="devtest", cache_dir=cache_dir)
        # FLORES+ uses "sentence" in older releases, "text" in openlanguagedata/flores_plus
        text_col = next(
            (c for c in ("sentence", "text") if c in ds.column_names),
            None,
        )
        if text_col is None:
            raise KeyError(
                f"FLORES+ devtest for '{lang}' has no 'sentence' or 'text' column. "
                f"Available columns: {ds.column_names}"
            )
        sentences = [row[text_col] for row in ds]
        result[lang] = sentences
        logger.info("  %s: %d sentences", lang, len(sentences))

    # Sanity check: all languages must have the same number of sentences
    counts = {lang: len(sents) for lang, sents in result.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"FLORES+ devtest sentence counts differ across languages: {counts}. "
            "This indicates a data loading problem."
        )

    logger.info("FLORES+ devtest loaded: %d aligned sentences per language.", list(counts.values())[0])
    return result


# ---------------------------------------------------------------------------
# OPUS-100 bridge pairs
# ---------------------------------------------------------------------------

def load_opus_bridge(
    bridge_lang: str,
    tokenizer=None,
    n: int = SUBSET_SIZE,
    seed: int = SEED,
    cache_dir: str | None = None,
) -> list[tuple[str, str]]:
    """
    Load an OPUS Books bridge corpus ({bridge_lang}↔Italian), subsampled to n pairs.

    bridge_lang must be one of FRA, SPA, POR (NLLB codes).
    Returns cleaned (bridge_text, italian_text) pairs in that order.
    """
    if bridge_lang not in OPUS_CONFIGS:
        raise ValueError(
            f"Unknown bridge language: {bridge_lang}. "
            f"Must be one of {list(OPUS_CONFIGS.keys())}."
        )

    config_name = OPUS_CONFIGS[bridge_lang]
    bridge_key = OPUS_LANG_KEYS[bridge_lang]
    it_key = OPUS_LANG_KEYS[ITA]

    logger.info("Loading OPUS Books bridge corpus: %s (config=%s)", bridge_lang, config_name)
    ds = load_dataset("Helsinki-NLP/opus_books", config_name, split="train", cache_dir=cache_dir)

    raw = _extract_translation_pairs(ds, bridge_key, it_key)
    logger.info("OPUS Books %s: %d raw pairs before preprocessing", config_name, len(raw))

    cleaned = preprocess_pairs(raw, tokenizer=tokenizer)
    subsampled = deterministic_subsample(cleaned, n=n, seed=seed)
    logger.info(
        "OPUS Books %s: %d pairs after preprocessing + subsample to %d",
        config_name, len(cleaned), len(subsampled),
    )
    return subsampled


# ---------------------------------------------------------------------------
# No-leakage assertion
# ---------------------------------------------------------------------------

def assert_no_flores_leakage(
    train_pairs: list[tuple[str, str]],
    flores_sentences: dict[str, list[str]],
) -> None:
    """
    Assert that no FLORES+ devtest sentence appears verbatim in any training pair.

    FLORES+ is eval-only. This check guards against accidental contamination.
    """
    flores_set: set[str] = set()
    for sents in flores_sentences.values():
        flores_set.update(sents)

    leaks = [
        pair for pair in train_pairs
        if pair[0] in flores_set or pair[1] in flores_set
    ]
    if leaks:
        raise AssertionError(
            f"FLORES+ leakage detected! {len(leaks)} training pairs contain "
            "verbatim FLORES+ devtest sentences. Remove them before training."
        )
    logger.info("No-leakage check passed: FLORES+ devtest not found in training data.")
