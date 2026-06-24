"""
Linguistic similarity between Ladin and {Italian, French, Spanish, Portuguese}:

1. Subword overlap — Jaccard similarity of NLLB subword-type sets across FLORES+ devtest.
2. Embedding distance — mean cosine distance between NLLB encoder representations,
   averaged over the 1,012 aligned FLORES+ devtest sentence pairs.

Both metrics intentionally use the NLLB model under study (internal consistency).
A Spearman rank correlation between similarity rank and transfer-gain rank is
computed as a directional check, not a powered statistical test (n=3 bridge langs).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import NllbTokenizer

from src.config import LADIN, ITA, FRA, SPA, POR, BATCH_SIZE

logger = logging.getLogger(__name__)

COMPARISON_LANGS: tuple[str, ...] = (ITA, FRA, SPA, POR)


# ---------------------------------------------------------------------------
# Subword Jaccard
# ---------------------------------------------------------------------------

def _subword_type_set(
    sentences: list[str],
    lang: str,
    tokenizer: NllbTokenizer,
) -> set[int]:
    """Return the set of unique subword token IDs across all sentences for a language."""
    tokenizer.src_lang = lang
    token_ids: set[int] = set()
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        token_ids.update(ids)
    return token_ids


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_subword_jaccard(
    flores_sentences: dict[str, list[str]],
    tokenizer: NllbTokenizer,
) -> dict[str, float]:
    """
    Compute Jaccard similarity between Ladin's subword type set and each
    comparison language's subword type set, using FLORES+ devtest.
    """
    ladin_types = _subword_type_set(flores_sentences[LADIN], LADIN, tokenizer)
    logger.info("Ladin subword type set size: %d", len(ladin_types))

    results: dict[str, float] = {}
    for lang in COMPARISON_LANGS:
        if lang not in flores_sentences:
            logger.warning("Skipping Jaccard for %s — not in FLORES+ data", lang)
            continue
        lang_types = _subword_type_set(flores_sentences[lang], lang, tokenizer)
        j = jaccard_similarity(ladin_types, lang_types)
        results[lang] = round(j, 6)
        logger.info("Jaccard(%s, %s) = %.4f", LADIN, lang, j)

    return results


# ---------------------------------------------------------------------------
# Encoder cosine distance
# ---------------------------------------------------------------------------

def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the last hidden states, masking padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_sentences(
    model,
    tokenizer: NllbTokenizer,
    sentences: list[str],
    lang: str,
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """
    Encode sentences with the NLLB encoder; return (N, hidden_dim) array
    of mean-pooled representations.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    tokenizer.src_lang = lang
    all_embeds: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            # Use the encoder only (no decoder pass needed)
            encoder_out = model.model.encoder(**enc)
            pooled = _mean_pool(encoder_out.last_hidden_state, enc["attention_mask"])
            all_embeds.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeds, axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine *distance* (1 - cosine similarity) between paired rows."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cos_sim = (a_norm * b_norm).sum(axis=1)
    return float(round(1.0 - cos_sim.mean(), 6))


def compute_cosine_distances(
    model,
    flores_sentences: dict[str, list[str]],
    tokenizer: NllbTokenizer,
) -> dict[str, float]:
    """
    Compute mean cosine distance between Ladin and each comparison language
    over aligned FLORES+ devtest sentence pairs.
    """
    ladin_embeds = _encode_sentences(model, tokenizer, flores_sentences[LADIN], LADIN)
    logger.info("Encoded Ladin: %s", ladin_embeds.shape)

    results: dict[str, float] = {}
    for lang in COMPARISON_LANGS:
        if lang not in flores_sentences:
            logger.warning("Skipping cosine distance for %s — not in FLORES+ data", lang)
            continue
        lang_embeds = _encode_sentences(model, tokenizer, flores_sentences[lang], lang)
        dist = cosine_distance(ladin_embeds, lang_embeds)
        results[lang] = dist
        logger.info("Cosine distance(%s, %s) = %.4f", LADIN, lang, dist)

    return results


# ---------------------------------------------------------------------------
# Rank + Spearman correlation
# ---------------------------------------------------------------------------

def rank_by_similarity(
    jaccard: dict[str, float],
    cosine_dist: dict[str, float],
) -> dict[str, int]:
    """
    Rank bridge languages {FRA, SPA, POR} by similarity to Ladin.

    Higher Jaccard = more similar. Lower cosine distance = more similar.
    We average normalised scores to produce a combined rank (rank 1 = most similar).
    Italian is excluded from ranking (it is the reference, not a bridge candidate).
    """
    bridge_langs = [FRA, SPA, POR]

    j_vals = np.array([jaccard.get(l, 0.0) for l in bridge_langs])
    d_vals = np.array([cosine_dist.get(l, 1.0) for l in bridge_langs])

    # Normalize to [0, 1] (higher = more similar for both after inverting distance)
    j_norm = (j_vals - j_vals.min()) / (j_vals.max() - j_vals.min() + 1e-9)
    d_norm = 1.0 - (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-9)
    combined = (j_norm + d_norm) / 2.0

    order = np.argsort(-combined)  # descending: most similar first
    ranks = {bridge_langs[i]: int(np.where(order == i)[0][0]) + 1 for i in range(len(bridge_langs))}
    return ranks


def compute_spearman(
    similarity_ranks: dict[str, int],
    transfer_gains: dict[str, float],
) -> float | None:
    """
    Spearman ρ between similarity rank and transfer gain for bridge languages.

    Returns None if fewer than 3 data points (cannot compute).
    Reported as a weak directional correlate only — n=3 has no statistical power.
    """
    langs = [l for l in [FRA, SPA, POR] if l in similarity_ranks and l in transfer_gains]
    if len(langs) < 3:
        logger.warning("Cannot compute Spearman: only %d bridge languages available.", len(langs))
        return None

    sim_r = [similarity_ranks[l] for l in langs]
    gain_r = [transfer_gains[l] for l in langs]
    rho, _ = spearmanr(sim_r, gain_r)
    logger.info("Spearman ρ (similarity vs. transfer gain): %.4f  [n=%d, interpretive only]", rho, len(langs))
    return float(round(rho, 4))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_similarity_analysis(
    model,
    tokenizer: NllbTokenizer,
    flores_sentences: dict[str, list[str]],
    results_dir: str | Path,
    transfer_gains: dict[str, float] | None = None,
) -> dict:
    """
    Run both similarity metrics, compute ranks, optionally compute Spearman,
    and write results/similarity.json + .csv.

    transfer_gains: {lang_code: chrf_gain_over_direct} — provide after eval.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    jaccard = compute_subword_jaccard(flores_sentences, tokenizer)
    cosine_dist = compute_cosine_distances(model, flores_sentences, tokenizer)
    ranks = rank_by_similarity(jaccard, cosine_dist)

    spearman = None
    if transfer_gains is not None:
        spearman = compute_spearman(ranks, transfer_gains)

    output = {
        "jaccard": jaccard,
        "cosine_distance": cosine_dist,
        "similarity_rank": ranks,
        "spearman_rho_similarity_vs_transfer": spearman,
        "note": (
            "Spearman rho is a weak directional check only — n=3 bridge languages "
            "has no statistical power for a definitive claim."
        ),
    }

    json_path = results_dir / "similarity.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Similarity results written to %s", json_path)

    # Also write CSV
    import pandas as pd
    rows = []
    for lang in COMPARISON_LANGS:
        rows.append({
            "language": lang,
            "jaccard": jaccard.get(lang),
            "cosine_distance": cosine_dist.get(lang),
            "similarity_rank": ranks.get(lang),
        })
    pd.DataFrame(rows).to_csv(results_dir / "similarity.csv", index=False)

    return output
