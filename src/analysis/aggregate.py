"""
Collect all results JSONs and produce master CSV tables:
  - results/main_results.csv  — condition × direction × {BLEU, chrF}
  - results/ablation.csv      — condition × data_setting × scores
  - results/similarity.csv    — already written by similarity.py (no-op here)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _load_all_results(results_dir: Path) -> list[dict]:
    """Load every *.json in results_dir (excluding similarity.json)."""
    jsons = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name in ("similarity.json",):
            continue
        with open(p) as f:
            jsons.append(json.load(f))
    logger.info("Loaded %d result JSON files from %s", len(jsons), results_dir)
    return jsons


def build_main_results(results: list[dict]) -> pd.DataFrame:
    """
    Flatten results JSONs into a tidy long-form table:
    condition | data_setting | direction | bleu | chrf
    """
    rows = []
    for r in results:
        condition = r.get("condition", "unknown")
        data_setting = r.get("data_setting", "unknown")
        for direction, scores in r.get("scores", {}).items():
            rows.append({
                "condition": condition,
                "data_setting": data_setting,
                "direction": direction,
                "bleu": scores.get("bleu"),
                "chrf": scores.get("chrf"),
            })
    return pd.DataFrame(rows)


def build_ablation_table(results: list[dict]) -> pd.DataFrame:
    """
    Filter to ablation-relevant rows (direct + best bridge, both data_settings).
    """
    ablation_conditions = {"direct", "bridge_fr", "bridge_es", "bridge_pt"}
    ablation_rows = [
        r for r in results
        if r.get("condition") in ablation_conditions
        and r.get("data_setting") in {"adita_only", "adita_plus_synth"}
    ]
    return build_main_results(ablation_rows)


def aggregate(results_dir: str | Path) -> None:
    """
    Read all result JSONs and write main_results.csv and ablation.csv.

    Idempotent — safe to call multiple times as results accumulate.
    """
    results_dir = Path(results_dir)
    all_results = _load_all_results(results_dir)

    if not all_results:
        logger.warning("No result JSON files found in %s", results_dir)
        return

    main_df = build_main_results(all_results)
    main_path = results_dir / "main_results.csv"
    main_df.to_csv(main_path, index=False)
    logger.info("main_results.csv written (%d rows) → %s", len(main_df), main_path)

    ablation_df = build_ablation_table(all_results)
    ablation_path = results_dir / "ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    logger.info("ablation.csv written (%d rows) → %s", len(ablation_df), ablation_path)
