"""
Two-stage training pipeline for bridge conditions.

Stage 1: Fine-tune on {bridge_lang}↔Italian (both directions) — EPOCHS_BRIDGE passes.
Stage 2: Continue on the same adapter with ADIta_Lad — EPOCHS_LADIN passes.

The same LoRA adapter object is mutated across both stages (no reload between them).
The adapter is saved after each stage for checkpointing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import (
    EPOCHS_BRIDGE, EPOCHS_LADIN, ITA, LADIN, SEED,
    BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, MAX_SEQ_LEN,
    SUBSET_SIZE,
)
from src.languages import BRIDGE_CONDITION_MAP
from src.train.trainer import train_stage

logger = logging.getLogger(__name__)


def run_direct(
    model,
    tokenizer,
    ladin_pairs: list[tuple[str, str]],
    output_dir: str | Path,
    run_config: dict | None = None,
    epochs: int = EPOCHS_LADIN,
) -> None:
    """
    Direct condition: one-stage LoRA fine-tune on ADIta_Lad only.

    pairs are (ita_Latn, lld_Latn) — both directions are used in training.
    """
    logger.info("=== DIRECT CONDITION ===")
    train_stage(
        model=model,
        tokenizer=tokenizer,
        pairs=ladin_pairs,
        src_lang=ITA,
        tgt_lang=LADIN,
        output_dir=output_dir,
        num_epochs=epochs,
        stage_name="direct",
        run_config=run_config,
    )


def run_bridge(
    model,
    tokenizer,
    bridge_pairs: list[tuple[str, str]],
    ladin_pairs: list[tuple[str, str]],
    bridge_lang: str,
    output_dir: str | Path,
    run_config: dict | None = None,
    epochs_bridge: int = EPOCHS_BRIDGE,
    epochs_ladin: int = EPOCHS_LADIN,
) -> None:
    """
    Bridge condition: two-stage fine-tune on the same LoRA adapter.

    Stage 1 — bridge corpus ({bridge_lang}↔Italian) for epochs_bridge passes.
    Stage 2 — ADIta_Lad (Ladin↔Italian) for epochs_ladin passes.

    bridge_pairs are (bridge_lang_text, ita_text).
    ladin_pairs are (ita_text, lld_text).
    """
    output_dir = Path(output_dir)
    condition_label = f"bridge_{bridge_lang.split('_')[0].lower()[:2]}"
    logger.info("=== BRIDGE CONDITION: %s ===", bridge_lang)

    # Stage 1: bridge language ↔ Italian
    stage1_dir = output_dir / "stage1_bridge"
    train_stage(
        model=model,
        tokenizer=tokenizer,
        pairs=bridge_pairs,
        src_lang=bridge_lang,
        tgt_lang=ITA,
        output_dir=stage1_dir,
        num_epochs=epochs_bridge,
        stage_name=f"{condition_label}_stage1",
        run_config=run_config,
    )

    # Stage 2: Ladin ↔ Italian (continuing the same adapter)
    stage2_dir = output_dir / "stage2_ladin"
    train_stage(
        model=model,
        tokenizer=tokenizer,
        pairs=ladin_pairs,
        src_lang=ITA,
        tgt_lang=LADIN,
        output_dir=stage2_dir,
        num_epochs=epochs_ladin,
        stage_name=f"{condition_label}_stage2",
        run_config=run_config,
    )
