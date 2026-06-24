"""LoRA configuration for NLLB seq2seq fine-tuning."""

from __future__ import annotations

from peft import LoraConfig, TaskType

from src.config import LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES


def get_lora_config(
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Return a LoRA config targeting NLLB's attention and FFN projection layers."""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
