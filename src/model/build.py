"""
Load NLLB-200, verify/add lld_Latn, resize embeddings, seed from ita_Latn,
and attach LoRA adapters.

New-language recipe based on David Dale's NLLB extension tutorial (2023) and
its 2025 successor: add the token, resize, seed from the closest high-resource
relative (Italian), keep the new row trainable while LoRA adapts everything else.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from peft import get_peft_model
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from src.config import MODEL_NAME, LADIN, ITA
from src.model.lora import get_lora_config

logger = logging.getLogger(__name__)


def _lld_is_native(tokenizer: NllbTokenizer) -> bool:
    """Return True if lld_Latn already exists in the tokenizer's special tokens."""
    return LADIN in tokenizer.additional_special_tokens


def _seed_embedding_from_ita(
    model: AutoModelForSeq2SeqLM,
    tokenizer: NllbTokenizer,
) -> None:
    """
    Copy the ita_Latn row to lld_Latn in the shared embedding (and in the LM
    head if it is untied). Italian is the closest high-resource relative of
    Ladin and gives a much better starting point than a random vector.
    """
    ita_id = tokenizer.convert_tokens_to_ids(ITA)
    lld_id = tokenizer.convert_tokens_to_ids(LADIN)

    with torch.no_grad():
        shared = model.model.shared.weight.data
        shared[lld_id] = shared[ita_id].clone()

        # Only write to lm_head separately when the weights are not tied
        if model.lm_head.weight.data_ptr() != shared.data_ptr():
            model.lm_head.weight.data[lld_id] = model.lm_head.weight.data[ita_id].clone()
            logger.info("Seeded lld_Latn in shared embedding AND untied lm_head.")
        else:
            logger.info("Seeded lld_Latn in shared embedding (lm_head is tied — no separate update).")


def _register_single_row_grad_hook(
    model: AutoModelForSeq2SeqLM,
    tokenizer: NllbTokenizer,
) -> None:
    """
    Register a gradient hook that zeroes every gradient row in the shared
    embedding except lld_Latn.  This lets us learn the new language code
    without accumulating gradients for all 256k existing tokens (~1 GB).
    """
    lld_id = tokenizer.convert_tokens_to_ids(LADIN)
    embedding = model.model.shared
    embedding.weight.requires_grad_(True)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask[lld_id] = 1.0
        return grad * mask

    embedding.weight.register_hook(_mask_grad)
    logger.info(
        "Gradient hook registered: only lld_Latn row (token_id=%d) "
        "will accumulate a gradient in the shared embedding.",
        lld_id,
    )


def prepare_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    target_modules: list[str] | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
) -> Tuple[AutoModelForSeq2SeqLM, NllbTokenizer]:
    """
    Single entry point for all downstream code.

    Returns (peft_model, tokenizer) with:
    - lld_Latn added (if not already native) and seeded from ita_Latn
    - LoRA adapters attached to attention + FFN projections
    - Only the lld_Latn embedding row trainable from the base model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = NllbTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    ladin_added = False
    if _lld_is_native(tokenizer):
        logger.info("lld_Latn is natively present in %s — no token addition needed.", model_name)
    else:
        logger.info(
            "lld_Latn absent from %s. Adding as new special token, "
            "seeding embedding from ita_Latn (closest high-resource relative).",
            model_name,
        )
        tokenizer.add_special_tokens({"additional_special_tokens": [LADIN]})
        ladin_added = True
        logger.info("Tokenizer vocab size after addition: %d", len(tokenizer))

    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir)

    if ladin_added:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model embeddings to %d tokens.", len(tokenizer))
        _seed_embedding_from_ita(model, tokenizer)

    model = model.to(device)

    lora_kwargs: dict = {}
    if lora_r is not None:
        lora_kwargs["r"] = lora_r
    if lora_alpha is not None:
        lora_kwargs["alpha"] = lora_alpha
    if lora_dropout is not None:
        lora_kwargs["dropout"] = lora_dropout
    if target_modules is not None:
        lora_kwargs["target_modules"] = target_modules

    lora_cfg = get_lora_config(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)

    # Re-enable gradients for the lld_Latn embedding row (PEFT freezes everything)
    if ladin_added:
        _register_single_row_grad_hook(model.base_model.model, tokenizer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %d / %d (%.2f%%)",
        trainable, total, 100.0 * trainable / total,
    )

    return model, tokenizer
