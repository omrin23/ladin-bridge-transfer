"""
Single-stage LoRA fine-tuning with HuggingFace Seq2SeqTrainer.

Tokenization is done once before training (never inside the training loop).
Both translation directions are included in every training stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset as HFDataset
from transformers import (
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import (
    BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE,
    MAX_SEQ_LEN, SEED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize_pairs(
    pairs: list[tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    tokenizer: NllbTokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
) -> list[dict]:
    """
    Tokenize a list of (src_text, tgt_text) pairs.

    For NLLB:
    - Source: tokenizer encodes with src_lang BOS prepended.
    - Target labels: we temporarily set src_lang = tgt_lang so that the
      tokenizer prepends the target language ID (the forced-BOS trick used
      throughout NLLB fine-tuning).

    Padding is intentionally omitted here; DataCollatorForSeq2Seq pads per batch.
    """
    tokenizer.src_lang = src_lang
    src_enc = tokenizer(
        [p[0] for p in pairs],
        max_length=max_seq_len,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    # Temporarily use tgt_lang as src_lang to get target language BOS in labels
    tokenizer.src_lang = tgt_lang
    tgt_enc = tokenizer(
        [p[1] for p in pairs],
        max_length=max_seq_len,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    tokenizer.src_lang = src_lang  # restore

    records = []
    for i in range(len(pairs)):
        records.append({
            "input_ids": src_enc["input_ids"][i],
            "attention_mask": src_enc["attention_mask"][i],
            "labels": tgt_enc["input_ids"][i],
        })
    return records


def build_tokenized_dataset(
    pairs: list[tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    tokenizer: NllbTokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
    bidirectional: bool = True,
) -> HFDataset:
    """
    Build a HF Dataset from translation pairs, tokenized for seq2seq training.

    When bidirectional=True, both (src→tgt) and (tgt→src) examples are included,
    doubling the effective dataset size and teaching the model both directions.
    """
    records = _tokenize_pairs(pairs, src_lang, tgt_lang, tokenizer, max_seq_len)
    if bidirectional:
        reverse = _tokenize_pairs(
            [(tgt, src) for src, tgt in pairs],
            src_lang=tgt_lang,
            tgt_lang=src_lang,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        records = records + reverse
        logger.info("Bidirectional dataset: %d pairs → %d examples", len(pairs), len(records))
    else:
        logger.info("Unidirectional dataset: %d examples", len(records))

    return HFDataset.from_list(records)


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    model,
    tokenizer: NllbTokenizer,
    train_dataset: HFDataset,
    output_dir: str | Path,
    num_train_epochs: int,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    fp16: bool | None = None,
) -> Seq2SeqTrainer:
    """Return a Seq2SeqTrainer configured for LoRA fine-tuning."""
    if fp16 is None:
        # Model is loaded in fp16; PEFT handles dtype casting internally.
        # Enabling the trainer's own GradScaler (fp16=True) causes it to try
        # unscaling fp16 gradients, which PyTorch forbids. Keep it off.
        fp16 = False

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=fp16,
        predict_with_generate=False,  # generation only needed at eval, not training
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        seed=seed,
        report_to="none",
        dataloader_drop_last=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )


# ---------------------------------------------------------------------------
# Top-level training function
# ---------------------------------------------------------------------------

def train_stage(
    model,
    tokenizer: NllbTokenizer,
    pairs: list[tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    output_dir: str | Path,
    num_epochs: int,
    stage_name: str = "stage",
    run_config: dict | None = None,
    **trainer_kwargs,
) -> None:
    """
    Fine-tune model on pairs for num_epochs and save the adapter to output_dir.

    Saves:
    - The LoRA adapter weights
    - A config.json capturing all run metadata for traceability
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage: %s | epochs=%d | src=%s tgt=%s ===", stage_name, num_epochs, src_lang, tgt_lang)

    train_dataset = build_tokenized_dataset(pairs, src_lang, tgt_lang, tokenizer)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=output_dir / "checkpoints",
        num_train_epochs=num_epochs,
        **trainer_kwargs,
    )

    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("Adapter saved to %s", adapter_path)

    if run_config:
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)
        logger.info("Run config saved to %s", config_path)
