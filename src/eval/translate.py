"""
Batched translation using beam search.

Never loops one sentence at a time — always uses batched generation
for efficiency on the T4 GPU.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
from tqdm import tqdm
from transformers import NllbTokenizer

from src.config import BATCH_SIZE, BEAM_SIZE, MAX_SEQ_LEN

logger = logging.getLogger(__name__)


def _batched(items: list, batch_size: int) -> Iterator[list]:
    """Yield successive batches of size batch_size."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def translate_batch(
    model,
    tokenizer: NllbTokenizer,
    sources: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = BATCH_SIZE,
    beam_size: int = BEAM_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    device: str | None = None,
) -> list[str]:
    """
    Translate a list of source strings from src_lang to tgt_lang.

    Uses batched beam search. Returns hypotheses in the same order as sources.
    Special tokens are stripped from outputs (skip_special_tokens=True).

    forced_bos_token_id is obtained via tokenizer.convert_tokens_to_ids(tgt_lang)
    rather than the deprecated lang_code_to_id attribute.
    """
    if device is None:
        device = next(model.parameters()).device

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Target language code '{tgt_lang}' is not in the tokenizer vocabulary. "
            "Ensure lld_Latn was added via prepare_model_and_tokenizer() before eval."
        )

    model.eval()
    hypotheses: list[str] = []

    with torch.no_grad():
        for batch in tqdm(
            list(_batched(sources, batch_size)),
            desc=f"{src_lang}→{tgt_lang}",
            leave=False,
        ):
            tokenizer.src_lang = src_lang
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            gen_ids = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=beam_size,
                max_length=max_seq_len,
            )

            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            hypotheses.extend(decoded)

    logger.info(
        "translate_batch: %d sentences | %s→%s | beam=%d",
        len(sources), src_lang, tgt_lang, beam_size,
    )
    return hypotheses
