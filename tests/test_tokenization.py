"""
Unit tests for tokenization correctness and forced_bos_token_id resolution.

These tests use the real NLLB tokenizer (downloaded on first run, ~several MB).
They verify the exact API we rely on throughout the codebase.
"""

import pytest


def _load_tokenizer():
    """Load the NLLB tokenizer; skip tests if transformers is not installed."""
    try:
        from transformers import NllbTokenizer
        return NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    except Exception as e:
        pytest.skip(f"Could not load NLLB tokenizer: {e}")


def test_ita_latn_is_native():
    """Italian should be a native language code in NLLB-200."""
    tok = _load_tokenizer()
    assert "ita_Latn" in tok.additional_special_tokens


def test_lld_latn_not_native():
    """Ladin (lld_Latn) should NOT be a native NLLB-200 language code."""
    tok = _load_tokenizer()
    # This is the expected state before we add it; if it passes, update docs.
    assert "lld_Latn" not in tok.additional_special_tokens


def test_add_lld_latn_token():
    """After adding lld_Latn, convert_tokens_to_ids should return a valid (non-UNK) id."""
    tok = _load_tokenizer()
    unk_id = tok.unk_token_id
    tok.add_special_tokens({"additional_special_tokens": ["lld_Latn"]})
    lld_id = tok.convert_tokens_to_ids("lld_Latn")
    assert lld_id != unk_id
    assert isinstance(lld_id, int)


def test_forced_bos_via_convert_tokens_to_ids():
    """
    Verify that forced_bos_token_id for a known language is obtained via
    convert_tokens_to_ids, not the deprecated lang_code_to_id attribute.
    """
    tok = _load_tokenizer()
    ita_id = tok.convert_tokens_to_ids("ita_Latn")
    assert isinstance(ita_id, int)
    assert ita_id != tok.unk_token_id


def test_src_lang_setting_changes_tokenization():
    """Setting src_lang should change which language BOS is prepended."""
    tok = _load_tokenizer()
    text = "Hello world"
    tok.src_lang = "fra_Latn"
    ids_fr = tok.encode(text)
    tok.src_lang = "ita_Latn"
    ids_it = tok.encode(text)
    # First token should differ (it is the language BOS)
    assert ids_fr[0] != ids_it[0]


def test_tokenize_and_decode_roundtrip():
    """Tokenize → decode should approximately recover the original string."""
    tok = _load_tokenizer()
    tok.src_lang = "ita_Latn"
    text = "Il gatto dorme sul tappeto."
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    # Round-trip won't be perfect due to subword tokenization, but
    # must reconstruct the text when detokenized
    assert decoded.strip() == text.strip()
