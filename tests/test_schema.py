"""Unit tests for results JSON schema validation."""

import json
import tempfile
from pathlib import Path
import pytest


REQUIRED_TOP_LEVEL_FIELDS = [
    "condition",
    "data_setting",
    "bridge_language",
    "base_model",
    "ladin_added_as_new_code",
    "ladin_seed_source",
    "seed",
    "hyperparameters",
    "data_sizes",
    "sacrebleu_signature",
    "scores",
    "eval_set",
    "n_eval_sentences",
    "timestamp",
]

REQUIRED_HYPERPARAMETER_FIELDS = [
    "lora_r", "lora_alpha", "lr",
    "epochs_bridge", "epochs_ladin",
    "batch_size", "beam_size", "max_seq_len",
]

REQUIRED_DATA_SIZE_FIELDS = ["ladin_pairs", "bridge_pairs", "synthetic_pairs"]

REQUIRED_SCORE_DIRECTIONS = [
    "lld_Latn->ita_Latn",
    "ita_Latn->lld_Latn",
    "lld_Latn->fra_Latn",
    "lld_Latn->spa_Latn",
    "lld_Latn->por_Latn",
]


def _make_valid_result() -> dict:
    return {
        "condition": "direct",
        "data_setting": "adita_only",
        "bridge_language": None,
        "base_model": "facebook/nllb-200-distilled-600M",
        "ladin_added_as_new_code": True,
        "ladin_seed_source": "ita_Latn",
        "seed": 42,
        "hyperparameters": {
            "lora_r": 16, "lora_alpha": 32, "lr": 2e-4,
            "epochs_bridge": 1, "epochs_ladin": 3,
            "batch_size": 16, "beam_size": 5, "max_seq_len": 128,
        },
        "data_sizes": {"ladin_pairs": 18139, "bridge_pairs": 0, "synthetic_pairs": 0},
        "sacrebleu_signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3",
        "scores": {
            "lld_Latn->ita_Latn": {"bleu": 12.5, "chrf": 34.2},
            "ita_Latn->lld_Latn": {"bleu": 8.3, "chrf": 28.1},
            "lld_Latn->fra_Latn": {"bleu": 5.1, "chrf": 20.4},
            "lld_Latn->spa_Latn": {"bleu": 4.8, "chrf": 19.7},
            "lld_Latn->por_Latn": {"bleu": 4.2, "chrf": 18.9},
        },
        "eval_set": "flores_plus_devtest",
        "n_eval_sentences": 1012,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def validate_result_schema(result: dict) -> list[str]:
    """Return a list of schema violation messages (empty = valid)."""
    errors = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in result:
            errors.append(f"Missing top-level field: '{field}'")

    if "hyperparameters" in result:
        for field in REQUIRED_HYPERPARAMETER_FIELDS:
            if field not in result["hyperparameters"]:
                errors.append(f"Missing hyperparameter: '{field}'")

    if "data_sizes" in result:
        for field in REQUIRED_DATA_SIZE_FIELDS:
            if field not in result["data_sizes"]:
                errors.append(f"Missing data_sizes field: '{field}'")

    if "scores" in result:
        for direction in REQUIRED_SCORE_DIRECTIONS:
            if direction not in result["scores"]:
                errors.append(f"Missing score direction: '{direction}'")
            else:
                for metric in ["bleu", "chrf"]:
                    if metric not in result["scores"][direction]:
                        errors.append(f"Missing metric '{metric}' for direction '{direction}'")

    return errors


def test_valid_schema_passes():
    result = _make_valid_result()
    errors = validate_result_schema(result)
    assert errors == [], f"Schema errors: {errors}"


def test_missing_field_detected():
    result = _make_valid_result()
    del result["seed"]
    errors = validate_result_schema(result)
    assert any("seed" in e for e in errors)


def test_missing_score_direction_detected():
    result = _make_valid_result()
    del result["scores"]["lld_Latn->ita_Latn"]
    errors = validate_result_schema(result)
    assert any("lld_Latn->ita_Latn" in e for e in errors)


def test_written_json_is_valid():
    """Test that a JSON written to disk round-trips and validates."""
    result = _make_valid_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "direct__adita_only.json"
        with open(p, "w") as f:
            json.dump(result, f)
        with open(p) as f:
            loaded = json.load(f)
    errors = validate_result_schema(loaded)
    assert errors == []


def test_condition_values():
    valid_conditions = {"zero_shot", "direct", "bridge_fr", "bridge_es", "bridge_pt"}
    result = _make_valid_result()
    assert result["condition"] in valid_conditions


def test_data_setting_values():
    valid_settings = {"adita_only", "adita_plus_synth"}
    result = _make_valid_result()
    assert result["data_setting"] in valid_settings
