# 01 — Coding Instructions (Build Spec)

Implementation spec for the pipeline. Read `00_GENERAL_INSTRUCTIONS.md` first for
context, scope, and the locked design decisions. This file defines **what to
build, how to structure it, and the exact output schemas** so that analysis and
the paper are mechanical to produce.

Coding standards: PEP 8 / `black` (≤88 cols), type annotations, docstrings on
every module/class/function, comment *why* not *what*, named constants for all
hyperparameters, explicit edge-case handling. Build a **repo of `.py` modules**,
not one monolithic notebook (a thin `run_colab.ipynb` may orchestrate calls).

---

## 1. Environment

- **Runtime:** Google Colab, single **T4 GPU (16 GB)**. Code must run there
  end-to-end. No local training (M1 CPU only).
- **Pin versions** in `requirements.txt`. Suggested (verify latest-compatible on
  Colab, then pin exact):
  - `torch`, `transformers`, `peft`, `datasets`, `accelerate`,
    `sentencepiece`, `sacrebleu`, `pandas`, `numpy`, `scikit-learn`,
    `matplotlib`.
- **Tokenizer API caution:** the NLLB tokenizer's `lang_code_to_id` is
  deprecated in newer `transformers`. Use `tokenizer.src_lang = ...` and obtain
  the forced BOS id via `tokenizer.convert_tokens_to_ids("<code>")`. **Pin
  `transformers`** to avoid behavior drift, and centralize this in one helper.
- **Colab/Drive:** mount Google Drive; checkpoint models, logs, and **all results
  files** to Drive so a session disconnect never loses results. Re-download
  model/datasets per session is expected — cache to Drive where possible.
- **HF login:** FLORES+ requires accepting terms + `huggingface_hub` login.
  Document this in the README; handle a missing-token error gracefully.

---

## 2. Repository structure

```
ladin-bridge-transfer/
  README.md
  requirements.txt
  configs/
    default.yaml              # all hyperparameters + paths + seeds
  src/
    config.py                 # load/validate config; named constants
    seeding.py                # set_all_seeds(seed)
    languages.py              # lang codes, roles, lld_Latn add+seed recipe
    data/
      loaders.py              # ADIta_Lad, SDLad-Ita, FLORES+, OPUS bridge
      preprocess.py           # normalize, length-filter, dedup
      subsample.py            # deterministic size-matching to 18,139
    model/
      build.py                # load NLLB, add lld_Latn, resize+seed, attach LoRA
      lora.py                 # LoRA config + target modules
    train/
      trainer.py              # single-stage LoRA fine-tune
      pipeline.py             # bridge-stage -> ladin-stage sequencing
    eval/
      translate.py            # batched generation (beam search)
      metrics.py              # BLEU + chrF via sacrebleu
      evaluate.py             # run eval over all pairs, write results JSON
    analysis/
      similarity.py           # subword Jaccard + encoder cosine
      qualitative.py          # sample 25/variant -> CSV for manual error typing
      aggregate.py            # collect results JSONs -> master tables (CSV)
    experiments/
      run_all.py              # orchestrates the full condition matrix
  results/                    # all machine-readable outputs (see schema)
  tests/                      # unit tests
```

---

## 3. Step 0 — Verify Ladin support, then add the language code

Implement in `model/build.py` + `languages.py`:

1. Load `facebook/nllb-200-distilled-600M` model + tokenizer.
2. Check whether `lld_Latn` is a known language code (inspect the tokenizer's
   additional special tokens / language-code set).
3. **If present:** use it directly (and update `00_GENERAL_INSTRUCTIONS.md`
   §4 note — this would be a pleasant surprise).
4. **If absent (expected):**
   - Add `lld_Latn` as a new special language token.
   - Resize model token embeddings (`model.resize_token_embeddings(...)`).
   - **Seed** the new embedding row by copying the `ita_Latn` row (Italian =
     closest relative). Do the same for the LM head row if untied.
   - Log exactly what was added and how it was seeded.
5. Expose a single `prepare_model_and_tokenizer()` that returns a model+tokenizer
   in a known-good state for all downstream code.

Document this as a known constraint with a citation to the NLLB new-language
recipe in code comments.

---

## 4. Data module

`data/loaders.py`, `preprocess.py`, `subsample.py`:

- **Loaders** for: ADIta_Lad (primary, Italian–Ladin), SDLad-Ita (synthetic,
  ablation only), FLORES+ devtest (multi-parallel eval; load `lld_Latn`,
  `ita_Latn`, `fra_Latn`, `spa_Latn`, `por_Latn`), OPUS bridge pairs
  ({fr,es,pt}↔it).
- **Ladin variety check:** confirm which FLORES+ Ladin variety matches ADIta_Lad
  (Val Badia vs. Gherdëina) and load the matching config. Fail loudly if
  ambiguous.
- **Preprocess:** Unicode NFKC normalization; strip control chars; drop empty
  strings; length-filter (drop pairs where either side > `MAX_SEQ_LEN` tokens or
  length ratio is extreme); dedup exact pairs. Handle OOV/edge cases explicitly.
- **Subsample:** deterministically sample exactly `SUBSET_SIZE = 18139` pairs
  from each OPUS bridge corpus with `SEED`. If a bridge corpus is smaller than
  that (it won't be for fr/es/pt↔it — all are high-resource), document and use
  all available.
- **No leakage:** assert FLORES+ devtest sentences never appear in any training
  set.

---

## 5. Tokenization & named constants

Centralize constants in `config.py` / `configs/default.yaml`:

```python
MODEL_NAME      = "facebook/nllb-200-distilled-600M"
MAX_SEQ_LEN     = 128          # FLORES sentences ~21 words; safe upper bound
BATCH_SIZE      = 16           # T4-safe
GRAD_ACCUM      = 2            # effective batch 32
LEARNING_RATE   = 2e-4         # typical LoRA LR (higher than full-FT)
EPOCHS_BRIDGE   = 1            # bridge stage is a light prime
EPOCHS_LADIN    = 3            # small data -> a few passes
BEAM_SIZE       = 5
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
TARGET_MODULES  = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
SEED            = 42
SUBSET_SIZE     = 18139        # == ADIta_Lad size
LADIN, ITA, FRA, SPA, POR = "lld_Latn","ita_Latn","fra_Latn","spa_Latn","por_Latn"
```

For NLLB translation: set `tokenizer.src_lang`, pass
`forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)` to `generate`.
Truncate to `MAX_SEQ_LEN`; document the padding strategy. Tokenize **outside**
the training loop (never re-tokenize per step).

---

## 6. Model / LoRA

`model/build.py`, `model/lora.py`:

- Freeze base weights; attach LoRA to `TARGET_MODULES`.
- The resized embedding rows for `lld_Latn` should remain trainable (this is how
  the new language is actually learned) — be explicit about which params are
  trainable and log the count.
- Be explicit about device (`cuda`) and dtype (fp16 for training/inference on T4;
  guard against fp16 instability if it appears).

---

## 7. Training

`train/trainer.py` (single stage) + `train/pipeline.py` (sequencing):

- **Direct condition:** one stage — LoRA fine-tune on ADIta_Lad (both
  directions: Ladin→Italian and Italian→Ladin in the training data).
- **Bridge conditions:** two stages on the **same adapter** — (1) fine-tune on
  `{bridge}↔Italian` (both directions) for `EPOCHS_BRIDGE`; (2) continue on
  ADIta_Lad for `EPOCHS_LADIN`. Save the adapter after each stage.
- Log full config (all constants above + dataset versions + seed) at the start
  of every run.
- Save a per-run `config.json` next to the adapter for traceability.

---

## 8. Evaluation

`eval/translate.py`, `metrics.py`, `evaluate.py`:

- **Batched** generation (beam = `BEAM_SIZE`); never loop one sentence at a time.
- Decode with `skip_special_tokens=True`.
- Compute **BLEU** and **chrF** with `sacrebleu` (use its default tokenization;
  record the sacrebleu signature for reproducibility).
- Evaluate every condition on FLORES+ devtest for: **lld↔ita** (primary, both
  directions) and **lld↔{fra,spa,por}** (secondary, both directions).
- Write one results JSON per condition (schema below).

---

## 9. Similarity analysis

`analysis/similarity.py` — compute on FLORES+ devtest (1,012 multi-parallel
sentences), between Ladin and each of {ita, fra, spa, por}:

- **Subword overlap:** tokenize each language's devtest with the **NLLB
  tokenizer**; compute **Jaccard similarity** of the sets of subword types
  (Ladin vs. each comparison language). Report per-language.
- **Embedding distance:** encode each devtest sentence with the **NLLB encoder**;
  **mean-pool** the final hidden states (mask padding); compute **cosine
  distance** between Ladin and each comparison language, averaged over the 1,012
  aligned sentence pairs.
- Output a similarity table (`results/similarity.json` + `.csv`) with both
  metrics per language, plus the resulting **similarity ranking** of
  {fra, spa, por} (Italian included as reference).
- This is a **weak directional correlate** — compute Spearman rank correlation
  between the similarity ranking and the transfer-gain ranking, but report it as
  suggestive only (n is tiny).

---

## 10. Qualitative analysis

`analysis/qualitative.py`:

- For each variant, take a **fixed-seed random sample of 25** devtest items
  (lld→ita primary direction; optionally ita→lld too).
- Dump `source, reference, hypothesis` to a CSV per variant for **manual** error
  typing into: `lexical`, `morphological/agreement`, `omission`,
  `hallucination`, `other`. Leave an empty `error_type` column to fill by hand.

---

## 11. Ablation

`experiments/run_all.py` — after the main runs, for **Direct** and the
**best bridge** condition only, run two data settings: (a) ADIta_Lad alone,
(b) ADIta_Lad + SDLad-Ita. Same eval; write results with a `data_setting` field.

---

## 12. Results file schema (exact — do not deviate)

One JSON per evaluated model variant in `results/`, named
`{condition}__{data_setting}.json` (e.g. `bridge_fr__adita_only.json`):

```json
{
  "condition": "bridge_fr",          // zero_shot | direct | bridge_fr|es|pt
  "data_setting": "adita_only",      // adita_only | adita_plus_synth
  "bridge_language": "fra_Latn",     // null for zero_shot/direct
  "base_model": "facebook/nllb-200-distilled-600M",
  "ladin_added_as_new_code": true,
  "ladin_seed_source": "ita_Latn",
  "seed": 42,
  "hyperparameters": { "lora_r": 16, "lora_alpha": 32, "lr": 2e-4,
                       "epochs_bridge": 1, "epochs_ladin": 3,
                       "batch_size": 16, "beam_size": 5, "max_seq_len": 128 },
  "data_sizes": { "ladin_pairs": 18139, "bridge_pairs": 18139,
                  "synthetic_pairs": 0 },
  "sacrebleu_signature": "…",
  "scores": {
    "lld_Latn->ita_Latn": { "bleu": 0.0, "chrf": 0.0 },
    "ita_Latn->lld_Latn": { "bleu": 0.0, "chrf": 0.0 },
    "lld_Latn->fra_Latn": { "bleu": 0.0, "chrf": 0.0 },
    "lld_Latn->spa_Latn": { "bleu": 0.0, "chrf": 0.0 },
    "lld_Latn->por_Latn": { "bleu": 0.0, "chrf": 0.0 }
  },
  "eval_set": "flores_plus_devtest",
  "n_eval_sentences": 1012,
  "timestamp": "ISO-8601"
}
```

`analysis/aggregate.py` collects all these into master CSVs the paper consumes:
- `results/main_results.csv` — condition × direction × {BLEU, chrF}.
- `results/similarity.csv` — language × {jaccard, cosine_distance, rank}.
- `results/ablation.csv` — condition × data_setting × scores.

---

## 13. Testing

`tests/` — unit tests for:
- preprocessing (empty string, max-length, special chars, length-ratio filter),
- tokenization round-trip and correct `forced_bos_token_id` per target,
- metric computation on a tiny known example (BLEU/chrF sanity values),
- subsampling determinism (same seed → same subset),
- results-schema validation (every written JSON matches the schema).

---

## 14. Runtime & compute notes

- Full matrix: zero-shot eval + (direct + 3 bridges) training&eval +
  similarity + ablation (2 extra runs) ≈ a few GPU-hours on T4. Keep
  `EPOCHS_*` small; increase only if under-fitting.
- If a second seed is feasible for the key conditions (direct + best bridge),
  run it to report variance — strengthens the paper.
- Checkpoint adapters + results to Drive after **every** stage.

---

## 15. Do NOT

- Train any model from scratch.
- Put FLORES+ devtest (or its sentences) into any training set.
- Store HF tokens or secrets in the repo.
- Re-tokenize inside the training loop.
- Use a different tokenizer/encoder for the similarity metrics than the NLLB
  model under study (internal consistency is the point).
