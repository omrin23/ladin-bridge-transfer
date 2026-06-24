# 00 — General Project Instructions (North Star)

This document is the shared context for the whole project. Both the coding work
(see `01_CODING_INSTRUCTIONS.md`) and the paper writing (see
`02_LATEX_REQUIREMENTS.md`) must stay consistent with it. If any later decision
contradicts this file, flag the discrepancy explicitly rather than silently
diverging.

---

## 1. One-line summary

We test whether **intermediate fine-tuning on a closely related Romance language
improves cross-lingual transfer to Ladin** (a low-resource Rhaeto-Romance
language), and whether **measured linguistic similarity predicts which bridge
language helps most** — using NLLB-200-distilled-600M with LoRA.

---

## 2. Research questions & hypotheses

- **RQ1 (does bridging help?).** Does fine-tuning on a related Romance language
  before fine-tuning on Ladin↔Italian beat fine-tuning on Ladin↔Italian directly?
- **RQ2 (is similarity predictive?).** Does a similarity ranking of the bridge
  languages to Ladin predict the ranking of their transfer benefit?

- **H1.** Bridged variants ≥ direct fine-tuning on Ladin↔Italian (BLEU/chrF).
- **H2.** The bridge language closest to Ladin (by our metrics) yields the
  largest gain. **Stated up front as a weak, directional correlate** — with only
  3 bridge languages we have no statistical power for a strong claim.

A **null result is acceptable and publishable** for an 8-page course report
(e.g., "bridging gave no significant gain over direct fine-tuning; here is why").
Do not over-claim.

---

## 3. Scope & constraints

- **Romance languages only.** Bridges are drawn from {French, Spanish,
  Portuguese}. German and other non-Romance contact languages are **out of
  scope** by deliberate design (the project's framing is genealogical proximity
  within Romance, not contact-induced borrowing).
- **No training from scratch.** Base model is pretrained; we only do LoRA
  adapter fine-tuning.
- **Small compute.** Target runtime is a free Google Colab **T4 (16 GB)**.
  NLLB-200-distilled-600M + LoRA fits comfortably; the full run set is a few
  GPU-hours.
- **Workflow.** Claude Code generates the repo locally (on an M1 MacBook, CPU
  only — no local training). Execution happens on **Colab**: push the repo,
  `!pip install -r requirements.txt`, run scripts, checkpoint everything to
  Google Drive (free Colab wipes sessions).
- **Deliverable.** A research-style report, **max 8 pages**, in LaTeX using the
  course-provided Overleaf template, submitted via Piazza.

---

## 4. The Ladin language-code situation (important technical context)

`lld_Latn` is present in **FLORES+** (the OLDI-managed evaluation benchmark) but
is **likely not a native language code in the NLLB-200-distilled-600M model**
(its original 200 languages skew toward African and other low-resource
languages; Ladin was added to FLORES+ later by OLDI).

**Decision:**
1. The code verifies native support first (check the tokenizer's language
   codes for `lld_Latn`).
2. If absent (expected), **add `lld_Latn` as a new language code, resize the
   model embeddings, and seed the new code's embedding from `ita_Latn`**
   (Italian, the closest high-resource relative). This follows the standard
   NLLB new-language recipe (David Dale's tutorial and its 2025 successor).
3. The **zero-shot baseline** therefore translates Ladin using the **Italian
   proxy token** ("treat Ladin as Italian") — a reasonable lower bound, and
   must be **labeled as a proxy** in all results and in the paper.

**Ladin variety must match.** FLORES+ has two Ladin varieties (Val Badia and
Gherdëina). The fine-tuning corpus (ADIta_Lad) corresponds to a specific written
variety — verify which, and evaluate on the **matching** FLORES+ config. Document
the choice; a mismatch silently inflates error.

---

## 5. Languages & their roles

| Language | Code | Role |
|---|---|---|
| Ladin | `lld_Latn` | Target low-resource language (added + seeded from Italian) |
| Italian | `ita_Latn` | Fixed anchor / pivot side; similarity-reference ceiling |
| French | `fra_Latn` | Bridge candidate |
| Spanish | `spa_Latn` | Bridge candidate |
| Portuguese | `por_Latn` | Bridge candidate |

Italian is **not** a bridge condition (bridging Italian→Italian is degenerate);
it is the constant target side and a reference point in the similarity analysis.

---

## 6. Datasets

| Dataset | Pair | Size | Role | License / source |
|---|---|---|---|---|
| ADIta_Lad | Italian–Ladin (authentic) | 18,139 pairs | Primary Ladin fine-tuning data | 2026 Ladin MT paper / HF |
| SDLad-Ita | Ladin–Italian (synthetic) | supplement | Ablation only | same paper; flag as synthetic |
| OPUS subsets | {fr,es,pt}↔it | 18,139 each (subsampled) | Bridge fine-tuning | OPUS (mixed-domain) |
| FLORES+ devtest | multi-parallel | 1,012 sentences | Evaluation (all pairs) | CC-BY-SA; HF login required |

**Size-matching:** all bridge subsets are deterministically subsampled to exactly
18,139 pairs (= ADIta_Lad size) with a fixed seed, for a fair comparison. OPUS is
mixed-domain (subtitles, EU text, etc.), so it will not match ADIta_Lad's domain
— note this as a limitation; the bridge signal is *linguistic*, not topical.

---

## 7. Model

- **Base:** `facebook/nllb-200-distilled-600M`.
- **Adaptation:** LoRA (PEFT) — adapters only; base weights frozen (except the
  resized embedding rows for the new `lld_Latn` code).

---

## 8. Experimental conditions

1. **Zero-shot** (no fine-tuning): NLLB-200 with Ladin-as-Italian proxy, eval
   Ladin↔{it,fr,es,pt}.
2. **Direct** (no bridge): LoRA on ADIta_Lad (Ladin↔Italian), eval same.
3. **Bridge-FR:** LoRA on French↔Italian → continue on ADIta_Lad.
4. **Bridge-ES:** LoRA on Spanish↔Italian → continue on ADIta_Lad.
5. **Bridge-PT:** LoRA on Portuguese↔Italian → continue on ADIta_Lad.
6. **Similarity analysis:** subword overlap + encoder-embedding distance between
   Ladin and each of {it, fr, es, pt}.
7. **Ablation:** ADIta_Lad alone vs. ADIta_Lad + SDLad-Ita, run **only** for the
   Direct condition and the best-performing bridge condition (not all five).

Primary evaluation direction: **Ladin↔Italian, both directions**. Secondary:
Ladin↔{fr,es,pt} (zero-shot generalization only — we have no fine-tuning data
for those targets).

---

## 9. Evaluation

- **Metrics:** BLEU and chrF via `sacrebleu` (report chrF as the more reliable
  metric for a morphologically rich low-resource target; BLEU for comparability).
- **Test set:** FLORES+ **devtest**, 1,012 sentences, multi-parallel across all
  five languages (same sentences). FLORES+ is **eval-only — never used in
  training**.
- **Qualitative:** 25 translations per variant, fixed-seed sample, manually
  labeled by error type (lexical, morphological/agreement, omission,
  hallucination).

---

## 10. Reproducibility principles (apply everywhere)

- Set and log all seeds (`torch`, `numpy`, `random`, `transformers.set_seed`).
- Log every hyperparameter, dataset version, and subset seed at run start.
- Pin all library versions (`requirements.txt`).
- All splits/subsamples are deterministic (fixed seed) — never random without one.
- Every experiment writes a machine-readable results file (schema in
  `01_CODING_INSTRUCTIONS.md`) so analysis and paper tables are mechanical.

---

## 11. Honesty / limitations to carry into the paper

- Ladin is **added**, not native to NLLB-200 → zero-shot baseline is an
  Italian-proxy lower bound, not true zero-shot.
- Only **3 bridge languages** → H2 is a directional trend, not a powered test.
- **SDLad-Ita is synthetic** → used only in the ablation, never as primary
  evidence.
- **Domain mismatch** between OPUS bridge data and ADIta_Lad.
- Small fine-tuning data and a single base model → results may be noisy; report
  variance where feasible (e.g., a second seed for the key conditions if compute
  allows).
