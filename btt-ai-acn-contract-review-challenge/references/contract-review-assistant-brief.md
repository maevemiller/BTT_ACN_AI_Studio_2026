# Intelligent Contract Review Assistant
### Break Through Tech AI Studio — Fall 2026 Challenge Project
**Industry Partner:** Accenture · **ML Problem Type:** Natural Language Processing (text classification / span extraction) · **Environment:** Free-tier Google Colab

---

## 1. Challenge Overview

Enterprise legal, procurement, and finance teams review tens of thousands of contracts a year, hunting through long documents for the handful of clauses that actually carry risk. The work is slow, expensive, and inconsistent from one reviewer to the next. A reliable automated "first pass" lets human experts spend their time only on the clauses that matter.

In this project, your team will build a machine learning system that reads a commercial contract, locates and labels the key clauses inside it, and flags which ones deserve a human's attention. You will work with real, expert-annotated contract data, confront a genuinely hard NLP problem (the clauses you care about are rare and buried in very long documents), and produce an end-to-end pipeline that an organization could actually use to triage its contract backlog.

The project has two components:

1. **Clause detection (core ML task).** Given a contract, find each important clause — or correctly determine that it is absent — and label it by type. This is a supervised problem with real ground-truth labels, so your work is rigorously measurable with precision, recall, and F1.
2. **Explainable risk scoring (heuristic layer).** On top of the detector, combine a few computable signals into a transparent severity score that sorts each clause into Low / Medium / High. Because the logic is auditable, you can show a stakeholder not just *that* a clause flagged, but *why*.

By December you will have a working pipeline (contract in → labeled clause register with risk tags out), a solid evaluation of the detector, and a calibration of your risk scores against your Accenture advisor's expert judgment. The entire project runs on free Google Colab using open models — no paid APIs required.

---

## 2. Scope

**In scope (core deliverables):**
- Clause detection / classification against real CUAD labels
- An explainable, rule-based risk-scoring layer over the detected clauses
- End-to-end pipeline and evaluation

**Out of scope (stretch goals only — see Section 8):**
- LLM-based plain-language summarization
- A trained (rather than heuristic) risk model or favorability model

Deployment is **not** required.

---

## 3. Business Relevance & Success Criteria

**Business relevance.** Surfacing the small set of clauses that carry risk — liability caps, indemnification, termination, auto-renewal, IP assignment, governing law, change-of-control — and flagging unusual or missing terms is exactly the repetitive expert judgment that ML can accelerate. The artifacts produced here map directly to how AI is being applied in legal-operations and procurement modernization.

**Success criteria.**
- A clause detector evaluated against CUAD gold labels, reported as **per-category precision, recall, and F1**, beating a documented baseline.
- A **risk-scoring layer** that assigns each detected clause Low / Medium / High via transparent, defensible logic.
- A **contract-level risk roll-up** usable for triage (which contracts need a human first).
- A short **validation** of the risk scores against the advisor's expert ranking (calibration, not accuracy — see Section 6).
- A clean, public, **reproducible GitHub repo** that runs on free Colab.

---

## 4. Dataset

**Contract Understanding Atticus Dataset (CUAD) v1** — 510 real commercial contracts with 13,000+ expert annotations across 41 clause categories, manually labeled under attorney supervision. Licensed CC BY 4.0; underlying contracts are public EDGAR filings (no PII or confidentiality concerns). Available in PDF, TXT, a master-clauses CSV, and a SQuAD-2.0-style JSON for span extraction. Published ReadMe/Datasheet documents format and known limitations. Well under 10 GB and machine-readable.

**Access points:**
- Hugging Face (easiest for Colab): https://huggingface.co/datasets/theatticusproject/cuad
- Official page: https://www.atticusprojectai.org/cuad
- Zenodo (DOI archive): https://zenodo.org/records/4595826
- GitHub (data + code + trained model): https://github.com/TheAtticusProject/cuad
- Paper: https://arxiv.org/abs/2103.06268

**The defining characteristic of this data.** The atomic unit is a **(contract, clause-category) pair**, so each contract generates 41 examples. Any given contract contains only a few of the 41 clause types, so the large majority of pairs are *empty* (`is_impossible = true`), and a clause that *is* present may be only a sentence or two inside tens of thousands of tokens. The original authors call this "finding needles in a haystack." Handling this severe class imbalance — at both the pair level and the token level — is the central modeling challenge of the project.

---

## 5. Two Modeling Framings

CUAD supports two ways of framing the detection task. **Start with the first; attempt the second as a stretch.**

**A. Chunk classification (recommended starting point).** Split each contract into sentences or paragraphs and label each chunk with its clause category (or "none"). A training step is: embed the chunk → multi-label classifier → binary cross-entropy. Inference is: classify every chunk → group by predicted label. Gentler on-ramp; the master-clauses CSV makes building chunk labels straightforward. You lose exact span boundaries.

**B. Extractive span prediction (stretch / native format).** Use CUAD's SQuAD-style format directly: for each (contract, category) pair, predict the start/end token positions of the answer span, or "no answer." Requires sliding-window chunking of long contracts and explicit no-answer handling. Heavier, but produces exact clause locations and a stronger final write-up ("we beat our classification baseline with span extraction").

Either framing trains comfortably on a free Colab T4 GPU — the only real training job is fine-tuning a base-size encoder (BERT/RoBERTa/DeBERTa, ~110M params). No large generative model is involved anywhere in the core.

---

## 6. Risk-Scoring Layer (Heuristic, Explainable)

CUAD has **no risk labels** — it tells you *what* a clause is, never whether it is dangerous. Rather than inventing labels, the team builds a transparent severity score from signals that are computable from artifacts they already have. The "model" here is auditable logic, not a trained network — which is itself a virtue in legal tech.

**Structure:** `clause_risk = base_weight[category] × Σ (signal_i × signal_weight_i)`

| Signal | How it's computed | Range | Weight |
|---|---|---|---|
| Category base | Static lookup, advisor-set (high-stakes types score higher) | 1–5 | multiplier |
| Absence flip | Detector found the clause? (protective clauses only — absence *is* the risk) | 0 / 1 | 0.30 |
| Favorability | Directional keyword pass on the span (favors us vs counterparty) | −1 / 0 / +1 | 0.30 |
| Template deviation | Cosine distance between the clause embedding and the category's "standard" centroid | 0–1 | 0.40 |

The three weighted signals form a 0–1 modifier, multiplied by the category base for a per-clause score (~0–5). Bucket: **≤1.0 Low, 1.0–3.0 Medium, >3.0 High.** Roll up to a contract-level score (use `max(clause_scores)` for conservative triage).

**Validation.** Because the score has no ground truth, you cannot compute an F1 on it. Instead, have the Accenture advisor hand-rank ~30 clauses, then check that the score's ordering agrees (Spearman correlation, or bucket-agreement). Frame this as **calibration against expert judgment**, not accuracy. Tuning the weights and running a sensitivity analysis (does the High/Medium boundary move a lot when a weight shifts?) is a meaty, resume-worthy piece of work.

---

## 7. Monthly Milestones & Steps

**August — Kickoff: Business Understanding & Data Access**
- Meet the Accenture advisor; align on the business problem and what a useful output looks like.
- Load CUAD from Hugging Face on Colab; mount Google Drive for checkpointing.
- Read the CUAD datasheet and paper; understand the SQuAD-style format and the (contract, category) pair structure.
- Run a setup-check notebook end-to-end (verify GPU, load and print sample examples).
- Decide the subset of clause categories to target first (e.g., the 8–12 highest-stakes types).

**September — Data Preparation & Exploratory Analysis**
- Clean and normalize contract text; align annotations to text.
- Run EDA to surface the core challenge: clauses-per-contract, the `is_impossible` empty-pair majority, clause-length distribution.
- Build train / validation / test splits (split by contract to avoid leakage).
- Implement the chunking/windowing strategy for the chosen framing.
- Build a deliberately simple baseline (TF-IDF or keyword matcher) with metrics printed — the bar to beat, and proof the pipeline runs on free Colab.

**October — Modeling: Clause Detection**
- Fine-tune a base-size encoder (BERT/RoBERTa/DeBERTa) or sentence-transformer classifier on the chunk-classification framing.
- Address class imbalance: class weighting, negative sampling, or balanced batching.
- Track **per-category precision/recall/F1** (not accuracy — it's misleading under imbalance).
- Do error analysis: which clause types are hard, where false positives/negatives cluster.
- Checkpoint to Drive each epoch to survive Colab session timeouts.

**November — Risk-Scoring Layer & Integration**
- Implement the four risk signals (category base, absence flip, favorability, template deviation).
- Build the category "standard" centroids from clause embeddings across the corpus.
- Assemble the end-to-end pipeline: contract in → detected clauses → risk-scored register out.
- Run the advisor calibration exercise (~30 hand-ranked clauses); compute rank correlation.
- Tune weights; run a sensitivity analysis on the bucket boundaries.

**December — Evaluation, Write-up & Demo**
- Finalize detector metrics versus baseline; finalize risk-score calibration results.
- Produce sample clause registers with risk tags for a few contracts.
- Build a lightweight demo (notebook walkthrough, or a small Streamlit app if time allows).
- Write the final report: methods, results, error analysis, limitations, and an estimate of reviewer time saved per contract.
- Clean and document the public GitHub repo for sharing on GitHub/LinkedIn.

---

## 8. Deliverables

- Clause detection model with per-category precision/recall/F1 against a documented baseline.
- Explainable risk-scoring layer producing Low/Medium/High tags, with documented logic.
- End-to-end pipeline: contract → labeled, risk-tagged clause register.
- Contract-level risk roll-up for triage.
- Evaluation report (detector metrics + risk-score calibration + error analysis + business framing).
- Lightweight demo and a clean, reproducible, public GitHub repo.

---

## 9. Applicable DS / ML / AI Skills

Text preprocessing and tokenization · exploratory data analysis on text · handling severe class imbalance · sentence embeddings · text classification and/or extractive span prediction · transformer fine-tuning (transfer learning) · similarity / deviation scoring with embeddings · evaluation methodology (precision/recall/F1, confusion analysis, calibration against expert ranking, sensitivity analysis) · reproducible Python workflows on Colab · lightweight demo building (Streamlit).

---

## 10. Resources & Environment

The entire core project trains and runs on the **free tier of Google Colab**. Its only training job is fine-tuning a base-size encoder, which the free T4 GPU handles natively. **No LLM and no hosted API are required anywhere in the core**, so the project is fully self-contained on free infrastructure.

Practical notes for the team:
- **Session limits:** free Colab disconnects on idle and caps runtime — checkpoint to Google Drive every epoch.
- **Long contracts:** the cost driver is windowing many overlapping chunks per contract, not model size — keep windowing sane.

---

## 11. Stretch Goals

- **Span extraction:** move from chunk classification to CUAD's native extractive format for exact clause boundaries.
- **Learned favorability signal:** replace the keyword-based favorability heuristic with a small trained classifier.
- **LLM summarization:** add a layer that turns the clause register into a one-page plain-language brief. This is an **inference-only** job — run a 4-bit-quantized small open model (e.g., a 7–8B instruct model) on the free T4; no training and no API key needed. If the team wants to fine-tune the summarizer, **QLoRA / LoRA** makes that feasible on a single T4 as an advanced exercise (a comparison of zero-shot vs fine-tuned summaries is a strong write-up). *Note: any reliance on a hosted LLM API — rather than the open-model path — would require Accenture-provided API keys per program guidance.*

---

## 12. Repository Structure (Suggested)

```
contract-review-assistant/
├── README.md                  ← narrative, scope box, 15-minute quickstart
├── data/README.md             ← how to get CUAD (don't commit the data)
├── notebooks/
│   ├── 00_setup_check.ipynb   ← verify Colab + GPU + data load
│   ├── 01_explore_cuad.ipynb  ← guided EDA
│   └── 02_baseline.ipynb      ← trivial working baseline to beat
├── src/                       ← starter scaffolding (stubs, not solutions)
├── docs/
│   ├── milestones.md
│   ├── risk_scoring.md
│   └── glossary.md            ← contract + ML terms
├── requirements.txt
└── LICENSE
```
