# Monthly Milestones

## August — Kickoff: Business Understanding & Data Access

- [ ] Meet Accenture advisor; align on business problem and useful output definition
- [ ] Load CUAD from Hugging Face on Colab; mount Google Drive for checkpointing
- [ ] Read CUAD datasheet and paper; understand SQuAD-style format and (contract, category) pair structure
- [ ] Run `00_setup_check.ipynb` end-to-end (verify GPU, load and print sample examples)
- [ ] Decide the subset of clause categories to target first (suggested: 8–12 highest-stakes types)

**Deliverable:** Setup notebook runs cleanly; team has a shared understanding of the data format and the target clause categories.

---

## September — Data Preparation & Exploratory Analysis

- [ ] Clean and normalize contract text; align annotations to text spans
- [ ] Run EDA (`01_explore_cuad.ipynb`) to surface the core challenge: clauses-per-contract, `is_impossible` majority, clause-length distribution
- [ ] Build train / validation / test splits (split by contract to avoid leakage)
- [ ] Implement chunking / windowing strategy for chosen framing (sentence or paragraph)
- [ ] Build deliberately simple baseline in `02_baseline.ipynb` (TF-IDF or keyword matcher) with metrics printed

**Deliverable:** EDA notebook with key imbalance statistics; baseline F1 per category documented as the bar to beat.

---

## October — Modeling: Clause Detection

- [ ] Fine-tune a base-size encoder (BERT / RoBERTa / DeBERTa) or sentence-transformer classifier on chunk-classification framing
- [ ] Address class imbalance: class weighting, negative sampling, or balanced batching
- [ ] Track per-category precision / recall / F1 (not accuracy — misleading under imbalance)
- [ ] Error analysis: which clause types are hard, where false positives / negatives cluster
- [ ] Checkpoint to Drive each epoch to survive Colab session timeouts

**Deliverable:** Fine-tuned model with per-category metrics table; error analysis write-up.

---

## November — Risk-Scoring Layer & Integration

- [ ] Implement four risk signals: category base weight, absence flip, favorability keyword pass, template deviation (cosine distance to category centroid)
- [ ] Build category "standard" centroids from clause embeddings across the corpus
- [ ] Assemble end-to-end pipeline: contract in → detected clauses → risk-scored register out
- [ ] Run advisor calibration exercise (~30 hand-ranked clauses); compute Spearman rank correlation
- [ ] Tune weights; run sensitivity analysis on bucket boundaries

**Deliverable:** Working pipeline; calibration results vs. advisor ranking; sensitivity analysis.

---

## December — Evaluation, Write-up & Demo

- [ ] Finalize detector metrics versus baseline
- [ ] Finalize risk-score calibration results
- [ ] Produce sample clause registers with risk tags for a few contracts
- [ ] Build lightweight demo (notebook walkthrough or Streamlit app)
- [ ] Write final report: methods, results, error analysis, limitations, estimated reviewer time saved
- [ ] Clean and document the public GitHub repo for sharing

**Deliverable:** Final report; demo; clean public repo.
