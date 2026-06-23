# Glossary

## Contract Terms

**Auto-renewal clause** — A provision that automatically extends the contract for another term unless a party gives notice by a specified deadline. Absence or short notice windows are common risk flags.

**Change-of-control clause** — Triggers specific rights or obligations (often termination) if one party is acquired or undergoes a significant ownership change.

**Clause** — A discrete provision in a contract that addresses a specific topic (e.g., payment terms, liability limits, governing law).

**Governing law** — Specifies which jurisdiction's law applies to the contract and where disputes must be resolved.

**Indemnification** — An obligation by one party to compensate the other for specified losses, damages, or legal costs. Broad indemnification language is a high-risk flag.

**IP ownership / assignment** — Determines who owns intellectual property created during the engagement. Assignment to the counterparty is high-risk for service providers.

**Limitation of liability** — A cap on the total damages one party can recover from the other. *Absence* of this clause exposes a party to unlimited liability — the absence-flip signal targets exactly this case.

**Termination for convenience** — Allows a party to end the contract without cause, usually with notice. Asymmetric rights (only one party can terminate) are a risk signal.

---

## ML / NLP Terms

**BPE (Byte-Pair Encoding)** — The subword tokenization algorithm used by most modern transformers (BERT, RoBERTa, DeBERTa). Splits rare words into frequent subword units.

**BERT / RoBERTa / DeBERTa** — Base-size (~110M parameter) transformer encoders pre-trained on large text corpora. Fine-tuned for clause classification in this project.

**Chunk classification** — Splitting a document into sentence/paragraph chunks and labeling each chunk with its clause category (or "none"). The recommended starting framing.

**Class imbalance** — When one label appears far more often than others. In CUAD, the vast majority of (contract, category) pairs are empty (`is_impossible = true`). Accuracy is a misleading metric here; use F1.

**Cosine similarity / distance** — Measures the angle between two embedding vectors. Used for template deviation: a clause far from its category centroid in embedding space scores high deviation.

**CUAD** — Contract Understanding Atticus Dataset. 510 contracts, 41 clause categories, expert-annotated. The dataset this project is built on.

**Embedding** — A dense numerical vector representation of text produced by a neural encoder. Captures semantic meaning; used for template-deviation scoring.

**F1 score** — Harmonic mean of precision and recall: `2 × P × R / (P + R)`. The primary evaluation metric for clause detection (per-category).

**Fine-tuning** — Continuing the training of a pre-trained model on a new, smaller dataset. Here: adapting a pre-trained encoder to the clause-classification task.

**is_impossible** — A boolean field in CUAD's SQuAD-style format. `True` means the queried clause category is absent from the contract. The majority class in the dataset.

**(Contract, category) pair** — The atomic unit in CUAD. Each contract generates 41 examples (one per category), most of which are empty.

**Precision** — Of all times the model predicted a clause type, what fraction were correct. `TP / (TP + FP)`.

**Recall** — Of all true clause instances, what fraction did the model find. `TP / (TP + FN)`.

**Sentence transformer** — An encoder fine-tuned with contrastive objectives to produce semantically meaningful sentence-level embeddings (e.g., `all-MiniLM-L6-v2`). Used for chunk embeddings and centroid computation.

**Sliding window** — A chunking strategy that creates overlapping windows of tokens across a long document. Required for span extraction (stretch goal) because transformers have a fixed maximum input length (~512 tokens).

**Span extraction** — Predicting the exact start/end character or token positions of a clause within the contract. Uses CUAD's SQuAD-style format. More precise than chunk classification; the stretch-goal framing.

**Spearman rank correlation** — A non-parametric measure of how well two orderings agree. Used to validate risk scores against the advisor's expert ranking (−1 = inverse, 0 = no relationship, 1 = perfect agreement).

**TF-IDF** — Term Frequency–Inverse Document Frequency. A simple keyword-frequency baseline used in `02_baseline.ipynb` as the bar to beat.
