# Data — CUAD v1

**The `data/` directory is intentionally empty in this repository.** Contract files and annotation CSVs are not committed here. Follow the instructions below to obtain the data.

---

## About CUAD

**Contract Understanding Atticus Dataset (CUAD) v1**
- 510 real commercial contracts sourced from public EDGAR filings
- 13,000+ expert annotations across **41 clause categories**, labeled under attorney supervision
- License: **CC BY 4.0**
- No PII or confidentiality concerns (all filings are public)

Formats available:
| Format | Use |
|---|---|
| `master_clauses.csv` | One row per (contract, category) pair — easiest for chunk classification |
| `cuad_v1.json` | SQuAD 2.0-style, with answer spans — use for span extraction (stretch goal) |
| Individual `.txt` files | Raw contract text |

---

## Access Options

### Option 1 — Hugging Face (recommended for Colab)

```python
from datasets import load_dataset
ds = load_dataset("theatticusproject/cuad")
```

No download required; streams directly into your Colab session.

### Option 2 — Zenodo (DOI-archived, full package)

```
https://zenodo.org/records/4595826
```

Download `CUAD_v1.zip` (~1 GB). Unzip into this `data/` directory.

### Option 3 — Official GitHub

```
git clone https://github.com/TheAtticusProject/cuad
```

Includes data, the original training code, and a pre-trained model checkpoint.

---

## Expected Local Layout (after download)

```
data/
├── README.md                  ← this file
├── master_clauses.csv         ← primary annotation table
├── cuad_v1.json               ← SQuAD-style spans (stretch goal)
└── full_contract_txt/         ← one .txt file per contract
    ├── ABILITYINC_...txt
    └── ...
```

---

## .gitignore note

`*.csv`, `*.json`, and `full_contract_txt/` are excluded from version control via `.gitignore`. Never commit raw contract files to this repository.
