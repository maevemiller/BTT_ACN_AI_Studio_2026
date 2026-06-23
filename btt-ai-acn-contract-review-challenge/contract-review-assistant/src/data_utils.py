"""
Data loading, normalization, and splitting utilities for CUAD.

Key outputs:
    - load_cuad_hf()         -> HuggingFace DatasetDict
    - load_master_clauses()  -> pd.DataFrame (one row per contract×category pair)
    - split_by_contract()    -> train / val / test DataFrames
    - normalize_text()       -> cleaned contract string
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset, DatasetDict


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cuad_hf(cache_dir: str | None = None) -> DatasetDict:
    """Load CUAD from Hugging Face (streams if no local cache)."""
    return load_dataset("theatticusproject/cuad", cache_dir=cache_dir)


def load_master_clauses(csv_path: str | Path) -> pd.DataFrame:
    """
    Load master_clauses.csv into a DataFrame.

    Columns of interest:
        'contract_name', '<CategoryName>_answer', '<CategoryName>_is_impossible', ...
    """
    df = pd.read_csv(csv_path)
    # TODO: melt wide format to long (contract, category, answer, is_impossible)
    return df


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Light normalization: unicode → ASCII, collapse whitespace, strip.
    Preserves sentence structure for downstream chunking.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Train / val / test splits
# ---------------------------------------------------------------------------

def split_by_contract(
    df: pd.DataFrame,
    contract_col: str = "contract_name",
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame by unique contract name to avoid leakage.

    Returns (train_df, val_df, test_df).
    """
    contracts = df[contract_col].unique()
    rng = pd.Series(contracts).sample(frac=1, random_state=random_state)

    n = len(rng)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_contracts = set(rng.iloc[:n_test])
    val_contracts = set(rng.iloc[n_test : n_test + n_val])

    test_df = df[df[contract_col].isin(test_contracts)]
    val_df = df[df[contract_col].isin(val_contracts)]
    train_df = df[~df[contract_col].isin(test_contracts | val_contracts)]

    return train_df, val_df, test_df
