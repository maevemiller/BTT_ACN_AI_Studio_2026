{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  },
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "accelerator": "GPU"
 },
 "cells": [
  {
   "cell_type": "markdown",
   "id": "baseline-header",
   "metadata": {},
   "source": [
    "# 02 — Trivial Baseline\n",
    "\n",
    "**Goal:** Build the simplest possible clause detector and measure it rigorously.\n",
    "\n",
    "This is the **bar your trained model must beat**. A baseline that runs on free Colab in minutes proves the pipeline is wired up end-to-end, gives you concrete F1 numbers to improve on, and prevents you from declaring victory over a weak model later.\n",
    "\n",
    "Approach: TF-IDF vectorization of clause context chunks → logistic regression per category.\n",
    "\n",
    "**Prerequisite:** Run `00_setup_check.ipynb` first."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-setup",
   "metadata": {},
   "source": [
    "## 0. Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "imports",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q datasets scikit-learn pandas\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from datasets import load_dataset\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, f1_score\n",
    "from collections import defaultdict"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-load",
   "metadata": {},
   "source": [
    "## 1. Load and Flatten Data\n",
    "\n",
    "CUAD's training set is already structured as (context, question, answer) triples.\n",
    "We'll treat each row as a binary classification example: does this context contain the named clause?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "load-flatten",
   "metadata": {},
   "outputs": [],
   "source": [
    "cuad = load_dataset(\"theatticusproject/cuad\")\n",
    "\n",
    "def to_dataframe(split):\n",
    "    rows = []\n",
    "    for ex in cuad[split]:\n",
    "        rows.append({\n",
    "            \"title\": ex[\"title\"],\n",
    "            \"question\": ex[\"question\"],\n",
    "            \"context\": ex[\"context\"],\n",
    "            \"label\": int(len(ex[\"answers\"][\"text\"]) > 0),\n",
    "        })\n",
    "    return pd.DataFrame(rows)\n",
    "\n",
    "train_df = to_dataframe(\"train\")\n",
    "test_df  = to_dataframe(\"test\")\n",
    "\n",
    "print(f\"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows\")\n",
    "print(f\"Train class balance: {train_df['label'].mean():.2%} positive\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-target-cats",
   "metadata": {},
   "source": [
    "## 2. Select Target Categories\n",
    "\n",
    "Start with a manageable subset. Expand once the pipeline is working."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "target-cats",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Adjust this list based on advisor input\n",
    "TARGET_CATEGORIES = [\n",
    "    \"Indemnification\",\n",
    "    \"Limitation of Liability\",\n",
    "    \"IP Ownership Assignment\",\n",
    "    \"Termination for Convenience\",\n",
    "    \"Auto-Renewal\",\n",
    "    \"Governing Law\",\n",
    "    \"Change of Control\",\n",
    "]\n",
    "\n",
    "# Filter to only rows whose question contains one of the target keywords\n",
    "def matches_target(question):\n",
    "    return any(t.lower() in question.lower() for t in TARGET_CATEGORIES)\n",
    "\n",
    "train_sub = train_df[train_df[\"question\"].apply(matches_target)].copy()\n",
    "test_sub  = test_df[test_df[\"question\"].apply(matches_target)].copy()\n",
    "\n",
    "print(f\"Filtered train: {len(train_sub):,} | Filtered test: {len(test_sub):,}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-features",
   "metadata": {},
   "source": [
    "## 3. Feature Engineering: TF-IDF on Contract Context"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "tfidf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Concatenate question + context so the model sees what it's looking for\n",
    "train_sub[\"text\"] = train_sub[\"question\"] + \" [SEP] \" + train_sub[\"context\"].str[:2000]\n",
    "test_sub[\"text\"]  = test_sub[\"question\"]  + \" [SEP] \" + test_sub[\"context\"].str[:2000]\n",
    "\n",
    "vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)\n",
    "X_train = vectorizer.fit_transform(train_sub[\"text\"])\n",
    "X_test  = vectorizer.transform(test_sub[\"text\"])\n",
    "\n",
    "y_train = train_sub[\"label\"].values\n",
    "y_test  = test_sub[\"label\"].values\n",
    "\n",
    "print(f\"Feature matrix: {X_train.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-model",
   "metadata": {},
   "source": [
    "## 4. Train Logistic Regression Baseline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "train-lr",
   "metadata": {},
   "outputs": [],
   "source": [
    "# class_weight='balanced' is essential under heavy class imbalance\n",
    "clf = LogisticRegression(\n",
    "    max_iter=1000,\n",
    "    class_weight=\"balanced\",\n",
    "    C=1.0,\n",
    "    solver=\"lbfgs\",\n",
    ")\n",
    "clf.fit(X_train, y_train)\n",
    "print(\"Training complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-eval",
   "metadata": {},
   "source": [
    "## 5. Evaluate: Per-Category Precision / Recall / F1\n",
    "\n",
    "**Do not report accuracy** — it is misleading when the majority class is absent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "overall-eval",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = clf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred, target_names=[\"Absent\", \"Present\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "per-category-eval",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Per-category breakdown\n",
    "results = []\n",
    "for category in TARGET_CATEGORIES:\n",
    "    mask = test_sub[\"question\"].str.contains(category, case=False)\n",
    "    if mask.sum() == 0:\n",
    "        continue\n",
    "    y_t = y_test[mask.values]\n",
    "    y_p = y_pred[mask.values]\n",
    "    f1 = f1_score(y_t, y_p, zero_division=0)\n",
    "    results.append({\"Category\": category, \"N_test\": int(mask.sum()), \"F1\": round(f1, 3)})\n",
    "\n",
    "results_df = pd.DataFrame(results).sort_values(\"F1\", ascending=False)\n",
    "print(results_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-save",
   "metadata": {},
   "source": [
    "## 6. Save Baseline Results\n",
    "\n",
    "Record these numbers. Your fine-tuned model must beat them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "save-results",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save to CSV for later comparison\n",
    "results_df[\"Model\"] = \"TF-IDF + LR (baseline)\"\n",
    "results_df.to_csv(\"baseline_results.csv\", index=False)\n",
    "print(\"Saved to baseline_results.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-next",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "| What we built | A TF-IDF + logistic regression classifier treating each (context, clause-category) pair as a binary presence/absence example. |\n",
    "|---|---|\n",
    "| Key metric | Per-category F1 on the test split. |\n",
    "| Imbalance handling | `class_weight='balanced'` in the LR. |\n",
    "| The bar to beat | The F1 values in `baseline_results.csv`. |\n",
    "\n",
    "**Next steps:** Fine-tune a transformer encoder in a new notebook to beat these F1 scores."
   ]
  }
 ]
}
