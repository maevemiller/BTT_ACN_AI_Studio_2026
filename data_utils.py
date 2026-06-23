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
   "id": "eda-header",
   "metadata": {},
   "source": [
    "# 01 — Exploratory Data Analysis: CUAD\n",
    "\n",
    "**Goal:** Build intuition for the data before writing any model code.\n",
    "\n",
    "By the end of this notebook you should be able to answer:\n",
    "- How many contracts are in the dataset, and how long are they?\n",
    "- Which clause categories appear most / least often?\n",
    "- How severe is the class imbalance?\n",
    "- What does a typical clause excerpt look like vs. a missing clause?\n",
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
    "!pip install -q datasets pandas matplotlib seaborn\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "from datasets import load_dataset\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "load-data",
   "metadata": {},
   "outputs": [],
   "source": [
    "cuad = load_dataset(\"theatticusproject/cuad\")\n",
    "train = cuad[\"train\"]\n",
    "print(f\"Train: {len(train):,} examples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-contracts",
   "metadata": {},
   "source": [
    "## 1. Contract-Level Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "contract-stats",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Each row in CUAD is a (contract, category) pair.\n",
    "# Extract unique contract titles and their context lengths.\n",
    "df = pd.DataFrame({\n",
    "    \"title\": [ex[\"title\"] for ex in train],\n",
    "    \"question\": [ex[\"question\"] for ex in train],\n",
    "    \"context_len\": [len(ex[\"context\"]) for ex in train],\n",
    "    \"has_answer\": [len(ex[\"answers\"][\"text\"]) > 0 for ex in train],\n",
    "})\n",
    "\n",
    "n_contracts = df[\"title\"].nunique()\n",
    "n_categories = df[\"question\"].nunique()\n",
    "print(f\"Unique contracts:  {n_contracts}\")\n",
    "print(f\"Clause categories: {n_categories}\")\n",
    "print(f\"Total pairs:       {len(df):,}  (= {n_contracts} × {n_categories})\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "contract-lengths",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Contract length distribution (characters)\n",
    "contract_lengths = df.drop_duplicates(\"title\")[\"context_len\"]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "ax.hist(contract_lengths / 1000, bins=40, color=\"steelblue\", edgecolor=\"white\")\n",
    "ax.set_xlabel(\"Contract length (thousands of characters)\")\n",
    "ax.set_ylabel(\"Number of contracts\")\n",
    "ax.set_title(\"Distribution of Contract Lengths\")\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"Median: {contract_lengths.median()/1000:.0f}k chars\")\n",
    "print(f\"Max:    {contract_lengths.max()/1000:.0f}k chars\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-imbalance",
   "metadata": {},
   "source": [
    "## 2. Class Imbalance\n",
    "\n",
    "This is the defining challenge. Most (contract, category) pairs are empty."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "imbalance-overall",
   "metadata": {},
   "outputs": [],
   "source": [
    "present = df[\"has_answer\"].sum()\n",
    "absent = (~df[\"has_answer\"]).sum()\n",
    "total = len(df)\n",
    "\n",
    "print(f\"Clause PRESENT: {present:,}  ({100*present/total:.1f}%)\")\n",
    "print(f\"Clause ABSENT:  {absent:,}  ({100*absent/total:.1f}%)\")\n",
    "print(f\"Imbalance ratio: 1:{absent//present}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "imbalance-per-category",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Presence rate per clause category\n",
    "cat_stats = df.groupby(\"question\")[\"has_answer\"].mean().sort_values(ascending=False)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "cat_stats.plot(kind=\"bar\", ax=ax, color=\"steelblue\")\n",
    "ax.set_ylabel(\"Fraction of contracts containing this clause\")\n",
    "ax.set_title(\"Clause Presence Rate per Category\")\n",
    "ax.axhline(0.5, color=\"red\", linestyle=\"--\", label=\"50% line\")\n",
    "plt.xticks(rotation=45, ha=\"right\", fontsize=7)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nTop 5 most common clauses:\")\n",
    "print(cat_stats.head())\n",
    "print(\"\\nBottom 5 rarest clauses:\")\n",
    "print(cat_stats.tail())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-clause-length",
   "metadata": {},
   "source": [
    "## 3. Clause Excerpt Length"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "clause-lengths",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Length of the answer spans that DO exist\n",
    "answer_lengths = []\n",
    "for ex in train:\n",
    "    for text in ex[\"answers\"][\"text\"]:\n",
    "        answer_lengths.append(len(text.split()))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "ax.hist(answer_lengths, bins=60, color=\"darkorange\", edgecolor=\"white\")\n",
    "ax.set_xlabel(\"Clause excerpt length (words)\")\n",
    "ax.set_ylabel(\"Count\")\n",
    "ax.set_title(\"Distribution of Clause Excerpt Lengths\")\n",
    "ax.set_xlim(0, 500)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"Median clause length: {np.median(answer_lengths):.0f} words\")\n",
    "print(f\"95th percentile:      {np.percentile(answer_lengths, 95):.0f} words\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-sample",
   "metadata": {},
   "source": [
    "## 4. Browse Sample Clauses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "browse-present",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print three examples of present clauses\n",
    "present_examples = [ex for ex in train if len(ex[\"answers\"][\"text\"]) > 0][:3]\n",
    "\n",
    "for i, ex in enumerate(present_examples):\n",
    "    print(f\"=== Example {i+1} ===\")\n",
    "    print(f\"Contract: {ex['title']}\")\n",
    "    print(f\"Category: {ex['question'][:80]}\")\n",
    "    print(f\"Clause excerpt: {ex['answers']['text'][0][:300]}\")\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "browse-absent",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print three examples of absent clauses (the majority class)\n",
    "absent_examples = [ex for ex in train if len(ex[\"answers\"][\"text\"]) == 0][:3]\n",
    "\n",
    "for i, ex in enumerate(absent_examples):\n",
    "    print(f\"=== Example {i+1} ===\")\n",
    "    print(f\"Contract: {ex['title']}\")\n",
    "    print(f\"Category: {ex['question'][:80]}\")\n",
    "    print(f\"=> Clause is ABSENT from this contract.\")\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-summary",
   "metadata": {},
   "source": [
    "## 5. Summary of Key Observations\n",
    "\n",
    "Fill in after running the cells above:\n",
    "\n",
    "| Finding | Value |\n",
    "|---|---|\n",
    "| Total unique contracts | ??? |\n",
    "| Total clause categories | ??? |\n",
    "| % of pairs that are absent | ??? |\n",
    "| Imbalance ratio | 1:??? |\n",
    "| Median contract length | ???k chars |\n",
    "| Median clause excerpt length | ??? words |\n",
    "\n",
    "**Next:** Open `02_baseline.ipynb` to build and evaluate a simple TF-IDF baseline."
   ]
  }
 ]
}
