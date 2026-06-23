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
   "id": "setup-header",
   "metadata": {},
   "source": [
    "# 00 — Setup Check\n",
    "\n",
    "Run this notebook first. It will:\n",
    "1. Verify your Colab GPU is available.\n",
    "2. Install all required packages.\n",
    "3. Load a sample from CUAD via Hugging Face.\n",
    "4. Print a few example (contract, category) pairs so you understand the data format.\n",
    "\n",
    "**Expected runtime:** ~3 minutes on a fresh Colab session (mostly pip installs).\n",
    "\n",
    "**Runtime → Change runtime type → T4 GPU → Save** before running."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-gpu",
   "metadata": {},
   "source": [
    "## 1. Verify GPU"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "check-gpu",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
    "else:\n",
    "    print(\"WARNING: No GPU found. Go to Runtime → Change runtime type → T4 GPU.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-install",
   "metadata": {},
   "source": [
    "## 2. Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "install-deps",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q transformers datasets sentence-transformers scikit-learn"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-load",
   "metadata": {},
   "source": [
    "## 3. Load CUAD from Hugging Face"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "load-cuad",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "# This will download ~1 GB on the first run; subsequent runs use the cache.\n",
    "cuad = load_dataset(\"theatticusproject/cuad\")\n",
    "print(cuad)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-inspect",
   "metadata": {},
   "source": [
    "## 4. Inspect the Data Format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "inspect-features",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature names — each question corresponds to one of the 41 clause categories\n",
    "train = cuad[\"train\"]\n",
    "print(\"Number of training examples:\", len(train))\n",
    "print(\"\\nFeature names:\")\n",
    "print(list(train.features.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "inspect-sample",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print one example\n",
    "sample = train[0]\n",
    "print(\"Contract title:\", sample[\"title\"])\n",
    "print(\"Question (clause category):\", sample[\"question\"])\n",
    "print(\"is_impossible (clause absent?):\", sample[\"answers\"][\"is_impossible\"] if \"is_impossible\" in sample.get(\"answers\", {}) else \"N/A\")\n",
    "print(\"\\nContext snippet (first 500 chars):\")\n",
    "print(sample[\"context\"][:500])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "count-impossible",
   "metadata": {},
   "outputs": [],
   "source": [
    "# How many examples have the clause present vs absent?\n",
    "# In CUAD's SQuAD format, answers['text'] is empty when the clause is absent.\n",
    "n_total = len(train)\n",
    "n_present = sum(1 for ex in train if len(ex[\"answers\"][\"text\"]) > 0)\n",
    "n_absent = n_total - n_present\n",
    "\n",
    "print(f\"Total training pairs: {n_total:,}\")\n",
    "print(f\"Clause PRESENT:       {n_present:,}  ({100*n_present/n_total:.1f}%)\")\n",
    "print(f\"Clause ABSENT:        {n_absent:,}  ({100*n_absent/n_total:.1f}%)\")\n",
    "print(\"\\n=> The majority class is ABSENT — class imbalance is the core challenge.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-drive",
   "metadata": {},
   "source": [
    "## 5. (Optional) Mount Google Drive for Checkpointing\n",
    "\n",
    "Colab sessions disconnect on idle. Save model checkpoints to Drive to persist work across sessions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "mount-drive",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Uncomment to mount Drive\n",
    "# from google.colab import drive\n",
    "# drive.mount(\"/content/drive\")\n",
    "# CHECKPOINT_DIR = \"/content/drive/MyDrive/contract-review-assistant/checkpoints\"\n",
    "# import os; os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
    "# print(\"Drive mounted. Checkpoints will save to:\", CHECKPOINT_DIR)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "section-done",
   "metadata": {},
   "source": [
    "## Setup Complete\n",
    "\n",
    "If all cells above ran without errors, your environment is ready.\n",
    "\n",
    "**Next:** Open `01_explore_cuad.ipynb` for guided EDA."
   ]
  }
 ]
}
