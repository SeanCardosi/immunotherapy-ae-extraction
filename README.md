# CTCAE v6 Adverse Event Extraction Pipeline

Code and data for the paper:

> **Structured Adverse Event Extraction from Unstructured Clinical Text: A CTCAE v6 NLP Pipeline Developed and Evaluated on Reddit Immunotherapy Posts**  
> Sean Cardosi — Sean.Cardosi@Moffitt.org

---

## Overview

This repository contains a two-stage NLP pipeline that extracts and normalizes adverse events (AEs) from free-form text to CTCAE v6 terminology with severity grades. The pipeline is developed and evaluated on 1,907 Reddit posts from cancer immunotherapy communities.

**Stage 1 — Free-form extraction (Qwen3):** Reads each post and identifies AEs in patient language, with supporting evidence quotes.

**Stage 2 — CTCAE normalization (Gemma4):** Embeds the extracted term, retrieves the top-10 candidate CTCAE v6 terms by cosine similarity, then uses Gemma4 to select the best match, assign a grade (1–5), and report confidence.

**Results on 763 evaluated posts:**
- Micro F1: **0.902** (Precision: 0.859, Recall: 0.950)
- Within-one grade accuracy: **93.6%** (Exact: 49.1%, MAE: 0.574)

---

## Prerequisites

### 1. Python

Python 3.10 or later is required (the code uses built-in type hints like `list[dict]`).

```bash
python --version   # must be 3.10+
```

### 2. Ollama

The pipeline calls local LLMs via [Ollama](https://ollama.com). Install Ollama, then pull the three required models:

```bash
ollama pull qwen3:8b          # Stage 1: free-form extraction
ollama pull gemma4:e2b        # Stage 2: CTCAE normalization
ollama pull nomic-embed-text  # Embedding retrieval
```

Ollama must be running before you execute the pipeline:

```bash
ollama serve   # starts the API on http://localhost:11434
```

### 3. Python dependencies

```bash
pip install pandas requests numpy
```

---

## Repository structure

```
.
├── main.py                        # Step 1 — Reddit data collection
├── arctic_shift.py                # Arctic Shift API client (used by main.py)
├── config.py                      # Subreddits, search terms, Ollama URL
├── qwen_mapper.py                 # Step 2 — AE extraction + normalization pipeline
├── compute_ctcae_f1.py            # Step 3a — F1 evaluation
├── compute_grade_accuracy.py      # Step 3b — Grade accuracy evaluation
├── ctcae_v6_with_descriptors.csv  # CTCAE v6 terminology + grade criteria
├── immunotherapy_posts.csv        # 1,907 Reddit posts (corpus)
├── correct_ctcae_results.csv      # Gold standard annotations
├── correct_ctcae_results_graded.csv  # Gold standard with independent grades
└── requirements.txt
```

---

## Reproducing the paper results

If you only want to reproduce the evaluation numbers from the paper, the pipeline output (`ctcae_results.csv`) is included. You can skip Steps 1 and 2 and go straight to Step 3.

### Step 1 — Data collection (optional — corpus already included)

`immunotherapy_posts.csv` is already in the repository. This step only needs to be re-run if you want to collect a fresh corpus.

```bash
python main.py
```

Collects posts from the subreddits and date windows defined in `config.py` using the Arctic Shift API. Output is written to `output/immunotherapy_posts.csv`. The script is resumable — if interrupted, re-running picks up where it left off.

---

### Step 2 — Run the extraction pipeline (optional — output already included)

`ctcae_results.csv` (the pipeline predictions) is included. This step re-runs the full LLM pipeline, which takes significant time (~2–4 minutes per post with local models).

```bash
python qwen_mapper.py
```

Options:

| Flag | Description |
|------|-------------|
| `--limit N` | Process only the first N posts (useful for testing) |
| `--restart` | Clear the checkpoint and output CSV and start from scratch |
| `--debug` | Print live model output for each post |

The pipeline writes to `ctcae_results.csv` and checkpoints progress in `ctcae_checkpoint.txt`. If interrupted, re-running resumes from the last completed post.

**Quick test (3 posts):**

```bash
python qwen_mapper.py --limit 3 --debug
```

---

### Step 3 — Evaluate

#### 3a — CTCAE term extraction F1

Compares `ctcae_results.csv` (pipeline predictions) against `correct_ctcae_results.csv` (gold standard).

```bash
python compute_ctcae_f1.py
```

Expected output:

```
========================================
CTCAE EXTRACTION F1
========================================
Posts evaluated : 763
  TP            : 1827
  FP            : 301
  FN            : 97
Precision       : 0.859
Recall          : 0.950
Micro  F1       : 0.902
Macro  F1       : 0.881
Weighted F1     : 0.916
========================================
```

#### 3b — Grade accuracy

Evaluates grade assignments on true positives using `correct_ctcae_results_graded.csv`.

```bash
python compute_grade_accuracy.py
```

Expected output:

```
Exact match    : 761 / 1551  (49.1%)
Within-one     : 1452 / 1551  (93.6%)
MAE            : 0.574
```

---

## Gold standard files

| File | Description |
|------|-------------|
| `correct_ctcae_results.csv` | Manual gold standard. Same format as `ctcae_results.csv`. Posts with `ctcae_normalized = __NONE__` are caregiver-symptom sentinels — any pipeline prediction on these is a false positive. |
| `correct_ctcae_results_graded.csv` | Gold standard with independently assigned grades using a rule-based grading function applied to each evidence quote. Used only for grade accuracy evaluation. |

---

## Configuration

`config.py` contains all tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma4:12b` | Model used by the old `ctcae_mapper.py` (not used by `qwen_mapper.py`) |
| `SUBREDDITS` | See file | Subreddits searched during data collection |
| `SEARCH_TERMS` | See file | Immunotherapy drug names and modalities |
| `DATE_WINDOWS` | 2023–2026 | Date ranges for post collection |

The extraction and normalization models used by `qwen_mapper.py` are configured at the top of that file:

```python
QWEN_MODEL   = "qwen3:8b"       # Stage 1 extraction
NORM_MODEL   = "gemma4:e2b"     # Stage 2 normalization
EMBED_MODEL  = "nomic-embed-text"
```

---

## Citation

If you use this code or data, please cite:

```
@article{cardosi2026ctcae,
  title   = {Structured Adverse Event Extraction from Unstructured Clinical Text:
             A CTCAE v6 NLP Pipeline Developed and Evaluated on Reddit
             Immunotherapy Posts},
  author  = {Cardosi, Sean},
  year    = {2026}
}
```
