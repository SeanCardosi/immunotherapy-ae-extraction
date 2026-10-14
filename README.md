# CTCAE v6 Adverse Event Extraction Pipeline

Code and data for the manuscript:

> **Structured Adverse Event Extraction from Unstructured Clinical Text: A CTCAE v6 NLP Pipeline Developed and Evaluated on Reddit Immunotherapy Posts**
> Sean Cardosi (University of Florida), Ghulam Rasool, PhD (Moffitt Cancer Center)

---

## Overview

This repository contains a two-stage NLP pipeline that extracts adverse events (AEs) from free-form text, normalizes them to CTCAE v6 terms, assigns severity grades where the text supports one, and records the drug or treatment each AE most likely came from. It was developed and evaluated on 1,907 public Reddit posts from cancer communities.

**Stage 1: free-form extraction (Qwen3).** Reads each post and lists the adverse events the patient reports having personally experienced, each with a grade (or null), a verbatim evidence quote, the likely responsible drug or treatment, and a confidence level.

**Stage 2: CTCAE normalization (Gemma 4).** For each extraction that is not already an exact CTCAE term, embeds the term and evidence, retrieves the ten closest CTCAE v6 terms by cosine similarity, and has Gemma 4 choose the best match, assign a grade (or null), and report confidence.

### Results (920 evaluated posts, every prediction scored)

| Task | Result |
|------|--------|
| Term extraction | Micro F1 **0.831** (precision 0.739, recall 0.950); macro F1 0.731; weighted F1 0.916 |
| Grade agreement (346 true positives with both grades) | 60.7% exact, **97.1%** within one grade, MAE 0.422 |
| Attribution: immunotherapy named | Accuracy 0.827, precision 0.669, recall 0.994, kappa 0.658 |
| Attribution: any treatment named | Accuracy 0.728, precision 0.730, recall 0.975, kappa 0.149 |

**Read these numbers with the reference standard in mind** (see [Reference standard](#reference-standard)): it was built by reviewing the pipeline's own predictions, so recall is likely optimistic and precision is the more reliable figure. Reference grades were left blank when a post did not support one, and the pipeline graded most of those terms anyway; likewise the pipeline attributed most events the reviewer judged unrelated to treatment to a treatment.

---

## Prerequisites

### 1. Python

Python 3.10 or later is required (the code uses built-in type hints like `list[dict]`).

```bash
python --version
```

### 2. Ollama

The pipeline calls local models through [Ollama](https://ollama.com). Install Ollama, then pull the three models:

```bash
ollama pull qwen3:8b          # Stage 1: free-form extraction
ollama pull gemma4:e2b        # Stage 2: CTCAE normalization
ollama pull nomic-embed-text  # embedding retrieval
```

Ollama must be running before you start the pipeline (it starts automatically after installation on most systems; otherwise run `ollama serve`). The API listens on `http://localhost:11434`.

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

`evaluate.py` uses only the standard library, so you can reproduce the paper's numbers without Ollama or any installed packages.

---

## Repository structure

```
.
├── main.py                           # Step 1: Reddit data collection (optional)
├── arctic_shift.py                   # Arctic Shift API client used by main.py
├── config.py                         # Subreddits, search terms, date windows, Ollama URL
├── qwen_mapper.py                    # Step 2: extraction + normalization pipeline
├── evaluate.py                       # Step 3: terms, grades, and attribution evaluation
├── ctcae_v6_with_descriptors.csv     # CTCAE v6 terms, definitions, and grade criteria
├── immunotherapy_posts.csv           # Corpus: 1,907 Reddit posts
├── ctcae_results.csv                 # Pipeline output evaluated in the paper
├── ctcae_results_gold.csv            # Reference standard (terms, grades, attributions)
└── requirements.txt
```

---

## Reproducing the paper's results

The pipeline output (`ctcae_results.csv`) and the reference standard (`ctcae_results_gold.csv`) are included, so you can go straight to the evaluation:

```bash
python evaluate.py --gold ctcae_results_gold.csv
```

Expected output (abridged):

```
=== TERM EXTRACTION ===
Posts evaluated (gold U pred): 920
TP / FP / FN:                  1827 / 645 / 97
Precision / Recall:            0.739 / 0.950
Micro F1:                      0.831
Macro F1:                      0.731
Weighted F1:                   0.916

=== GRADE AGREEMENT ===
TPs with both grades:          346 of 1827
Exact match:                   210/346 (60.7%)
Within one:                    336/346 (97.1%)
MAE:                           0.422

=== TREATMENT ATTRIBUTION (true-positive terms) ===
Immunotherapy named:   accuracy 0.827  precision 0.669  recall 0.994  kappa 0.658
Any treatment named:   accuracy 0.728  precision 0.730  recall 0.975  kappa 0.149
```

The script prints the full confusion matrix and counts as well.

**How scoring works**
- **Terms:** exact match on (post, CTCAE term) pairs. Every pipeline prediction counts, including predictions in posts where no AE was confirmed (those are false positives). Rows with the term `__NONE__` mark posts recorded as having no adverse event.
- **Grades:** computed on true positives where both the reference and the pipeline have a grade. Blank grades are skipped.
- **Attribution:** `likely_relation` is free text, so two coarse questions are scored with keyword matching (the keyword lists are at the top of `evaluate.py`): whether an immunotherapy is named, and whether any treatment is named rather than only disease, infection, or an unclear cause.

---

## Re-running the pipeline (optional)

Re-running the pipeline will regenerate `ctcae_results.csv`. The paper's numbers come from the included file; local LLM output is not perfectly deterministic, so a fresh run may differ slightly.

```bash
python qwen_mapper.py
```

| Flag | Description |
|------|-------------|
| `--limit N` | Process only the first N posts (useful for testing) |
| `--restart` | Delete the checkpoint and output CSV and start from scratch |
| `--debug` | Print live model output for each post |

Progress is checkpointed in `ctcae_checkpoint.txt`, so an interrupted run resumes where it stopped. **The checkpoint file is not included in this repository, so a run without `--restart` appends a second copy of every post's results to the existing `ctcae_results.csv`. Back up the original (or use `--restart`, which deletes it) before re-running.** Runtime depends on your hardware and is substantial for the full corpus.

A quick test on three posts:

```bash
python qwen_mapper.py --limit 3 --debug
```

---

## Data collection (optional)

`immunotherapy_posts.csv` is included, so this step is only needed to collect a fresh corpus.

```bash
python main.py
```

`main.py` searches the subreddits, terms, and date windows in `config.py` through the Arctic Shift archive and writes `output/immunotherapy_posts.csv`. It is resumable. To use a freshly collected corpus with the pipeline, copy that file over `immunotherapy_posts.csv`.

---

## Reference standard

`ctcae_results_gold.csv` has one row per confirmed adverse event:

| Column | Description |
|--------|-------------|
| `post_id`, `subreddit`, `permalink` | Reddit post identification |
| `ctcae` | CTCAE v6 term, or `__NONE__` for a post recorded as having no AE |
| `grade` | CTCAE grade 1-5, or blank when the post did not support a grade |
| `likely_relation` | Reviewed drug/treatment attribution, or a non-treatment cause (disease, infection, unclear) |

**How it was built.** A single reviewer (S.C.) built it in two steps:
1. Starting from the pipeline's output, the reviewer confirmed or rejected each predicted term against the post text.
2. An LLM (Claude) was given only the raw post text and the CTCAE v6 definitions, without the pipeline output, and asked to suggest terms for AEs the pipeline may have missed; the reviewer confirmed or rejected each suggestion.

Grades were assigned from the post evidence and left blank when the text did not support one. For every confirmed term the reviewer also confirmed or corrected the pipeline's treatment attribution.

**What this means.** The reference standard is not independent of the pipeline: an AE that neither the pipeline nor the LLM proposed cannot appear in it. Recall is therefore likely optimistic, while precision is well grounded because every false positive is a prediction the reviewer rejected. There is one reviewer and no clinical expert review. See the paper's Limitations section.

Of the 763 posts in the file, 744 have at least one confirmed AE and 19 are recorded as `__NONE__` (caregiver posts in which the pipeline extracted the caregiver's own symptoms). A further 157 posts had predictions that were all rejected and do not appear in the file; the evaluation counts those predictions as false positives.

---

## Configuration

`config.py` holds the collection and Ollama settings:

| Parameter | Description |
|-----------|-------------|
| `OLLAMA_URL` | Ollama API endpoint (default `http://localhost:11434/api/generate`) |
| `SUBREDDITS` | Subreddits searched during data collection |
| `SEARCH_TERMS` | Immunotherapy drug names and modalities |
| `DATE_WINDOWS` | Three one-year windows from 2023-01-01 to 2026-01-01 |
| `POST_LIMIT_PER_SEARCH` | Maximum posts per (subreddit, term, window) search |

The models are set at the top of `qwen_mapper.py`:

```python
QWEN_MODEL   = "qwen3:8b"        # Stage 1 extraction
NORM_MODEL   = "gemma4:e2b"      # Stage 2 normalization
EMBED_MODEL  = "nomic-embed-text"
```

---

## Data and licensing notes

- Code is released under the MIT License (see `LICENSE`).
- CTCAE v6.0 is published by the U.S. National Cancer Institute. `ctcae_v6_with_descriptors.csv` adds `possible_descriptors` and `patient_synonyms` columns.
- Reddit posts remain subject to Reddit's and Arctic Shift's terms and to their authors' rights. Do not use the data to identify or contact authors.

---

## Citation

If you use this code or data, please cite the manuscript (under review):

```bibtex
@unpublished{cardosi2026ctcae,
  title  = {Structured Adverse Event Extraction from Unstructured Clinical Text:
            A CTCAE v6 NLP Pipeline Developed and Evaluated on Reddit
            Immunotherapy Posts},
  author = {Cardosi, Sean and Rasool, Ghulam},
  year   = {2026},
  note   = {Manuscript under review}
}
```
