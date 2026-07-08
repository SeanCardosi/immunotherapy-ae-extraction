"""
CTCAE Extraction F1 Score
=========================
Compares pipeline predictions (ctcae_results.csv) against the gold standard
(correct_ctcae_results.csv) using set-matching on ctcae_normalized per post.

Posts with ctcae_normalized='__NONE__' are caregiver-symptom sentinels:
pipeline predictions on those count as FP with no corresponding FN.
"""

import csv
from collections import defaultdict

CSV_PRED = 'ctcae_results.csv'
CSV_GOLD = 'correct_ctcae_results.csv'
SENTINEL = '__none__'


def load_gold_pids(path):
    pids = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            pids.add(row['post_id'])
    return pids


def load_terms(path, pid_filter=None):
    d = defaultdict(set)
    with open(path) as f:
        for row in csv.DictReader(f):
            pid = row['post_id']
            if pid_filter is None or pid in pid_filter:
                norm = row['ctcae_normalized'].strip().lower()
                if norm != SENTINEL:
                    d[pid].add(norm)
    return d


gold_pids = load_gold_pids(CSV_GOLD)
gold = load_terms(CSV_GOLD)
pred = load_terms(CSV_PRED, gold_pids)

tp = fp = fn = 0
per_post = []

for pid in gold_pids:
    p = pred.get(pid, set())
    g = gold.get(pid, set())
    t   = len(p & g)
    f_p = len(p - g)
    f_n = len(g - p)
    tp += t
    fp += f_p
    fn += f_n
    pr = t / (t + f_p) if (t + f_p) else (1.0 if not g else 0.0)
    rc = t / (t + f_n) if (t + f_n) else (1.0 if not g else 0.0)
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    per_post.append((pid, f1, len(g)))

micro_p  = tp / (tp + fp) if (tp + fp) else 0.0
micro_r  = tp / (tp + fn) if (tp + fn) else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
macro    = sum(x[1] for x in per_post) / len(per_post)
total_g  = sum(x[2] for x in per_post)
weighted = sum(x[1] * x[2] for x in per_post) / total_g if total_g else 0.0

print("=" * 40)
print("CTCAE EXTRACTION F1")
print("=" * 40)
print(f"Posts evaluated : {len(gold_pids)}")
print(f"  TP            : {tp}")
print(f"  FP            : {fp}")
print(f"  FN            : {fn}")
print(f"Precision       : {micro_p:.3f}")
print(f"Recall          : {micro_r:.3f}")
print(f"Micro  F1       : {micro_f1:.3f}")
print(f"Macro  F1       : {macro:.3f}")
print(f"Weighted F1     : {weighted:.3f}")
print("=" * 40)
