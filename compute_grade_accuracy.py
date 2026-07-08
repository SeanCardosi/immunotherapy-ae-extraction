"""
CTCAE Grade Accuracy
====================
For every TP (pipeline correctly identified the CTCAE term), compares the
predicted grade against independently-assigned gold grades in
correct_ctcae_results_graded.csv.

Metrics:
  - Exact match  : grade predicted == gold grade
  - Within-1     : |predicted - gold| <= 1  (clinically acceptable)
  - MAE          : mean absolute error across graded TPs
  - Confusion    : counts of (gold_grade, pred_grade) pairs
"""

import csv
from collections import defaultdict, Counter

CSV_PRED = 'ctcae_results.csv'
CSV_GOLD = 'correct_ctcae_results_graded.csv'   # independent gold grades
SENTINEL = '__none__'


def load_gold_pids(path):
    pids = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            pids.add(row['post_id'])
    return pids


def load_graded(path, pid_filter=None):
    d = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            pid  = row['post_id']
            norm = row['ctcae_normalized'].strip().lower()
            if norm == SENTINEL:
                continue
            if pid_filter is None or pid in pid_filter:
                d[pid][norm] = row.get('grade', '').strip()
    return d


gold_pids = load_gold_pids(CSV_GOLD)
gold = load_graded(CSV_GOLD)
pred = load_graded(CSV_PRED, gold_pids)

exact = within1 = graded_tp = 0
skipped_no_grade = 0
confusion = Counter()
errors = []

for pid in gold_pids:
    g_terms = gold.get(pid, {})
    p_terms = pred.get(pid, {})
    tp_terms = set(g_terms.keys()) & set(p_terms.keys())

    for term in tp_terms:
        gg = g_terms[term]
        pg = p_terms[term]

        if not gg or not pg:
            skipped_no_grade += 1
            continue
        try:
            g, p = int(gg), int(pg)
        except ValueError:
            skipped_no_grade += 1
            continue

        graded_tp += 1
        confusion[(g, p)] += 1
        diff = abs(g - p)
        if diff == 0:
            exact += 1
        if diff <= 1:
            within1 += 1
        else:
            errors.append((pid, term, g, p))

mae = (
    sum(abs(g - p) * n for (g, p), n in confusion.items()) / graded_tp
    if graded_tp else 0.0
)

exact_pct   = exact   / graded_tp * 100 if graded_tp else 0
within1_pct = within1 / graded_tp * 100 if graded_tp else 0

print("=" * 50)
print("CTCAE GRADE ACCURACY  (independent gold grades)")
print("=" * 50)
print(f"TPs with both grades present : {graded_tp}")
print(f"TPs missing grade (skipped)  : {skipped_no_grade}")
print(f"Exact match                  : {exact} / {graded_tp}  ({exact_pct:.1f}%)")
print(f"Within-1 accuracy            : {within1} / {graded_tp}  ({within1_pct:.1f}%)")
print(f"Mean absolute error (MAE)    : {mae:.3f}")
print()
print("Grade confusion matrix (rows = gold, cols = pred):")
print("       " + "".join(f" P={p}" for p in range(1, 6)))
for g in range(1, 6):
    row_vals = "".join(f"{confusion.get((g, p), 0):5}" for p in range(1, 6))
    print(f"  G={g}  {row_vals}")
print()
if errors:
    errors.sort(key=lambda x: -abs(x[2] - x[3]))
    print(f"Largest mismatches (|diff| >= 2) — top 15:")
    for pid, term, g, p in errors[:15]:
        print(f"  [{pid}] {term:<40} gold={g}  pred={p}  diff={abs(g-p)}")
print("=" * 50)
