"""
evaluate.py

Scores the pipeline output (ctcae_results.csv) against a gold-standard CSV. Every
pipeline prediction counts, including predictions in posts with no gold terms.
Rows whose term is __NONE__ mark posts with no confirmed adverse event.

1. Term extraction: exact match on (post_id, CTCAE term) pairs.
   - micro P/R/F1 over all pairs
   - macro F1: mean of per-post F1 over posts with any gold or predicted term
   - weighted F1: per-post F1 weighted by number of gold terms in the post
2. Grade agreement: true positives where BOTH gold and predicted grades are present
   (exact match, within-one, MAE, confusion matrix). Blank grades are skipped.
3. Treatment attribution on true positives (likely_relation is free text, so two
   coarse questions are scored with keyword matching):
   - immunotherapy named: an immune checkpoint inhibitor or general immunotherapy term
   - any treatment named: a treatment rather than only disease/unclear/infection

Usage:
    python evaluate.py --gold ctcae_results_gold.csv
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_CSV = os.path.join(BASE_DIR, "ctcae_results.csv")
CTCAE_CSV = os.path.join(BASE_DIR, "ctcae_v6_with_descriptors.csv")
SENTINEL = "__none__"

ICI = re.compile(
    r"keytruda|pembro|opdivo|nivo|yervoy|ipilimumab|\bipi\b|imfinzi|durva|tecentriq|"
    r"atezo|libtayo|cemiplimab|opdualag|relatlimab|imjudo|tremelimumab|balstilimab|"
    r"dostarlimab|jemperli|avelumab|bavencio|\bpd-?l?1\b|checkpoint|immunotherapy|"
    r"immune therapy",
    re.I,
)
NON_TREATMENT = re.compile(
    r"^\s*(disease|unclear|infection|cancer|gvhd|unspecified|unknown|none|n/?a|"
    r"covid|flu|unrelated)\b",
    re.I,
)


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_grade(x):
    try:
        g = int(str(x).strip())
        return g if 1 <= g <= 5 else None
    except ValueError:
        return None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def is_ici(s):
    return bool(ICI.search(s))


def is_treatment(s):
    """True if the entry names at least one treatment (not only non-treatment causes)."""
    parts = [p.strip() for p in re.split(r"[,/;+]| and ", re.sub(r"\([^)]*\)", "", s)) if p.strip()]
    return any(not NON_TREATMENT.match(p) for p in parts)


def binary_report(pairs, label):
    """pairs: (reviewed_positive, pipeline_positive) per true-positive term."""
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    n = len(pairs)
    prec, rec, f1 = prf(tp, fp, fn)
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    print(f"\n{label}  (n = {n})")
    print(f"  reviewed positive / pipeline positive: {tp + fn} / {tp + fp}")
    print(f"  TP / FP / FN / TN:             {tp} / {fp} / {fn} / {tn}")
    print(f"  accuracy {acc:.3f}  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}  kappa {kappa:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", default=PRED_CSV)
    args = ap.parse_args()

    gold_rows = read(args.gold)
    term_col = next(c for c in ("ctcae_normalized", "ctcae_term", "ctcae") if c in gold_rows[0])
    gold_posts = {r["post_id"] for r in gold_rows}
    gold, gold_attr = {}, {}
    for r in gold_rows:
        t = r[term_col].strip().lower()
        if t != SENTINEL:
            gold[(r["post_id"], t)] = to_grade(r["grade"])
            gold_attr[(r["post_id"], t)] = r.get("likely_relation", "").strip()

    pred, pred_attr = {}, {}
    for r in read(args.pred):
        k = (r["post_id"], r["ctcae_normalized"].strip().lower())
        pred[k] = to_grade(r["grade"])
        pred_attr[k] = r.get("likely_relation", "").strip()

    # --- 1. Term extraction -------------------------------------------------------
    G, P = set(gold), set(pred)
    tp, fp, fn = len(G & P), len(P - G), len(G - P)
    prec, rec, micro = prf(tp, fp, fn)

    bg, bp = defaultdict(set), defaultdict(set)
    for pid, t in G: bg[pid].add(t)
    for pid, t in P: bp[pid].add(t)
    posts = set(bg) | set(bp)
    f1s = [prf(len(bg.get(i, set()) & bp.get(i, set())),
               len(bp.get(i, set()) - bg.get(i, set())),
               len(bg.get(i, set()) - bp.get(i, set())))[2] for i in posts]
    w = [len(bg.get(i, set())) for i in posts]
    macro = sum(f1s) / len(f1s)
    weighted = sum(f * x for f, x in zip(f1s, w)) / sum(w)

    print("=== TERM EXTRACTION ===")
    print(f"Posts with gold terms:         {len(bg)}  (posts recorded as having none: {len(gold_posts - set(bg))})")
    print(f"Posts with pipeline preds:     {len(bp)}")
    print(f"Posts evaluated (gold U pred): {len(posts)}")
    print(f"Gold rows / predicted rows:    {len(G)} / {len(P)}")
    print(f"TP / FP / FN:                  {tp} / {fp} / {fn}")
    print(f"Precision / Recall:            {prec:.3f} / {rec:.3f}")
    print(f"Micro F1:                      {micro:.3f}")
    print(f"Macro F1:                      {macro:.3f}")
    print(f"Weighted F1:                   {weighted:.3f}")
    print(f"Gold posts with no preds:      {len(set(bg) - set(bp))}")
    valid = {r["adverse_effect"].strip().lower() for r in read(CTCAE_CSV)}
    print(f"Predicted terms not in CTCAE list: {len({t for _, t in P} - valid)}")

    # --- 2. Grade agreement -------------------------------------------------------
    both = [(gold[k], pred[k]) for k in G & P if gold[k] and pred[k]]
    print("\n=== GRADE AGREEMENT ===")
    print(f"Gold rows graded:              {sum(1 for v in gold.values() if v)} / {len(gold)}")
    print(f"Pred rows graded:              {sum(1 for v in pred.values() if v)} / {len(pred)}")
    print(f"TPs with both grades:          {len(both)} of {tp}")
    if both:
        ex = sum(a == b for a, b in both)
        w1 = sum(abs(a - b) <= 1 for a, b in both)
        print(f"Exact match:                   {ex}/{len(both)} ({ex/len(both):.1%})")
        print(f"Within one:                    {w1}/{len(both)} ({w1/len(both):.1%})")
        print(f"MAE:                           {sum(abs(a - b) for a, b in both)/len(both):.3f}")
        print(f"Over-graded / under-graded:    {sum(b > a for a, b in both)} / {sum(b < a for a, b in both)}")
        c = Counter(both)
        print("\nConfusion (rows=gold, cols=pred 1..5)")
        for g in range(1, 6):
            print(f"G={g}: " + "  ".join(f"{c[(g, p)]:4d}" for p in range(1, 6)))

    # --- 3. Treatment attribution -------------------------------------------------
    keys = sorted(G & P)
    print("\n=== TREATMENT ATTRIBUTION (true-positive terms) ===")
    print(f"Terms compared:                {len(keys)}")
    print(f"Reviewed attribution blank:    {sum(1 for k in keys if not gold_attr[k])}")
    print(f"Pipeline attribution blank:    {sum(1 for k in keys if not pred_attr[k])}")
    print(f"Identical text (ignoring case): {sum(gold_attr[k].lower() == pred_attr[k].lower() for k in keys)}")
    binary_report([(is_ici(gold_attr[k]), is_ici(pred_attr[k])) for k in keys],
                  "Immunotherapy (ICI/immunotherapy) named")
    binary_report([(is_treatment(gold_attr[k]), is_treatment(pred_attr[k]) if pred_attr[k] else False)
                   for k in keys],
                  "Any treatment named (vs disease/unclear/infection only)")
    nt = [k for k in keys if gold_attr[k] and not is_treatment(gold_attr[k])]
    named = sum(1 for k in nt if pred_attr[k] and is_treatment(pred_attr[k]))
    print(f"\nReviewed as non-treatment cause only: {len(nt)} of {len(keys)} ({len(nt)/len(keys):.1%}); "
          f"pipeline still named a treatment for {named} ({named/len(nt):.1%})")


if __name__ == "__main__":
    main()
