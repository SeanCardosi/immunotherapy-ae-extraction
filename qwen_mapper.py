"""
qwen_mapper.py

Two-stage adverse event extraction pipeline:

  Stage 1 — Extraction (Qwen3):
    Reads each Reddit post and extracts adverse event descriptions in
    free-form patient language.
    Output per match: ctcae_described | grade | evidence | likely_relation | confidence

  Stage 2 — CTCAE Normalization (Gemma4):
    Embeds the extracted term, retrieves top-10 candidate CTCAE v6 terms
    by cosine similarity, then uses Gemma4 to select the best match,
    assign a CTCAE grade (1-5), and report confidence.

Output: ctcae_results.csv
Checkpoint: ctcae_checkpoint.txt  (resumable — re-run to continue)

Usage:
    python qwen_mapper.py
    python qwen_mapper.py --limit 3     # test on first 3 posts
    python qwen_mapper.py --restart     # clear output and start fresh
    python qwen_mapper.py --debug       # print live model output
"""

import argparse
import csv
import json
import os
import pickle
import re
import time
from datetime import datetime

import numpy as np
import requests

from config import OLLAMA_URL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QWEN_MODEL   = "qwen3:8b"        # extraction model
NORM_MODEL   = "gemma4:e2b"     # normalization model — supports thinking
EMBED_MODEL  = "nomic-embed-text"
EMBED_URL    = OLLAMA_URL.replace("/api/generate", "/api/embed")
EMBED_CANDIDATES = 10           # how many CTCAE candidates to pass to the norm model
DEBUG = False                   # set via --debug flag

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CTCAE_CSV       = os.path.join(BASE_DIR, "ctcae_v6_with_descriptors.csv")
POSTS_CSV       = os.path.join(BASE_DIR, "immunotherapy_posts.csv")
OUTPUT_CSV      = os.path.join(BASE_DIR, "ctcae_results.csv")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "ctcae_checkpoint.txt")
EMBED_CACHE     = os.path.join(BASE_DIR, "ctcae_embeddings_v2.pkl")

MAX_POST_CHARS = 3000

OUTPUT_FIELDNAMES = [
    "post_id",
    "subreddit",
    "permalink",
    "ctcae_described",
    "ctcae_normalized",
    "grade",
    "evidence",
    "likely_relation",
    "confidence",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_ctcae(path: str) -> list[dict]:
    entries = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            entries.append({
                "adverse_effect": row["adverse_effect"].strip(),
                "definition":     row["definition"].strip(),
                "grade_1":        row.get("grade_1", "").strip(),
                "grade_2":        row.get("grade_2", "").strip(),
                "grade_3":        row.get("grade_3", "").strip(),
                "grade_4":        row.get("grade_4", "").strip(),
                "grade_5":        row.get("grade_5", "").strip(),
            })
    return entries


def load_posts(path: str) -> list[dict]:
    posts = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = row.get("selftext", "").strip()
            if text and text.lower() not in ("[deleted]", "[removed]", ""):
                posts.append({
                    "post_id":   row.get("post_id", ""),
                    "subreddit": row.get("subreddit", ""),
                    "permalink": row.get("permalink", ""),
                    "selftext":  text,
                })
    return posts


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(path: str, post_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{post_id}\n")


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------
def _call_ollama(prompt: str, retries: int = 3, live: bool = False, label: str = "", think: bool = True, model: str = None) -> str:
    payload = {
        "model":   model or QWEN_MODEL,
        "prompt":  prompt,
        "stream":  True,
        "think":   think,
        # qwen3 recommended sampling params for thinking mode
        "options": {"temperature": 0.0, "top_p": 0.95, "top_k": 20},
    }
    if live:
        tag = f" [{label}]" if label else ""
        print(f"\n{'─'*60}")
        print(f"YOU{tag}:\n{prompt.strip()}")
        print(f"\n{'─'*60}")
        print(f"\033[36m{(model or QWEN_MODEL).upper()}:\033[0m")

    for attempt in range(1, retries + 1):
        try:
            chunks = []
            if live:
                print("  ", end="", flush=True)
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    thinking_token = obj.get("thinking", "")
                    response_token = obj.get("response", "")
                    chunks.append(response_token)
                    if live:
                        if thinking_token:
                            print(f"\033[2m{thinking_token}\033[0m", end="", flush=True)
                        elif response_token:
                            print(f"\033[36m{response_token}\033[0m", end="", flush=True)
                    if obj.get("done"):
                        break
            if live:
                print()
            result = "".join(chunks).strip()
            if result:
                return result
        except requests.RequestException as e:
            print(f"    [!] Ollama error (attempt {attempt}): {e}")
        if attempt < retries:
            time.sleep(2)
    return ""


# ---------------------------------------------------------------------------
# Step 1 — Extraction
# ---------------------------------------------------------------------------
def build_extraction_prompt(post_text: str) -> str:
    return f"""You are a clinical oncology expert extracting adverse events from patient-reported text.

Read the patient text below and identify every clinical CTCAE (Common Terminology Criteria for Adverse Events) adverse effect the patient personally experienced. Use standard CTCAE medical terminology for the term name.

Decision rules
- ONLY include effects the patient states they are currently experiencing or have already experienced
- EXCLUDE anything the patient is asking about, worried about, or heard from others — if they haven't had it themselves, skip it
- List each adverse effect as a separate entry, even if symptoms overlap
- Translate patient language into specific CTCAE terminology — do not echo the patient's exact words unless they are a CTCAE term
- If you are unsure whether something qualifies, include it with confidence: low
- Do not debate whether to combine or split entries — always split
- Do not list the same CTCAE term more than once — each term appears at most one time in your output

For each clinical adverse effect, output entries in this exact format:

---ctcae_term1 | grade | [verbatim quote from text] | drug_name | confidence
---ctcae_term2 | grade | [verbatim quote from text] | drug_name | confidence

Definitions:
- ctcae_term: standard clinical CTCAE adverse effect name
- grade: integer 1-5 matching severity the patient describes, or null if unclear
- evidence: verbatim quote from the patient text in [square brackets] — quote only, no added explanation
- likely_relation: from the drugs or treatments explicitly mentioned in the post, identify the one or combination most likely to have caused this specific adverse effect based on known toxicity profiles — if multiple are plausible, list them together; null only if no drug mentioned in the post plausibly causes this effect
- confidence: high if the patient describes the symptom with clear detail, medium if clearly mentioned but without detail, low if vague, uncertain, or only inferred

Patient text:
\"\"\"{post_text[:MAX_POST_CHARS]}\"\"\"

Output ONLY the entries in the format above. No headers, no extra explanation. If nothing matches, output: NONE"""


def extract_from_post(post_text: str) -> list[dict]:
    """Call Ollama to freely extract CTCAE adverse effects from post text."""
    raw = _call_ollama(build_extraction_prompt(post_text), live=DEBUG, label="extraction")
    if not raw or raw.strip().upper() == "NONE":
        return []

    results = []
    entries = re.split(r"\n?\s*---", raw)
    for entry in entries:
        line = entry.strip()
        if not line or line.upper() == "NONE":
            continue
        line = line.splitlines()[0].strip()
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) < 2:
            continue

        ctcae_described = parts[0]
        if not ctcae_described:
            continue

        grade_str = parts[1].lower().replace("grade", "").strip()
        try:
            grade = int(grade_str)
        except ValueError:
            grade = None

        evidence        = parts[2] if len(parts) > 2 else ""
        likely_relation = parts[3] if len(parts) > 3 else None
        confidence      = parts[4] if len(parts) > 4 else ""

        if likely_relation and likely_relation.lower() in ("null", "none", ""):
            likely_relation = None

        results.append({
            "ctcae_described": ctcae_described,
            "grade":           grade,
            "evidence":        evidence,
            "likely_relation": likely_relation,
            "confidence":      confidence,
        })

    return results


# ---------------------------------------------------------------------------
# Step 2 — Ollama normalization
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 2 — Embedding + qwen normalization
# ---------------------------------------------------------------------------
def _get_embedding(text: str) -> list[float]:
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": text}, timeout=(10, 30))
    r.raise_for_status()
    return r.json()["embeddings"][0]


def build_ctcae_embeddings(ctcae_list: list[dict]) -> tuple[list[str], list[str], np.ndarray]:
    """
    Pre-compute embeddings using term+definition for richer semantic matching.
    Returns (names, definitions, matrix).
    Definitions are always taken from ctcae_list (not cached).
    """
    names       = [c["adverse_effect"] for c in ctcae_list]
    definitions = [c["definition"]     for c in ctcae_list]

    if os.path.exists(EMBED_CACHE):
        print("Loading CTCAE embeddings from cache...")
        with open(EMBED_CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded {len(data['names'])} embeddings.")
        return names, definitions, np.array(data["matrix"])

    print(f"Computing embeddings for {len(ctcae_list)} CTCAE terms (one-time, will cache)...")
    matrix = []
    for i, entry in enumerate(ctcae_list, 1):
        emb = _get_embedding(f"{entry['adverse_effect']}: {entry['definition']}")
        matrix.append(emb)
        if i % 100 == 0:
            print(f"  {i}/{len(ctcae_list)}...")
    print(f"  Done. Saving to {EMBED_CACHE}")
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"names": names, "matrix": matrix}, f)
    return names, definitions, np.array(matrix)


def _top_candidates(query: str, ctcae_names: list[str], ctcae_defs: list[str],
                    ctcae_matrix: np.ndarray) -> list[tuple[str, str]]:
    """Embed the query and return top EMBED_CANDIDATES (name, definition) pairs by cosine sim."""
    emb   = np.array(_get_embedding(query))
    norms = np.linalg.norm(ctcae_matrix, axis=1)
    sims  = ctcae_matrix.dot(emb) / (norms * np.linalg.norm(emb) + 1e-9)
    top_i = np.argsort(sims)[::-1][:EMBED_CANDIDATES]
    return [(ctcae_names[i], ctcae_defs[i]) for i in top_i]


def normalize_term(term: str, evidence: str, ctcae_names: list[str],
                   ctcae_defs: list[str], ctcae_matrix: np.ndarray,
                   ctcae_lookup: dict) -> tuple[str, int | None, str | None]:
    """
    Map extracted term to an official CTCAE name, assign a grade using CTCAE
    grade criteria, and update confidence — all in one llama call.
    Returns (normalized_term, grade, confidence).
    Exact matches skip the llama call and return (term, None, None).
    """
    ctcae_set = set(ctcae_names)
    if term in ctcae_set:
        return term, None, None

    query      = f"{term}: {evidence}" if evidence else term
    candidates = _top_candidates(query, ctcae_names, ctcae_defs, ctcae_matrix)

    def grade_block(name: str) -> str:
        entry = ctcae_lookup.get(name, {})
        grades = [f"G{i}={entry[f'grade_{i}']}" for i in range(1, 6) if entry.get(f"grade_{i}")]
        return " | ".join(grades) if grades else "no grade criteria"

    candidate_block = "\n".join(
        f"- {name} [{grade_block(name)}]"
        for name, defn in candidates
    )

    prompt = f"""You are a clinical oncology expert. Given an extracted adverse event term and patient evidence, you must:
1. Choose the single best matching CTCAE from the candidates
2. Assign a CTCAE grade (1-5) based on the grade criteria shown and the patient evidence, or null if unclear
3. Assign confidence: high if the evidence clearly and specifically describes the symptom, medium if clearly mentioned but without detail, low if vague or inferred

Extracted term: "{term}"
Patient evidence: {evidence if evidence else "not provided"}

Candidates (with definitions and grade criteria):
{candidate_block}

Reply with ONLY this format, nothing else:
ctcae_term | grade | confidence"""

    raw = _call_ollama(prompt, think=True, live=DEBUG, label="normalize", model=NORM_MODEL).strip()

    # parse "term | grade | confidence"
    parts = [p.strip() for p in raw.split("|")]
    normalized = parts[0] if parts else ""
    if normalized not in ctcae_set:
        print(f"    [normalize] returned {normalized!r} — not in CTCAE list, keeping original")
        normalized = term

    grade = None
    if len(parts) > 1:
        raw_grade = re.sub(r"[Gg]rade\s*", "", parts[1]).strip().lstrip("Gg")
        try:
            g = int(raw_grade)
            grade = g if 1 <= g <= 5 else None
        except ValueError:
            grade = None

    confidence = parts[2].lower() if len(parts) > 2 and parts[2].lower() in ("high", "medium", "low") else None

    return normalized, grade, confidence


# ---------------------------------------------------------------------------
# Process one post
# ---------------------------------------------------------------------------
def process_post(post: dict, ctcae_names: list[str], ctcae_defs: list[str],
                 ctcae_matrix: np.ndarray, ctcae_lookup: dict) -> list[dict]:
    print("  [Step 1] Extracting adverse effects...")
    extracted = extract_from_post(post["selftext"])
    print(f"  Extracted {len(extracted)} adverse effect(s)")

    if not extracted:
        return []

    results = []
    seen: dict[str, int] = {}  # normalized_term -> index in results, for dedup

    conf_rank = {"high": 3, "medium": 2, "low": 1, "": 0}

    for match in extracted:
        term                       = match["ctcae_described"]
        evidence                   = match["evidence"]
        normalized, grade, confidence = normalize_term(
            term, evidence, ctcae_names, ctcae_defs, ctcae_matrix, ctcae_lookup
        )
        # fall back to extraction values if normalization didn't produce them
        if grade is None:
            grade = match["grade"]
        if confidence is None:
            confidence = match["confidence"]

        mapped = "→" if normalized != term else "="
        print(f"  {term!r} {mapped} {normalized!r}  grade={grade}  conf={confidence}")

        row = {
            "post_id":          post["post_id"],
            "subreddit":        post["subreddit"],
            "permalink":        post["permalink"],
            "ctcae_described":  term,
            "ctcae_normalized": normalized,
            "grade":            grade,
            "evidence":         evidence,
            "likely_relation":  match["likely_relation"],
            "confidence":       confidence,
        }

        if normalized in seen:
            # merge: keep higher grade and confidence, combine evidence
            existing = results[seen[normalized]]
            existing_grade = existing["grade"] if isinstance(existing["grade"], int) else 0
            new_grade      = grade if isinstance(grade, int) else 0
            existing_conf  = conf_rank.get(str(existing["confidence"]).lower(), 0)
            new_conf       = conf_rank.get(str(confidence).lower(), 0)
            merged_grade = new_grade if new_grade > existing_grade else existing_grade
            merged_conf  = confidence if new_conf > existing_conf else existing["confidence"]
            # combine evidence if different
            if evidence and evidence != existing["evidence"]:
                merged_evidence = f"{existing['evidence']} | {evidence}"
            else:
                merged_evidence = existing["evidence"]
            print(f"  [dedup] merging duplicate {normalized!r} — grade={merged_grade} conf={merged_conf}")
            existing["grade"]      = merged_grade or None
            existing["confidence"] = merged_conf
            existing["evidence"]   = merged_evidence
        else:
            seen[normalized] = len(results)
            results.append(row)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--restart", action="store_true",
                        help="Clear checkpoint and output CSV before running")
    parser.add_argument("--debug", action="store_true",
                        help="Show live model conversation output")
    args = parser.parse_args()

    if args.debug:
        global DEBUG
        DEBUG = True

    if args.restart:
        for path in (CHECKPOINT_FILE, OUTPUT_CSV):
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted {path}")
        print()

    start_time = time.time()

    print("=" * 60)
    print("CTCAE Mapper — qwen3 (free-form extraction)")
    print(f"Model:   {QWEN_MODEL}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.limit:
        print(f"Mode:    TEST — {args.limit} post(s)")
    print("=" * 60)

    ctcae_list              = load_ctcae(CTCAE_CSV)
    ctcae_names, ctcae_defs, ctcae_matrix = build_ctcae_embeddings(ctcae_list)
    ctcae_lookup            = {c["adverse_effect"]: c for c in ctcae_list}
    print(f"Loaded {len(ctcae_names)} official CTCAE terms")

    posts = load_posts(POSTS_CSV)
    print(f"Loaded {len(posts)} posts")

    done = load_checkpoint(CHECKPOINT_FILE)
    remaining = [p for p in posts if p["post_id"] not in done]
    if args.limit:
        remaining = remaining[:args.limit]
        print(f"Limiting to {len(remaining)} post(s)")
    else:
        print(f"{len(posts) - len(remaining)} done, {len(remaining)} to process")
    print("-" * 60)

    file_exists = os.path.exists(OUTPUT_CSV)
    out_f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    total_posts   = len(remaining)
    total_matches = 0

    try:
        for post_num, post in enumerate(remaining, start=1):
            post_start    = time.time()
            elapsed_total = time.time() - start_time

            if post_num > 1:
                avg = elapsed_total / (post_num - 1)
                eta = avg * (total_posts - post_num + 1)
                eta_str = f"ETA ~{eta / 60:.1f} min"
            else:
                eta_str = "ETA: calculating..."

            print(f"\n[Post {post_num}/{total_posts}] {post['post_id']}"
                  f"  |  r/{post['subreddit']}  |  {eta_str}")

            matches = process_post(post, ctcae_names, ctcae_defs, ctcae_matrix, ctcae_lookup)

            for row in matches:
                writer.writerow(row)
            out_f.flush()
            total_matches += len(matches)
            save_checkpoint(CHECKPOINT_FILE, post["post_id"])

            print(f"  → {len(matches)} match(es)  |  "
                  f"{time.time() - post_start:.1f}s  |  "
                  f"Total: {total_matches}")

    finally:
        out_f.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:    {elapsed / 60:.1f} min")
    print(f"Total matches: {total_matches}")
    print(f"Output:        {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
