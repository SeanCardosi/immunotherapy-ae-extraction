# main.py

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from arctic_shift import search_posts
from config import (
    SUBREDDITS,
    SEARCH_TERMS,
    DATE_WINDOWS,
    POST_LIMIT_PER_SEARCH,
    REQUEST_SLEEP_SECONDS,
)


OUTPUT_DIR    = "output"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "immunotherapy_posts.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "collection_progress.json")

FIELDNAMES = [
    "post_id",
    "subreddit",
    "created_utc",
    "created_date",
    "score",
    "num_comments",
    "title",
    "selftext",
    "combined_text",
    "permalink",
    "matched_search_term",
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_deleted_or_empty(text: str) -> bool:
    return normalize_text(text).lower() in ["", "[deleted]", "[removed]"]


def utc_to_date(created_utc) -> str:
    try:
        return datetime.fromtimestamp(int(created_utc), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# PROGRESS / RESUME
# ──────────────────────────────────────────────────────────────────────────────

def load_progress(path: str) -> tuple[set[str], set[str]]:
    """
    Returns:
      seen_post_ids  — post IDs already written to the CSV (skip these)
      done_searches  — (subreddit, term, after, before) combos already completed
    """
    if not Path(path).exists():
        return set(), set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("seen_post_ids", [])), set(
        tuple(x) for x in data.get("done_searches", [])
    )


def save_progress(path: str, seen_post_ids: set[str], done_searches: set[tuple]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "seen_post_ids": sorted(seen_post_ids),
                "done_searches": [list(x) for x in sorted(done_searches)],
            },
            f,
        )


# ──────────────────────────────────────────────────────────────────────────────
# COLLECTION
# ──────────────────────────────────────────────────────────────────────────────

def collect_immunotherapy_posts():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_post_ids, done_searches = load_progress(PROGRESS_FILE)
    resuming = len(seen_post_ids) > 0 or len(done_searches) > 0

    if resuming:
        print(f"[Resume] {len(seen_post_ids)} posts already collected, "
              f"{len(done_searches)} search windows already done.\n")

    # Open output CSV — append on resume, fresh write otherwise
    file_exists = Path(OUTPUT_FILE).exists()
    mode = "a" if (resuming and file_exists) else "w"
    out_fh = open(OUTPUT_FILE, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=FIELDNAMES, extrasaction="ignore")
    if mode == "w":
        writer.writeheader()

    total_searches = len(SUBREDDITS) * len(SEARCH_TERMS) * len(DATE_WINDOWS)
    search_num = 0
    new_posts  = 0

    try:
        for subreddit in SUBREDDITS:
            for search_term in SEARCH_TERMS:
                for after, before in DATE_WINDOWS:
                    search_num += 1
                    search_key = (subreddit, search_term, after, before)

                    if search_key in done_searches:
                        print(f"[{search_num}/{total_searches}] Skipping (done): "
                              f"r/{subreddit} | '{search_term}' | {after}→{before}")
                        continue

                    print(f"[{search_num}/{total_searches}] "
                          f"r/{subreddit} | '{search_term}' | {after}→{before}")

                    posts = search_posts(
                        subreddit=subreddit,
                        query=search_term,
                        after=after,
                        before=before,
                        limit=POST_LIMIT_PER_SEARCH,
                    )

                    for post in posts:
                        post_id = post.get("id")
                        if not post_id or post_id in seen_post_ids:
                            continue

                        title    = normalize_text(post.get("title", ""))
                        selftext = normalize_text(post.get("selftext", ""))

                        if is_deleted_or_empty(title) and is_deleted_or_empty(selftext):
                            continue

                        row = {
                            "post_id":             post_id,
                            "subreddit":           post.get("subreddit", subreddit),
                            "created_utc":         post.get("created_utc"),
                            "created_date":        utc_to_date(post.get("created_utc")),
                            "score":               post.get("score"),
                            "num_comments":        post.get("num_comments"),
                            "title":               title,
                            "selftext":            selftext,
                            "combined_text":       f"{title} {selftext}".strip(),
                            "permalink":           post.get("permalink"),
                            "matched_search_term": search_term,
                        }
                        writer.writerow(row)
                        seen_post_ids.add(post_id)
                        new_posts += 1

                    out_fh.flush()
                    done_searches.add(search_key)
                    save_progress(PROGRESS_FILE, seen_post_ids, done_searches)

                    time.sleep(REQUEST_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n[Interrupted] Progress saved — re-run to resume.")
    finally:
        out_fh.close()

    return len(seen_post_ids), new_posts


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Starting Arctic Shift immunotherapy post collection.")
    print("Collects broad immunotherapy-related posts only.")
    print("Does not filter relevance or extract symptoms — those are separate steps.\n")

    total, new_posts = collect_immunotherapy_posts()

    print("\nDone.")
    print(f"Total unique posts collected (all runs): {total}")
    print(f"New posts added this run:                {new_posts}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
