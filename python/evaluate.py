"""Ground-truth evaluation: turn "the search works" into numbers.

For every question in eval/ground_truth.jsonl (a question plus the title(s) of the article that
answers it) the question is embedded with bge-small-en-v1.5 and searched with search():

  * hit@1, hit@k   share of questions whose correct article is the first result / in the top k chunks
  * MRR            mean reciprocal rank of the first chunk from a correct article (1, 1/2, 1/3, ... or 0)
  * index recall   share of the exact top-k chunks that the approximate index also returns
  * latency        median and 95th percentile of search() time per query, over --repeat runs

"exact" (no vector index) is the ground truth for index recall; "index" uses the HNSW index with
the given mhnsw_ef_search values.

Usage (corpus loaded, from the repo root):
    python python/evaluate.py
    python python/evaluate.py --k 5 --ef-search 10 20 50 100 --repeat 5
    python python/evaluate.py --prefix      # bge query instruction in front of each question
    python python/evaluate.py --csv docs/eval_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection  # noqa: E402
from search import DIM, search  # noqa: E402

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "eval" / "ground_truth.jsonl"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------- pure helpers (unit-tested, no database needed) ----------

def read_questions(path) -> list[dict]:
    """One JSON object per line: {"id": ..., "question": ..., "answers": [title, ...]}."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            q = json.loads(line)
            if not q.get("question") or not q.get("answers"):
                raise ValueError(f"{path}:{line_no}: needs 'question' and a non-empty 'answers' list")
            questions.append(q)
    return questions


def first_hit_rank(result_titles: list[str], answers: list[str]) -> int | None:
    """1-based rank of the first result whose article is a correct answer, or None."""
    for rank, title in enumerate(result_titles, start=1):
        if title in answers:
            return rank
    return None


def score(ranks: list[int | None], k: int) -> dict:
    """hit@1, hit@k and MRR from the first-hit rank of each question."""
    n = len(ranks)
    if n == 0:
        return {"hit@1": 0.0, f"hit@{k}": 0.0, "mrr": 0.0}
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
    }


def overlap(approx_ids: list[int], exact_ids: list[int]) -> float:
    """Share of the exact top-k that the approximate search also returned (index recall)."""
    if not exact_ids:
        return 1.0
    return len(set(approx_ids) & set(exact_ids)) / len(exact_ids)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in 0..100."""
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))          # 1-based nearest rank
    return ordered[max(0, min(len(ordered), rank) - 1)]


# ---------- the evaluation ----------

def timed_search(conn, vector, k, mode, ef_search, repeat):
    """Run search() `repeat` times; return the result rows and the median time in ms."""
    times = []
    rows = []
    for _ in range(repeat):
        start = time.perf_counter()
        rows = search(conn, vector, k=k, mode=mode, ef_search=ef_search)
        times.append((time.perf_counter() - start) * 1000)
    return rows, statistics.median(times)


def evaluate(conn, questions, vectors, k, ef_values, repeat):
    """One result row per configuration: exact, then index with each ef_search value."""
    exact_ids, results = [], []
    configs = [("exact", None)] + [("index", ef) for ef in ef_values]
    for mode, ef in configs:
        ranks, times, recalls = [], [], []
        for i, (q, vec) in enumerate(zip(questions, vectors)):
            rows, ms = timed_search(conn, vec, k, mode, ef, repeat)
            ids = [r["chunk_id"] for r in rows]
            ranks.append(first_hit_rank([r["title"] for r in rows], q["answers"]))
            times.append(ms)
            if mode == "exact":
                exact_ids.append(ids)
            else:
                recalls.append(overlap(ids, exact_ids[i]))
        results.append({
            "mode": mode,
            "ef_search": ef if ef is not None else "",
            **score(ranks, k),
            "index_recall": statistics.mean(recalls) if recalls else 1.0,
            "median_ms": statistics.median(times),
            "p95_ms": percentile(times, 95),
        })
    conn.cursor().execute("SET SESSION mhnsw_ef_search = DEFAULT")
    return results


def missing_answers(conn, questions) -> list[tuple]:
    """(question id, title) for answer titles that are not in the articles table."""
    cur = conn.cursor()
    cur.execute("SELECT title FROM articles")
    titles = {row[0] for row in cur.fetchall()}
    return [(q["id"], a) for q in questions for a in q["answers"] if a not in titles]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-truth evaluation of search().")
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="ground-truth JSONL file")
    parser.add_argument("--k", type=int, default=5, help="top k chunks per question (default 5)")
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 50, 100],
                        help="mhnsw_ef_search values for the index (default 10 20 50 100)")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per query (default 3)")
    parser.add_argument("--prefix", action="store_true", help="add the bge query instruction to each question")
    parser.add_argument("--csv", help="also write the results table to this CSV file")
    args = parser.parse_args()

    from embed import embed_batch  # imported here: loading the model is slow

    questions = read_questions(args.file)
    conn = get_connection()
    missing = missing_answers(conn, questions)
    if missing:
        print("Answer titles not in the database (questions skipped):", missing)
        bad_ids = {qid for qid, _ in missing}
        questions = [q for q in questions if q["id"] not in bad_ids]

    texts = [(BGE_QUERY_PREFIX if args.prefix else "") + q["question"] for q in questions]
    vectors = embed_batch(texts)
    assert all(len(v) == DIM for v in vectors)

    cur = conn.cursor()
    cur.execute("SELECT VERSION(), (SELECT COUNT(*) FROM articles), (SELECT COUNT(*) FROM chunks)")
    version, n_articles, n_chunks = cur.fetchone()
    print(f"Setup: MariaDB {version} | {n_articles} articles, {n_chunks} chunks, {DIM} dims | "
          f"{platform.platform()} | {len(questions)} questions, k={args.k}, repeat={args.repeat}, "
          f"prefix={'yes' if args.prefix else 'no'}")

    results = evaluate(conn, questions, vectors, args.k, args.ef_search, args.repeat)
    conn.close()

    hit_k = f"hit@{args.k}"
    print(f"\n{'mode':<6} {'ef':>4} {'hit@1':>6} {hit_k:>6} {'MRR':>6} {'idx recall':>10} {'median ms':>10} {'p95 ms':>8}")
    for r in results:
        print(f"{r['mode']:<6} {str(r['ef_search']):>4} {r['hit@1']:>6.2f} {r[hit_k]:>6.2f} {r['mrr']:>6.2f} "
              f"{r['index_recall']:>10.3f} {r['median_ms']:>10.2f} {r['p95_ms']:>8.2f}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
