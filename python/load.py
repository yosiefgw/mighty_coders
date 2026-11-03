"""Load articles and embedded chunks into MariaDB (tables from sql/init/01_vector_schema.sql).

Usage (from the repo root, database running):
    python python/load.py data/articles.jsonl data/chunks.jsonl
    python python/load.py data/articles.jsonl data/chunks.jsonl --force   # reload all chunks

Input format (JSONL = one JSON object per line). PROPOSAL, to be confirmed by the team.

  articles.jsonl, one line per article:
    {"id": 19331, "revision_id": 1250000000, "title": "Mars",
     "url": "https://en.wikipedia.org/wiki/Mars", "word_count": 15234,
     "last_edited": "2025-06-01T12:34:56Z",
     "categories": ["Planets of the Solar System", "Mars"],
     "links_to": ["Phobos (moon)", "Deimos (moon)", "Jupiter"]}

  chunks.jsonl, one line per chunk:
    {"article_id": 19331, "chunk_index": 0, "word_count": 182,
     "text": "Mars is the fourth planet ...", "embedding": [0.0123, -0.0456, ...]}   # 384 numbers

What the loader does, all in ONE transaction (everything is loaded, or nothing):
  1. Checks the files first: every chunk belongs to an article in the file, has 384 numbers,
     and (article_id, chunk_index) appears only once. On a problem: error, nothing written.
  2. Articles: INSERT ... ON DUPLICATE KEY UPDATE (an existing article is updated, never deleted).
     We never DELETE an article row, because that would also remove the links pointing to it
     (ON DELETE CASCADE on links).
  3. Chunks: only for articles that are new or have a new revision_id (or with --force).
     Old chunks are removed with a direct DELETE on chunks. Never through a cascade: a cascade
     delete breaks the vector index (see the comment in 01_vector_schema.sql and
     test_known_bug_cascade_breaks_vector_index in tests/test_schema.py).
  4. Categories and links of the articles in the file are replaced.
     A link is stored only if its target title is also in our corpus.
     Titles are compared EXACTLY (title uses COLLATE utf8mb4_bin), so "links_to" must use the
     same form as "title": spaces, not underscores ("Phobos (moon)", not "Phobos_(moon)").
     A link to a redirect page (e.g. "Phobos" -> "Phobos (moon)") is lost unless the downloader
     resolves redirects first (MediaWiki API parameter "redirects").
  5. COMMIT. On any error: ROLLBACK.
Articles that are in the database but not in the file are left as they are.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection  # noqa: E402

DIM = 384  # bge-small-en-v1.5; must match VECTOR(384) in the schema


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_datetime(text):
    """'2025-06-01T12:34:56Z' (MediaWiki format, UTC) -> '2025-06-01 12:34:56' (MariaDB DATETIME)."""
    return text.replace("T", " ").rstrip("Z")


def check(articles, chunks):
    """Raise ValueError if the files do not fit together. Nothing is written before this passes."""
    article_ids = [a["id"] for a in articles]
    duplicates = [i for i, n in Counter(article_ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"article id(s) appear more than once: {duplicates[:5]}")
    known = set(article_ids)
    seen = set()
    for n, c in enumerate(chunks, start=1):
        key = (c["article_id"], c["chunk_index"])
        if c["article_id"] not in known:
            raise ValueError(f"chunk line {n}: article_id {c['article_id']} is not in the articles file")
        if key in seen:
            raise ValueError(f"chunk line {n}: (article_id, chunk_index) {key} appears twice")
        if len(c["embedding"]) != DIM:
            raise ValueError(f"chunk line {n}: embedding has {len(c['embedding'])} numbers, expected {DIM}")
        seen.add(key)


def load(conn, articles, chunks, force=False, commit=True):
    """Write `articles` and `chunks` (lists of dicts, format above). Returns a dict with counts.
    commit=False is for tests: the caller decides about COMMIT / ROLLBACK."""
    check(articles, chunks)
    chunks_by_article = {}
    for c in chunks:
        chunks_by_article.setdefault(c["article_id"], []).append(c)

    stats = Counter()
    cur = conn.cursor()
    try:
        # Which articles need new chunks? (new article, new revision, or --force)
        ids = [a["id"] for a in articles]
        old_revision = {}
        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(f"SELECT id, revision_id FROM articles WHERE id IN ({placeholders})", ids)
            old_revision = dict(cur.fetchall())

        for a in articles:
            changed = force or old_revision.get(a["id"]) != a["revision_id"]
            cur.execute(
                "INSERT INTO articles (id, revision_id, title, url, word_count, last_edited) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE revision_id = VALUES(revision_id), title = VALUES(title), "
                "url = VALUES(url), word_count = VALUES(word_count), "
                "last_edited = VALUES(last_edited), fetched_at = CURRENT_TIMESTAMP",
                (a["id"], a["revision_id"], a["title"], a["url"], a["word_count"], to_datetime(a["last_edited"])),
            )
            stats["articles"] += 1

            if changed:
                # Direct DELETE on chunks, never a cascade (see module docstring).
                cur.execute("DELETE FROM chunks WHERE article_id = %s", (a["id"],))
                stats["chunks_deleted"] += cur.rowcount
                rows = [
                    (c["article_id"], c["chunk_index"], c["word_count"], c["text"],
                     json.dumps([float(x) for x in c["embedding"]]))
                    for c in chunks_by_article.get(a["id"], [])
                ]
                if rows:
                    cur.executemany(
                        "INSERT INTO chunks (article_id, chunk_index, word_count, chunk_text, embedding) "
                        "VALUES (%s, %s, %s, %s, VEC_FromText(%s))",
                        rows,
                    )
                stats["chunks_inserted"] += len(rows)
            else:
                stats["articles_unchanged"] += 1

            # Categories: replace this article's list.
            cur.execute("DELETE FROM article_categories WHERE article_id = %s", (a["id"],))
            for name in a.get("categories", []):
                cur.execute("INSERT IGNORE INTO categories (name) VALUES (%s)", (name,))
                cur.execute(
                    "INSERT IGNORE INTO article_categories (article_id, category_id) "
                    "SELECT %s, id FROM categories WHERE name = %s",
                    (a["id"], name),
                )
                stats["article_categories"] += 1

        # Links last, when every article of the file has its row. Only links inside the corpus.
        for a in articles:
            cur.execute("DELETE FROM links WHERE from_article_id = %s", (a["id"],))
            for target in a.get("links_to", []):
                cur.execute(
                    "INSERT IGNORE INTO links (from_article_id, to_article_id) "
                    "SELECT %s, id FROM articles WHERE title = %s AND id <> %s",
                    (a["id"], target, a["id"]),
                )
                stats["links"] += cur.rowcount

        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Load articles and embedded chunks into MariaDB.")
    parser.add_argument("articles", help="articles.jsonl")
    parser.add_argument("chunks", help="chunks.jsonl")
    parser.add_argument("--force", action="store_true", help="reload chunks even if revision_id is unchanged")
    args = parser.parse_args()

    articles, chunks = read_jsonl(args.articles), read_jsonl(args.chunks)
    start = time.perf_counter()
    conn = get_connection(autocommit=False)
    try:
        stats = load(conn, articles, chunks, force=args.force)
    finally:
        conn.close()
    print(f"Loaded in {time.perf_counter() - start:.1f} s:")
    for key in ["articles", "articles_unchanged", "chunks_deleted", "chunks_inserted",
                "article_categories", "links"]:
        print(f"  {key:<20} {stats.get(key, 0)}")


if __name__ == "__main__":
    main()
