"""Download Wikipedia articles from a category, chunk them, and embed with bge-small-en-v1.5.

Usage (from the repo root, .venv activated):
    python python/fetch.py                          # Solar System, depth 1, writes to data/
    python python/fetch.py --depth 2                # also recurse one level into subcategories
    python python/fetch.py --category "Gas giants"  # any Wikipedia category name
    python python/fetch.py --output /tmp/out        # custom output directory

Outputs (JSONL, one JSON object per line, format matches load.py):
    articles.jsonl  — one article per line
    chunks.jsonl    — one chunk per line, includes 384-dim embedding

Chunking strategy: sliding window, target 175 words, 30-word overlap (150-200 words per chunk).
Embedding: BAAI/bge-small-en-v1.5 via embed.py (batch mode for speed).
Rate limiting: one request per second by default (polite MediaWiki API usage).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_batch  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "mighty_coders/1.0 (Constructor University; github.com/playboifufalo/mighty_coders)"

CHUNK_TARGET = 175   # words per chunk (aim for 150-200)
CHUNK_OVERLAP = 30   # words shared between consecutive chunks
BATCH_SIZE = 50      # titles per MediaWiki API request (hard limit: 50 for most props)
REQUEST_DELAY = 1.0  # seconds between requests


def _get(params: dict) -> dict:
    """GET the MediaWiki API with standard headers and a small delay."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(
        API,
        params={"format": "json", "formatversion": "2", **params},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Category traversal
# ---------------------------------------------------------------------------

def _category_members(category: str, cm_type: str) -> Iterator[dict]:
    """Yield all members of one type ("page" or "subcat") from a category, handling pagination."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": cm_type,
        "cmlimit": "500",
        "cmprop": "title",
    }
    while True:
        data = _get(params)
        yield from data["query"]["categorymembers"]
        cont = data.get("continue", {}).get("cmcontinue")
        if cont is None:
            break
        params["cmcontinue"] = cont


def collect_titles(category: str, depth: int) -> list[str]:
    """Return article titles in *category* and subcategories up to *depth* levels deep."""
    titles: set[str] = set()
    seen_cats: set[str] = {category}

    def _recurse(cat: str, remaining: int) -> None:
        for page in _category_members(cat, "page"):
            titles.add(page["title"])
        if remaining > 0:
            for subcat in _category_members(cat, "subcat"):
                name = subcat["title"].removeprefix("Category:")
                if name not in seen_cats:
                    seen_cats.add(name)
                    _recurse(name, remaining - 1)

    _recurse(category, depth)
    return sorted(titles)


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def _fetch_batch(titles: list[str]) -> list[dict]:
    """Fetch revisions, plain-text extract, categories, and links for up to 50 titles."""
    data = _get({
        "action": "query",
        "titles": "|".join(titles),
        "prop": "revisions|extracts|categories|links",
        # revisions: latest id + timestamp
        "rvprop": "ids|timestamp",
        "rvlimit": "1",
        # plain text, no section markers
        "explaintext": "true",
        "exsectionformat": "plain",
        # categories: skip hidden maintenance cats
        "cllimit": "max",
        "clshow": "!hidden",
        # links: main namespace only (no File:, Template:, etc.)
        "pllimit": "max",
        "plnamespace": "0",
        "redirects": "true",   # resolve redirects automatically
    })

    articles = []
    pages = data.get("query", {}).get("pages", [])
    for page in pages:
        # Skip missing/special pages
        if page.get("missing") or not page.get("extract", "").strip():
            continue

        rev = (page.get("revisions") or [{}])[0]
        revision_id = rev.get("revid", 0)
        last_edited = rev.get("timestamp", "1970-01-01T00:00:00Z")

        cats = [c["title"].removeprefix("Category:") for c in page.get("categories", [])]

        # Wikipedia link titles use spaces (not underscores); the API already returns them that way.
        links = [lk["title"] for lk in page.get("links", [])]

        text = page["extract"].strip()
        word_count = len(text.split())

        articles.append({
            "id": page["pageid"],
            "revision_id": revision_id,
            "title": page["title"],
            "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
            "word_count": word_count,
            "last_edited": last_edited,
            "categories": cats,
            "links_to": links,
            "text": text,  # kept only during processing; not written to articles.jsonl
        })
    return articles


def fetch_articles(titles: list[str]) -> list[dict]:
    """Fetch all *titles* in batches of BATCH_SIZE, printing progress."""
    result = []
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i: i + BATCH_SIZE]
        fetched = _fetch_batch(batch)
        result.extend(fetched)
        print(f"  fetched {len(result)}/{len(titles)} articles", end="\r", flush=True)
    print()
    return result


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks of roughly *target* words.

    Returns an empty list for very short texts (fewer than overlap words).
    """
    words = text.split()
    if len(words) <= overlap:
        return [text] if words else []

    step = max(1, target - overlap)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_chunks(articles: list[dict]) -> list[dict]:
    """Chunk and embed all articles; return a flat list of chunk dicts."""
    all_texts: list[str] = []
    index_map: list[tuple[int, int, int]] = []  # (article_id, chunk_index, word_count)

    for art in articles:
        for ci, chunk in enumerate(chunk_text(art["text"])):
            all_texts.append(chunk)
            index_map.append((art["id"], ci, len(chunk.split())))

    if not all_texts:
        return []

    print(f"  embedding {len(all_texts)} chunks …", flush=True)
    embeddings = embed_batch(all_texts)

    return [
        {
            "article_id": article_id,
            "chunk_index": chunk_index,
            "word_count": wc,
            "text": text,
            "embedding": emb,
        }
        for (article_id, chunk_index, wc), text, emb in zip(index_map, all_texts, embeddings)
    ]


def write_jsonl(path: Path, records: list[dict], exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: v for k, v in rec.items() if k not in exclude}) + "\n")
    print(f"  wrote {len(records):,} records → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia articles and build JSONL for load.py.")
    parser.add_argument("--category", default="Solar System", help="Wikipedia category name (default: Solar System)")
    parser.add_argument("--depth", type=int, default=1, help="subcategory recursion depth (default: 1)")
    parser.add_argument("--output", default="data", help="output directory (default: data/)")
    args = parser.parse_args()

    out = Path(args.output)

    print(f"Collecting titles from Category:{args.category} (depth={args.depth}) …")
    titles = collect_titles(args.category, args.depth)
    print(f"  found {len(titles)} articles")

    if not titles:
        print("Nothing to fetch. Check the category name.")
        sys.exit(1)

    print("Fetching article content …")
    articles = fetch_articles(titles)
    print(f"  {len(articles)} articles with text")

    print("Chunking and embedding …")
    chunks = build_chunks(articles)
    print(f"  {len(chunks)} chunks total")

    print("Writing output …")
    write_jsonl(out / "articles.jsonl", articles, exclude={"text"})
    write_jsonl(out / "chunks.jsonl", chunks)

    print("Done.")


if __name__ == "__main__":
    main()
