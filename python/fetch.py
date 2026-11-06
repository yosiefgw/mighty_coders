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
MAX_RETRIES = 5      # retries when Wikipedia answers 429 (too many requests) or 503


def _retry_wait(resp: requests.Response, attempt: int) -> float:
    """Seconds to wait before a retry: the server's Retry-After if given, else 5, 10, 20, ... s."""
    try:
        return max(1.0, float(resp.headers.get("Retry-After", "")))
    except ValueError:
        return 5.0 * 2 ** attempt


def _get(params: dict) -> dict:
    """GET the MediaWiki API with standard headers, a small delay, and retries on 429/503."""
    for attempt in range(MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY)
        resp = requests.get(
            API,
            params={"format": "json", "formatversion": "2", **params},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
            wait = _retry_wait(resp, attempt)
            print(f"\n  Wikipedia says {resp.status_code}, waiting {wait:.0f} s (retry {attempt + 1}/{MAX_RETRIES}) ...",
                  flush=True)
            time.sleep(wait)
            continue
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
        "cmnamespace": "0",
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

def _fetch_one(title: str) -> dict | None:
    """Fetch one article: revision, full plain-text extract, categories, and links.

    Fetches a single title per request (TextExtracts returns only one full extract
    per multi-title request). Follows ``continue`` tokens to paginate through all
    links and categories. Raises RuntimeError if the API returns an error object.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|extracts|categories|links",
        # revisions: latest id + timestamp (rvlimit not set — only valid for a single page)
        "rvprop": "ids|timestamp",
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
    }

    page_data: dict | None = None
    cats: list[str] = []
    links: list[str] = []

    while True:
        data = _get(params)
        if "error" in data:
            raise RuntimeError(f"MediaWiki API error: {data['error']}")

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            break

        page = pages[0]
        if page.get("missing"):
            return None

        if page_data is None:
            page_data = page  # first call: capture revisions + extract

        # Accumulate paginated categories and links across continuation calls
        cats.extend(c["title"].removeprefix("Category:") for c in page.get("categories", []))
        links.extend(lk["title"] for lk in page.get("links", []))

        cont = data.get("continue")
        if not cont:
            break
        params = {**params, **{k: v for k, v in cont.items()}}

    if page_data is None or not page_data.get("extract", "").strip():
        return None

    rev = (page_data.get("revisions") or [{}])[0]
    text = page_data["extract"].strip()

    return {
        "id": page_data["pageid"],
        "revision_id": rev.get("revid", 0),
        "title": page_data["title"],
        "url": f"https://en.wikipedia.org/wiki/{page_data['title'].replace(' ', '_')}",
        "word_count": len(text.split()),
        "last_edited": rev.get("timestamp", "1970-01-01T00:00:00Z"),
        "categories": cats,
        "links_to": links,
        "text": text,  # kept only during processing; not written to articles.jsonl
    }


def _fetch_batch(titles: list[str]) -> list[dict]:
    """Fetch revisions, plain-text extract, categories, and links for each title."""
    articles = []
    for title in titles:
        article = _fetch_one(title)
        if article is not None:
            articles.append(article)
    return articles


def fetch_articles(titles: list[str]) -> list[dict]:
    """Fetch all *titles* in batches of BATCH_SIZE, printing progress.

    A category can list a redirect page next to its target. With redirects=true both titles
    resolve to the same page, so each page id is kept only once (load.py refuses duplicates).
    """
    result = []
    seen_ids: set[int] = set()
    duplicates = 0
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i: i + BATCH_SIZE]
        for article in _fetch_batch(batch):
            if article["id"] in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(article["id"])
            result.append(article)
        print(f"  fetched {len(result)}/{len(titles)} articles", end="\r", flush=True)
    print()
    if duplicates:
        print(f"  skipped {duplicates} duplicate(s): redirect titles that point to an article already fetched")
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
