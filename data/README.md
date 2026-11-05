# Dataset

**Source:** English Wikipedia, downloaded with the [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page)
(`https://en.wikipedia.org/w/api.php`): article text (TextExtracts, plain text), revision id and time,
categories, and links.

**License:** Wikipedia text is available under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Every article keeps its Wikipedia URL in
the `articles.url` column, and the web page links back to it.

**What we use:** the categories "Solar System" (depth 1) and "Planets of the Solar System" (depth 1).
In our download (November 2026): 166 articles, 3,753 chunks of about 175 words (30 words overlap),
each with a 384-number embedding from `BAAI/bge-small-en-v1.5`.

**How to get it** (from the repository root; about 40 minutes, because Wikipedia rate-limits the API and
the downloader waits and retries):

```bash
python python/fetch.py --category "Solar System" --depth 1 --output data
python python/fetch.py --category "Planets of the Solar System" --depth 1 --output data/planets
```

Each run writes two files:

- `articles.jsonl`: one article per line: `id` (page id), `revision_id`, `title`, `url`, `word_count`,
  `last_edited`, `categories`, `links_to`
- `chunks.jsonl`: one chunk per line: `article_id`, `chunk_index`, `word_count`, `text`, `embedding` (384 numbers)

**How to load it into MariaDB:** see "How to run it" in the main [README](../README.md#how-to-run-it).

**Repeatability:** Wikipedia changes every day, so a later download can differ slightly. `revision_id`
in `articles` records exactly which version of each article we stored. The downloaded files are not
committed (see `.gitignore`).
