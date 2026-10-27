Project 4: [Vector search and RAG, natively](https://mariadb.org/bachelor_hackathon_2026-09/)

# mighty_coders

Retrieval over Wikipedia with MariaDB's native VECTOR type.
Capstone I, Constructor University, with the MariaDB Foundation.

## What this is
TODO

## What we found
TODO

## Versions and setup
- MariaDB: 11.8 (Docker image `mariadb:11.8`). VECTOR needs 11.7 or newer.
- Python 3 with `pymysql` and `pytest`.
- Dataset: TODO, see [data/README.md](data/README.md).

## Data model
TODO

## How to run it
```bash
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose down -v && docker compose up -d     # -v gives a clean start; init scripts only run on first start
pytest -q
```
TODO: ingest and query commands

## Layout
- `sql/init/`: numbered SQL files, run in order on first start (`01_...sql`, `02_...sql`)
- `python/`: ingest, search, evaluation and query tool, plus `db.py`
- `tests/`: pytest tests
- `data/`: dataset instructions (large files are not committed)
- `docs/`: charts and screenshots for this README
- `CONTRIBUTING.md`: how we work (branches, pull requests, reviews)

## How it was made
TODO

## Team
- Yosief Gebrewahid Kidane ([@yosiefgw](https://github.com/yosiefgw))
- Timofey Stukalin ([@tstukalin1](https://github.com/tstukalin1))
- Dinu Panackal Devassia ([@DinuPD](https://github.com/DinuPD))