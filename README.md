# mighty_coders: Vector search and RAG, natively in MariaDB

Capstone I (Constructor University) with the MariaDB Foundation. Team: Yosief Gebrewahid Kidane, Timofey Stukalin, Dinu Panackal Devassia.

> Status: skeleton. Replace every TODO before submitting.

## What this project shows
TODO: one paragraph. Native `VECTOR` type and vector index, similarity search combined with ordinary SQL filters, retrieval for RAG, compared with doing it outside the database.

## Versions and setup
- MariaDB: 11.8 (Docker image `mariadb:11.8`). VECTOR needs 11.7 or newer.
- Python 3 with `pymysql` and `pytest`.
- Dataset: TODO, see [data/README.md](data/README.md).

## Run it
```bash
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose down -v && docker compose up -d     # -v gives a clean start; init scripts only run on first start
pytest -q
```

## Layout
- `sql/init/` numbered SQL files run in order on first start (`01_...sql`, `02_...sql`)
- `python/` before/after demos per feature, plus `db.py`
- `tests/` pytest tests
- `data/` dataset instructions (large files are not committed)
- `CONTRIBUTING.md`

## Results: before and after
TODO: comparison against the same task without the native feature (for example brute-force similarity in Python or another database). Show numbers, how many repeats, and the machine/setup.

## Limits
TODO: be explicit about what does not work or was not measured.
