
"""Load repeatable synthetic data for benchmarking development.

Demo articles are clearly labelled and kept separate from real data.
Safe to run repeatedly: demo chunks are replaced, not duplicated.
"""

import json

import numpy as np

from db import get_connection
from search import DIM, to_vector_text


DEMO_PREFIX = "DEMO - Benchmark Article "
NUM_ARTICLES = 20
CHUNKS_PER_ARTICLE = 2
RANDOM_SEED = 42


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate deterministic, normalized synthetic embeddings.
    vectors = rng.normal(size=(NUM_ARTICLES * CHUNKS_PER_ARTICLE, DIM))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            for i in range(NUM_ARTICLES):
                article_id = 3_000_000_000 + i
                title = f"{DEMO_PREFIX}{i + 1:02d}"
                url = f"https://example.invalid/demo/{i + 1}"
                revision_id = article_id
                word_count = 500 + i * 10

                # Insert/update only records bearing our demo IDs.
                cursor.execute(
                    """
                    INSERT INTO articles
                        (id, revision_id, title, url, word_count, last_edited)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                        revision_id = VALUES(revision_id),
                        title = VALUES(title),
                        url = VALUES(url),
                        word_count = VALUES(word_count),
                        last_edited = NOW()
                    """,
                    (article_id, revision_id, title, url, word_count),
                )

                # Remove old demo chunks for this article before reloading.
                cursor.execute(
                    "DELETE FROM chunks WHERE article_id = %s",
                    (article_id,),
                )

                for chunk_no in range(CHUNKS_PER_ARTICLE):
                    vector_index = i * CHUNKS_PER_ARTICLE + chunk_no
                    vector_text = to_vector_text(
                        vectors[vector_index].astype(np.float32).tolist()
                    )

                    cursor.execute(
                        """
                        INSERT INTO chunks
                            (article_id, chunk_index, word_count, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s, VEC_FromText(%s))
                        """,
                        (
                            article_id,
                            chunk_no,
                            15,
                            (
                                f"Synthetic demo text for article {i + 1}, "
                                f"chunk {chunk_no + 1}. This is test data only."
                            ),
                            vector_text,
                        ),
                    )

        print(
            f"Demo data loaded: {NUM_ARTICLES} articles, "
            f"{NUM_ARTICLES * CHUNKS_PER_ARTICLE} chunks."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
