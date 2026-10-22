-- 01_vector_schema.sql
-- Project 4: Vector search and RAG, natively (MariaDB 11.8)
-- Corpus: Wikipedia articles in the category "Solar System" (and its subcategories).
-- Embedding model: BAAI/bge-small-en-v1.5 -> 384 numbers per chunk, cosine distance.
-- Chunks: 150-200 words, 30 words overlap.

-- One row per Wikipedia article.
CREATE TABLE articles (
    id           INT UNSIGNED  NOT NULL,            -- Wikipedia page id
    revision_id  BIGINT UNSIGNED NOT NULL,          -- revision of the stored text, so the download can be repeated
    title        VARCHAR(255)  COLLATE utf8mb4_bin NOT NULL,  -- exact match: 'Titan' and 'TITAN' are different pages
    url          VARCHAR(512)  NOT NULL,
    word_count   INT UNSIGNED  NOT NULL,            -- length of the whole article
    last_edited  DATETIME      NOT NULL,            -- time of the latest revision (UTC)
    fetched_at   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_articles_title (title),
    KEY idx_articles_last_edited (last_edited),
    KEY idx_articles_word_count (word_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Wikipedia categories (an article can be in many).
CREATE TABLE categories (
    id    INT UNSIGNED NOT NULL AUTO_INCREMENT,
    name  VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,          -- exact match, like title
    PRIMARY KEY (id),
    UNIQUE KEY uq_categories_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE article_categories (
    article_id   INT UNSIGNED NOT NULL,
    category_id  INT UNSIGNED NOT NULL,
    PRIMARY KEY (article_id, category_id),
    KEY idx_ac_category (category_id),
    CONSTRAINT fk_ac_article  FOREIGN KEY (article_id)  REFERENCES articles (id)   ON DELETE CASCADE,
    CONSTRAINT fk_ac_category FOREIGN KEY (category_id) REFERENCES categories (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Links between articles of our corpus ("article A links to article B").
-- Links to pages outside the corpus are not stored.
CREATE TABLE links (
    from_article_id  INT UNSIGNED NOT NULL,
    to_article_id    INT UNSIGNED NOT NULL,
    PRIMARY KEY (from_article_id, to_article_id),
    KEY idx_links_to (to_article_id),
    CONSTRAINT fk_links_from FOREIGN KEY (from_article_id) REFERENCES articles (id) ON DELETE CASCADE,
    CONSTRAINT fk_links_to   FOREIGN KEY (to_article_id)   REFERENCES articles (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Text chunks with their embedding.
-- Per the MariaDB docs (page "CREATE TABLE with Vectors"): only one vector index per table,
-- the indexed column must be NOT NULL, and every vector must have exactly 384 numbers.
-- Per the docs (page "Vector Overview"): queries must use VEC_DISTANCE_COSINE (the same distance
-- as the index) with ORDER BY ... LIMIT, or the index is not used.
-- M = 8 is our starting value. Per the docs: allowed range 3..200, server default 6
-- (variable mhnsw_default_m, page "Vector System Variables"). We will measure other values.
-- NO "ON DELETE CASCADE" here, on purpose. Our test (MariaDB 11.8.9, tests/test_schema.py) showed that
-- rows deleted by a foreign key cascade break the vector index: afterwards the index finds only
-- 0-1 of 15 live rows (cosine and euclidean). A direct DELETE on chunks works correctly.
-- So: delete an article's chunks first, then the article.
CREATE TABLE chunks (
    id           INT UNSIGNED       NOT NULL AUTO_INCREMENT,
    article_id   INT UNSIGNED       NOT NULL,
    chunk_index  SMALLINT UNSIGNED  NOT NULL,       -- 0, 1, 2, ... inside the article
    word_count   SMALLINT UNSIGNED  NOT NULL,
    chunk_text   TEXT               NOT NULL,
    embedding    VECTOR(384)        NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_chunks_article_chunk (article_id, chunk_index),
    VECTOR INDEX idx_chunks_embedding (embedding) M=8 DISTANCE=cosine,
    CONSTRAINT fk_chunks_article FOREIGN KEY (article_id) REFERENCES articles (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
