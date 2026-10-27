-- Project 4: Vector search and RAG, natively (MariaDB 11.8)
-- Corpus: Wikipedia category "Solar System". Model: BAAI/bge-small-en-v1.5 (384 dimensions).
-- Chunks: 150-200 words, 30 words overlap. Distance: cosine.
-- Docs: https://mariadb.com/docs/server/reference/sql-structure/vectors/create-table-with-vectors


-- One row per Wikipedia article.

CREATE TABLE articles (
    id           INT UNSIGNED    NOT NULL,   -- Wikipedia page id
    revision_id  BIGINT UNSIGNED NOT NULL,   -- stored revision, so the download is repeatable
    title        VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,  -- exact match
    url          VARCHAR(512)    NOT NULL,
    word_count   INT UNSIGNED    NOT NULL,   -- length of the whole article
    last_edited  DATETIME        NOT NULL,   -- time of the stored revision (UTC)
    fetched_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_articles_title (title),
    KEY idx_articles_last_edited (last_edited),
    KEY idx_articles_word_count (word_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Wikipedia categories (an article can have many).

CREATE TABLE categories (
    id    INT UNSIGNED NOT NULL AUTO_INCREMENT,
    name  VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
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


-- Links between articles of the corpus (links to outside pages are not stored).

CREATE TABLE links (
    from_article_id  INT UNSIGNED NOT NULL,
    to_article_id    INT UNSIGNED NOT NULL,
    PRIMARY KEY (from_article_id, to_article_id),
    KEY idx_links_to (to_article_id),
    CONSTRAINT fk_links_from FOREIGN KEY (from_article_id) REFERENCES articles (id) ON DELETE CASCADE,
    CONSTRAINT fk_links_to   FOREIGN KEY (to_article_id)   REFERENCES articles (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



-- Text chunks with their embedding.
-- Vector rules: one vector index per table, column NOT NULL, exactly 384 numbers.
-- Queries must use VEC_DISTANCE_COSINE with ORDER BY ... LIMIT to use the index.
-- DISTANCE=cosine must be written: the default is euclidean.
-- M = 8 is a starting value (default 6); we will measure other values.
-- No ON DELETE CASCADE on article_id: in our test (11.8.9), rows removed by a cascade
-- broke the vector index. Delete the chunks first, then the article.

CREATE TABLE chunks (
    id           INT UNSIGNED      NOT NULL AUTO_INCREMENT,
    article_id   INT UNSIGNED      NOT NULL,
    chunk_index  SMALLINT UNSIGNED NOT NULL,   -- 0, 1, 2, ... inside the article
    word_count   SMALLINT UNSIGNED NOT NULL,
    chunk_text   TEXT              NOT NULL,
    embedding    VECTOR(384)       NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_chunks_article_chunk (article_id, chunk_index),
    VECTOR INDEX idx_chunks_embedding (embedding) M=8 DISTANCE=cosine,
    CONSTRAINT fk_chunks_article FOREIGN KEY (article_id) REFERENCES articles (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;