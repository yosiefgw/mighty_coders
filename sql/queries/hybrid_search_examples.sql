-- hybrid_search_examples.sql
-- Project 4: Vector search and RAG, natively (MariaDB 11.8)
-- Vector search combined with ordinary SQL filters and joins, in ONE statement.
--
-- NOT in sql/init/ on purpose: these queries need loaded data, so they are run by hand:
--   docker exec -i mighty_coders_vector_db mariadb -umcuser -pmcpassword -t mighty_coders < sql/queries/hybrid_search_examples.sql
-- or paste them one by one into the SQL shell:
--   docker exec -it mighty_coders_vector_db mariadb -umcuser -pmcpassword mighty_coders
--
-- The same query is built in Python by python/search.py (function build_query).
-- Docs: https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-overview
--       https://mariadb.com/docs/server/reference/sql-structure/vectors/vector-system-variables
--
-- Rules that shape every query below:
--   (a) The vector index is used only for  ORDER BY VEC_DISTANCE_COSINE(embedding, <vector>) LIMIT k
--       (ascending, plain function call or its alias, with LIMIT, and the same distance as the
--       index: DISTANCE=cosine in sql/init/01_vector_schema.sql). [docs, Vector Overview]
--   (b) Filters go in the SAME statement (WHERE ... ORDER BY distance LIMIT k).
--       The docs warn that the LIMIT is a hard cap applied before the WHERE filter, so rows could be
--       "silently dropped". We measured MariaDB 11.8.9 (300 non-matching chunks nearer than the 3
--       matching ones, FORCE INDEX, mhnsw_ef_search = 1): all 3 matching rows came back. The index
--       search continues until k rows pass the filter. tests/test_search.py checks this
--       (test_index_with_strict_filter_still_returns_k_rows). So we do NOT use a subquery with a
--       bigger inner LIMIT: that inner LIMIT would itself be a hard cap.
--   (c) The optimizer decides whether to use the index. With a strict filter (few matching
--       articles) it may skip the index and compute exact distances for the few filtered chunks.
--       Check with EXPLAIN (section 9).
--
-- Article titles and categories below are examples from the "Solar System" corpus.
-- If one is not in your data, change it (SELECT title FROM articles; SELECT name FROM categories;).


-- -----------------------------------------------------------------------------------------
-- 0. The query vector.
--    Normally Python embeds a question with bge-small-en-v1.5 and sends 384 numbers:
--      SET @q = VEC_FromText('[0.0123, -0.0456, ...]');
--    Here, so the file runs without Python, we use the embedding of an existing chunk:
--    "find text similar to the first chunk of the article about Titan".
-- -----------------------------------------------------------------------------------------
SET @q = (SELECT c.embedding FROM chunks c JOIN articles a ON a.id = c.article_id
          WHERE a.title = 'Titan (moon)' AND c.chunk_index = 0);
SELECT @q IS NOT NULL AS query_vector_found;   -- must be 1; if 0, change the title above


-- -----------------------------------------------------------------------------------------
-- 1. Plain vector search, joined back to the article title.
-- -----------------------------------------------------------------------------------------
SELECT a.title, c.chunk_index, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 2. Filter: only articles in one category (join through article_categories).
-- -----------------------------------------------------------------------------------------
SELECT a.title, c.chunk_index, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE EXISTS (SELECT 1 FROM article_categories ac JOIN categories cat ON cat.id = ac.category_id
              WHERE ac.article_id = a.id AND cat.name = 'Moons of Saturn')
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 3. Filter: only long articles (at least 3000 words).
-- -----------------------------------------------------------------------------------------
SELECT a.title, a.word_count, c.chunk_index, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE a.word_count >= 3000
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 4. Filter: only articles edited since a date.
-- -----------------------------------------------------------------------------------------
SELECT a.title, a.last_edited, c.chunk_index, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE a.last_edited >= '2025-01-01'
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 5. Filter: only articles that the article "Saturn" links to (join through links).
-- -----------------------------------------------------------------------------------------
SELECT a.title, c.chunk_index, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE EXISTS (SELECT 1 FROM links l JOIN articles src ON src.id = l.from_article_id
              WHERE src.title = 'Saturn' AND l.to_article_id = a.id)
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 6. All filters combined in one statement (this is what search() in python/search.py sends).
-- -----------------------------------------------------------------------------------------
SELECT a.title, a.word_count, a.last_edited, c.chunk_index,
       ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE EXISTS (SELECT 1 FROM article_categories ac JOIN categories cat ON cat.id = ac.category_id
              WHERE ac.article_id = a.id AND cat.name = 'Moons of Saturn')
  AND a.word_count >= 1000
  AND a.last_edited >= '2024-01-01'
  AND EXISTS (SELECT 1 FROM links l JOIN articles src ON src.id = l.from_article_id
              WHERE src.title = 'Saturn' AND l.to_article_id = a.id)
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 7. Enrich the results with relational data: how many corpus articles link TO each result,
--    and its categories. A separate vector store would need a second system for this.
-- -----------------------------------------------------------------------------------------
SELECT a.title, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance,
       (SELECT COUNT(*) FROM links l WHERE l.to_article_id = a.id) AS linked_from_n_articles,
       (SELECT GROUP_CONCAT(cat.name ORDER BY cat.name SEPARATOR '; ')
          FROM article_categories ac JOIN categories cat ON cat.id = ac.category_id
         WHERE ac.article_id = a.id) AS categories
FROM chunks c
JOIN articles a ON a.id = c.article_id
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 8. Best ARTICLES instead of best chunks: one row per article.
--    GROUP BY cannot be answered by the index directly (the index orders chunks, not groups).
--    So: the 50 nearest chunks (index), then group them by article. Here the inner LIMIT 50 is
--    intended: "which articles appear among the 50 nearest chunks".
-- -----------------------------------------------------------------------------------------
SELECT a.title, COUNT(*) AS chunks_in_top_50, ROUND(MIN(s.distance), 4) AS best_distance
FROM (SELECT article_id, VEC_DISTANCE_COSINE(embedding, @q) AS distance
      FROM chunks ORDER BY distance LIMIT 50) s
JOIN articles a ON a.id = s.article_id
GROUP BY a.id, a.title
ORDER BY best_distance
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 9. Which plan does MariaDB use? Look at the "key" column of the chunks row (c).
--    9a: optimizer decides  9b: FORCE INDEX (approximate, HNSW)  9c: IGNORE INDEX (exact)
--    Expected: 9b shows idx_chunks_embedding, 9c does not. 9a depends on how strict the filter is.
--    These are the three modes of search(): mode="auto", "index", "exact".
-- -----------------------------------------------------------------------------------------
EXPLAIN
SELECT a.title, VEC_DISTANCE_COSINE(c.embedding, @q) AS distance
FROM chunks c
JOIN articles a ON a.id = c.article_id
WHERE a.word_count >= 3000
ORDER BY distance
LIMIT 5;

EXPLAIN
SELECT a.title, VEC_DISTANCE_COSINE(c.embedding, @q) AS distance
FROM chunks c FORCE INDEX (idx_chunks_embedding)
JOIN articles a ON a.id = c.article_id
WHERE a.word_count >= 3000
ORDER BY distance
LIMIT 5;

EXPLAIN
SELECT a.title, VEC_DISTANCE_COSINE(c.embedding, @q) AS distance
FROM chunks c IGNORE INDEX (idx_chunks_embedding)
JOIN articles a ON a.id = c.article_id
WHERE a.word_count >= 3000
ORDER BY distance
LIMIT 5;


-- -----------------------------------------------------------------------------------------
-- 10. Approximate vs exact: the same query with and without the index.
--     If the two lists differ, the approximate index missed a chunk. A higher search effort
--     (mhnsw_ef_search, default 20, range 1..10000) makes misses rarer but queries slower.
--     Measuring this over many questions is the "index recall" in our results.
-- -----------------------------------------------------------------------------------------
SET SESSION mhnsw_ef_search = 20;
SELECT c.id, a.title, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c FORCE INDEX (idx_chunks_embedding)
JOIN articles a ON a.id = c.article_id
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 10;

SELECT c.id, a.title, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c IGNORE INDEX (idx_chunks_embedding)
JOIN articles a ON a.id = c.article_id
ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q)
LIMIT 10;

SET SESSION mhnsw_ef_search = DEFAULT;


-- -----------------------------------------------------------------------------------------
-- 11. Vectors are transactional, like every other row.
--     Inside a transaction a new chunk is found at once; after ROLLBACK it is gone.
--     A separate vector store has to be kept in sync with the database by hand.
-- -----------------------------------------------------------------------------------------
START TRANSACTION;
INSERT INTO chunks (article_id, chunk_index, word_count, chunk_text, embedding)
SELECT a.id, 9999, 3, 'transaction demo chunk', @q
FROM articles a WHERE a.title = 'Titan (moon)';

SELECT c.chunk_index, c.chunk_text, ROUND(VEC_DISTANCE_COSINE(c.embedding, @q), 4) AS distance
FROM chunks c ORDER BY VEC_DISTANCE_COSINE(c.embedding, @q) LIMIT 3;   -- the demo chunk is here

ROLLBACK;

SELECT COUNT(*) AS demo_chunks_left FROM chunks WHERE chunk_index = 9999;   -- 0
