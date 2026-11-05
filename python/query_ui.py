
"""Streamlit UI for the Solar System Wikipedia corpus.

Supports:
1. Question-based hybrid vector search through MariaDB.
2. Direct category browsing with five articles per page.
3. Full article display by merging overlapping text chunks.

No LLM-generated answers are used.
"""

from time import perf_counter

import streamlit as st

from db import get_connection
from embed import embed
from search import search


# ---------------------------------------------------------
# Database helper functions
# ---------------------------------------------------------

@st.cache_data(ttl=300)
def load_categories():
    """Load available category names from MariaDB."""
    conn = get_connection()

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM categories ORDER BY name")
        categories = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return categories
    finally:
        conn.close()


def load_category_articles(
    category,
    min_words=None,
    edited_since=None,
    limit=5,
    offset=0,
):
    """Retrieve a filtered page of articles belonging to a category."""
    conn = get_connection()

    try:
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT
                a.id,
                a.title,
                a.url,
                a.word_count,
                a.last_edited
            FROM articles AS a
            JOIN article_categories AS ac
                ON a.id = ac.article_id
            JOIN categories AS cat
                ON ac.category_id = cat.id
            WHERE cat.name = %s
        """

        params = [category]

        if min_words is not None and min_words > 0:
            query += " AND a.word_count >= %s"
            params.append(min_words)

        if edited_since is not None:
            query += " AND DATE(a.last_edited) >= %s"
            params.append(edited_since)

        query += " ORDER BY a.title LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cursor.execute(query, tuple(params))
        articles = cursor.fetchall()
        cursor.close()

        return articles

    finally:
        conn.close()


def load_article_content(article_id):
    """Retrieve article chunks and merge their overlapping text."""
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT chunk_text
            FROM chunks
            WHERE article_id = %s
            ORDER BY chunk_index
            """,
            (article_id,),
        )

        chunks = [row[0] for row in cursor.fetchall()]
        cursor.close()

        if not chunks:
            return ""

        combined_words = chunks[0].split()

        for chunk in chunks[1:]:
            next_words = chunk.split()

            max_overlap = min(
                50,
                len(combined_words),
                len(next_words),
            )

            overlap = 0

            for size in range(max_overlap, 0, -1):
                previous = [
                    word.lower().strip(".,!?;:()[]{}\"'")
                    for word in combined_words[-size:]
                ]

                current = [
                    word.lower().strip(".,!?;:()[]{}\"'")
                    for word in next_words[:size]
                ]

                if previous == current:
                    overlap = size
                    break

            combined_words.extend(next_words[overlap:])

        return " ".join(combined_words)

    finally:
        conn.close()


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Solar System Explorer",
    page_icon="🪐",
    layout="wide",
)

st.title("🪐 Solar System Explorer")

st.write(
    "Explore the Solar System Wikipedia knowledge base using "
    "MariaDB hybrid vector search or direct category browsing."
)

st.divider()


# ---------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------

session_defaults = {
    "search_results": [],
    "search_question": "",
    "search_elapsed_time": None,
    "selected_result_index": 0,
    "category_articles": [],
    "category_offset": 0,
    "category_has_more": False,
    "display_mode": None,
    "selected_category_article_id": None,
    "active_category": None,
    "active_min_words": 0,
    "active_edited_since": None,
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ---------------------------------------------------------
# Search input
# ---------------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="What is Jupiter made of?",
)


# ---------------------------------------------------------
# Search filters
# ---------------------------------------------------------

st.subheader("Search filters")

TOP_K = 5
PAGE_SIZE = 5

try:
    categories = load_categories()
except Exception as exc:
    st.error(f"Could not load categories: {exc}")
    categories = []

category_options = ["All categories"] + categories

selected_category = st.selectbox(
    "Select category",
    options=category_options,
)

category_filter = (
    None
    if selected_category == "All categories"
    else selected_category
)

col1, col2 = st.columns(2)

with col1:
    min_words = st.number_input(
        "Minimum article word count",
        min_value=0,
        value=0,
        step=100,
    )

with col2:
    edited_since = st.date_input(
        "Articles edited since (optional)",
        value=None,
    )

edited_since_value = (
    edited_since.isoformat()
    if edited_since is not None
    else None
)


# ---------------------------------------------------------
# Search button and search execution
# ---------------------------------------------------------

search_clicked = st.button(
    "🔍 Search",
    type="primary",
)


if search_clicked:

    # -----------------------------------------------------
    # Mode 1: Category browsing without a question
    # -----------------------------------------------------

    if not question.strip() and category_filter is not None:

        st.session_state.display_mode = "category"

        st.session_state.search_results = []
        st.session_state.search_elapsed_time = None

        st.session_state.category_articles = []
        st.session_state.category_offset = 0
        st.session_state.category_has_more = False

        st.session_state.selected_category_article_id = None

        # Save the filters used for this browsing session.
        st.session_state.active_category = category_filter
        st.session_state.active_min_words = min_words
        st.session_state.active_edited_since = edited_since_value

        try:
            with st.spinner("Loading articles from MariaDB..."):

                fetched_articles = load_category_articles(
                    category_filter,
                    min_words=min_words if min_words > 0 else None,
                    edited_since=edited_since_value,
                    limit=PAGE_SIZE + 1,
                    offset=0,
                )

            # Display five articles initially.
            st.session_state.category_articles = (
                fetched_articles[:PAGE_SIZE]
            )

            st.session_state.category_offset = len(
                st.session_state.category_articles
            )

            # Fetching one extra row tells us whether more exist.
            st.session_state.category_has_more = (
                len(fetched_articles) > PAGE_SIZE
            )

            if st.session_state.category_articles:
                st.session_state.selected_category_article_id = (
                    st.session_state.category_articles[0][0]
                )

        except Exception as exc:
            st.session_state.category_articles = []
            st.error(f"Could not load category articles: {exc}")

    # -----------------------------------------------------
    # No question and no category selected
    # -----------------------------------------------------

    elif not question.strip() and category_filter is None:

        st.warning(
            "Please enter a question or select a category."
        )

    # -----------------------------------------------------
    # Mode 2: Question-based vector search
    # -----------------------------------------------------

    else:

        st.session_state.display_mode = "search"

        st.session_state.category_articles = []
        st.session_state.category_offset = 0
        st.session_state.category_has_more = False

        st.session_state.active_category = None

        conn = None
        start_time = perf_counter()

        print("\n" + "=" * 60, flush=True)
        print("             SOLAR SYSTEM EXPLORER", flush=True)
        print("=" * 60, flush=True)

        print("\n[1] QUESTION RECEIVED", flush=True)
        print(f"    {question.strip()}", flush=True)

        try:

            with st.spinner(
                "Generating embedding and searching MariaDB..."
            ):

                # Step 1: Generate query embedding.
                print(
                    "\n[2] GENERATING QUERY EMBEDDING",
                    flush=True,
                )

                query_vector = embed(question.strip())

                print(
                    "    Model: BAAI/bge-small-en-v1.5",
                    flush=True,
                )

                print(
                    f"    Embedding dimensions: {len(query_vector)}",
                    flush=True,
                )

                # Step 2: Connect to MariaDB.
                print(
                    "\n[3] CONNECTING TO MARIADB",
                    flush=True,
                )

                conn = get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT DATABASE(), @@hostname, @@port, VERSION()"
                )

                db_info = cursor.fetchone()
                cursor.close()

                print(
                    "    Database connection successful.",
                    flush=True,
                )

                print(f"    Database: {db_info[0]}", flush=True)
                print(f"    Server hostname: {db_info[1]}", flush=True)
                print(f"    Server port: {db_info[2]}", flush=True)
                print(f"    MariaDB version: {db_info[3]}", flush=True)

                # Step 3: Execute MariaDB vector search.
                print(
                    "\n[4] EXECUTING MARIADB VECTOR SEARCH",
                    flush=True,
                )

                print(
                    "    Retrieving matching chunks...",
                    flush=True,
                )

                results = search(
                    conn,
                    query_vector,
                    k=TOP_K * 5,
                    category=category_filter,
                    min_words=min_words if min_words > 0 else None,
                    edited_since=edited_since_value,
                )

                print(
                    f"    Chunks returned by database: {len(results)}",
                    flush=True,
                )

                # Step 4: Keep the best chunk for each article.
                unique_results = []
                seen_articles = set()

                for result in results:

                    title = result["title"]

                    if title not in seen_articles:
                        seen_articles.add(title)
                        unique_results.append(result)

                    if len(unique_results) >= TOP_K:
                        break

                elapsed_time = perf_counter() - start_time

                # Save results in session state.
                st.session_state.search_results = unique_results

                st.session_state.search_question = question.strip()

                st.session_state.search_elapsed_time = elapsed_time

                st.session_state.selected_result_index = 0

                st.session_state["article_suggestion_selector"] = 0

                # Step 5: Log results in CMD.
                print(
                    "\n[5] RESULTS RETRIEVED FROM MARIADB",
                    flush=True,
                )

                if not unique_results:

                    print(
                        "    No matching results found.",
                        flush=True,
                    )

                else:

                    for i, result in enumerate(
                        unique_results,
                        start=1,
                    ):

                        print(f"\n    RESULT {i}", flush=True)

                        print(
                            f"    Article: {result['title']}",
                            flush=True,
                        )

                        print(
                            f"    URL: {result['url']}",
                            flush=True,
                        )

                        print(
                            f"    Distance: "
                            f"{float(result['distance']):.4f}",
                            flush=True,
                        )

                        print(
                            f"    Chunk index: {result['chunk_index']}",
                            flush=True,
                        )

                        print(
                            f"    Word count: {result['word_count']}",
                            flush=True,
                        )

                        print(
                            f"    Last edited: {result['last_edited']}",
                            flush=True,
                        )

                        print(
                            f"    Chunk text: "
                            f"{result['chunk_text'][:250]}",
                            flush=True,
                        )

                print("\n" + "=" * 60, flush=True)

                print(
                    "       SEARCH COMPLETED SUCCESSFULLY",
                    flush=True,
                )

                print(
                    f"       Unique articles found: {len(unique_results)}",
                    flush=True,
                )

                print(
                    f"       Execution time: {elapsed_time:.3f} seconds",
                    flush=True,
                )

                print("=" * 60 + "\n", flush=True)

        except Exception as exc:

            st.session_state.search_results = []
            st.session_state.search_elapsed_time = None

            print(
                f"\n[ERROR] Search failed: {exc}",
                flush=True,
            )

            st.error(f"Search failed: {exc}")

        finally:

            if conn is not None:
                conn.close()


# ---------------------------------------------------------
# Display category browsing results
# ---------------------------------------------------------

if st.session_state.display_mode == "category":

    active_category = st.session_state.active_category

    st.subheader(f"Articles in: {active_category}")

    articles = st.session_state.category_articles

    if articles:

        st.caption(
            f"Showing {len(articles)} article(s). "
            "Select an article to read its stored content."
        )

        # Create article ID lookup.
        article_by_id = {
            article[0]: article
            for article in articles
        }

        article_ids = list(article_by_id.keys())

        current_selected_id = (
            st.session_state.selected_category_article_id
        )

        if current_selected_id not in article_ids:

            current_selected_id = article_ids[0]

            st.session_state.selected_category_article_id = (
                current_selected_id
            )

        selected_article_id = st.selectbox(
            "Choose an article to read",
            options=article_ids,
            format_func=lambda article_id: (
                article_by_id[article_id][1]
            ),
            index=article_ids.index(current_selected_id),
            key="category_article_selector",
        )

        st.session_state.selected_category_article_id = (
            selected_article_id
        )

        selected_article = article_by_id[selected_article_id]

        (
            article_id,
            title,
            url,
            word_count,
            last_edited,
        ) = selected_article

        st.divider()

        st.markdown(f"## {title}")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.metric("Article word count", word_count)

        with detail_col2:
            st.write(f"**Last edited:** {last_edited}")

        st.markdown(
            f"[Open original Wikipedia page]({url})"
        )

        st.markdown("### Article content")

        try:

            with st.spinner(
                "Loading article content from MariaDB..."
            ):

                article_content = load_article_content(article_id)

            if article_content:

                st.markdown(article_content)

            else:

                st.info(
                    "No text chunks were found for this article in MariaDB."
                )

        except Exception as exc:

            st.error(
                f"Could not load article content: {exc}"
            )

        # -------------------------------------------------
        # Load more articles
        # -------------------------------------------------

        if st.session_state.category_has_more:

            if st.button("Load more articles"):

                try:

                    with st.spinner("Loading more articles..."):

                        fetched_articles = load_category_articles(
                            active_category,
                            min_words=(
                                st.session_state.active_min_words
                                if st.session_state.active_min_words > 0
                                else None
                            ),
                            edited_since=(
                                st.session_state.active_edited_since
                            ),
                            limit=PAGE_SIZE + 1,
                            offset=st.session_state.category_offset,
                        )

                    new_articles = fetched_articles[:PAGE_SIZE]

                    existing_ids = {
                        article[0]
                        for article in st.session_state.category_articles
                    }

                    new_articles = [
                        article
                        for article in new_articles
                        if article[0] not in existing_ids
                    ]

                    st.session_state.category_articles.extend(
                        new_articles
                    )

                    st.session_state.category_offset += len(
                        fetched_articles[:PAGE_SIZE]
                    )

                    st.session_state.category_has_more = (
                        len(fetched_articles) > PAGE_SIZE
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Could not load more articles: {exc}"
                    )

    else:

        st.info(
            "No articles found matching the selected category and filters."
        )


# ---------------------------------------------------------
# Display vector search results
# ---------------------------------------------------------

results = st.session_state.search_results

if (
    st.session_state.display_mode == "search"
    and results
):

    st.sidebar.header("Article suggestions")

    st.sidebar.caption(
        "The closest match is selected first. Choose another article "
        "to view its matching passage."
    )

    selected_index = st.sidebar.radio(
        "Select an article",
        options=list(range(len(results))),
        index=min(
            st.session_state.selected_result_index,
            len(results) - 1,
        ),
        format_func=lambda index: (
            f"{index + 1}. {results[index]['title']} "
            f"({float(results[index]['distance']):.4f})"
        ),
        key="article_suggestion_selector",
    )

    st.session_state.selected_result_index = selected_index

    selected_result = results[selected_index]

    if selected_index == 0:

        st.subheader("Most relevant article")

    else:

        st.subheader("Selected suggested article")

    st.caption(
        f"Search: {st.session_state.search_question} | "
        f"{len(results)} unique matching articles found"
    )

    if st.session_state.search_elapsed_time is not None:

        st.caption(
            "Search execution time: "
            f"{st.session_state.search_elapsed_time:.3f} seconds"
        )

    st.divider()

    title = selected_result["title"]

    wikipedia_url = selected_result["url"]

    distance = float(selected_result["distance"])

    st.markdown(f"## {title}")

    st.caption(f"Vector distance: {distance:.4f}")

    st.markdown(
        f"**Wikipedia article:** [{title}]({wikipedia_url})"
    )

    st.markdown("### Matching passage")

    st.write(selected_result["chunk_text"])

    st.divider()

    st.markdown("### Article and chunk details")

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        st.metric(
            "Chunk index",
            selected_result["chunk_index"],
        )

    with detail_col2:
        st.metric(
            "Article word count",
            selected_result["word_count"],
        )

    with detail_col3:
        st.metric(
            "Vector distance",
            f"{distance:.4f}",
        )

    st.caption(
        f"Last edited: {selected_result['last_edited']}"
    )

    st.info(
        "These are retrieved Wikipedia passages ranked by vector distance. "
        "The application retrieves relevant content; it does not generate "
        "a separate natural-language answer."
    )


elif (
    st.session_state.display_mode == "search"
    and not results
):

    st.info(
        "No matching results found. "
        "Try changing your question or filters."
    )
