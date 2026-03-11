"""
app.py  –  Netflix-Style Movie Recommendation System
Run with:  streamlit run app.py
"""

import random
import streamlit as st
import pandas as pd

from recommender import (
    load_and_preprocess,
    build_similarity_matrix,
    recommend_movies,
    get_top_rated,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  —  Netflix dark-theme + card styles
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Base dark theme ──────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #141414 !important;
        color: #e5e5e5 !important;
    }
    [data-testid="stSidebar"] { background-color: #1a1a1a !important; }

    /* ── Hide Streamlit chrome ────────────────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #e50914; border-radius: 3px; }

    /* ── Netflix logo / hero ──────────────────────────────────────────────── */
    .netflix-logo {
        font-size: 2.6rem;
        font-weight: 900;
        color: #e50914;
        letter-spacing: -1px;
        font-family: 'Arial Black', sans-serif;
    }
    .hero-subtitle {
        color: #999;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #e5e5e5;
        margin: 32px 0 14px 0;
        border-left: 4px solid #e50914;
        padding-left: 10px;
    }

    /* ── Movie card ───────────────────────────────────────────────────────── */
    .movie-card {
        background: #1f1f1f;
        border-radius: 6px;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        cursor: pointer;
        margin-bottom: 16px;
        position: relative;
    }
    .movie-card:hover {
        transform: scale(1.06);
        box-shadow: 0 12px 40px rgba(229,9,20,0.35);
        z-index: 10;
    }
    .movie-card img {
        width: 100%;
        height: 210px;
        object-fit: cover;
        display: block;
    }
    .card-body {
        padding: 9px 10px 11px;
    }
    .card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #e5e5e5;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 4px;
    }
    .card-meta {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .badge-rating {
        background: #e50914;
        color: #fff;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 3px;
    }
    .badge-genre {
        background: #333;
        color: #ccc;
        font-size: 0.65rem;
        padding: 1px 6px;
        border-radius: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 130px;
    }

    /* ── Search & controls ────────────────────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] select,
    [data-testid="stSelectbox"] > div > div {
        background-color: #2a2a2a !important;
        color: #e5e5e5 !important;
        border: 1px solid #444 !important;
        border-radius: 4px !important;
    }
    [data-testid="stSelectbox"] > div > div:hover { border-color: #e50914 !important; }

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .stButton > button {
        background-color: #e50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 10px 28px !important;
        transition: background 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover { background-color: #f40612 !important; opacity: 0.92; }

    /* ── Divider ──────────────────────────────────────────────────────────── */
    hr { border-color: #2a2a2a !important; }

    /* ── Selected movie banner ────────────────────────────────────────────── */
    .selected-banner {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d0d0d 100%);
        border: 1px solid #e50914;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .selected-banner h3 { color: #e50914; margin: 0 0 4px 0; font-size: 1.1rem; }
    .selected-banner p  { color: #aaa; margin: 0; font-size: 0.85rem; }

    /* ── Spinner ──────────────────────────────────────────────────────────── */
    .stSpinner > div { border-top-color: #e50914 !important; }

    /* ── Stats bar ────────────────────────────────────────────────────────── */
    .stats-bar {
        display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap;
    }
    .stat-item { text-align: center; }
    .stat-value { font-size: 1.6rem; font-weight: 900; color: #e50914; }
    .stat-label { font-size: 0.72rem; color: #888; }

    /* ── No results ───────────────────────────────────────────────────────── */
    .no-results {
        text-align: center; padding: 60px 0; color: #666; font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLACEHOLDER_IMG = "https://via.placeholder.com/140x209/1f1f1f/555555?text=No+Poster"


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL  (cached so they only load once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    return load_and_preprocess("dataset.csv")


@st.cache_resource(show_spinner=False)
def load_model(df: pd.DataFrame):
    _, sim = build_similarity_matrix(df)
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def poster_url(url: str) -> str:
    """Return poster URL, falling back to a grey placeholder."""
    return url.strip() if url and url.strip() else PLACEHOLDER_IMG


def first_genre(genre_str: str) -> str:
    """Return only the first genre tag to keep the card tidy."""
    if not genre_str:
        return "—"
    return str(genre_str).split(",")[0].strip().title()


def render_movie_card(title: str, poster: str, rating, genre: str):
    """Render a single Netflix-style movie card via HTML."""
    rating_str = f"{rating:.1f}" if pd.notna(rating) else "N/A"
    genre_clean = first_genre(genre)
    poster_src  = poster_url(poster)

    st.markdown(
        f"""
        <div class="movie-card">
            <img src="{poster_src}"
                 alt="{title}"
                 onerror="this.src='{PLACEHOLDER_IMG}'">
            <div class="card-body">
                <div class="card-title" title="{title}">{title}</div>
                <div class="card-meta">
                    <span class="badge-rating">⭐ {rating_str}</span>
                    <span class="badge-genre">{genre_clean}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grid(movies_df: pd.DataFrame, cols: int = 5):
    """Render a responsive grid of movie cards."""
    if movies_df.empty:
        st.markdown('<div class="no-results">No movies found.</div>',
                    unsafe_allow_html=True)
        return

    rows = [movies_df.iloc[i : i + cols] for i in range(0, len(movies_df), cols)]
    for row in rows:
        grid = st.columns(cols)
        for col_idx, (_, movie) in enumerate(row.iterrows()):
            with grid[col_idx]:
                render_movie_card(
                    title=movie.get("Title", "Unknown"),
                    poster=movie.get("Poster", ""),
                    rating=movie.get("Rating", None),
                    genre=movie.get("Genre", ""),
                )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("🎬 Loading movies…"):
        df = load_data()

    with st.spinner("🤖 Building recommendation engine…"):
        sim_matrix = load_model(df)

    # ── Header / Hero ─────────────────────────────────────────────────────────
    col_logo, col_stats = st.columns([3, 2])
    with col_logo:
        st.markdown('<div class="netflix-logo">🎬 CineMatch</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">AI-powered movie recommendations</div>',
            unsafe_allow_html=True,
        )
    with col_stats:
        genres  = df["Genre"].dropna().str.split(",").explode().str.strip().nunique()
        avg_rtg = df["Rating"].mean()
        st.markdown(
            f"""
            <div class="stats-bar" style="justify-content:flex-end; margin-top:10px;">
                <div class="stat-item">
                    <div class="stat-value">{len(df):,}</div>
                    <div class="stat-label">MOVIES</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{genres}</div>
                    <div class="stat-label">GENRES</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{avg_rtg:.1f}</div>
                    <div class="stat-label">AVG RATING</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Control panel ─────────────────────────────────────────────────────────
    all_titles = sorted(df["Title"].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])

    with c1:
        search_query = st.text_input(
            "🔍 Search movies",
            placeholder="Type to search a movie title…",
            label_visibility="collapsed",
        )

    with c2:
        num_recs = st.selectbox(
            "Results",
            [5, 10, 15, 20],
            index=1,
            label_visibility="collapsed",
        )

    with c3:
        recommend_btn = st.button("🎯 Recommend", use_container_width=True)

    with c4:
        random_btn = st.button("🎲 Random", use_container_width=True)

    # ── Filter dropdown based on search ───────────────────────────────────────
    if search_query:
        filtered = [t for t in all_titles if search_query.lower() in t.lower()]
    else:
        filtered = all_titles

    if not filtered:
        st.warning("No movies matched your search.")
        filtered = all_titles

    selected_movie = st.selectbox(
        "Pick a movie",
        filtered,
        label_visibility="collapsed",
    )

    # ── Random movie shortcut ─────────────────────────────────────────────────
    if random_btn:
        selected_movie = random.choice(all_titles)
        st.session_state["random_pick"] = selected_movie
        st.info(f"🎲 Random pick: **{selected_movie}**")
        recommend_btn = True   # auto-trigger recommendation

    # ── Recommendation results ────────────────────────────────────────────────
    if recommend_btn and selected_movie:
        # Selected movie info banner
        sel_row = df[df["Title"].str.lower() == selected_movie.lower()]
        if not sel_row.empty:
            sel = sel_row.iloc[0]
            rating_display = f"{sel['Rating']:.1f}" if pd.notna(sel.get("Rating")) else "N/A"
            genre_display  = sel.get("Genre", "—") or "—"
            year_display   = int(sel["Year"]) if pd.notna(sel.get("Year")) else ""
            st.markdown(
                f"""
                <div class="selected-banner">
                    <h3>📽️ Because you selected: {selected_movie}</h3>
                    <p>⭐ {rating_display} &nbsp;|&nbsp; 🎭 {genre_display}
                       {'&nbsp;|&nbsp; 📅 ' + str(year_display) if year_display else ''}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.spinner(f"Finding movies similar to **{selected_movie}**…"):
            recs = recommend_movies(selected_movie, df, sim_matrix, num_recs)

        if recs.empty:
            st.error("Could not find recommendations. Please try another movie.")
        else:
            st.markdown(
                f'<div class="section-title">🍿 Recommended for You ({len(recs)} movies)</div>',
                unsafe_allow_html=True,
            )
            render_grid(recs, cols=5)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Top Rated section ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Top Rated Movies</div>', unsafe_allow_html=True)
    top_rated = get_top_rated(df, n=10)
    render_grid(top_rated, cols=5)

    # ── Genre Explorer ────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎭 Browse by Genre</div>', unsafe_allow_html=True)

    all_genres = sorted(
        df["Genre"].dropna()
           .str.split(",")
           .explode()
           .str.strip()
           .str.title()
           .unique()
           .tolist()
    )
    chosen_genre = st.selectbox(
        "Select genre",
        all_genres,
        label_visibility="collapsed",
    )

    genre_movies = df[
        df["Genre"].str.contains(chosen_genre, case=False, na=False)
    ].sort_values("Rating", ascending=False).head(10)

    if not genre_movies.empty:
        render_grid(genre_movies, cols=5)
    else:
        st.info(f"No movies found for genre: {chosen_genre}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align:center;color:#555;font-size:0.78rem;'>
        🎬 CineMatch &nbsp;•&nbsp; Powered by TF-IDF + Cosine Similarity
        &nbsp;•&nbsp; Built with Streamlit
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
