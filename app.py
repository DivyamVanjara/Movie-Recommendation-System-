"""
╔══════════════════════════════════════════════════════════════╗
║        🎬  CineMatch — Netflix-Style Movie Recommender       ║
║   Single-file Streamlit app  |  streamlit run app.py         ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install streamlit pandas numpy scikit-learn

Run:
    streamlit run app.py
"""

import re
import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be FIRST Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎬 CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — full Netflix dark theme + responsive card grid
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Dark base ─────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main, .block-container {
    background-color: #141414 !important;
    color: #e5e5e5 !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
[data-testid="stSidebar"]   { background:#0d0d0d !important; }
#MainMenu, footer, header   { visibility:hidden; }
.block-container            { padding-top: 0 !important; max-width:100% !important; }

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar       { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#1a1a1a; }
::-webkit-scrollbar-thumb { background:#e50914; border-radius:3px; }

/* ── Hero ──────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(180deg,#200000 0%,#141414 100%);
    padding: 30px 0 20px; margin-bottom:6px;
}
.logo {
    font-size:2.8rem; font-weight:900; color:#e50914;
    letter-spacing:-2px; line-height:1;
    font-family:'Arial Black','Arial Bold',sans-serif;
    text-shadow:0 2px 20px rgba(229,9,20,.5);
}
.tagline { color:#777; font-size:.92rem; margin-top:4px; letter-spacing:.4px; }
.stats   { display:flex; gap:36px; margin-top:16px; flex-wrap:wrap; }
.stat    { text-align:center; }
.stat-n  { font-size:1.65rem; font-weight:900; color:#e50914; line-height:1; }
.stat-l  { font-size:.64rem; color:#555; letter-spacing:1px; }

/* ── Section titles ────────────────────────────────────────── */
.stitle {
    font-size:1.3rem; font-weight:800; color:#e5e5e5;
    border-left:4px solid #e50914; padding-left:12px;
    margin:28px 0 14px;
}

/* ── Selected-movie banner ─────────────────────────────────── */
.sel {
    background:linear-gradient(135deg,#1c0000,#250d0d);
    border:1.5px solid #e50914; border-radius:8px;
    padding:15px 20px; margin-bottom:18px;
}
.sel h3 { color:#e50914; margin:0 0 5px; font-size:1rem; }
.sel p  { color:#999; margin:0; font-size:.82rem; }

/* ── Responsive card grid ──────────────────────────────────── */
.grid {
    display:grid;
    grid-template-columns:repeat(5,1fr);
    gap:14px; margin-bottom:10px;
}
@media(max-width:1200px){ .grid{grid-template-columns:repeat(4,1fr)} }
@media(max-width: 900px){ .grid{grid-template-columns:repeat(3,1fr)} }
@media(max-width: 600px){ .grid{grid-template-columns:repeat(2,1fr)} }

/* ── Movie card ────────────────────────────────────────────── */
.card {
    background:#1c1c1c; border-radius:6px; overflow:hidden;
    transition:transform .22s ease, box-shadow .22s ease;
    cursor:pointer; position:relative;
}
.card:hover {
    transform:scale(1.07);
    box-shadow:0 14px 44px rgba(229,9,20,.4);
    z-index:20;
}
/* Poster image — fixed 2:3 aspect ratio */
.card img {
    width:100%; aspect-ratio:2/3; object-fit:cover;
    display:block; background:#252525;
}
.card-body   { padding:9px 10px 12px; }
.card-title  {
    font-size:.79rem; font-weight:700; color:#e5e5e5;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    margin-bottom:5px;
}
.card-meta   { display:flex; gap:5px; align-items:center; flex-wrap:wrap; }
.br {
    background:#e50914; color:#fff; font-size:.63rem;
    font-weight:800; padding:2px 6px; border-radius:3px; white-space:nowrap;
}
.bg {
    background:#2a2a2a; color:#bbb; font-size:.60rem;
    padding:2px 7px; border-radius:3px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:110px;
}
.by { color:#555; font-size:.60rem; margin-left:auto; }

/* ── Form controls ─────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {
    background:#212121 !important; color:#e5e5e5 !important;
    border:1px solid #3a3a3a !important; border-radius:5px !important;
}
[data-testid="stSelectbox"] > div > div:hover { border-color:#e50914 !important; }

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
    background:#e50914 !important; color:#fff !important;
    border:none !important; border-radius:5px !important;
    font-weight:800 !important; font-size:.88rem !important;
    padding:10px 0 !important; width:100% !important;
    transition:opacity .18s;
}
.stButton > button:hover { opacity:.85; }

/* ── Misc ──────────────────────────────────────────────────── */
hr          { border-color:#222 !important; margin:24px 0 !important; }
.no-movies  { color:#555; text-align:center; padding:48px 0; }
.stSpinner > div { border-top-color:#e50914 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DATASET    = "imdb-movies-dataset.csv"
NO_POSTER  = "https://placehold.co/300x444/1c1c1c/555555?text=No+Poster"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV, clean text, upgrade poster resolution, build 'content' feature.
    Cached — runs once per server session.
    """
    df = pd.read_csv(path)

    # Keep relevant columns
    keep = ["Title","Genre","Description","Director","Cast",
            "Rating","Poster","Year","Duration (min)","Certificate"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Remove duplicate titles
    df.drop_duplicates(subset=["Title"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill missing text fields
    for col in ["Genre","Description","Director","Cast"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Normalize text for TF-IDF
    def norm(t):
        t = str(t).lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    for col in ["Genre","Description","Director","Cast"]:
        if col in df.columns:
            df[col] = df[col].apply(norm)

    # Build content feature: Genre (×3) + Director (×2) + Cast + Description
    df["content"] = (
        df["Genre"].apply(lambda x: (x+" ")*3) +
        df["Director"].apply(lambda x: (x+" ")*2) +
        df["Cast"] + " " + df["Description"]
    )

    # Numeric rating
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Upsize IMDb thumbnail URLs → larger poster (300×444 px)
    def upsize(url):
        if not isinstance(url, str) or not url.startswith("http"):
            return NO_POSTER
        return re.sub(r"_V1_.*\.jpg", "_V1_UX300_CR0,0,300,444_AL_.jpg", url)

    df["Poster"] = df["Poster"].apply(upsize)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF + COSINE SIMILARITY MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_model(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorise 'content' with TF-IDF, compute cosine-similarity matrix.
    @st.cache_resource keeps the matrix across reruns (not re-serialised).
    """
    tfidf = TfidfVectorizer(
        max_features=15_000,
        stop_words="english",
        ngram_range=(1, 2),     # unigrams + bigrams
        dtype=np.float32,
    )
    mat = tfidf.fit_transform(df["content"].fillna(""))
    return cosine_similarity(mat, mat)          # (n × n) float32 matrix


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def get_recommendations(
    title: str,
    df: pd.DataFrame,
    sim: np.ndarray,
    n: int = 10,
) -> pd.DataFrame:
    """
    Find top-n movies most similar to `title` using the cosine-similarity matrix.
    Returns a DataFrame with: Title, Poster, Genre, Rating, Year, Director.
    """
    mask = df["Title"].str.lower() == title.lower()
    if not mask.any():
        return pd.DataFrame()

    idx    = int(mask.idxmax())
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
    top    = [i for i, _ in scores if i != idx][:n]

    cols = [c for c in ["Title","Poster","Genre","Rating","Year","Director","Cast"]
            if c in df.columns]
    return df.iloc[top][cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_rating(v) -> str:
    try:    return f"{float(v):.1f}"
    except: return "N/A"

def fmt_genre(g: str) -> str:
    if not g: return "—"
    return str(g).split(",")[0].strip().title()

def fmt_year(y) -> str:
    try:    return str(int(float(y)))
    except: return ""


def movie_card(title: str, poster: str, rating, genre: str, year) -> str:
    """Return HTML string for a single Netflix-style movie card."""
    r = fmt_rating(rating)
    g = fmt_genre(genre)
    y = fmt_year(year)
    p = poster if poster and poster.startswith("http") else NO_POSTER
    safe_title = title.replace('"', '&quot;')

    return f"""
    <div class="card">
        <img src="{p}"
             alt="{safe_title}"
             loading="lazy"
             onerror="this.onerror=null;this.src='{NO_POSTER}'">
        <div class="card-body">
            <div class="card-title" title="{safe_title}">{title}</div>
            <div class="card-meta">
                <span class="br">⭐ {r}</span>
                <span class="bg">{g}</span>
                <span class="by">{y}</span>
            </div>
        </div>
    </div>"""


def render_grid(movies: pd.DataFrame) -> None:
    """Render a full responsive grid of movie cards."""
    if movies.empty:
        st.markdown('<p class="no-movies">No movies found.</p>', unsafe_allow_html=True)
        return

    html = '<div class="grid">'
    for _, m in movies.iterrows():
        html += movie_card(
            title  = m.get("Title",  "Unknown"),
            poster = m.get("Poster", NO_POSTER),
            rating = m.get("Rating", None),
            genre  = m.get("Genre",  ""),
            year   = m.get("Year",   ""),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def main():

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("🎬 Loading movie database…"):
        df = load_data(DATASET)

    with st.spinner("🤖 Building recommendation engine…"):
        sim = build_model(df)

    all_titles = sorted(df["Title"].dropna().unique().tolist())
    genres_all = sorted(
        df["Genre"].dropna()
          .str.split(",").explode()
          .str.strip().str.title()
          .unique().tolist()
    )

    # ── Hero section ──────────────────────────────────────────────────────────
    avg_r = df["Rating"].mean()
    n_gen = df["Genre"].dropna().str.split(",").explode().str.strip().nunique()
    n_dir = df["Director"].dropna().nunique()

    st.markdown(f"""
    <div class="hero">
        <div class="logo">🎬 CineMatch</div>
        <div class="tagline">AI-powered movie recommendations — {len(df):,} films</div>
        <div class="stats">
            <div class="stat"><div class="stat-n">{len(df):,}</div><div class="stat-l">MOVIES</div></div>
            <div class="stat"><div class="stat-n">{n_gen}</div><div class="stat-l">GENRES</div></div>
            <div class="stat"><div class="stat-n">{avg_r:.1f}</div><div class="stat-l">AVG RATING</div></div>
            <div class="stat"><div class="stat-n">{n_dir:,}</div><div class="stat-l">DIRECTORS</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search & controls ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([4, 1, 1, 1])

    with c1:
        search = st.text_input("", placeholder="🔍  Search for a movie…",
                               label_visibility="collapsed", key="search")
    with c2:
        num_recs = st.selectbox("", [5, 10, 15, 20], index=1,
                                label_visibility="collapsed", key="nrec")
    with c3:
        go_btn  = st.button("🎯 Recommend", use_container_width=True)
    with c4:
        rnd_btn = st.button("🎲 Surprise", use_container_width=True)

    # Filter list based on search
    filtered = ([t for t in all_titles if search.lower() in t.lower()]
                if search else all_titles) or all_titles

    # Random pick
    if rnd_btn:
        st.session_state["pick"] = random.choice(all_titles)

    pick        = st.session_state.get("pick")
    def_idx     = (filtered.index(pick) if pick and pick in filtered else 0)

    selected = st.selectbox("", filtered, index=def_idx,
                             label_visibility="collapsed", key="sel")

    # ── Recommendation results ─────────────────────────────────────────────────
    trigger = go_btn or rnd_btn
    if trigger and selected:

        # Info banner for selected movie
        row = df[df["Title"].str.lower() == selected.lower()]
        if not row.empty:
            s = row.iloc[0]
            st.markdown(f"""
            <div class="sel">
                <h3>📽️ Because you selected: {selected}</h3>
                <p>⭐ {fmt_rating(s.get('Rating'))} &nbsp;|&nbsp;
                   🎭 {s.get('Genre','—') or '—'} &nbsp;|&nbsp;
                   📅 {fmt_year(s.get('Year',''))}
                   {'&nbsp;|&nbsp; 🎬 ' + str(s.get('Director','')) if s.get('Director') else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with st.spinner(f"Finding movies similar to **{selected}**…"):
            recs = get_recommendations(selected, df, sim, num_recs)

        st.markdown(
            f'<div class="stitle">🍿 Recommended for You ({len(recs)} movies)</div>',
            unsafe_allow_html=True)
        render_grid(recs)
        st.markdown("<hr>", unsafe_allow_html=True)

    # ── Top Rated ──────────────────────────────────────────────────────────────
    st.markdown('<div class="stitle">🏆 Top Rated Movies</div>', unsafe_allow_html=True)
    top_rated = (df[df["Rating"].notna()]
                   .sort_values("Rating", ascending=False)
                   .head(10))
    render_grid(top_rated)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Genre Explorer ─────────────────────────────────────────────────────────
    st.markdown('<div class="stitle">🎭 Browse by Genre</div>', unsafe_allow_html=True)

    ga, gb = st.columns([3, 1])
    with ga:
        chosen_genre = st.selectbox("", genres_all,
                                    label_visibility="collapsed", key="genre")
    with gb:
        genre_n = st.selectbox("", [5, 10, 15, 20], index=1,
                               label_visibility="collapsed", key="gn")

    genre_movies = (
        df[df["Genre"].str.contains(chosen_genre, case=False, na=False)]
          .sort_values("Rating", ascending=False)
          .head(genre_n)
    )
    render_grid(genre_movies)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#3a3a3a;font-size:.74rem;'>"
        "🎬 CineMatch &nbsp;•&nbsp; TF-IDF + Cosine Similarity"
        " &nbsp;•&nbsp; Built with Streamlit</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
