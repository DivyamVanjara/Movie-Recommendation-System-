"""
recommender.py
Core recommendation engine using TF-IDF + Cosine Similarity.
Handles data loading, preprocessing, vectorization, and similarity scoring.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, clean data, and build a combined 'content' feature column.
    """
    df = pd.read_csv(csv_path)

    # Keep only the columns we care about
    useful_cols = ["Title", "Genre", "Description", "Director", "Cast",
                   "Rating", "Poster", "Year", "Duration (min)", "Certificate"]
    df = df[[c for c in useful_cols if c in df.columns]].copy()

    # ── Deduplication ──────────────────────────────────────────────────────────
    df.drop_duplicates(subset=["Title"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Fill missing text fields ───────────────────────────────────────────────
    text_cols = ["Genre", "Description", "Director", "Cast"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ── Normalize text ─────────────────────────────────────────────────────────
    def normalize(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation
        text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
        return text

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize)

    # ── Build combined content feature ────────────────────────────────────────
    # Weight genre and director more heavily by repeating them
    df["content"] = (
        df["Genre"].apply(lambda x: (x + " ") * 3) +
        df["Director"].apply(lambda x: (x + " ") * 2) +
        df["Cast"] + " " +
        df["Description"]
    )

    # ── Normalize numeric fields ───────────────────────────────────────────────
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # ── Clean poster URLs (use placeholder if missing) ────────────────────────
    if "Poster" in df.columns:
        df["Poster"] = df["Poster"].fillna("")
    else:
        df["Poster"] = ""

    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING — TF-IDF + COSINE SIM
# ─────────────────────────────────────────────

def build_similarity_matrix(df: pd.DataFrame):
    """
    Vectorize the 'content' column with TF-IDF and compute cosine similarity.

    Returns:
        tfidf        – fitted TfidfVectorizer
        sim_matrix   – (n_movies × n_movies) float32 cosine-similarity array
    """
    tfidf = TfidfVectorizer(
        max_features=15_000,   # cap vocabulary for memory efficiency
        stop_words="english",
        ngram_range=(1, 2),    # unigrams + bigrams for richer matching
        dtype=np.float32,
    )

    tfidf_matrix = tfidf.fit_transform(df["content"].fillna(""))

    # Compute cosine similarity (returns dense float32 array)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return tfidf, sim_matrix


# ─────────────────────────────────────────────
# 3. RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────

def recommend_movies(
    movie_title: str,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    num_recommendations: int = 10,
) -> pd.DataFrame:
    """
    Return the top-N most similar movies to `movie_title`.

    Parameters
    ----------
    movie_title          : exact title string (must exist in df["Title"])
    df                   : preprocessed movies DataFrame
    sim_matrix           : cosine-similarity matrix from build_similarity_matrix()
    num_recommendations  : how many results to return

    Returns
    -------
    DataFrame with columns: Title, Poster, Genre, Rating, Year, Director, Cast
    """
    # ── Find movie index (case-insensitive) ───────────────────────────────────
    titles_lower = df["Title"].str.lower()
    matches = titles_lower[titles_lower == movie_title.lower()]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    # ── Similarity scores for every other movie ───────────────────────────────
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Exclude the queried movie itself
    scores = [(i, s) for i, s in scores if i != idx][:num_recommendations]

    recommended_indices = [i for i, _ in scores]

    result_cols = [c for c in ["Title", "Poster", "Genre", "Rating", "Year",
                                "Director", "Cast", "Duration (min)", "Certificate"]
                   if c in df.columns]

    return df.iloc[recommended_indices][result_cols].reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. HELPER — TOP-RATED MOVIES
# ─────────────────────────────────────────────

def get_top_rated(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-N movies sorted by Rating."""
    result_cols = [c for c in ["Title", "Poster", "Genre", "Rating", "Year",
                                "Director", "Cast"]
                   if c in df.columns]
    top = (df[df["Rating"].notna()]
           .sort_values("Rating", ascending=False)
           .head(n)[result_cols]
           .reset_index(drop=True))
    return top
