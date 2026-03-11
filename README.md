# 🎬 CineMatch — Netflix-Style Movie Recommendation System

A production-ready content-based movie recommender with a Netflix-inspired dark UI,
powered by **TF-IDF vectorisation** and **cosine similarity**.

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd movie_recommender

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**.

---

## 📁 Project Structure

```
movie_recommender/
├── app.py            ← Streamlit UI (Netflix dark theme, grid layout)
├── recommender.py    ← Data processing + ML recommendation engine
├── dataset.csv       ← IMDb movie dataset (10 000 movies)
├── requirements.txt
└── README.md
```

---

## 🧠 How the Recommendation Model Works

### Step 1 — Data Preprocessing
- Loads CSV with pandas; removes duplicate titles.
- Fills missing text fields (genre, director, cast, description) with empty strings.
- Normalises all text: lowercasing, stripping punctuation, collapsing whitespace.

### Step 2 — Combined Feature ("content")
Each movie gets a single text string that merges:

```
Genre (×3) + Director (×2) + Cast + Description
```

Genre and director are repeated to **increase their weight** relative to
description, nudging the model to prioritise thematic and stylistic similarity.

### Step 3 — TF-IDF Vectorisation
`TfidfVectorizer` converts every movie's `content` string into a sparse
high-dimensional vector (up to 15 000 features, unigrams + bigrams).

- **TF (Term Frequency)** — how often a word appears in a movie's content.
- **IDF (Inverse Document Frequency)** — down-weights words that appear in
  almost every movie (e.g. "the", "a") so genre-specific terms get boosted.

### Step 4 — Cosine Similarity Matrix
The dot-product of the normalised TF-IDF matrix with itself produces an
*n × n* similarity matrix where entry `[i, j]` is the cosine similarity
between movie *i* and movie *j* (0 = completely different, 1 = identical).

### Step 5 — Recommendation
Given a query movie at index `i`, we:
1. Fetch row `i` of the similarity matrix.
2. Sort all other movies by descending score.
3. Return the top-N titles (with poster, genre, rating).

### Why Cosine Similarity?
It measures the **angle** between two vectors, not their magnitude — so a
short description and a long description can still match perfectly if they
use the same key terms.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎯 Content-based recommendations | TF-IDF + Cosine Similarity |
| 🔍 Live search bar | Filter the 10k-title dropdown instantly |
| 🎲 Random movie button | Picks a random movie and recommends immediately |
| 🏆 Top Rated section | Always-visible best-rated movies |
| 🎭 Genre Explorer | Browse any genre, sorted by rating |
| 🖼️ Movie posters | Uses IMDb poster URLs from the dataset |
| 🌑 Netflix dark theme | Full custom CSS, hover effects, red accents |
| ⚡ Cached model | Similarity matrix computed once, reused across sessions |

---

## ⚙️ Performance Notes

- **10 000 movies** → similarity matrix built in ~3 seconds on a modern CPU.
- For **50 000 movies**, the matrix grows to 50k × 50k = 2.5 GB (float32).
  Consider switching to `linear_kernel` (no square matrix needed) or using
  approximate nearest-neighbour libraries like `faiss` / `annoy`.
- Streamlit's `@st.cache_resource` ensures the model is only built once per
  server session, not on every page reload.
  
