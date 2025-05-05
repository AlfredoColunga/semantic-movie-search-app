import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import streamlit as st

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="data_embeddings")
db = chroma_client.get_collection(name="movies_db")

def search(query, genre, rating, max_results):
    # Generate embedding from query
    query_vector = model.encode(query).tolist()

    filter_rating = rating

    if genre:
        if isinstance(genre, str):
            genre = [genre]
        genre = [g.lower() for g in genre]
    else:
        genre = []

    # Query
    results = db.query(
        query_embeddings=[query_vector],
        n_results=max_results * 2,
        include=["documents", "metadatas", "distances"]
    )

    # Apply filters
    response_data = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # Filter by genre
        if genre and not any(g in meta.get("Generes", "").lower() for g in genre):
            continue
        # Filter by rating
        if float(meta.get("Rating", 0)) < filter_rating:
            continue

        # Append results into list
        response_data.append({
            "Title": meta.get("movie title"),
            "Overview": meta.get("Overview"),
            "Director": meta.get("Director"),
            "Genre": meta.get("Generes"),
            "Year": meta.get("year"),
            "Rating": meta.get("Rating"),
            "Score": dist
        })

    df = pd.DataFrame(response_data[:max_results])

    return df

genres = ['Action', 'Drama', 'Adventure', 'Sci-Fi', 'Animation', 'Crime',
          'Comedy', 'Thriller', 'Fantasy', 'Horror', 'History', 'Mystery',
          'Biography', 'War', 'Western', 'Sport', 'Family', 'Romance',
          'Music', 'Musical', 'Film-Noir', 'Game-Show', 'Adult',
          'Reality-TV']

st.title(":movie_camera: Semantic Movie Search Engine")

st.subheader("")

with st.form("form"):
    col_query, col_genre = st.columns(2)
    query_input = col_query.text_area("Query:")
    genre_input = col_genre.multiselect("Genre:", options=genres, default=None)

    col_score, col_results = st.columns(2)
    score_input = col_score.slider("Minimum rating:", min_value=0, max_value=10)
    results_input = col_results.number_input("Desired results:", min_value=3, max_value=10)

    search_btn = st.form_submit_button("Search")

if search_btn:
    if query_input:
        responses = search(
            query=query_input,
            genre=genre_input,
            rating=score_input,
            max_results=results_input
        )

        st.dataframe(responses, use_container_width=True, hide_index=True)