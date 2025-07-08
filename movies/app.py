# app.py

import numpy as np
import pandas as pd
import difflib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the data
@st.cache_data
def load_data():
    movies_data = pd.read_csv('movies.csv')

    # Select relevant features
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combine selected features
    combined_features = (
        movies_data['genres'] + ' ' +
        movies_data['keywords'] + ' ' +
        movies_data['tagline'] + ' ' +
        movies_data['cast'] + ' ' +
        movies_data['director']
    )

    # Vectorization
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Similarity score
    similarity = cosine_similarity(feature_vectors)

    return movies_data, similarity

# Recommend movies
def recommend_movies(movie_name, movies_data, similarity):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return ["❌ No match found. Please try another movie."]

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for movie in sorted_similar_movies[1:31]:  # Skip the input movie itself
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommended_movies.append(title_from_index)

    return recommended_movies

# Streamlit UI
def main():
    st.title("🎬 Movie Recommendation System")
    st.subheader("Get movie suggestions based on your favourite movie")
    st.markdown("Built with **Python, Machine Learning & Streamlit**")

    movies_data, similarity = load_data()

    movie_name = st.text_input("🔍 Enter your favourite movie:")

    if st.button("Recommend"):
        if movie_name.strip() == "":
            st.warning("⚠️ Please enter a movie name.")
        else:
            recommendations = recommend_movies(movie_name, movies_data, similarity)
            st.subheader("🎥 Recommended Movies:")
            for i, title in enumerate(recommendations):
                st.write(f"{i+1}. {title}")

if __name__ == "__main__":
    main()
