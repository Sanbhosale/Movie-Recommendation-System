import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data():
    df = pd.read_csv("movies_metadata.csv", low_memory=False)
    df = df[['title', 'genres', 'overview']]
    df.dropna(subset=['overview'], inplace=True)
    df = df[df['overview'].apply(lambda x: isinstance(x, str))]
    return df

# Preprocess genres
def preprocess_genres(genres_str):
    try:
        import ast
        genres = ast.literal_eval(genres_str)
        return ' '.join([g['name'] for g in genres])
    except:
        return ''

# Recommend movies
def recommend_movies(df, user_genres, watched_titles, top_n=5):
    df['genres_clean'] = df['genres'].apply(preprocess_genres)
    df['combined'] = df['overview'] + ' ' + df['genres_clean']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])

    # Create a pseudo-profile for the user
    user_profile = ' '.join(user_genres)
    user_vec = tfidf.transform([user_profile])

    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df['similarity'] = cosine_sim
    recommendations = df[~df['title'].isin(watched_titles)].sort_values(by='similarity', ascending=False)
    return recommendations[['title', 'genres_clean']].head(top_n)

# Streamlit UI
def main():
    st.title("🎬 Movie Recommendation System")

    df = load_data()

    st.sidebar.header("User Preferences")
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
              'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
    selected_genres = st.sidebar.multiselect("Preferred Genres", genres)
    age_group = st.sidebar.selectbox("Age Group", options=['Under 18', '18-25', '26-35', '36-50', '50+'],index=None, placeholder="Select your age group"
)
    platform = st.sidebar.selectbox("Viewing Platform", options=['Netflix', 'Amazon Prime', 'Disney+', 'Other'],index=None, placeholder="Select your age group")

    watched_movies = st.sidebar.text_area("Previously Watched Movies (comma-separated)")
    rating = st.sidebar.slider("Rating Given to Last Movie", 1, 5, 4)

    if st.sidebar.button("Get Recommendations"):
        watched_list = [title.strip() for title in watched_movies.split(',')]
        recommendations = recommend_movies(df, selected_genres, watched_list)

        st.subheader("📽️ Recommended Genre:")
        st.write(", ".join(selected_genres) if selected_genres else "Mixed")

        st.subheader("🎯 Recommended Movies:")
        for idx, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}** — _Genres: {row['genres_clean']}_")

if __name__ == "__main__":
    main()
