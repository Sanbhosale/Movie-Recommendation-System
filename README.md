# Movie Recommendation System

A personalized movie recommendation system built using Streamlit and content-based filtering. Users can select genres, input previously watched movies, and receive tailored suggestions.

## Features
- Genre-based recommendations
- Interactive Streamlit UI
- Content-based filtering via TF-IDF and cosine similarity

## How It Works
1. Loads movie metadata from the MovieLens dataset.
2. Combines movie overviews and genres into one text field.
3. Uses TF-IDF vectorization to represent movie descriptions.
4. Compares a user’s pseudo-profile with all movies using cosine similarity.

## Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn

## Dataset
**Source:** [The Movies Dataset – Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  
**File Used:** `movies_metadata.csv`
