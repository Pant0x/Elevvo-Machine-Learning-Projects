# app.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import streamlit as st

# ------------------ Load Data ------------------ #
@st.cache_data
def load_data():
    base_path = os.path.join(os.path.dirname(__file__), "ml-100k")
    ratings_file = os.path.join(base_path, "u.data")
    movies_file = os.path.join(base_path, "u.item")

    ratings_df = pd.read_csv(
        ratings_file, 
        sep='\t', 
        names=['user_id','movie_id','rating','timestamp'], 
        encoding='latin-1'
    )

    movies_df = pd.read_csv(
        movies_file, 
        sep='|', 
        names=['movie_id','movie_title','release_date','video_release_date',
               'IMDb_URL','unknown','Action','Adventure','Animation',
               'Childrens','Comedy','Crime','Documentary','Drama',
               'Fantasy','Film-Noir','Horror','Musical','Mystery',
               'Romance','Sci-Fi','Thriller','War','Western'],
        encoding='latin-1', 
        usecols=['movie_id','movie_title']
    )

    df = pd.merge(ratings_df, movies_df, on='movie_id', how='left')
    return df, movies_df

# ------------------ User-Movie Matrix ------------------ #
@st.cache_data
def create_user_movie_matrix(df):
    user_movie_matrix = df.pivot_table(index='user_id', columns='movie_title', values='rating')
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    return user_movie_matrix, user_movie_matrix_filled

# ------------------ Similarity & SVD ------------------ #
@st.cache_data
def calculate_user_similarity(user_movie_matrix_filled):
    user_similarity = cosine_similarity(user_movie_matrix_filled)
    return pd.DataFrame(user_similarity, index=user_movie_matrix_filled.index, columns=user_movie_matrix_filled.index)

@st.cache_resource
def train_svd_model(user_movie_matrix, n_components=20):
    user_movie_matrix_svd = user_movie_matrix.fillna(user_movie_matrix.mean())
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(user_movie_matrix_svd)
    predicted_ratings = np.dot(latent_matrix, svd.components_)
    return pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

# ------------------ User-Based Recommendation ------------------ #
def recommend_movies_user_based(user_id, user_movie_matrix, user_similarity_df, num_recommendations=10, num_similar_users=5):
    if user_id not in user_movie_matrix.index:
        return pd.Series(), f"User ID {user_id} not found."

    similar_users = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    top_similar_users = similar_users.head(num_similar_users).index

    user_rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index

    predicted_ratings = {}
    for similar_user in top_similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        unseen_movies = similar_user_ratings[~similar_user_ratings.index.isin(user_rated_movies)]
        unseen_movies = unseen_movies[unseen_movies > 0]

        for movie_title, rating in unseen_movies.items():
            predicted_ratings[movie_title] = predicted_ratings.get(movie_title, 0) + rating * user_similarity_df.loc[user_id, similar_user]

    normalized_predicted_ratings = {}
    for movie_title, sum_weighted_rating in predicted_ratings.items():
        users_who_rated = user_movie_matrix[movie_title][user_movie_matrix[movie_title] > 0].index
        relevant_similar_users = [u for u in top_similar_users if u in users_who_rated]
        if relevant_similar_users:
            sum_sim = user_similarity_df.loc[user_id, relevant_similar_users].sum()
            normalized_predicted_ratings[movie_title] = sum_weighted_rating / sum_sim if sum_sim > 0 else 0
        else:
            normalized_predicted_ratings[movie_title] = 0

    recommendations = pd.Series(normalized_predicted_ratings).sort_values(ascending=False)
    return recommendations.head(num_recommendations), None

# ------------------ Item-Based Recommendation ------------------ #
def recommend_movies_item_based(user_id, user_movie_matrix, num_recommendations=10, num_similar_items=5):
    if user_id not in user_movie_matrix.index:
        return pd.Series(), f"User ID {user_id} not found."

    item_user_matrix = user_movie_matrix.T.fillna(0)
    if item_user_matrix.shape[0] < 2:
        return pd.Series(), "Not enough items."

    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

    user_ratings = user_movie_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]

    predicted_ratings = {}
    for movie_title, rating in rated_movies.items():
        if movie_title not in item_similarity_df.index:
            continue
        similar_items = item_similarity_df[movie_title].drop(movie_title).sort_values(ascending=False)
        top_similar_items = similar_items.head(num_similar_items).index
        for sim_movie in top_similar_items:
            if sim_movie not in rated_movies.index:
                predicted_ratings[sim_movie] = predicted_ratings.get(sim_movie, 0) + item_similarity_df.loc[movie_title, sim_movie] * rating

    normalized_predicted_ratings = {}
    for movie_title, sum_weighted in predicted_ratings.items():
        contributing_movies = [m for m in rated_movies.index if m in item_similarity_df.columns and movie_title in item_similarity_df.index]
        sum_sim = sum(item_similarity_df.loc[m, movie_title] for m in contributing_movies)
        normalized_predicted_ratings[movie_title] = sum_weighted / sum_sim if sum_sim > 0 else 0

    recommendations = pd.Series(normalized_predicted_ratings).sort_values(ascending=False)
    return recommendations.head(num_recommendations), None

# ------------------ SVD Recommendation ------------------ #
def recommend_movies_svd(user_id, user_movie_matrix, predicted_ratings_df_svd, num_recommendations=10):
    if user_id not in user_movie_matrix.index:
        return pd.Series(), f"User ID {user_id} not found."

    user_rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    user_predictions = predicted_ratings_df_svd.loc[user_id]
    unseen_predictions = user_predictions.drop(user_rated_movies, errors='ignore')
    recommendations = unseen_predictions.sort_values(ascending=False)
    return recommendations.head(num_recommendations), None

# ------------------ Display Recommendations ------------------ #
def display_recommendations(recommendations, title):
    st.markdown(f"### {title}")
    if recommendations.empty:
        st.info("No recommendations available.")
    else:
        for movie, rating in recommendations.items():
            st.write(f"{movie} — Predicted rating: {rating:.2f}/5")

# ------------------ Main App ------------------ #
def main():
    st.title("🎬 FilmFinder AI")
    with st.spinner("Loading movie dataset..."):
        df, movies_df = load_data()
        user_movie_matrix, user_movie_matrix_filled = create_user_movie_matrix(df)
        user_similarity_df = calculate_user_similarity(user_movie_matrix_filled)
        predicted_ratings_svd = train_svd_model(user_movie_matrix)

    st.success("✨ Dataset loaded successfully!")

    with st.sidebar:
        st.header("Settings")
        all_user_ids = sorted(df['user_id'].unique())
        target_user = st.selectbox("Select User ID", all_user_ids, index=0)
        num_recs = st.slider("Number of Recommendations", 1, 20, 10)

    st.header(f"Recommendations for User {target_user}")

    col1, col2, col3 = st.columns(3)
    with col1:
        user_recs, _ = recommend_movies_user_based(target_user, user_movie_matrix, user_similarity_df, num_recs)
        display_recommendations(user_recs, "User-Based CF")
    with col2:
        item_recs, _ = recommend_movies_item_based(target_user, user_movie_matrix, num_recs)
        display_recommendations(item_recs, "Item-Based CF")
    with col3:
        svd_recs, _ = recommend_movies_svd(target_user, user_movie_matrix, predicted_ratings_svd, num_recs)
        display_recommendations(svd_recs, "SVD CF")

if __name__ == "__main__":
    main()
