import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def run_collaborative_filtering(ratings):

    # Train/Test Split
    train_data, test_data = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    # Create User-Item Matrix
    user_item_matrix = train_data.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )

    # Calculate user mean ratings
    user_means = user_item_matrix.mean(axis=1)

    # Mean Centering
    user_item_centered = user_item_matrix.sub(user_means, axis=0)

    # Replace NaN with 0 AFTER centering
    user_item_centered = user_item_centered.fillna(0)

    # Compute Item Similarity
    item_similarity = cosine_similarity(user_item_centered.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    # Prediction Function
    def predict_rating(user_id, movie_id):

        if movie_id not in item_similarity_df.columns:
            return user_means.get(user_id, 0)

        if user_id not in user_item_centered.index:
            return 0

        similar_movies = item_similarity_df[movie_id]
        user_ratings = user_item_centered.loc[user_id]

        numerator = np.dot(similar_movies, user_ratings)
        denominator = np.sum(np.abs(similar_movies))

        if denominator == 0:
            return user_means[user_id]

        prediction = user_means[user_id] + (numerator / denominator)

        return prediction

    # Calculate RMSE
    predictions = []
    actuals = []

    for row in test_data.itertuples():
        pred = predict_rating(row.userId, row.movieId)
        predictions.append(pred)
        actuals.append(row.rating)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    print(f"\nCollaborative Filtering RMSE: {rmse:.4f}")