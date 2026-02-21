import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def run_svd_model(ratings):

   
    train_data, test_data = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

  
    user_item_matrix = train_data.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )

   
    user_means = user_item_matrix.mean(axis=1)

    user_item_centered = user_item_matrix.sub(user_means, axis=0)
    user_item_centered = user_item_centered.fillna(0)

    # ---------------------------
    #  Apply SVD
    # ---------------------------
    svd = TruncatedSVD(
        n_components=50,
        random_state=42
    )

    latent_matrix = svd.fit_transform(user_item_centered)
    reconstructed = np.dot(latent_matrix, svd.components_)

    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    # Add user means back
    reconstructed_df = reconstructed_df.add(user_means, axis=0)

    # ---------------------------
    # RMSE Evaluation
    # ---------------------------
    predictions = []
    actuals = []

    for row in test_data.itertuples():

        if (
            row.userId in reconstructed_df.index
            and row.movieId in reconstructed_df.columns
        ):
            pred = reconstructed_df.loc[row.userId, row.movieId]
        else:
            pred = user_means.get(row.userId, 0)

        predictions.append(pred)
        actuals.append(row.rating)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    print(f"SVD RMSE: {rmse:.4f}")
