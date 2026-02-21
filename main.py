import pandas as pd

from models.collaborative import run_collaborative_filtering
from models.svd_model import run_svd_model
from models.content_based import recommend_by_genre


# Load Data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")


# Run Models
run_collaborative_filtering(ratings)
run_svd_model(ratings)


# Genre-Based Recommendation
print("\n--- Genre-Based Movie Recommendation ---")
genre_input = input("Enter a genre (e.g., action, drama, comedy): ")

results = recommend_by_genre(movies, ratings, genre_input)

if results is not None:
    print("\nTop Movies in this Genre:")
    print(results)