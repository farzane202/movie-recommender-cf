def recommend_by_genre(movies, ratings, genre_input, n=5):

    genre_input = genre_input.lower()

    filtered_movies = movies[
        movies["genres"].str.lower().str.contains(genre_input, na=False)
    ]

    if filtered_movies.empty:
        print("No movies found for this genre.")
        return None

    avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

    merged = filtered_movies.merge(avg_ratings, on="movieId")

    top_movies = merged.sort_values(by="rating", ascending=False).head(n)

    return top_movies[["title", "genres", "rating"]]