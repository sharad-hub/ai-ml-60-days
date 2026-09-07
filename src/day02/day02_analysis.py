import pandas as pd

from pathlib import Path

movies_path = Path(__file__).resolve().parents[2] / "datasets" / "ml-latest-small" / "movies.csv"
ratings_path = Path(__file__).resolve().parents[2] / "datasets" / "ml-latest-small" / "ratings.csv"
movies= pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

top_10_highest_rated = (
    ratings.groupby("movieId")
    .filter(lambda x: len(x) >= 30)
    .groupby("movieId")["rating"].mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .merge(movies[["movieId", "title"]], on="movieId")
)
print(top_10_highest_rated[["title", "rating"]])

# 2. This one's a real bug: you computed by year, not decade.
# The task asked for decade, but groupby("year") groups every individual year separately, and since year is a string, sorting will also be alphabetical, not chronological. Fix:

ratings_with_year = ratings.merge(movies, on="movieId")
ratings_with_year["year"] = ratings_with_year["title"].str.extract(r"\((\d{4})\)").astype(float)
ratings_with_year["decade"] = (ratings_with_year["year"] // 10 * 10).astype("Int64")

average_rating_by_decade = ratings_with_year.groupby("decade")["rating"].mean().sort_index()
print(average_rating_by_decade)

## 3
counts = ratings.groupby("movieId")["rating"].transform("count")
filtered = ratings[counts >= 30]
print("User with most ratings:", filtered["userId"].value_counts().idxmax())
# # Top 10 most popular genres (hint: genres column needs splitting on "|", then count)
# top_10_genres = movies["genres"].str.split("|").explode().value_counts().head(10)   
# print("Top 10 most popular genres:\n", top_10_genres)
# # Top 10 highest-rated movies with at least 30 ratings
# top_10_highest_rated = ratings.groupby("movieId").filter(lambda x: len(x) >= 30).groupby("movieId")["rating"].mean().sort_values(ascending=False).head(10)
# print("Top 10 highest-rated movies with at least 30 ratings:\n", top_10_highest_rated)
# # Average rating by decade (hint: you'll need to extract the year from the title string, e.g. "Toy Story (1995)" — try a regex or string split)
# ratings_with_year = ratings.merge(movies, on="movieId")
# ratings_with_year["year"] = ratings_with_year["title"].str.extract(r"\((\d{4})\)")
# average_rating_by_decade = ratings_with_year.groupby("year")["rating"].mean()
# print("Average rating by decade:\n", average_rating_by_decade)
# # Which single user (userId) has rated the most movies
# most_active_user = ratings["userId"].value_counts().idxmax()
# print("User with most ratings:", most_active_user)