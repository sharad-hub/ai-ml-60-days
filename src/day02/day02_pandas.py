import pandas as pd
from pathlib import Path

movies_path = Path(__file__).resolve().parents[2] / "datasets" / "ml-latest-small" / "movies.csv"
ratings_path = Path(__file__).resolve().parents[2] / "datasets" / "ml-latest-small" / "ratings.csv"

movies= pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# basic filtering and aggregation
comedies = movies[movies["genres"].str.contains("Comedy")]
print("Number of comedies:", comedies.shape[0])
print("Number of comedies:", len(comedies))

average_rating = ratings["rating"].mean()
print("Average rating:", average_rating)

# Multiple conditions
action_or_scifi = movies[(movies["genres"].str.contains("Action")) | (movies["genres"].str.contains("Sci-Fi"))]
print("Number of Action or Sci-Fi movies:", action_or_scifi.shape[0])

# filtering 
# filtering ratings
high_ratings = ratings[ratings["rating"] >= 4.5]
print(f"Number of ratings >= 4.5: {len(high_ratings)}")

# .loc vs .iloc — know the difference
print(movies.loc[0:5, ["title", "genres"]])   # label-based
print(movies.iloc[0:5, 0:2])  

# General aggregation

# average rating per movie
avg_rating_per_movie = ratings.groupby("movieId")["rating"].mean()
print(avg_rating_per_movie.head())

# count of ratings per movie (popularity proxy)
rating_counts = ratings.groupby("movieId")["rating"].count()
print(rating_counts.sort_values(ascending=False).head(10))

# multiple aggregations at once
movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count", "std"])
print(movie_stats.head())

# groupby with sorting — most-rated movies
print(movie_stats.sort_values("count", ascending=False).head(10))



###
# merge movie titles into the ratings/stats table
merged = movie_stats.merge(movies, on="movieId")
print(merged.head())

# now answer a real question: best movies with a minimum popularity threshold
popular_and_good = merged[merged["count"] >= 50].sort_values("mean", ascending=False)
print(popular_and_good[["title", "mean", "count"]].head(10))