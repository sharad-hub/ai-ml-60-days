from pathlib import Path

import pandas as pd

movies_path = Path(__file__).resolve().parents[2] / "datasets" / "ml-latest-small" / "movies.csv"
movies = pd.read_csv(movies_path)
ratings_path = Path(__file__).resolve().parents[2] / "datasets" / "ml-latest-small" / "ratings.csv"
ratings = pd.read_csv(ratings_path)

print(movies.head(10))
print(movies.info())
print(movies.describe(include="all"))

print(ratings.head(10))
print(ratings.info())
print(ratings.describe())


print("Number of movies:", movies.shape[0])
print("Number of ratings:", ratings.shape[0])
print("Average rating:", ratings["rating"].mean())
print("Unique genres sample:", movies["genres"].str.split("|").explode().unique()[:10])