import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path

movies_path = Path(__file__).resolve().parents[1] / "datasets" / "ml-latest-small" / "movies.csv"
ratings_path = Path(__file__).resolve().parents[1] / "datasets" / "ml-latest-small" / "ratings.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# build a tiny, deliberately crude feature set: avg rating per movie + number of ratings
movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
movie_stats = movie_stats.merge(movies, on="movieId")

# extract year as a numeric feature (you already know how from Day 2)
movie_stats["year"] = movie_stats["title"].str.extract(r"\((\d{4})\)").astype(float)
movie_stats = movie_stats.dropna(subset=["year"])  # drop rows with no year found

# X = features we THINK might predict rating, y = the answer we're trying to predict
X = movie_stats[["count", "year"]]
y = movie_stats["mean"]

# THE split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} movies, testing on {len(X_test)} movies")

model = LinearRegression()
model.fit(X_train, y_train)          # learning happens here

predictions = model.predict(X_test)   # inference — model never saw X_test's answers

# how honest is the model, on data it never saw?
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error on test set: {mae:.3f}")

# sanity check: compare a few real vs predicted
comparison = pd.DataFrame({"actual": y_test.values[:10], "predicted": predictions[:10]})
print(comparison)