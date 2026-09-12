from multiprocessing import dummy

import pandas as pd
from sklearn import dummy
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
print("movie_stats.head():", movie_stats.head())
movie_stats = movie_stats.merge(movies, on="movieId")
print("movie_stats.head() --- after merge:", movie_stats.head())

# extract year as a numeric feature (you already know how from Day 2)
movie_stats["year"] = movie_stats["title"].str.extract(r"\((\d{4})\)").astype(float)
print("movie_stats.head() --- after year extraction:", movie_stats.head())
movie_stats = movie_stats.dropna(subset=["year"])  # drop rows with no year found

# X = features we THINK might predict rating, y = the answer we're trying to predict
X = movie_stats[["count", "year"]]
y = movie_stats["mean"]
print(f"X.shape: {X.shape}, y.shape: {y.shape}")
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

##################### D4
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.dummy import DummyRegressor

def evaluate(model, X_test, y_test, name=""):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}
dummy = DummyRegressor(strategy="mean")
results = []
results.append(evaluate(dummy, X_test, y_test, "Dummy (mean)"))

lr = LinearRegression()
lr.fit(X_train, y_train)
results.append(evaluate(lr, X_test, y_test, "LinearRegression (count+year)"))