import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(model, X_test, y_test, name=""):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}

movies_path = Path(__file__).resolve().parents[1] / "datasets" / "ml-latest-small" / "movies.csv"
ratings_path = Path(__file__).resolve().parents[1] / "datasets" / "ml-latest-small" / "ratings.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
movie_stats = movie_stats.merge(movies, on="movieId")
movie_stats["year"] = movie_stats["title"].str.extract(r"\((\d{4})\)").astype(float)
movie_stats = movie_stats.dropna(subset=["year"])

X = movie_stats[["count", "year"]]
y = movie_stats["mean"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# THE baseline — always predicts the mean, no learning at all
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_mae = mean_absolute_error(y_test, dummy.predict(X_test))
print(f"Dummy baseline MAE: {dummy_mae:.3f}")



results = []
results.append(evaluate(dummy, X_test, y_test, "Dummy (mean)"))

lr = LinearRegression()
lr.fit(X_train, y_train)
results.append(evaluate(lr, X_test, y_test, "LinearRegression (count+year)"))


# one-hot encode genres — each genre becomes its own 0/1 column
genre_dummies = movie_stats["genres"].str.get_dummies(sep="|")
print(genre_dummies.columns.tolist())  # see what genres exist

# combine with your existing numeric features
X_full = pd.concat([movie_stats[["count", "year"]], genre_dummies], axis=1)
y = movie_stats["mean"]

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

lr_genres = LinearRegression()
lr_genres.fit(X_train, y_train)
results.append(evaluate(lr_genres, X_test, y_test, "LinearRegression (count+year+genres)"))


coefs = pd.Series(lr_genres.coef_, index=X_full.columns).sort_values(ascending=False)
print("Genres associated with HIGHER predicted rating:")
print(coefs.head(5))
print("\nGenres associated with LOWER predicted rating:")
print(coefs.tail(5))

#--------------------
coefs = pd.Series(lr_genres.coef_, index=X_full.columns).sort_values(ascending=False)
print("Genres associated with HIGHER predicted rating:")
print(coefs.head(5))
print("\nGenres associated with LOWER predicted rating:")
print(coefs.tail(5))

results_df = pd.DataFrame(results)
print(results_df)