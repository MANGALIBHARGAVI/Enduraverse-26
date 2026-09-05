from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import math


def train_catboost(X_train, y_train):
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        iterations=180,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_catboost(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return predictions, metrics