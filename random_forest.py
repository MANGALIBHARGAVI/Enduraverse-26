from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        max_depth=14,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_random_forest(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return predictions, metrics