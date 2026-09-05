from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import math


def train_lightgbm(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_lightgbm(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return predictions, metrics