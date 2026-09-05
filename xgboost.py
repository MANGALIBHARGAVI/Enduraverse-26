from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import math


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_xgboost(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return predictions, metrics