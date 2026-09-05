import argparse
import importlib.util
import json
from pathlib import Path
import sys
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _load_local_module(module_name: str, file_name: str):
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if "" in sys.path:
    sys.path.remove("")


linear_regression_module = _load_local_module("linear_regression_module", "linear_regression.py")
random_forest_module = _load_local_module("random_forest_module", "random_forest.py")
gradient_boosting_module = _load_local_module("gradient_boosting_module", "gradient_boosting.py")
xgboost_module = _load_local_module("xgboost_module", "xgboost.py")
lightgbm_module = _load_local_module("lightgbm_module", "lightgbm.py")
catboost_module = _load_local_module("catboost_module", "catboost.py")
svr_module = _load_local_module("svr_module", "svr.py")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _extract_feature_importance(model, feature_cols):
    values = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef.ravel())
    elif hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "feature_importances_"):
            values = np.asarray(final_step.feature_importances_, dtype=float)
        elif hasattr(final_step, "coef_"):
            coef = np.asarray(final_step.coef_, dtype=float)
            values = np.abs(coef.ravel())

    if values is None or len(values) != len(feature_cols):
        return []

    paired = [
        {"feature": feature_cols[index], "importance": float(values[index])}
        for index in range(len(feature_cols))
    ]
    paired.sort(key=lambda item: item["importance"], reverse=True)
    return paired


def _safe_train(
    name,
    train_func,
    eval_func,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_cols,
    model_store_dir,
):
    try:
        model = train_func(X_train, y_train)
        predictions, metrics = eval_func(model, X_test, y_test)

        model_file_name = f"{_slugify(name)}.joblib"
        model_path = model_store_dir / model_file_name
        joblib.dump(model, model_path)

        predictions_np = np.asarray(predictions, dtype=float)
        actuals_np = np.asarray(y_test, dtype=float)

        sample_predictions = predictions_np[:80]
        sample_actuals = actuals_np[:80]
        sample_errors = sample_predictions - sample_actuals

        scatter_points = [
            {
                "actual": float(sample_actuals[index]),
                "predicted": float(sample_predictions[index]),
            }
            for index in range(len(sample_predictions))
        ]

        return {
            "name": name,
            "status": "ok",
            "metrics": metrics,
            "model_file": f"models/{model_file_name}",
            "sample_predictions": [float(value) for value in sample_predictions],
            "sample_actuals": [float(value) for value in sample_actuals],
            "sample_errors": [float(value) for value in sample_errors],
            "scatter_points": scatter_points,
            "feature_importance": _extract_feature_importance(model, feature_cols),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "error": str(exc),
            "metrics": None,
            "model_file": None,
            "sample_predictions": [],
            "sample_actuals": [],
            "sample_errors": [],
            "scatter_points": [],
            "feature_importance": [],
        }


def train_multi_model(dataset_path: Path, output_path: Path, max_rows: int | None = None):
    frame = pd.read_csv(dataset_path)

    if "Target_RUL_Cycles" not in frame.columns:
        raise ValueError("Target_RUL_Cycles column not found in dataset")

    frame = frame.dropna()
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42)

    target_col = "Target_RUL_Cycles"

    feature_cols = [
        col for col in frame.columns if col != target_col and np.issubdtype(frame[col].dtype, np.number)
    ]

    X = frame[feature_cols]
    y = frame[target_col]

    model_store_dir = output_path.parent / "models"
    model_store_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    results = []
    results.append(
        _safe_train(
            "Linear Regression",
            linear_regression_module.train_linear_regression,
            linear_regression_module.evaluate_linear_regression,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "Random Forest",
            random_forest_module.train_random_forest,
            random_forest_module.evaluate_random_forest,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "Gradient Boosting",
            gradient_boosting_module.train_gradient_boosting,
            gradient_boosting_module.evaluate_gradient_boosting,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "XGBoost",
            xgboost_module.train_xgboost,
            xgboost_module.evaluate_xgboost,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "LightGBM",
            lightgbm_module.train_lightgbm,
            lightgbm_module.evaluate_lightgbm,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "CatBoost",
            catboost_module.train_catboost,
            catboost_module.evaluate_catboost,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "SVR",
            svr_module.train_svr,
            svr_module.evaluate_svr,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )

    successful = [item for item in results if item["status"] == "ok"]
    successful = sorted(
        successful,
        key=lambda item: item["metrics"]["rmse"],
    )

    best_model = successful[0]["name"] if successful else None

    payload = {
        "dataset": str(dataset_path),
        "rows_used": int(len(frame)),
        "target_column": target_col,
        "feature_columns": feature_cols,
        "test_size": 0.2,
        "successful_model_count": len(successful),
        "best_model": best_model,
        "models": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train multi-model RUL regressors")
    parser.add_argument(
        "--dataset",
        default="../battery_dataset_final.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="../rul-dashboard-backend/data/multi-model-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=25000,
        help="Optional cap for sampled training rows to speed up training",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()

    train_multi_model(dataset_path, output_path, max_rows=args.max_rows)
    print(f"Training complete. Results written to {output_path}")


if __name__ == "__main__":
    main()