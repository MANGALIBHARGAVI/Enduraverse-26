import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if "" in sys.path:
    sys.path.remove("")


def predict(registry_path: Path, model_name: str, features_json: str):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    selected = None
    for model in registry.get("models", []):
        if model.get("name", "").lower() == model_name.lower():
            selected = model
            break

    if not selected:
        raise ValueError(f"Model '{model_name}' was not found")

    if selected.get("status") != "ok":
        raise ValueError(f"Model '{selected.get('name')}' is not available for inference")

    model_file = selected.get("model_file")
    if not model_file:
        raise ValueError("Model file path is missing in registry")

    model_path = registry_path.parent / model_file
    if not model_path.exists():
        raise ValueError(f"Persisted model not found at {model_path}")

    feature_columns = registry.get("feature_columns", [])
    incoming = json.loads(features_json)

    row = {}
    for column in feature_columns:
        if column not in incoming:
            raise ValueError(f"Missing required feature: {column}")
        row[column] = float(incoming[column])

    model = joblib.load(model_path)
    frame = pd.DataFrame([row], columns=feature_columns)
    prediction = float(model.predict(frame)[0])

    payload = {
        "model": selected.get("name"),
        "prediction": prediction,
        "features_used": row,
    }
    print(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description="Predict RUL from persisted model")
    parser.add_argument("--registry", required=True, help="Path to multi-model-results.json")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--features-json", required=True, help="JSON string with feature values")

    args = parser.parse_args()

    predict(
        registry_path=Path(args.registry).resolve(),
        model_name=args.model,
        features_json=args.features_json,
    )


if __name__ == "__main__":
    main()