from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import FEATURE_COLUMNS, prepare_training_data


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uber.csv"
MODEL_PATH = BASE_DIR / "best_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.csv"
REPORT_PATH = BASE_DIR / "model_report.json"
RANDOM_STATE = 42


def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": rmse,
        "r2": r2_score(y_test, predictions),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    x, y = prepare_training_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        metrics = evaluate_model(model, x_train, x_test, y_train, y_test)
        trained_models[model_name] = model
        results.append(
            {
                "model": model_name,
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "r2": round(metrics["r2"], 4),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by=["rmse", "mae", "r2"])
    best_model_name = results_df.iloc[0]["model"]
    deployment_model_name = best_model_name
    deployment_model = trained_models[deployment_model_name]

    artifact = {
        "model_name": deployment_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "model": deployment_model,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    results_df.to_csv(METRICS_PATH, index=False)

    report = {
        "dataset_path": str(DATA_PATH),
        "total_rows_after_cleaning": int(len(x)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "best_model": best_model_name,
        "deployment_model": deployment_model_name,
        "metrics": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training completed.")
    print(results_df.to_string(index=False))
    print(f"\nBest comparison model: {best_model_name}")
    print(f"Deployment model saved to: {MODEL_PATH.name} ({deployment_model_name})")


if __name__ == "__main__":
    main()
