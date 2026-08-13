"""
Train a Random Forest model for synthetic cement slurry thickening-time prediction.

Outputs:
    cement_thickening_time_rf.joblib

IMPORTANT:
This model is trained on synthetic portfolio/demo data and is not intended
for field, laboratory, safety-critical, or operational cementing decisions.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "slurry_performance_data.csv"
MODEL_FILE = BASE_DIR / "cement_thickening_time_rf.joblib"
METRICS_FILE = BASE_DIR / "model_metrics.json"

FEATURES = [
    "BHCT_F",
    "Depth_ft",
    "Retarder_pct_BWOC",
    "Slurry_Density_ppg",
]
TARGET = "Thickening_Time_min"


def main():
    df = pd.read_csv(DATA_FILE)

    # Basic data-quality checks.
    required_columns = FEATURES + [TARGET]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[required_columns].isna().any().any():
        raise ValueError("Dataset contains missing values in modeling columns.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    feature_importance = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
        .round(4)
        .to_dict()
    )

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_bounds": {
            "BHCT_F": [100.0, 300.0],
            "Depth_ft": [5000, 15000],
            "Retarder_pct_BWOC": [0.1, 1.5],
            "Slurry_Density_ppg": [15.0, 17.5],
        },
        "disclaimer": (
            "Synthetic portfolio/demo model only. "
            "Not validated for operational cementing decisions."
        ),
    }

    joblib.dump(artifact, MODEL_FILE)

    metrics = {
        "MAE_min": round(mae, 3),
        "RMSE_min": round(rmse, 3),
        "R2": round(r2, 4),
        "feature_importance": feature_importance,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"MAE:  {mae:.2f} min")
    print(f"RMSE: {rmse:.2f} min")
    print(f"R²:   {r2:.3f}")
    print("\nFeature importance:")
    for feature, importance in feature_importance.items():
        print(f"  {feature:24s} {importance:.3f}")


if __name__ == "__main__":
    main()
