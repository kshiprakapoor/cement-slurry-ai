"""
Generate explainability and sensitivity artifacts for the synthetic
cement slurry thickening-time Random Forest.

Outputs:
    artifacts/shap_global_importance.csv
    artifacts/shap_summary_bar.png
    artifacts/shap_beeswarm.png
    artifacts/temperature_retarder_sensitivity.csv
    artifacts/temperature_retarder_sensitivity.png

IMPORTANT:
This project uses synthetic data and is for portfolio/demo purposes only.
It is not validated for field or laboratory cementing decisions.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "slurry_performance_data.csv"
MODEL_FILE = BASE_DIR / "cement_thickening_time_rf.joblib"
OUTPUT_DIR = BASE_DIR / "artifacts"

OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_LABELS = {
    "BHCT_F": "BHCT (°F)",
    "Depth_ft": "Depth (ft)",
    "Retarder_pct_BWOC": "Retarder (% BWOC)",
    "Slurry_Density_ppg": "Slurry density (ppg)",
}


def load_assets():
    df = pd.read_csv(DATA_FILE)
    artifact = joblib.load(MODEL_FILE)
    return df, artifact["model"], artifact["features"]


def generate_shap_artifacts(df, model, features):
    sample = df[features].sample(n=min(400, len(df)), random_state=42)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(sample)

    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    global_importance = (
        pd.DataFrame(
            {
                "Feature": features,
                "Mean_Absolute_SHAP_min": mean_abs_shap,
            }
        )
        .sort_values("Mean_Absolute_SHAP_min", ascending=False)
        .reset_index(drop=True)
    )
    global_importance["Feature_Label"] = global_importance["Feature"].map(FEATURE_LABELS)
    global_importance.to_csv(OUTPUT_DIR / "shap_global_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    shap.plots.bar(explanation, max_display=len(features), show=False)
    plt.title("Global SHAP Feature Impact")
    plt.xlabel("Mean |SHAP value| (minutes)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(explanation, max_display=len(features), show=False)
    plt.title("SHAP Impact Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    return global_importance


def generate_sensitivity_artifacts(model, features):
    bhct_values = np.arange(100, 301, 5)
    retarder_values = np.round(np.arange(0.10, 1.501, 0.05), 2)
    depth_ft = 10000
    density_ppg = 16.2

    rows = []
    for retarder in retarder_values:
        grid = pd.DataFrame(
            {
                "BHCT_F": bhct_values.astype(float),
                "Depth_ft": depth_ft,
                "Retarder_pct_BWOC": float(retarder),
                "Slurry_Density_ppg": density_ppg,
            }
        )[features]
        predictions = model.predict(grid)

        for bhct, pred in zip(bhct_values, predictions):
            rows.append(
                {
                    "BHCT_F": float(bhct),
                    "Depth_ft": depth_ft,
                    "Retarder_pct_BWOC": float(retarder),
                    "Slurry_Density_ppg": density_ppg,
                    "Predicted_Thickening_Time_min": round(float(pred), 2),
                }
            )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(
        OUTPUT_DIR / "temperature_retarder_sensitivity.csv",
        index=False,
    )

    pivot = sensitivity.pivot(
        index="Retarder_pct_BWOC",
        columns="BHCT_F",
        values="Predicted_Thickening_Time_min",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
    )
    ax.set_xlabel("BHCT (°F)")
    ax.set_ylabel("Retarder concentration (% BWOC)")
    ax.set_title(
        "Temperature–Retarder Sensitivity\n"
        "Predicted Thickening Time at 10,000 ft and 16.2 ppg"
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Predicted thickening time (min)")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "temperature_retarder_sensitivity.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    return sensitivity


def main():
    df, model, features = load_assets()
    importance = generate_shap_artifacts(df, model, features)
    sensitivity = generate_sensitivity_artifacts(model, features)

    print("Explainability artifacts generated.")
    print("\nGlobal mean absolute SHAP values:")
    print(
        importance[
            ["Feature_Label", "Mean_Absolute_SHAP_min"]
        ].round(2).to_string(index=False)
    )
    print(f"\nSensitivity grid rows: {len(sensitivity):,}")


if __name__ == "__main__":
    main()
