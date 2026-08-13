"""
Streamlit app: AI-Assisted Cement Slurry Thickening-Time Predictor

Run:
    streamlit run app.py

Synthetic portfolio/demo project only. Not validated for operational,
laboratory, safety-critical, or field cementing decisions.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "cement_thickening_time_rf.joblib"
DATA_FILE = BASE_DIR / "slurry_performance_data.csv"

FEATURE_LABELS = {
    "BHCT_F": "BHCT (°F)",
    "Depth_ft": "Depth (ft)",
    "Retarder_pct_BWOC": "Retarder (% BWOC)",
    "Slurry_Density_ppg": "Slurry density (ppg)",
}

st.set_page_config(
    page_title="Cement Slurry AI | Thickening-Time Predictor",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_FILE)
    explainer = shap.TreeExplainer(artifact["model"])
    return artifact, explainer


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def make_input_frame(features, bhct, depth, retarder, density):
    row = {
        "BHCT_F": bhct,
        "Depth_ft": depth,
        "Retarder_pct_BWOC": retarder,
        "Slurry_Density_ppg": density,
    }
    return pd.DataFrame([row])[features]


def make_sensitivity_grid(model, features, depth, density):
    bhct_values = np.arange(100, 301, 5)
    retarder_values = np.round(np.arange(0.10, 1.501, 0.05), 2)

    rows = []
    for retarder in retarder_values:
        grid = pd.DataFrame(
            {
                "BHCT_F": bhct_values.astype(float),
                "Depth_ft": depth,
                "Retarder_pct_BWOC": float(retarder),
                "Slurry_Density_ppg": density,
            }
        )[features]
        predictions = model.predict(grid)

        for bhct, pred in zip(bhct_values, predictions):
            rows.append(
                {
                    "BHCT_F": float(bhct),
                    "Retarder_pct_BWOC": float(retarder),
                    "Predicted_Thickening_Time_min": float(pred),
                }
            )

    return pd.DataFrame(rows)


artifact, explainer = load_model()
model = artifact["model"]
features = artifact["features"]
data = load_data()

st.title("AI-Assisted Cement Slurry Thickening-Time Prediction")
st.caption(
    "Synthetic engineering data → Random Forest → SHAP explainability → "
    "sensitivity analysis → engineer-facing deployment"
)

st.warning(
    "Portfolio/demo tool only. The model is trained entirely on synthetic data, "
    "is not calibrated to laboratory or field cementing data, and must not be "
    "used for operational decisions."
)

tab_predict, tab_explain, tab_sensitivity, tab_about = st.tabs(
    ["Predict", "SHAP Explainability", "Sensitivity Analysis", "About"]
)

with tab_predict:
    st.subheader("Engineering inputs")

    c1, c2 = st.columns(2)
    with c1:
        bhct = st.slider(
            "Bottomhole Circulating Temperature (BHCT), °F",
            100.0,
            300.0,
            200.0,
            1.0,
        )
        retarder = st.slider(
            "Retarder concentration, % BWOC",
            0.10,
            1.50,
            0.70,
            0.05,
        )

    with c2:
        depth = st.slider(
            "Depth, ft",
            5000,
            15000,
            10000,
            100,
        )
        density = st.slider(
            "Slurry density, ppg",
            15.0,
            17.5,
            16.2,
            0.1,
        )

    input_df = make_input_frame(features, bhct, depth, retarder, density)
    predicted_minutes = float(model.predict(input_df)[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted thickening time", f"{predicted_minutes:.0f} min")
    m2.metric("Equivalent duration", f"{predicted_minutes / 60:.2f} h")
    m3.metric("Model", "Random Forest")

    st.dataframe(
        input_df.rename(columns=FEATURE_LABELS),
        use_container_width=True,
        hide_index=True,
    )

with tab_explain:
    st.subheader("Why did the model make this prediction?")

    explain_df = make_input_frame(features, bhct, depth, retarder, density)
    explanation = explainer(explain_df)
    base_value = float(np.ravel(explanation.base_values)[0])
    shap_values = np.ravel(explanation.values[0])
    prediction = float(model.predict(explain_df)[0])

    local = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(f, f) for f in features],
            "Input": [float(explain_df.iloc[0][f]) for f in features],
            "SHAP contribution (min)": shap_values,
        }
    )
    local["Absolute impact (min)"] = local["SHAP contribution (min)"].abs()
    local = local.sort_values("Absolute impact (min)", ascending=False)

    e1, e2 = st.columns(2)
    e1.metric("Model baseline", f"{base_value:.1f} min")
    e2.metric("Current prediction", f"{prediction:.1f} min")

    st.caption(
        "Positive SHAP values push the prediction toward a longer thickening time; "
        "negative values push it toward a shorter thickening time."
    )

    chart_df = (
        local.set_index("Feature")[["SHAP contribution (min)"]]
        .sort_values("SHAP contribution (min)")
    )
    st.bar_chart(chart_df)

    display_local = local[
        ["Feature", "Input", "SHAP contribution (min)"]
    ].copy()
    display_local["SHAP contribution (min)"] = (
        display_local["SHAP contribution (min)"].round(2)
    )
    st.dataframe(display_local, use_container_width=True, hide_index=True)

    st.subheader("Global feature impact")
    sample = data[features].sample(n=min(300, len(data)), random_state=42)
    global_explanation = explainer(sample)
    mean_abs_shap = np.abs(global_explanation.values).mean(axis=0)

    global_df = (
        pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in features],
                "Mean absolute SHAP (min)": mean_abs_shap,
            }
        )
        .sort_values("Mean absolute SHAP (min)", ascending=False)
        .set_index("Feature")
    )
    st.bar_chart(global_df)

with tab_sensitivity:
    st.subheader("BHCT × Retarder sensitivity")

    st.write(
        "The surface varies **BHCT** and **retarder concentration** while holding "
        f"depth at **{depth:,} ft** and slurry density at **{density:.1f} ppg**."
    )

    sensitivity = make_sensitivity_grid(model, features, depth, density)
    pivot = sensitivity.pivot(
        index="Retarder_pct_BWOC",
        columns="BHCT_F",
        values="Predicted_Thickening_Time_min",
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
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
    ax.set_title("Predicted Thickening-Time Sensitivity")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Predicted thickening time (min)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "This visualizes model sensitivity within the synthetic training domain. "
        "It is not a laboratory or operational cement design chart."
    )

with tab_about:
    st.subheader("Project scope")
    st.markdown(
        """
This project demonstrates:

- engineering-aware synthetic data generation with explicit bounds;
- a formal data dictionary and basic data-quality checks;
- Random Forest regression with holdout evaluation;
- `joblib` model serialization;
- local and global **SHAP** explainability;
- two-factor **BHCT × retarder sensitivity analysis**; and
- a deployable **Streamlit** engineer-facing interface.

**Generated input domain**

- BHCT: 100–300 °F
- Depth: 5,000–15,000 ft
- Retarder: 0.1–1.5% BWOC
- Slurry density: 15–17.5 ppg

Real slurry behavior depends on many additional variables including cement chemistry,
additive package, pressure, water chemistry, rheology, mixing history, and laboratory
temperature/pressure schedules. This project intentionally does not claim field validation.
        """
    )
