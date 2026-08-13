# Cement Slurry AI: Explainable Thickening-Time Prediction

A compact **upstream petroleum data-science portfolio project** that demonstrates an end-to-end engineering ML workflow.

> **Important:** all data and target values are synthetic. This is a portfolio/demo system, not a laboratory or field cement-design tool.

## Workflow

**Synthetic engineering data → Data dictionary → Random Forest → Holdout validation → SHAP explainability → BHCT/retarder sensitivity → Streamlit deployment**

## What the App Does

An engineer can vary:

- Bottomhole Circulating Temperature (**BHCT**)
- **Depth**
- **Retarder concentration** (% BWOC)
- **Slurry density** (ppg)

and receive a synthetic predicted cement slurry **thickening time**.

The app also includes:

- **local SHAP explainability** for each current prediction,
- **global SHAP feature impact**, and
- a **BHCT × retarder sensitivity surface** at the selected depth and slurry density.

## Synthetic Holdout Performance

- **R²:** 0.958
- **MAE:** 16.72 min
- **RMSE:** 21.43 min

These metrics apply only to the included synthetic dataset.

## Repository Structure

```text
cement-slurry-ai/
├── .streamlit/
│   └── config.toml
├── artifacts/
│   ├── shap_beeswarm.png
│   ├── shap_global_importance.csv
│   ├── shap_summary_bar.png
│   ├── temperature_retarder_sensitivity.csv
│   └── temperature_retarder_sensitivity.png
├── app.py
├── DATA_DICTIONARY.md
├── DEPLOYMENT.md
├── explain_model.py
├── generate_data.py
├── model_metrics.json
├── cement_thickening_time_rf.joblib
├── requirements.txt
├── slurry_performance_data.csv
└── train_model.py
```

## Engineering Data Dictionary

See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

| Feature | Unit | Generated Range |
|---|---:|---:|
| BHCT | °F | 100–300 |
| Depth | ft | 5,000–15,000 |
| Retarder concentration | % BWOC | 0.1–1.5 |
| Slurry density | ppg | 15.0–17.5 |
| Thickening time | min | 45–720 target bound |

## Explainability

### Global SHAP feature impact

![Global SHAP feature impact](artifacts/shap_summary_bar.png)

### SHAP impact distribution

![SHAP impact distribution](artifacts/shap_beeswarm.png)

SHAP values are expressed in **minutes of predicted thickening time**. Positive values
push the prediction above the model baseline; negative values push it below the baseline.

## Temperature–Retarder Sensitivity

![BHCT and retarder sensitivity](artifacts/temperature_retarder_sensitivity.png)

The static sensitivity artifact holds depth at **10,000 ft** and slurry density at
**16.2 ppg**. The live Streamlit app recomputes the surface at the engineer's selected
depth and density.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py
python train_model.py
python explain_model.py

streamlit run app.py
```

## Deploy

See [`DEPLOYMENT.md`](DEPLOYMENT.md). The repo is structured for direct deployment
through Streamlit Community Cloud.

## Portfolio Objective

The objective is to demonstrate the ability to:

1. translate an engineering problem into structured features and a target;
2. establish data definitions, units, and operating bounds;
3. implement reproducible model development and evaluation;
4. interrogate the model rather than treat it as a black box;
5. visualize coupled parameter sensitivity; and
6. deploy an engineer-facing inference interface.

## Limitations

Real cement slurry behavior depends on many additional factors such as cement chemistry
and class, additives, water chemistry, pressure, temperature schedule, mixing history,
rheology, fluid loss, strength development, and job-specific operational requirements.

**Do not use this model for field, laboratory, safety-critical, or operational decisions.**
