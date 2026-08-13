# Deployment Guide

## Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload all files and folders from this project.
3. Commit and push the repository.
4. In Streamlit Community Cloud, choose **Create app** from the GitHub repository.
5. Set the entry point to:

```text
app.py
```

6. Deploy.

The application requires no API key or external data service.

## Recommended Repository Name

`cement-slurry-ai`

## Recommended Repository Description

`Explainable ML portfolio project for synthetic cement slurry thickening-time prediction using Random Forest, SHAP, sensitivity analysis, and Streamlit.`

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

On Windows, activate with:

```text
.venv\Scripts\activate
```

## Reproducibility

If you change the synthetic data-generating assumptions:

1. rerun `generate_data.py`;
2. rerun `train_model.py`;
3. rerun `explain_model.py`; and
4. commit the regenerated CSV/model/artifacts.

## Scope

The package is a portfolio demonstration only. It is not validated against laboratory
or field cementing data and is not suitable for operational or safety-critical decisions.
