# Breast Cancer Detector (WDBC) — Flask Website

Educational demo: a simple logistic regression model trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

**Not medical advice.**

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 2) Run the website

```bash
python app.py
```

Then open http://127.0.0.1:5000

## 3) Batch predictions

Upload `sample_inputs.csv` from this folder (or your own CSV) with headers matching the feature names.

## 4) Re-train (optional)

If you want to train from the original ZIP you uploaded:

```bash
python train_model.py --zip "/path/to/breast+cancer+wisconsin+original.zip" --out model.pkl
```

## API (optional)
POST JSON to `/api/predict`:

```json
{ "features": [30 numbers] }
```