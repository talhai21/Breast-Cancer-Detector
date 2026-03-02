import pickle
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load model - handle both local and Vercel paths
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    scaler = bundle["scaler"]
    model = bundle["model"]
except FileNotFoundError:
    print(f"Warning: model.pkl not found at {model_path}")
    scaler = None
    model = None

def predict_one(features):
    if model is None or scaler is None:
        raise Exception("Model not loaded. model.pkl is missing.")
    X = np.array(features, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    prob_malignant = float(model.predict_proba(Xs)[0, 1])
    label = "Malignant" if prob_malignant >= 0.5 else "Benign"
    confidence = prob_malignant if label == "Malignant" else (1 - prob_malignant)
    return label, confidence

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list):
        return jsonify({"error": "Send JSON like {'features': [30 numbers]}"}), 400
    if len(features) != 30:
        return jsonify({"error": "Please provide exactly 30 numbers."}), 400

    try:
        label, confidence = predict_one(features)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/api/predict_batch")
def api_predict_batch():
    data = request.get_json(silent=True) or {}
    rows = data.get("rows")

    if not isinstance(rows, list):
        return jsonify({"error": "Send JSON like {'rows': [[30 numbers], [30 numbers], ...]}"}), 400

    out = []
    try:
        for i, r in enumerate(rows, start=1):
            if not isinstance(r, list) or len(r) != 30:
                return jsonify({"error": f"Row {i} must have exactly 30 numbers."}), 400
            label, confidence = predict_one(r)
            out.append({"row": i, "label": label, "confidence": confidence})
        return jsonify({"results": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
