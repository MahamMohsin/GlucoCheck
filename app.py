"""
GlucoCheck - Flask Web UI
"""

import os, pickle, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "trained_models.pkl")

#Load models once at startup 
models = None

def load_models():
    global models
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "trained_models.pkl not found.\n"
            "Run  python src/glucocheck_model.py  first to train and save the models."
        )
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    print("✓ Models loaded:", list(models.keys()))

#Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        values = [float(data[f]) for f in feature_names]
        patient_df = pd.DataFrame([values], columns=feature_names)

        lr     = models["lr_model"]
        scaler = models["scaler"]
        rf     = models["rf_model"]

        # Logistic Regression (needs scaled input)
        patient_sc = scaler.transform(patient_df)
        lr_pred    = int(lr.predict(patient_sc)[0])
        lr_prob    = float(lr.predict_proba(patient_sc)[0][1])

        # Random Forest (raw input)
        rf_pred    = int(rf.predict(patient_df)[0])
        rf_prob    = float(rf.predict_proba(patient_df)[0][1])

        # Ensemble
        avg_prob   = (lr_prob + rf_prob) / 2
        final      = 1 if avg_prob >= 0.5 else 0

        # Top risk factors from RF importances
        importances = rf.feature_importances_
        scaled_vals = patient_sc[0]
        contributions = sorted(
            zip(feature_names, importances, scaled_vals),
            key=lambda x: abs(x[2]) * x[1], reverse=True
        )
        top_factors = [{"name": n, "score": round(abs(v)*i*100, 1)}
                       for n, i, v in contributions[:4]]

        return jsonify({
            "lr_pred":    lr_pred,
            "lr_prob":    round(lr_prob * 100, 1),
            "rf_pred":    rf_pred,
            "rf_prob":    round(rf_prob * 100, 1),
            "avg_prob":   round(avg_prob * 100, 1),
            "final":      final,
            "top_factors": top_factors,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_models()
    print("\n  GlucoCheck running at  http://localhost:5007\n")
    app.run(debug=True, port=5007)