"""
GlucoCheck — Patient Prediction Interface
Run this after glucocheck_model.py to predict for a new patient.
Usage: python src/predict.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "trained_models.pkl")


def load_models():
    if not os.path.exists(MODEL_PATH):
        print("No trained models found. Run glucocheck_model.py first.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_input(prompt, min_val, max_val, dtype=float):
    while True:
        try:
            val = dtype(input(f"  {prompt}: ").strip())
            if min_val <= val <= max_val:
                return val
            print(f"    Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("    Invalid input. Please enter a number.")


def predict_patient(models):
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       GlucoCheck — Patient Prediction Tool          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\nEnter patient details below:\n")

    fields = [
        ("Pregnancies (0–17)",                   0,   17,  int),
        ("Glucose level mg/dL (44–199)",          44,  199, float),
        ("Blood Pressure mmHg (24–122)",          24,  122, float),
        ("Skin Thickness mm (7–99)",               7,   99, float),
        ("Insulin µU/mL (14–846)",                14,  846, float),
        ("BMI kg/m² (18–67)",                     18,   67, float),
        ("Diabetes Pedigree Function (0.08–2.42)",0.08,2.42,float),
        ("Age (21–81)",                           21,   81, int),
    ]

    values = []
    for label, lo, hi, dt in fields:
        values.append(get_input(label, lo, hi, dt))

    patient = np.array(values).reshape(1, -1)
    feature_names = [
        'Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age'
    ]
    patient_df = pd.DataFrame(patient, columns=feature_names)

    lr     = models['lr_model']
    scaler = models['scaler']
    rf     = models['rf_model']

    # LR prediction
    patient_sc   = scaler.transform(patient_df)
    lr_pred      = lr.predict(patient_sc)[0]
    lr_prob      = lr.predict_proba(patient_sc)[0][1]

    # RF prediction
    rf_pred      = rf.predict(patient_df)[0]
    rf_prob      = rf.predict_proba(patient_df)[0][1]

    # Ensemble (average probabilities)
    avg_prob     = (lr_prob + rf_prob) / 2
    final        = 1 if avg_prob >= 0.5 else 0

    print("\n══════════════════════════════════════════════════════")
    print("  PREDICTION RESULTS")
    print("══════════════════════════════════════════════════════")
    print(f"  Logistic Regression : {'Diabetic' if lr_pred else 'Non-Diabetic':15s} ({lr_prob*100:.1f}% risk)")
    print(f"  Random Forest       : {'Diabetic' if rf_pred else 'Non-Diabetic':15s} ({rf_prob*100:.1f}% risk)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Ensemble (avg)      : {'Diabetic' if final else 'Non-Diabetic':15s} ({avg_prob*100:.1f}% risk)")

    if final == 1:
        print("\n  [!] HIGH RISK — Recommend medical consultation.")
    else:
        print("\n  [✓] LOW RISK — Routine health monitoring advised.")

    print("══════════════════════════════════════════════════════")
    print("\n  Disclaimer: This is a ML prediction tool and does")
    print("  not replace professional medical advice.\n")

    again = input("Predict another patient? (y/n): ").strip().lower()
    if again == 'y':
        predict_patient(models)


if __name__ == '__main__':
    models = load_models()
    predict_patient(models)
