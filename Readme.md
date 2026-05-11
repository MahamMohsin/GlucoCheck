# GlucoCheck - Diabetes Risk Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A machine learning web application that predicts diabetes risk using an ensemble of **Logistic Regression** and **Random Forest** models, trained on the Pima Indians Diabetes Dataset.

---

## Overview

GlucoCheck takes 8 standard clinical measurements from a patient and predicts whether they are at risk of diabetes, along with a percentage risk score and the top contributing health factors. It is served as a Flask web app with an interactive slider-based UI.

**Key features:**
- Ensemble model (LR + RF) for higher accuracy than either model alone
- Interactive web UI with real-time sliders for all 8 input features
- Top contributing risk factors shown per prediction
- Command-line prediction tool (no browser needed)
- Auto-generates synthetic data if the CSV dataset is not found
- Evaluation charts saved automatically on training

---

## Demo

| Low Risk | High Risk |
|---|---|
| Green banner, < 50% ensemble probability | Red banner, ≥ 50% ensemble probability |

The UI shows individual predictions from both models plus the ensemble average, with a visual breakdown of which features contributed most to the result.

---

## Project Structure

```
GLUCOCHECK/
├── data/
│   └── diabetes.csv              # Pima Indians Diabetes Dataset
├── src/
│   ├── glucocheck_model.py       # Model training, evaluation & chart generation
│   └── predict.py                # Terminal-based prediction interface
├── templates/
│   └── index.html                # Web UI (sliders + results panel)
├── outputs/                      # Auto-generated after training
│   ├── trained_models.pkl        # Saved LR model, RF model, scaler
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── metrics_comparison.png
│   └── results_summary.json
├── app.py                        # Flask web server
├── requirements.txt
└── README.md
```

---

## Dataset

**Pima Indians Diabetes Dataset** — 768 female patients, 9 columns.

| Feature | Description | Range |
|---|---|---|
| Pregnancies | Number of pregnancies | 0 – 17 |
| Glucose | Blood sugar level (mg/dL) | 44 – 199 |
| BloodPressure | Diastolic blood pressure (mmHg) | 24 – 122 |
| SkinThickness | Triceps skinfold thickness (mm) | 7 – 99 |
| Insulin | 2-hour serum insulin (µU/mL) | 14 – 846 |
| BMI | Body Mass Index (kg/m²) | 18 – 67 |
| DiabetesPedigreeFunction | Genetic/family history score | 0.08 – 2.42 |
| Age | Age in years | 21 – 81 |
| Outcome | Target — 1 = Diabetic, 0 = Non-Diabetic | — |

> If `data/diabetes.csv` is not found, the model script automatically generates a realistic synthetic dataset with the same statistical properties.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/glucocheck.git
cd glucocheck
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the models

Run this once. It trains both models, evaluates them, saves the `.pkl` file, and generates all output charts.

```bash
python src/glucocheck_model.py
```

Expected output:
```
✓ Dataset loaded: (768, 9)
✓ Data cleaned — zero values imputed
✓ Models trained and saved to outputs/trained_models.pkl
✓ Charts saved to outputs/
```

### Step 2 — Launch the web app

```bash
python app.py
```

Then open your browser at:
```
http://localhost:5007
```

### Optional — Terminal prediction tool

Skip the browser entirely and enter patient values directly in the terminal:

```bash
python src/predict.py
```

---

## How It Works

### Data Cleaning
Columns like Glucose, BMI, and Insulin can have biologically impossible zero values in the raw dataset. These are replaced with the column mean (imputation).

### Models

**Logistic Regression**
- Draws a linear decision boundary between diabetic and non-diabetic patients
- Requires `StandardScaler` to normalise all features to the same range
- Fast, interpretable, shows exact feature weights

**Random Forest**
- Builds 100–200 decision trees, each trained on a random data subset
- Final prediction is a majority vote across all trees
- Hyperparameters auto-tuned using `GridSearchCV`
- Provides feature importances showing which inputs matter most

**Ensemble**
- The final risk probability is the average of LR and RF probabilities
- Final prediction: Diabetic if average ≥ 50%, Non-Diabetic otherwise
- More reliable than either model alone

### Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | % of all patients classified correctly |
| AUC | Area under ROC curve — separability score (target ≥ 0.82) |
| Precision | Of predicted diabetics, how many actually were |
| Recall | Of actual diabetics, how many were caught |
| F1 Score | Harmonic mean of Precision and Recall |
| Cross-Validation | 5-fold CV mean ± std to confirm model stability |

---

## Test Cases

Use these to verify the UI is working correctly after setup:

**High Risk (expect: Diabetic)**
```
Pregnancies: 6 | Glucose: 168 | Blood Pressure: 82 | Skin Thickness: 35
Insulin: 200   | BMI: 38.5    | Pedigree: 1.20    | Age: 50
```

**Low Risk (expect: Non-Diabetic)**
```
Pregnancies: 1 | Glucose: 89 | Blood Pressure: 66 | Skin Thickness: 23
Insulin: 94    | BMI: 22.6   | Pedigree: 0.17     | Age: 25
```

**Borderline (models may disagree — tests the ensemble)**
```
Pregnancies: 3 | Glucose: 130 | Blood Pressure: 72 | Skin Thickness: 28
Insulin: 140   | BMI: 31.2   | Pedigree: 0.55      | Age: 38
```

---

## Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
flask>=3.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Output Charts

After running `glucocheck_model.py`, the `outputs/` folder contains:

- `confusion_matrices.png` — True/false positive and negative breakdown for each model
- `roc_curves.png` — ROC curve comparison with AUC scores
- `feature_importance.png` — RF importances + LR coefficients side by side
- `metrics_comparison.png` — Accuracy, AUC, Precision, Recall, F1 bar chart with target line
- `results_summary.json` — All metrics in machine-readable format

---

## Disclaimer

> GlucoCheck is built for **educational and research purposes only**. It is not a medical device and does not constitute medical advice. Predictions should never be used as a substitute for professional clinical diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

## Acknowledgements

- Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — originally from the National Institute of Diabetes and Digestive and Kidney Diseases
- Built with [scikit-learn](https://scikit-learn.org/), [Flask](https://flask.palletsprojects.com/), and [Pandas](https://pandas.pydata.org/)
