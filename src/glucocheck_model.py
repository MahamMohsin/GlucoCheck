"""
GlucoCheck - Diabetes Prediction using Pima Indians Dataset
Binary Classification: Diabetic (1) / Non-Diabetic (0)
Models: Logistic Regression (primary) + Random Forest (secondary)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_fscore_support
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATASET  (download if not present)
# ─────────────────────────────────────────────
def load_data():
    """Load Pima Indians Diabetes Dataset."""
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    # Try local file first
    local_path = os.path.join(BASE_DIR, "data", "diabetes.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"✓ Dataset loaded from local file: {df.shape}")
        return df

    # Generate realistic synthetic data matching Pima dataset statistics
    print("⚠  Local dataset not found – generating synthetic Pima-like data for demo...")
    np.random.seed(42)
    n = 768

    data = {
        'Pregnancies': np.random.randint(0, 18, n),
        'Glucose': np.clip(np.random.normal(120, 32, n), 44, 199).astype(int),
        'BloodPressure': np.clip(np.random.normal(69, 19, n), 0, 122).astype(int),
        'SkinThickness': np.clip(np.random.normal(20, 16, n), 0, 99).astype(int),
        'Insulin': np.clip(np.random.normal(79, 115, n), 0, 846).astype(int),
        'BMI': np.clip(np.random.normal(32, 7.9, n), 0, 67.1).round(1),
        'DiabetesPedigreeFunction': np.clip(np.random.exponential(0.5, n), 0.08, 2.42).round(3),
        'Age': np.clip(np.random.normal(33, 12, n), 21, 81).astype(int),
    }
    df = pd.DataFrame(data)
    # Simulate outcome based on risk factors
    score = (
        (df['Glucose'] > 140).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) +
        (df['Age'] > 45).astype(int) +
        (df['DiabetesPedigreeFunction'] > 0.8).astype(int) +
        np.random.normal(0, 0.5, n)
    )
    df['Outcome'] = (score > 2.5).astype(int)
    df.to_csv(local_path, index=False)
    print(f"✓ Synthetic dataset created: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """Replace biologically invalid zeros and scale features."""
    print("\n── Preprocessing ──────────────────────────────────────")
    zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid:
        n_zeros = (df[col] == 0).sum()
        if n_zeros:
            mean_val = df[col][df[col] != 0].mean()
            df[col] = df[col].replace(0, mean_val)
            print(f"  Replaced {n_zeros:3d} zero(s) in '{col}' with mean={mean_val:.2f}")

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print(f"\n  Class distribution:")
    counts = y.value_counts()
    print(f"    Non-Diabetic (0): {counts[0]}  ({counts[0]/len(y)*100:.1f}%)")
    print(f"    Diabetic     (1): {counts[1]}  ({counts[1]/len(y)*100:.1f}%)")

    return X, y

# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE
# ─────────────────────────────────────────────
def train_and_evaluate(X, y):
    results = {}

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Logistic Regression ──────────────────────
    print("\n── Logistic Regression ─────────────────────────────────")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)

    lr_pred  = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_acc   = accuracy_score(y_test, lr_pred)
    lr_auc   = roc_auc_score(y_test, lr_proba)
    lr_cv    = cross_val_score(lr, X_train_sc, y_train, cv=skf, scoring='accuracy')

    p, r, f, _ = precision_recall_fscore_support(y_test, lr_pred, average='weighted')
    print(f"  Accuracy : {lr_acc:.4f}")
    print(f"  AUC      : {lr_auc:.4f}")
    print(f"  CV Mean  : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
    print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}")

    results['lr'] = dict(
        model=lr, scaler=scaler,
        y_test=y_test, y_pred=lr_pred, y_proba=lr_proba,
        accuracy=lr_acc, auc=lr_auc,
        cv_mean=lr_cv.mean(), cv_std=lr_cv.std(),
        precision=p, recall=r, f1=f,
        cm=confusion_matrix(y_test, lr_pred).tolist(),
        feature_names=list(X.columns),
        coef=lr.coef_[0].tolist()
    )

    # ── Random Forest ────────────────────────────
    print("\n── Random Forest ───────────────────────────────────────")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_base = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    rf = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")

    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_acc   = accuracy_score(y_test, rf_pred)
    rf_auc   = roc_auc_score(y_test, rf_proba)
    rf_cv    = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')

    p2, r2, f2, _ = precision_recall_fscore_support(y_test, rf_pred, average='weighted')
    print(f"  Accuracy : {rf_acc:.4f}")
    print(f"  AUC      : {rf_auc:.4f}")
    print(f"  CV Mean  : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    print(f"  Precision: {p2:.4f}  Recall: {r2:.4f}  F1: {f2:.4f}")

    results['rf'] = dict(
        model=rf,
        y_test=y_test, y_pred=rf_pred, y_proba=rf_proba,
        accuracy=rf_acc, auc=rf_auc,
        cv_mean=rf_cv.mean(), cv_std=rf_cv.std(),
        precision=p2, recall=r2, f1=f2,
        cm=confusion_matrix(y_test, rf_pred).tolist(),
        feature_names=list(X.columns),
        importances=rf.feature_importances_.tolist()
    )

    # Save models for predict.py
    model_path = os.path.join(OUTPUT_DIR, "trained_models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            'lr_model': results['lr']['model'],
            'rf_model': results['rf']['model'],
            'scaler':   results['lr']['scaler'],
            'feature_names': list(X.columns),
        }, f)
    print(f"\n  ✓ Models saved to: {model_path}")

    return results

# ─────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────
PALETTE = {
    'bg':      '#0D1117',
    'panel':   '#161B22',
    'accent1': '#00E5A0',  # teal-green
    'accent2': '#FF6B6B',  # coral
    'accent3': '#4ECDC4',
    'text':    '#E6EDF3',
    'muted':   '#8B949E',
}

def _style_ax(ax, title=''):
    ax.set_facecolor(PALETTE['panel'])
    ax.tick_params(colors=PALETTE['muted'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')
    if title:
        ax.set_title(title, color=PALETTE['text'], fontsize=12,
                     fontfamily='monospace', pad=10)

def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Confusion Matrices', color=PALETTE['text'],
                 fontsize=16, fontfamily='monospace', y=1.02)

    labels = ['Non-Diabetic', 'Diabetic']
    models_info = [
        ('lr', 'Logistic Regression', PALETTE['accent1']),
        ('rf', 'Random Forest',        PALETTE['accent2']),
    ]

    for ax, (key, name, color) in zip(axes, models_info):
        cm = np.array(results[key]['cm'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if key == 'lr' else 'Reds',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False,
                    annot_kws={'size': 16, 'color': 'white', 'weight': 'bold'})
        ax.set_facecolor(PALETTE['panel'])
        ax.set_title(f'{name}\nAccuracy: {results[key]["accuracy"]:.3f}',
                     color=PALETTE['text'], fontfamily='monospace', fontsize=11)
        ax.tick_params(colors=PALETTE['muted'])
        ax.set_xlabel('Predicted', color=PALETTE['muted'])
        ax.set_ylabel('Actual', color=PALETTE['muted'])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {path}")

def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    for key, name, color in [
        ('lr', f"Logistic Regression (AUC={results['lr']['auc']:.3f})", PALETTE['accent1']),
        ('rf', f"Random Forest       (AUC={results['rf']['auc']:.3f})", PALETTE['accent2']),
    ]:
        fpr, tpr, _ = roc_curve(results[key]['y_test'], results[key]['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2.5, label=name)

    ax.plot([0,1],[0,1], '--', color=PALETTE['muted'], lw=1.2, label='Random Baseline')
    ax.fill_between([0,1],[0,1], alpha=0.05, color=PALETTE['muted'])

    _style_ax(ax, 'ROC Curves — GlucoCheck')
    ax.set_xlabel('False Positive Rate', color=PALETTE['muted'])
    ax.set_ylabel('True Positive Rate', color=PALETTE['muted'])
    ax.legend(facecolor=PALETTE['panel'], edgecolor='#30363D',
              labelcolor=PALETTE['text'], fontsize=9, loc='lower right')
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {path}")

def plot_feature_importance(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle('Feature Importance / Coefficients', color=PALETTE['text'],
                 fontsize=15, fontfamily='monospace')

    features = results['rf']['feature_names']

    # RF importances
    ax = axes[0]
    imp = np.array(results['rf']['importances'])
    idx = np.argsort(imp)
    bars = ax.barh([features[i] for i in idx], imp[idx], color=PALETTE['accent2'],
                   alpha=0.85, edgecolor='none', height=0.65)
    _style_ax(ax, 'Random Forest — Feature Importances')
    ax.set_xlabel('Importance', color=PALETTE['muted'])
    ax.tick_params(axis='y', colors=PALETTE['text'])
    ax.tick_params(axis='x', colors=PALETTE['muted'])
    for bar, val in zip(bars, imp[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color=PALETTE['muted'], fontsize=8)

    # LR coefficients
    ax = axes[1]
    coef = np.array(results['lr']['coef'])
    idx2 = np.argsort(coef)
    colors = [PALETTE['accent2'] if c > 0 else PALETTE['accent3'] for c in coef[idx2]]
    bars2 = ax.barh([features[i] for i in idx2], coef[idx2], color=colors,
                    alpha=0.85, edgecolor='none', height=0.65)
    _style_ax(ax, 'Logistic Regression — Coefficients')
    ax.set_xlabel('Coefficient Value', color=PALETTE['muted'])
    ax.tick_params(axis='y', colors=PALETTE['text'])
    ax.tick_params(axis='x', colors=PALETTE['muted'])
    ax.axvline(0, color=PALETTE['muted'], lw=0.8, linestyle='--')
    pos_patch = mpatches.Patch(color=PALETTE['accent2'], label='Increases Risk')
    neg_patch = mpatches.Patch(color=PALETTE['accent3'], label='Decreases Risk')
    ax.legend(handles=[pos_patch, neg_patch], facecolor=PALETTE['panel'],
              edgecolor='#30363D', labelcolor=PALETTE['text'], fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {path}")

def plot_metrics_comparison(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['panel'])

    metrics   = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
    lr_vals   = [results['lr']['accuracy'], results['lr']['auc'],
                 results['lr']['precision'], results['lr']['recall'], results['lr']['f1']]
    rf_vals   = [results['rf']['accuracy'], results['rf']['auc'],
                 results['rf']['precision'], results['rf']['recall'], results['rf']['f1']]

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, lr_vals, w, label='Logistic Regression',
                color=PALETTE['accent1'], alpha=0.85, edgecolor='none')
    b2 = ax.bar(x + w/2, rf_vals, w, label='Random Forest',
                color=PALETTE['accent2'], alpha=0.85, edgecolor='none')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom',
                color=PALETTE['text'], fontsize=8, fontfamily='monospace')

    _style_ax(ax, 'Model Comparison — Evaluation Metrics')
    ax.set_xticks(x); ax.set_xticklabels(metrics, color=PALETTE['text'])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', color=PALETTE['muted'])
    ax.axhline(0.78, color='#FFD700', lw=1.2, linestyle='--', alpha=0.7)
    ax.text(4.6, 0.785, 'Target 78%', color='#FFD700', fontsize=8)
    ax.legend(facecolor=PALETTE['panel'], edgecolor='#30363D',
              labelcolor=PALETTE['text'], fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Saved: {path}")

# ─────────────────────────────────────────────
# 5. SAVE JSON SUMMARY
# ─────────────────────────────────────────────
def save_summary(results):
    summary = {}
    for key in ('lr', 'rf'):
        r = results[key]
        summary[key] = {
            'accuracy':  round(r['accuracy'], 4),
            'auc':       round(r['auc'], 4),
            'cv_mean':   round(r['cv_mean'], 4),
            'cv_std':    round(r['cv_std'], 4),
            'precision': round(r['precision'], 4),
            'recall':    round(r['recall'], 4),
            'f1':        round(r['f1'], 4),
            'confusion_matrix': r['cm'],
            'feature_names': r['feature_names'],
        }
        if key == 'lr':
            summary[key]['coefficients'] = [round(c, 4) for c in r['coef']]
        else:
            summary[key]['feature_importances'] = [round(i, 4) for i in r['importances']]

    path = os.path.join(OUTPUT_DIR, 'results_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {path}")
    return summary

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║          GlucoCheck — Diabetes Classifier            ║")
    print("╚══════════════════════════════════════════════════════╝")

    df      = load_data()
    X, y    = preprocess(df)
    results = train_and_evaluate(X, y)

    print("\n── Generating Plots ────────────────────────────────────")
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_feature_importance(results)
    plot_metrics_comparison(results)
    summary = save_summary(results)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║                  FINAL RESULTS                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    for name, key in [('Logistic Regression', 'lr'), ('Random Forest', 'rf')]:
        s = summary[key]
        target_acc = "✓" if s['accuracy'] >= 0.78 else "✗"
        target_auc = "✓" if s['auc'] >= 0.82 else "✗"
        print(f"║  {name:<22} Acc={s['accuracy']:.3f}{target_acc}  AUC={s['auc']:.3f}{target_auc}  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
