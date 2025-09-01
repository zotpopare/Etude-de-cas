# shap/shap_explain.py
"""
Script utilitaire pour générer des explications SHAP à partir d'un modèle sauvegardé.
- Charge `models/model.pkl` et (optionnel) `models/preproc.pkl`.
- Calcule les valeurs SHAP pour un exemple ou un batch d'exemples.
- Sauvegarde un PNG des contributions et produit un JSON de sortie contenant :
  - top features positives/négatives,
  - chemin vers l'image générée,
  - (optionnel) image encodée en base64.

Usage (depuis la racine du repo):
  python shap/shap_explain.py --model models/model.pkl --input examples/example.json

Remarques:
- Adapte `FEATURE_NAMES` à l'ordre exact utilisé lors de l'entraînement.
- Pour les modèles non-tree, KernelExplainer peut être très lent : fournissez un `--background` (CSV) ou un prééchantillon.
"""

import argparse
import json
import os
import joblib
import numpy as np
import base64
from pathlib import Path

try:
    import shap
except Exception as e:
    raise SystemExit("SHAP is required for this script. Install via `pip install shap`
" + str(e))

import matplotlib.pyplot as plt

FEATURE_NAMES = ['age','income','credit_score','existing_loans','loan_amount_requested','employment_salaried','housing_own']


def load_model(path_model: str, path_preproc: str | None):
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"Model file not found: {path_model}")
    model = joblib.load(path_model)
    preproc = None
    if path_preproc and os.path.exists(path_preproc):
        try:
            preproc = joblib.load(path_preproc)
        except Exception:
            preproc = None
    return model, preproc


def read_input_json(path_json: str):
    with open(path_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # support single dict or list of dicts
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError('Input JSON must be a dict or a list of dicts')


def build_matrix_from_records(records: list, feature_names: list):
    X = []
    for rec in records:
        # map fields to feature order; if missing, uses 0 or sensible default
        row = [
            rec.get('age', 0),
            rec.get('income', 0.0),
            rec.get('credit_score', 0.0),
            rec.get('existing_loans', 0),
            rec.get('loan_amount_requested', 0.0),
            1 if str(rec.get('employment_status','')).lower() == 'salaried' else 0,
            1 if str(rec.get('housing_status','')).lower() == 'own' else 0,
        ]
        X.append(row)
    return np.array(X)


def ensure_2d(a: np.ndarray):
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def explain_with_shap(model, preproc, X_raw: np.ndarray, feature_names: list, background=None):
    # apply preproc if present
    X_proc = X_raw
    if preproc is not None:
        try:
            X_proc = preproc.transform(X_raw)
        except Exception:
            # fallback: use raw
            X_proc = X_raw

    # pick explainer
    try:
        if 'Tree' in str(type(model)) or 'XGB' in str(type(model)) or 'LGBM' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_proc)
        else:
            # KernelExplainer requires a background dataset
            if background is None:
                # use first row as background (not ideal) — warning user
                background = X_proc[0:1]
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_proc)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'initialisation explainer SHAP: {e}")

    # normalize shap_values shape
    if isinstance(shap_values, list):
        # binary classification -> shap_values[1]
        try:
            sv = shap_values[1]
        except Exception:
            sv = shap_values[0]
    else:
        sv = shap_values

    sv = np.array(sv)
    sv = ensure_2d(sv)
    return X_proc, sv


def summarize_shap_row(shap_row: np.ndarray, feature_names: list, top_k: int = 5):
    pairs = list(zip(feature_names, shap_row.tolist()))
    # sort by contribution (descending)
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    top_pos = [(f, float(v)) for f, v in sorted_pairs if v > 0][:top_k]
    top_neg = [(f, float(v)) for f, v in sorted_pairs if v < 0][:top_k]
    return top_pos, top_neg


def plot_shap_row(shap_row: np.ndarray, feature_names: list, out_png: str, title: str = 'SHAP contributions'):
    pairs = list(zip(feature_names, shap_row.tolist()))
    pairs_sorted = sorted(pairs, key=lambda x: x[1])  # ascending for horizontal bar
    labels = [p[0] for p in pairs_sorted]
    vals = [p[1] for p in pairs_sorted]

    plt.figure(figsize=(8, max(3, len(labels)*0.3)))
    plt.barh(labels, vals)
    plt.xlabel('SHAP value')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def encode_png_base64(path_png: str) -> str:
    with open(path_png, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description='Generate SHAP explanations for saved model')
    parser.add_argument('--model', type=str, default='models/model.pkl')
    parser.add_argument('--preproc', type=str, default='models/preproc.pkl')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file (single record or list)')
    parser.add_argument('--background', type=str, default=None, help='Optional CSV file to use as background for KernelExplainer')
    parser.add_argument('--out-png', type=str, default='shap/shap_contribs.png')
    parser.add_argument('--out-json', type=str, default='shap/shap_output.json')
    parser.add_argument('--top-k', type=int, default=5)

    args = parser.parse_args()

    model, preproc = load_model(args.model, args.preproc)
    records = read_input_json(args.input)
    X_raw = build_matrix_from_records(records, FEATURE_NAMES)

    background = None
    if args.background and os.path.exists(args.background):
        import pandas as pd
        bg_df = pd.read_csv(args.background)
        # build matrix using same mapping
        background = build_matrix_from_records(bg_df.to_dict(orient='records'), FEATURE_NAMES)

    X_proc, shap_vals = explain_with_shap(model, preproc, X_raw, FEATURE_NAMES, background=background)

    outputs = []
    for i in range(shap_vals.shape[0]):
        row_sv = shap_vals[i]
        top_pos, top_neg = summarize_shap_row(row_sv, FEATURE_NAMES, top_k=args.top_k)
        out_png = args.out_png
        if shap_vals.shape[0] > 1:
            # multiple outputs: append index to filename
            base, ext = os.path.splitext(args.out_png)
            out_png = f"{base}_{i}{ext}"
        plot_shap_row(row_sv, FEATURE_NAMES, out_png, title=f'SHAP contributions (sample {i})')
        img_b64 = encode_png_base64(out_png)
        outputs.append({
            'index': i,
            'top_positive_features': top_pos,
            'top_negative_features': top_neg,
            'png': out_png,
            'png_base64': img_b64
        })

    # save JSON summary
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f'Saved summary to {args.out_json}')


if __name__ == '__main__':
    main()File: shap/shap_explain.py

# shap/shap_explain.py
# Script pour produire top features via SHAP et sauvegarder un png
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import shap
except Exception:
    raise RuntimeError("SHAP required: pip install shap")

MODEL = 'models/model.pkl'
PREPROC = 'models/preproc.pkl'

if not os.path.exists(MODEL):
    raise FileNotFoundError('Place model.pkl in models/')

model = joblib.load(MODEL)
preproc = None
if os.path.exists(PREPROC):
    try:
        preproc = joblib.load(PREPROC)
    except Exception:
        preproc = None

# Example input vector - adapt order to celui utilisé lors de l'entraînement
feature_names = ['age','income','credit_score','existing_loans','loan_amount_requested','employment_salaried','housing_own']
X_sample = np.array([[30, 35000, 650, 1, 8000, 1, 0]])
if preproc is not None:
    try:
        X_proc = preproc.transform(X_sample)
    except Exception:
        X_proc = X_sample
else:
    X_proc = X_sample

# Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_proc)
if isinstance(shap_values, list):
    sv = shap_values[1][0]
else:
    sv = shap_values[0]

contribs = list(zip(feature_names, sv.tolist()))
contribs = sorted(contribs, key=lambda x: x[1], reverse=True)

# plot top 10 contributions
labels = [c[0] for c in contribs][:10]
vals = [c[1] for c in contribs][:10]
plt.figure(figsize=(8,4))
plt.barh(labels[::-1], vals[::-1])
plt.title('SHAP contributions (sample)')
plt.tight_layout()
plt.savefig('shap/shap_contribs.png', dpi=150)
print('Saved shap/shap_contribs.png')
