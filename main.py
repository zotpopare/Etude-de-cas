# api/main.py
import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

API_KEY = os.getenv("API_KEY", "dev-secret-key")  # en prod : définir via env

app = FastAPI(title="NeoBanque Loan Scoring API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement pipeline
PIPELINE_PATH = os.path.join("models", "pipeline.pkl")
pipeline = None
explainer = None

if os.path.exists(PIPELINE_PATH):
    try:
        pipeline = joblib.load(PIPELINE_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du pipeline : {e}")
else:
    print("pipeline.pkl introuvable !")

# SHAP explainer
if SHAP_AVAILABLE and pipeline is not None:
    try:
        if hasattr(pipeline, "predict_proba") and (
            "Tree" in str(type(pipeline)) or "XGB" in str(type(pipeline))
        ):
            explainer = shap.TreeExplainer(pipeline)
    except Exception as e:
        print(f"Impossible d'initialiser SHAP explainer : {e}")

# Pydantic schema
class ClientData(BaseModel):
    client_id: str
    age: int
    income: float
    credit_score: float
    existing_loans: int
    loan_amount_requested: float
    employment_status: str
    housing_status: str

# Auth dependency
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def preprocess_input(data: ClientData):
    """Prépare les données sous forme de DataFrame pour le pipeline"""
    # Si ton pipeline attend un DataFrame, adapte cette partie
    import pandas as pd
    df = pd.DataFrame([{
        "age": data.age,
        "income": data.income,
        "credit_score": data.credit_score,
        "existing_loans": data.existing_loans,
        "loan_amount_requested": data.loan_amount_requested,
        "employment_status": data.employment_status,
        "housing_status": data.housing_status,
    }])
    return df

def generate_plain_text(explain):
    pos = ", ".join([f"{f[0]}" for f in explain.get("top_positive_features", [])])
    neg = ", ".join([f"{f[0]}" for f in explain.get("top_negative_features", [])])
    text = ""
    if pos:
        text += f"Principaux points positifs qui augmentent l'éligibilité : {pos}. "
    if neg:
        text += f"Points à surveiller : {neg}."
    if text == "":
        text = "Aucune explication disponible."
    return text

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: ClientData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline non chargé sur le serveur.")
    try:
        X = preprocess_input(data)
        proba = float(pipeline.predict_proba(X)[0][1])
        label = "Eligible" if proba >= 0.5 else "Non-eligible"

        explain = {"top_positive_features": [], "top_negative_features": []}
        if SHAP_AVAILABLE and explainer is not None:
            try:
                shap_values = explainer.shap_values(X)
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                feature_names = list(X.columns)
                contributions = list(zip(feature_names, sv.tolist()))
                contributions_sorted = sorted(contributions, key=lambda x: x[1], reverse=True)
                top_pos = [(f, float(v)) for f, v in contributions_sorted if v > 0][:5]
                top_neg = [(f, float(v)) for f, v in contributions_sorted if v < 0][:5]
                explain["top_positive_features"] = [[f, round(v, 4)] for f, v in top_pos]
                explain["top_negative_features"] = [[f, round(v, 4)] for f, v in top_neg]
            except Exception:
                pass
        else:
            # fallback heuristics
            if data.income > 40000:
                explain["top_positive_features"].append(["income", 0.12])
            if data.existing_loans <= 1:
                explain["top_positive_features"].append(["existing_loans", 0.05])
            if data.credit_score < 600:
                explain["top_negative_features"].append(["credit_score", -0.10])

        return {
            "client_id": data.client_id,
            "score": round(proba, 4),
            "label": label,
            "explain_text": generate_plain_text(explain),
            "explain_details": explain,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}
