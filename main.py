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


# CORS - en dev allow *, en prod restreindre aux domaines connus (ex: ton Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement modèle et préproc
MODEL_PATH = os.path.join("..", "models", "model.pkl")
PREPROC_PATH = os.path.join("..", "models", "preproc.pkl")

# try alternative path when running from root
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("models", "model.pkl")
if not os.path.exists(PREPROC_PATH):
    PREPROC_PATH = os.path.join("models", "preproc.pkl")

model = None
preproc = None
explainer = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")

if os.path.exists(PREPROC_PATH):
    try:
        preproc = joblib.load(PREPROC_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du préprocesseur : {e}")

# init SHAP explainer if possible
if SHAP_AVAILABLE and model is not None:
    try:
        if hasattr(model, "predict_proba") and (
            "Tree" in str(type(model)) or "XGB" in str(type(model))
        ):
            explainer = shap.TreeExplainer(model)
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
    features = [
        data.age,
        data.income,
        data.credit_score,
        data.existing_loans,
        data.loan_amount_requested,
        1 if data.employment_status.lower() == "salaried" else 0,
        1 if data.housing_status.lower() == "own" else 0,
    ]
    X = np.array(features).reshape(1, -1)
    if preproc is not None:
        try:
            X = preproc.transform(X)
        except Exception:
            pass
    return X

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
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    try:
        X = preprocess_input(data)
        proba = float(model.predict_proba(X)[0][1])
        label = "Eligible" if proba >= 0.5 else "Non-eligible"

        explain = {"top_positive_features": [], "top_negative_features": []}
        if SHAP_AVAILABLE and explainer is not None:
            try:
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]
                feature_names = [
                    "age", "income", "credit_score",
                    "existing_loans", "loan_amount_requested",
                    "employment_salaried", "housing_own"
                ]
                contributions = list(zip(feature_names, sv.tolist()))
                contributions_sorted = sorted(contributions, key=lambda x: x[1], reverse=True)
                top_pos = [(f, float(v)) for f, v in contributions_sorted if v > 0][:5]
                top_neg = [(f, float(v)) for f, v in contributions_sorted if v < 0][:5]
                explain["top_positive_features"] = [[f, round(v, 4)] for f, v in top_pos]
                explain["top_negative_features"] = [[f, round(v, 4)] for f, v in top_neg]
            except Exception:
                pass
        else:
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
