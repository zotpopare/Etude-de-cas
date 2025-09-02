import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"
API_KEY = "dev-secret-key"

st.title("NeoBanque Dashboard Conseiller")

client_id = st.text_input("Client ID")
age = st.number_input("Âge", 18, 100, 30)
income = st.number_input("Revenu", 0, 500000, 50000)
credit_score = st.number_input("Credit Score", 300, 850, 650)
existing_loans = st.number_input("Nombre de crédits", 0, 10, 1)
loan_amount_requested = st.number_input("Montant demandé", 1000, 500000, 10000)
employment_status = st.selectbox("Statut emploi", ["salaried", "unemployed", "self-employed"])
housing_status = st.selectbox("Logement", ["own", "rent"])

if st.button("Prédire"):
    data = {
        "client_id": client_id,
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_amount_requested": loan_amount_requested,
        "employment_status": employment_status,
        "housing_status": housing_status
    }
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={"x-api-key": API_KEY},
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"{result['label']} (score = {result['score']})")
            st.write(result["explain_text"])
        else:
            st.error(f"Erreur {response.status_code} : {response.text}")
    except Exception as e:
        st.error(f"Erreur API : {e}")
