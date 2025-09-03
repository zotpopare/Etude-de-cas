import streamlit as st
import requests

# URL de ton API Render (sans doublon) et clé API
API_URL = "https://etude-de-cas.onrender.com"
API_KEY = "dev-secret-key"

st.title("NeoBanque Dashboard Conseiller")

# Inputs client
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
        # Requête POST vers ton endpoint /predict
        response = requests.post(
            f"{API_URL}/predict",
            headers={"x-api-key": API_KEY},
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("Résultat du scoring")
            st.success(f"Client {result['client_id']}: {result['label']} (score = {result['score']})")
            st.subheader("Explication")
            st.write(result["explain_text"])
        else:
            st.error(f"Erreur {response.status_code} : {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API : {e}")
