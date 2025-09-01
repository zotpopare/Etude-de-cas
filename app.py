import streamlit as st
import os
import requests
import pandas as pd


st.set_page_config(page_title="Dashboard Conseiller - NeoBanque", layout="wide")


st.title("NeoBanque — Dashboard Conseiller")


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
API_KEY = os.getenv("API_KEY", "dev-secret-key")


st.sidebar.header("Sélection Client")
client_id = st.sidebar.text_input("Client ID")


uploaded = st.sidebar.file_uploader("Ou téléverser CSV client", type=["csv"])


# For convenience, let user set attributes in sidebar (to send to API)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Revenu annuel", value=30000.0)
credit_score = st.sidebar.number_input("Credit score", value=600.0)
existing_loans = st.sidebar.number_input("Nb prêts", min_value=0, value=0)
loan_amount_requested = st.sidebar.number_input("Montant demandé", value=5000.0)
employment_status = st.sidebar.selectbox("Emploi", ["salaried", "self-employed", "unemployed"])
housing_status = st.sidebar.selectbox("Logement", ["own", "rent", "other"])


if uploaded is not None:
df = pd.read_csv(uploaded)
st.sidebar.write("Aperçu", df.head())
# Optionally allow selecting a row
if st.sidebar.button("Utiliser la 1ère ligne du CSV"):
row = df.iloc[0]
client_id = str(row.get('client_id', client_id))
age = int(row.get('age', age))
income = float(row.get('income', income))
credit_score = float(row.get('credit_score', credit_score))
existing_loans = int(row.get('existing_loans', existing_loans))
loan_amount_requested = float(row.get('loan_amount_requested', loan_amount_requested))
employment_status = row.get('employment_status', employment_status)
housing_status = row.get('housing_status', housing_status)


st.sidebar.markdown("---")


if st.sidebar.button("Calculer le score"):
if not client_id:
st.sidebar.error("Renseigne un client_id")
else:
payload = {
"client_id": client_id,
"age": int(age),
"income": float(income),
"credit_score": float(credit_score),
"existing_loans": int(existing_loans),
"loan_amount_requested": float(loan_amount_requested),
"employment_status": employment_status,
"housing_status": housing_status
}
try:
resp = requests.post(API_URL, json=payload, headers={"x-api-key": API_KEY}, timeout=20)
if resp.status_code == 200:
r = resp.json()
st.subheader(f"Score d'éligibilité : {r['score']*100:.1f}% — {r['label']}")
                st.info(r.get('explain_text', 'Aucune explication fournie'))
                with st.expander("Détails de l'explication"):
                    st.json(r.get('explain_details', {}))
            else:
                st.error(f"Erreur API: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Erreur réseau / timeout: {e}")

# Main area - placeholders pour graphiques
st.markdown("---")
st.write("Indicateurs descriptifs (exemples)")
st.write("Ici on pourra afficher : histogrammes des revenus, comparaison client vs population, etc.")
