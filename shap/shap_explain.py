import joblib, shap, numpy as np
import matplotlib.pyplot as plt
import json

model = joblib.load("models/model.pkl")
X_sample = np.random.rand(1,7)  # exemple, adapte selon features

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, feature_names=[
    "age","income","credit_score","existing_loans",
    "loan_amount_requested","employment_salaried","housing_own"
])
