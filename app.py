import streamlit as st
import torch
import numpy as np
import joblib
import pandas as pd
from utils.counterfactuals import generate_counterfactual, summarize_changes
from utils.visualize import plot_changes
from utils.evaluation import evaluate_counterfactual

# --- Page Config ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Predictor with Counterfactual Explanations")

# --- Load Assets ---
model = torch.load("saved_files/churn_model.pth")
model.eval()
preprocessor = joblib.load("saved_files/preprocessor.joblib")
X_test = np.load("saved_files/X_test.npy")
y_test = np.load("saved_files/y_test.npy")

# Load feature names
numeric_features = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
    'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]
categorical_features = [
    'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
    'PreferedOrderCat', 'MaritalStatus'
]
cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_names])

# --- Input Form ---
st.sidebar.header("Input Customer Data")
with st.sidebar.form("churn_form"):
    user_input = {}
    for feature in numeric_features:
        user_input[feature] = st.number_input(feature, value=1.0)

    for feature in categorical_features:
        user_input[feature] = st.selectbox(feature, options=["Option1", "Option2", "Option3"])

    submitted = st.form_submit_button("Predict")

# --- Predict & Explain ---
if submitted:
    df_input = pd.DataFrame([user_input])
    x_transformed = preprocessor.transform(df_input)
    x_tensor = torch.tensor(x_transformed[0], dtype=torch.float32)
    pred_score = model(x_tensor).item()

    st.subheader(f"Prediction: {'Churn' if pred_score > 0.5 else 'No Churn'} (Score: {pred_score:.3f})")

    if pred_score > 0.5:
        x_cf, _ = generate_counterfactual(x_transformed[0], model, feature_names=feature_names)
        if x_cf is not None:
            st.markdown("### 🔄 Counterfactual Suggestions to Prevent Churn")
            changes = summarize_changes(x_transformed[0], x_cf, feature_names)
            for change in changes:
                st.markdown(f"- {change}")

            st.markdown("### 📊 Visual Explanation")
            plot_changes(x_transformed[0], x_cf, feature_names)

            st.markdown("### ✅ Evaluation Metrics")
            metrics = evaluate_counterfactual(x_transformed[0], x_cf, feature_names)
            st.json(metrics)
        else:
            st.warning("Could not generate counterfactual within allowed steps.")
    else:
        st.success("Customer is predicted to stay. No counterfactual needed.")

# --- Footer ---
st.markdown("---")
st.markdown("            © BAIBHAV KASHYAP                ")
