import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd

from utils.counterfactuals import generate_counterfactual
from utils.evaluation import summarize_changes, evaluate_counterfactual
from utils.visualize import plot_changes

# --- Page Config ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Predictor with Counterfactual Explanations")

# --- Define the Model Architecture ---
class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super(ChurnNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --- Load Assets ---
# Feature info
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

# Preprocessor & test data
preprocessor = joblib.load("saved_files/preprocessor.joblib")
X_test = np.load("saved_files/X_test.npy")
y_test = np.load("saved_files/y_test.npy")

# Extract feature names after preprocessing
cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_names])

# Load the model
input_dim = X_test.shape[1]
model = ChurnNet(input_dim)
model.load_state_dict(torch.load("saved_files/churn_model.pth", map_location=torch.device("cpu")))
model.eval()

# For evaluation
feature_mins = np.zeros(len(feature_names))
feature_maxs = np.ones(len(feature_names))
immutable_features = [
    'Gender_Female', 'Gender_Male',
    'MaritalStatus_Divorced', 'MaritalStatus_Single', 'MaritalStatus_Married',
    'CityTier'
]

# --- Input Form ---
st.sidebar.header("Input Customer Data")
with st.sidebar.form("churn_form"):
    user_input = {}
    for feature in numeric_features:
        user_input[feature] = st.number_input(feature, value=1.0)

    for feature in categorical_features:
        user_input[feature] = st.selectbox(
            feature,
            options=preprocessor.named_transformers_['cat'].categories_[categorical_features.index(feature)]
        )

    submitted = st.form_submit_button("Predict")

# --- Predict & Explain ---
if submitted:
    df_input = pd.DataFrame([user_input])
    x_transformed = preprocessor.transform(df_input)
    x_tensor = torch.tensor(x_transformed[0], dtype=torch.float32).unsqueeze(0)

    pred_score = model(x_tensor).item()
    st.subheader(f"Prediction: {'Churn' if pred_score > 0.5 else 'No Churn'} (Score: {pred_score:.3f})")

    if pred_score > 0.5:
        x_cf = generate_counterfactual(x_transformed[0], model, feature_names=feature_names)
        if x_cf is not None:
            st.markdown("### 🔄 Counterfactual Suggestions to Prevent Churn")
            changes = summarize_changes(x_transformed[0], x_cf, feature_names)
            for change in changes:
                st.markdown(f"- {change}")

            st.markdown("### 📊 Visual Explanation")
            plot_changes(x_transformed[0], x_cf, feature_names)

            st.markdown("### ✅ Evaluation Metrics")
            metrics = evaluate_counterfactual(
                x_transformed[0], x_cf, feature_names,
                feature_mins, feature_maxs, immutable_features
            )
            st.json(metrics)
        else:
            st.warning("Could not generate counterfactual within allowed steps.")
    else:
        st.success("Customer is predicted to stay. No counterfactual needed.")

# --- Footer ---
st.markdown("---")
st.markdown("© BAIBHAV KASHYAP")
