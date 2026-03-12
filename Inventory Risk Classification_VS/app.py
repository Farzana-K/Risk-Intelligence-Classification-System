import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model & Mapping
# -----------------------------
with open("gradient_boosting_inventory_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tier_mapping.pkl", "rb") as f:
    tier_mapping = pickle.load(f)

reverse_mapping = {v: k for k, v in tier_mapping.items()}

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Inventory Risk Prediction", layout="wide")

st.title("📦 Inventory Risk Classification System")
st.write("Predict inventory risk tier using Machine Learning")

st.divider()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Inventory Details")

currentStock = st.sidebar.number_input("Current Stock", min_value=0.0, value=500.0)
averageDailyDemand = st.sidebar.number_input("Average Daily Demand", min_value=0.0, value=50.0)
leadTimeDays = st.sidebar.number_input("Lead Time (Days)", min_value=0.0, value=10.0)
demandVariance = st.sidebar.number_input("Demand Variance", min_value=0.0, value=5.0)
supplierRiskScore = st.sidebar.number_input("Supplier Risk Score", min_value=0.0, value=2.0)
safetyStock = st.sidebar.number_input("Safety Stock", min_value=0.0, value=100.0)
reorderPoint = st.sidebar.number_input("Reorder Point", min_value=0.0, value=300.0)
incomingStockDays = st.sidebar.number_input("Incoming Stock (Days)", min_value=0.0, value=5.0)
pendingOrderQty = st.sidebar.number_input("Pending Order Quantity", min_value=0.0, value=200.0)
isCriticalItem = st.sidebar.selectbox("Is Critical Item?", [0, 1])

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame([[
    currentStock,
    averageDailyDemand,
    leadTimeDays,
    demandVariance,
    supplierRiskScore,
    safetyStock,
    reorderPoint,
    incomingStockDays,
    pendingOrderQty,
    isCriticalItem
]], columns=[
    "currentStock",
    "averageDailyDemand",
    "leadTimeDays",
    "demandVariance",
    "supplierRiskScore",
    "safetyStock",
    "reorderPoint",
    "incomingStockDays",
    "pendingOrderQty",
    "isCriticalItem"
])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk Tier"):

    prediction = model.predict(input_data)[0]
    predicted_label = reverse_mapping[prediction]

    probability = model.predict_proba(input_data).max()

    st.subheader("Prediction Result")

    if predicted_label == "low":
        st.success(f"Predicted Risk Tier: {predicted_label.upper()}")

    elif predicted_label == "medium":
        st.warning(f"Predicted Risk Tier: {predicted_label.upper()}")

    elif predicted_label == "high":
        st.error(f"Predicted Risk Tier: {predicted_label.upper()}")

    else:
        st.error(f"Predicted Risk Tier: {predicted_label.upper()} 🚨")

    st.write(f"Confidence: {round(probability * 100, 2)}%")
