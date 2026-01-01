# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Customer Churn Prediction")

# Input fields for user to enter customer details
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=120, value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
contract = st.selectbox("Contract Type", ["One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Credit card (automatic)", 
                                                  "Bank transfer (automatic)", 
                                                  "Electronic check", 
                                                  "Mailed check"])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract_' + contract: [1],
    'PaymentMethod_' + payment_method: [1]
})

# Ensure all model features are present
model_features = model.feature_names_in_
for feature in model_features:
    if feature not in input_data.columns:
        input_data[feature] = 0
input_data = input_data[model_features]

# Prediction button
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    result = 'Yes' if prediction[0] == 1 else 'No'
    st.success(f"Churn Prediction: {result}")

