import streamlit as st
import pickle
import numpy as np

# 1. Set up the page title and look
st.set_page_config(page_title="Customer Retention Hub", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Predictive Engine")
st.markdown("---")
st.write("Adjust the client metrics below to predict their likelihood of leaving the service provider.")

# 2. Load our trained Logistic Regression model
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# 3. Create Input Fields for columns in our exact training format
# Splitting inputs into two organized columns visually
col1, col2 = st.columns(2)

with col1:
    account_length = st.number_input("Account Length (Months)", min_value=1, max_value=300, value=100)
    intl_plan = st.selectbox("International Plan?", ["No", "Yes"])
    vmail_plan = st.selectbox("Voice Mail Plan?", ["No", "Yes"])
    vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, value=0)
    
    # Day metrics
    day_minutes = st.slider("Total Day Minutes", 0.0, 400.0, 180.0)
    day_calls = st.slider("Total Day Calls", 0, 200, 100)
    day_charge = st.slider("Total Day Charge ($)", 0.0, 60.0, 30.0)

with col2:
    # Evening metrics
    eve_minutes = st.slider("Total Evening Minutes", 0.0, 400.0, 200.0)
    eve_calls = st.slider("Total Evening Calls", 0, 200, 100)
    eve_charge = st.slider("Total Evening Charge ($)", 0.0, 40.0, 17.0)
    
    # Night metrics
    night_minutes = st.slider("Total Night Minutes", 0.0, 400.0, 200.0)
    night_calls = st.slider("Total Night Calls", 0, 200, 100)
    night_charge = st.slider("Total Night Charge ($)", 0.0, 20.0, 9.0)
    
    # International & Support metrics
    intl_minutes = st.slider("Total Intl Minutes", 0.0, 30.0, 10.0)
    intl_calls = st.slider("Total Intl Calls", 0, 20, 3)
    intl_charge = st.slider("Total Intl Charge ($)", 0.0, 10.0, 2.7)
    
    cust_service_calls = st.selectbox("Customer Service Calls", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 4. Map 'Yes'/'No' text selections to 1 and 0 for the model math
intl_encoded = 1 if intl_plan == "Yes" else 0
vmail_encoded = 1 if vmail_plan == "Yes" else 0

# 5. Pack everything neatly into an array matching our training features shape
input_data = np.array([[
    account_length, intl_encoded, vmail_encoded, vmail_messages,
    day_minutes, day_calls, day_charge,
    eve_minutes, eve_calls, eve_charge,
    night_minutes, night_calls, night_charge,
    intl_minutes, intl_calls, intl_charge,
    cust_service_calls
]])

st.markdown("---")

# 6. Make predictions when the user clicks the button
if st.button("Analyze Customer Risk Profile", type="primary"):
    prediction = model.predict(input_data)
    
    # Let's show the output clearly with nice alert colors
    if prediction[0] == 1:
        st.error("⚠️ **High Risk Indicator:** This customer matches a heavy profile for churn. Intervention recommended.")
    else:
        st.success("✅ **Stable Account Profile:** This customer is likely to stay loyal to the company.")