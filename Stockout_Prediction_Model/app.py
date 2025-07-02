import streamlit as st
import pandas as pd
import joblib

# Load models
classifier = joblib.load('stockout_classifier.pkl')
regressor = joblib.load('stockout_regressor.pkl')

st.title("🛒 Walmart Stockout Prediction System")
st.markdown("Predicts whether a stockout is likely and estimates the number of days until it happens.")

# Input form
with st.form("stockout_form"):
    inventory_level = st.number_input("Current Inventory Level", min_value=0)
    reorder_point = st.number_input("Reorder Point", min_value=0)
    reorder_quantity = st.number_input("Reorder Quantity", min_value=0)
    quantity_sold = st.number_input("Quantity Sold (last week)", min_value=0)
    forecasted_demand = st.number_input("Forecasted Demand", min_value=0)
    actual_demand = st.number_input("Actual Demand", min_value=0)
    supplier_lead_time = st.number_input("Supplier Lead Time (days)", min_value=0)
    promotion_applied = st.selectbox("Promotion Applied", ['Yes', 'No'])
    promotion_type = st.selectbox("Promotion Type", ['Discount', 'BOGO', 'None'])
    holiday_indicator = st.selectbox("Holiday Season?", ['Yes', 'No'])
    weather_conditions = st.selectbox("Weather Conditions", ['Clear', 'Rainy', 'Snowy', 'Stormy'])
    weekday = st.selectbox("Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    store_location = st.selectbox("Store Location", ['Urban', 'Suburban', 'Rural'])

    submitted = st.form_submit_button("Predict Stockout")

if submitted:
    input_data = pd.DataFrame([{
        'inventory_level': inventory_level,
        'reorder_point': reorder_point,
        'reorder_quantity': reorder_quantity,
        'quantity_sold': quantity_sold,
        'forecasted_demand': forecasted_demand,
        'actual_demand': actual_demand,
        'supplier_lead_time': supplier_lead_time,
        'promotion_applied': 1 if promotion_applied == 'Yes' else 0,
        'promotion_type': promotion_type,
        'holiday_indicator': 1 if holiday_indicator == 'Yes' else 0,
        'weather_conditions': weather_conditions,
        'weekday': weekday,
        'store_location': store_location
    }])

    # Predictions
    stockout_prediction = classifier.predict(input_data)[0]
    days_estimate = int(regressor.predict(input_data)[0])

    # Output results
    if stockout_prediction == 1:
        st.warning(f"⚠️ Stockout likely! Estimated in {days_estimate} day(s).")
    else:
        st.success("✅ Stock levels are sufficient. No refill needed.")
