# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('stockout_model.pkl')

# App title
st.title("🛒 Walmart Stockout Prediction Demo")
st.write("Enter product & inventory details to check if a stockout will happen.")

# Input fields
inventory_level = st.number_input("Inventory Level", min_value=0)
reorder_point = st.number_input("Reorder Point", min_value=0)
reorder_quantity = st.number_input("Reorder Quantity", min_value=0)
quantity_sold = st.number_input("Quantity Sold", min_value=0)
forecasted_demand = st.number_input("Forecasted Demand", min_value=0)
actual_demand = st.number_input("Actual Demand", min_value=0)
supplier_lead_time = st.number_input("Supplier Lead Time (days)", min_value=0)

promotion_applied = st.selectbox("Promotion Applied?", ['Yes', 'No'])
holiday_indicator = st.selectbox("Holiday?", ['Yes', 'No'])

promotion_type = st.selectbox("Promotion Type", ['None', 'Discount', 'BOGO', 'Clearance'])
weather_conditions = st.selectbox("Weather Conditions", ['Clear', 'Rain', 'Snow', 'Storm'])
weekday = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
store_location = st.selectbox("Store Location", ['North', 'South', 'East', 'West'])

# Add these just before prediction
sales_vs_reorder = quantity_sold - reorder_point
demand_to_stock_ratio = actual_demand / (inventory_level + 1)


# Predict button
if st.button("Predict Stockout"):
    input_data = pd.DataFrame([{
    'inventory_level': inventory_level,
    'reorder_point': reorder_point,
    'reorder_quantity': reorder_quantity,
    'quantity_sold': quantity_sold,
    'forecasted_demand': forecasted_demand,
    'actual_demand': actual_demand,
    'supplier_lead_time': supplier_lead_time,
    'promotion_applied': 1 if promotion_applied == 'Yes' else 0,
    'holiday_indicator': 1 if holiday_indicator == 'Yes' else 0,
    'promotion_type': promotion_type,
    'weather_conditions': weather_conditions,
    'weekday': weekday,
    'store_location': store_location,
    'stock_to_sales_ratio': inventory_level / (quantity_sold + 1),
    'demand_surge': actual_demand - forecasted_demand,
    'rolling_avg_sales': quantity_sold,  # Approximation
    'sales_vs_reorder': sales_vs_reorder,
    'demand_to_stock_ratio': demand_to_stock_ratio
}])


    prediction = model.predict(input_data)[0]

    if prediction:
        st.error("⚠️ Stockout is likely. Consider restocking!")
    else:
        st.success("✅ Stock level is sufficient. No stockout expected.")
