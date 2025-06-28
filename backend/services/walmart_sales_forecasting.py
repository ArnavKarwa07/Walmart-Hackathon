import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load and preprocess data
data = pd.read_csv('Walmart.csv', parse_dates=['Date'], dayfirst=True)
data = data.sort_values('Date')  # Ensure dates are in order

# Aggregate sales across all stores
sales_by_date = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
sales_by_date = sales_by_date.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

# Split data into train (80%) and test (20%)
train_size = int(len(sales_by_date) * 0.8)
train, test = sales_by_date[:train_size], sales_by_date[train_size:]

# Plot raw sales data
plt.figure(figsize=(12, 6))
plt.plot(train['ds'], train['y'], label='Train')
plt.plot(test['ds'], test['y'], label='Test', color='orange')
plt.title('Walmart Weekly Sales (Train-Test Split)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# --- SARIMA Model ---
sarima_model = SARIMAX(
    train['y'],
    order=(1, 1, 1),              # Non-seasonal parameters (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # Seasonal parameters (P, D, Q, seasonal period)
)
sarima_results = sarima_model.fit(disp=False)
sarima_forecast = sarima_results.get_forecast(steps=len(test)).predicted_mean

# --- Prophet Model ---
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
prophet_model.fit(train)
future = prophet_model.make_future_dataframe(periods=len(test))
prophet_forecast = prophet_model.predict(future)['yhat'][-len(test):]

# --- Evaluate Forecasts ---
def evaluate_forecast(test, forecast, model_name):
    mae = mean_absolute_error(test['y'], forecast)
    rmse = np.sqrt(mean_squared_error(test['y'], forecast))
    mape = np.mean(np.abs((test['y'] - forecast) / test['y'])) * 100  # MAPE (%)
    r2 = r2_score(test['y'], forecast)
    
    print(f'\n{model_name} Performance:')
    print(f'  MAE: ${mae:,.2f}')
    print(f'  RMSE: ${rmse:,.2f}')
    print(f'  MAPE: {mape:.2f}%')
    print(f'  R²: {r2:.4f}')

# Evaluate both models
evaluate_forecast(test, sarima_forecast, 'SARIMA')
evaluate_forecast(test, prophet_forecast, 'Prophet')

# --- Plot Forecasts vs Actuals ---
plt.figure(figsize=(12, 6))
plt.plot(test['ds'], test['y'], label='Actual Sales', color='black', linewidth=2)
plt.plot(test['ds'], sarima_forecast, label='SARIMA Forecast', linestyle='--', color='red')
plt.plot(test['ds'], prophet_forecast, label='Prophet Forecast', linestyle='--', color='blue')
plt.title('Walmart Sales Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.show()