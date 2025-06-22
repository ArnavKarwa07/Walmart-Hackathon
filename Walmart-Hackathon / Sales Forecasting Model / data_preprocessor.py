import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import requests
from dotenv import load_dotenv
import os

load_dotenv()

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
    def load_sales_data(self, filepath):
        """Load historical sales data"""
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    
    def add_weather_data(self, df, location):
        """Enhance data with weather information"""
        dates = df['date'].dt.strftime('%Y-%m-%d').unique()
        
        weather_data = []
        for date in dates:
            try:
                response = requests.get(
                    f"http://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&dt={date}&appid={self.weather_api_key}"
                )
                weather = response.json()
                weather_data.append({
                    'date': date,
                    'temp': weather['main']['temp'],
                    'humidity': weather['main']['humidity'],
                    'weather_main': weather['weather'][0]['main']
                })
            except:
                print(f"[Warning] Weather data not found for {date}")
                weather_data.append({
                    'date': date,
                    'temp': np.nan,
                    'humidity': np.nan,
                    'weather_main': 'unknown'
                })
                
        weather_df = pd.DataFrame(weather_data)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        return pd.merge(df, weather_df, on='date', how='left')
    
    def add_time_features(self, df):
        """Add temporal features"""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df
    
    def add_holidays(self, df, country='US'):
        """Add holiday indicators"""
        df['is_holiday'] = 0  # Placeholder, can integrate with `holidays` library
        return df
    
    def preprocess_data(self, df):
        """Prepare final features"""
        
        # Drop columns or rows with all NaNs
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)

        # Safely get dummies for available categorical columns
        categorical_cols = [col for col in ['weather_main', 'category'] if col in df.columns]
        df = pd.get_dummies(df, columns=categorical_cols)

        # Normalize numerical features
        numerical_cols = [col for col in ['temp', 'humidity', 'historical_sales'] if col in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        # Fix deprecated fillna method
        df.ffill(inplace=True)

        return df
