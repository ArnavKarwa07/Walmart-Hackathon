import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import holidays
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.holiday_calendar = holidays.US()
    
    def load_raw_data(self):
        """Load all raw data files"""
        sales = pd.read_csv(RAW_DATA_DIR / 'sales_data.csv')
        weather = pd.read_csv(RAW_DATA_DIR / 'weather_data.csv')
        events = pd.read_csv(RAW_DATA_DIR / 'event_data.csv')
        return sales, weather, events
    
    def preprocess_all(self):
        """Run complete preprocessing pipeline"""
        sales, weather, events = self.load_raw_data()
        df = self._merge_data(sales, weather, events)
        df = self._add_features(df)
        df = self._normalize(df)
        self._save_processed(df)
        return df
    
    def _merge_data(self, sales, weather, events):
        """Merge all data sources"""
        weather['date'] = pd.to_datetime(weather['date'])
        events['date'] = pd.to_datetime(events['date'])
        
        df = pd.merge(sales, weather, on='date', how='left')
        df = pd.merge(df, events, on='date', how='left')
        return df
    
    def _add_features(self, df):
        """Add engineered features"""
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['date'].dt.weekday >= 5).astype(int)
        df['is_holiday'] = df['date'].apply(lambda x: x in self.holiday_calendar).astype(int)
        df['bad_weather'] = ((df['precipitation'] > 0.5) | (df['temperature'] < 32)).astype(int)
        return df
    
    def _normalize(self, df):
        """Normalize features"""
        numeric_cols = ['temperature', 'precipitation', 'event_attendance']
        for col in numeric_cols:
            self.scalers[col] = MinMaxScaler()
            df[col] = self.scalers[col].fit_transform(df[[col]])
        return df
    
    def _save_processed(self, df):
        """Save processed data"""
        df.to_csv(PROCESSED_DATA_DIR / 'processed_data.csv', index=False)