import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_preprocessor import DataPreprocessor
from src.model import WalmartPulseModel
from src.config import MODEL_CONFIG

class WalmartPulseTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model_config = MODEL_CONFIG
    
    def prepare_data(self):
        """Prepare training data"""
        df = self.preprocessor.preprocess_all()
        
        # Features and target
        features = ['day_of_week', 'month', 'is_weekend', 'is_holiday', 
                  'temperature', 'precipitation', 'bad_weather']
        target = 'sales_volume'
        
        # Create sequences
        X, y = self._create_sequences(
            df[features].values,
            df[target].values,
            self.model_config['sequence_length']
        )
        
        return train_test_split(
            X, y, 
            test_size=self.model_config['validation_split'],
            shuffle=False
        )
    
    def _create_sequences(self, data, targets, sequence_length):
        """Create time series sequences"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(targets[i+sequence_length])
        return np.array(X), np.array(y)
    
    def train(self):
        """Complete training pipeline"""
        X_train, X_val, y_train, y_val = self.prepare_data()
        
        model = WalmartPulseModel(
            input_shape=(self.model_config['sequence_length'], X_train.shape[2])
        )
        
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            verbose=1
        )
        
        model.save()
        return history