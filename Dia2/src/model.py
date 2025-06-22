from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from src.config import MODELS_DIR, MODEL_CONFIG

class WalmartPulseModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        self.sequence_length = MODEL_CONFIG['sequence_length']
    
    def build_model(self, input_shape):
        """Build the hybrid CNN-LSTM model"""
        inputs = keras.Input(shape=input_shape)
        
        # LSTM Branch
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64)(x)
        
        # CNN Branch
        y = layers.Conv1D(64, 3, activation='relu')(inputs)
        y = layers.MaxPooling1D(2)(y)
        y = layers.Flatten()(y)
        
        # Combine
        combined = layers.concatenate([x, y])
        
        # Dense layers
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.Dropout(0.3)(z)
        outputs = layers.Dense(1)(z)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='huber_loss',
            metrics=['mae']
        )
        
        return model
    
    def save(self):
        """Save model to models folder"""
        self.model.save(MODELS_DIR / MODEL_CONFIG['model_name'])
    
    @classmethod
    def load(cls, input_shape):
        """Load saved model"""
        model_path = MODELS_DIR / MODEL_CONFIG['model_name']
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        instance = cls(input_shape)
        instance.model = keras.models.load_model(model_path)
        return instance