import numpy as np
from pathlib import Path
from src.model import WalmartPulseModel
from src.config import MODEL_CONFIG

class WalmartPulsePredictor:
    def __init__(self, input_shape):
        self.model = WalmartPulseModel.load(input_shape)
        self.sequence_length = MODEL_CONFIG['sequence_length']
    
    def predict(self, latest_sequence):
        """Make prediction from input sequence"""
        if len(latest_sequence) != self.sequence_length:
            raise ValueError(f"Input sequence must be length {self.sequence_length}")
        
        # Add batch dimension
        input_data = np.expand_dims(latest_sequence, axis=0)
        return self.model.model.predict(input_data)[0][0]