


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

class LSTMForecaster:
    def __init__(self, time_steps=7, n_features=10):
        self.time_steps = time_steps
        self.n_features = n_features
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.time_steps, self.n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(len(data) - self.time_steps + 1):
            X.append(data[i:i+self.time_steps])
            y.append(target[i + self.time_steps - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        return history
    
    def predict(self, X):
        return self.model.predict(X)

