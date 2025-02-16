import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Configuração
SYMBOL = 'PSSA3'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'
LOOKBACK = 60  # Número de dias para prever
MODEL_PATH = 'Utils/lstm_model.h5'
SCALER_PATH = 'Utils/scaler.pkl'

# Criando diretório para Utils
os.makedirs('Utils', exist_ok=True)

# Coleta de dados
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    joblib.dump(scaler, SCALER_PATH)
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

X, y = preprocess_data(df)

# Separação dos dados
test_size = int(len(X) * 0.2)
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# Construção do modelo
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Salvando o modelo
model.save(MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")
