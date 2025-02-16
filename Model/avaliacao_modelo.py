import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import logging

# Carregar modelo e scaler
model = tf.keras.models.load_model('Model/model.h5')

with open('Model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Definir o ticker e período de validação
import yfinance as yf

ticker = 'PETR4.SA'
start_date = '2023-01-01'
end_date = '2025-01-31'

# Coleta dos dados históricos (apenas o fechamento)
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']]

# Normalizar os dados
data_scaled = scaler.transform(df)

# Criar sequências de entrada
SEQ_LENGTH = 50

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(data_scaled, SEQ_LENGTH)

# Fazer previsões
y_pred_scaled = model.predict(X_test)

# Reverter a escala das previsões e dos valores reais
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Cálculo das métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")


# CRIANDO MONITORAMENTO DO MODELO, CRIANDO ARQUIVO DE LOG PARA ARMAZENAR DADOS DE PERFORMANCE
# Configurar logging
logging.basicConfig(filename='Model/performance.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def monitor_prediction(model, X_test):
    start_time = time.time()
    prediction = model.predict(X_test)
    end_time = time.time()
    
    response_time = end_time - start_time
    logging.info(f"Tempo de resposta: {response_time:.4f} segundos")

    return prediction
