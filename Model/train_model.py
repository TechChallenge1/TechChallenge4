import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import os  # Para trabalhar com diretórios

# Definir o ticker e período de coleta
ticker = 'PETR4.SA'
start_date = '2023-01-01'
end_date = '2025-01-31'

# Coleta dos dados históricos (apenas o fechamento)
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']]

# Normaliza os dados utilizando MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#data_scaled = scaler.fit_transform(df)
scaler = StandardScaler()  # Alternativa ao MinMaxScaler
data_scaled = scaler.fit_transform(df)


# Criar pasta Model, caso não exista
os.makedirs('Model', exist_ok=True)

# Salva o scaler para uso na inferência dentro da pasta Model
with open('Model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Função para criar sequências para o treinamento
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50  # Tamanho da sequência
X, y = create_sequences(data_scaled, seq_length)

# Dividir os dados em treino e teste (80% treino)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Construção do modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.25))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.25))
#model.add(Dense(25))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Iniciando o treinamento do modelo...")
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
print("Treinamento concluído.")

# Salva o modelo no formato .h5 dentro da pasta Model
model.save('Model/model.h5')
print("Modelo salvo como Model/model.h5.")

# Salva o scaler como pickle para uso futuro dentro da pasta Model
with open('Model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Arquivos model.h5 e scaler.pkl salvos com sucesso.")


# Previsões do modelo
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Convertendo y_test para escala original
y_test_real = scaler.inverse_transform(y_test)

# Visualização dos resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label='Real')
plt.plot(y_pred, label='Previsto', linestyle='dashed')
plt.legend()
plt.title('Comparação entre valores reais e previstos')
plt.show()

