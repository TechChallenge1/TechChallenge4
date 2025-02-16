import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle

# Definir o ticker e período de coleta
ticker = 'PETR4.SA'
start_date = '2018-01-01'
end_date = '2024-07-20'

# Coleta dos dados históricos (apenas o fechamento)
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']]

# Normaliza os dados utilizando MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

# Salva o scaler para uso na inferência
with open('scaler.pkl', 'wb') as f:
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
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Iniciando o treinamento do modelo...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
print("Treinamento concluído.")

# Cria um wrapper para que o modelo Keras seja pickleável
class KerasModelWrapper:
    def __init__(self, model):  # Corrigido: __init__ em vez de _init_
        self.model = model

    def __getstate__(self):  # Corrigido: __getstate__ em vez de _getstate_
        # Salva a arquitetura e os pesos do modelo
        model_json = self.model.to_json()
        model_weights = self.model.get_weights()
        return {'model_json': model_json, 'model_weights': model_weights}

    def __setstate__(self, state):  # Corrigido: __setstate__ em vez de _setstate_
        from tensorflow.keras.models import model_from_json
        self.model = model_from_json(state['model_json'])
        self.model.set_weights(state['model_weights'])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def predict(self, x):
        return self.model.predict(x)

# Instancia o wrapper e salva-o via pickle
model_wrapper = KerasModelWrapper(model)
with open('model.pkl', 'wb') as f:
    pickle.dump(model_wrapper, f)

print("Arquivos model.pkl e scaler.pkl salvos com sucesso.")