# TECH CHALLENGE 4 - Previsão de Preços Futuros de Ações com FastAPI e LSTM

Este projeto tem como objetivo a previsão de preços futuros de ações utilizando uma rede neural LSTM (Long Short-Term Memory). A aplicação é desenvolvida com a API FastAPI, permitindo a interação com o modelo de previsão de forma simples e eficiente.

-------------------------------------------

Projeto Previsão de Preços Futuros de Ações
├── Main.py                # Script principal da API FastAPI que carrega o modelo e realiza previsões

├── Model/

│   ├── model.h5           # Modelo treinado no formato H5 (Modelo LSTM)

│   ├── scaler.pkl         # Scaler usado para normalizar os dados de entrada e reverter as previsões

│   ├── performance.log    # Log de monitoramento de performance do modelo

│   ├── train_model.py     # Script para treinamento do modelo LSTM utilizando dados históricos de ações

│   ├── avaliacao_modelo.py# Script para avaliação do modelo com métricas de desempenho (MAE, RMSE, MAPE)

│   └── test_values_ticker.py # Script para testar o modelo com valores de um ticker específico

├── requirements.txt       # Arquivo com as dependências do projeto

└── README.md              # Documentação do projeto


## Estrutura de Arquivos
- *Main.py*: Script principal da API FastAPI.
- *Model/*: Diretório que contém o modelo treinado, scaler, e os scripts de treinamento, avaliação e teste do modelo.
- *requirements.txt*: Arquivo com as dependências do projeto.
- *README.md*: Documentação do projeto.

-------------------------------------------

## Funcionalidades

- **Previsão de Preços**: O modelo LSTM realiza previsões de preços futuros com base em dados históricos de ações.
- **API FastAPI**: A aplicação expõe uma API com endpoints que permitem carregar o modelo, fazer previsões e acessar métricas de desempenho.
- **Avaliação de Modelo**: O desempenho do modelo é monitorado e pode ser avaliado com métricas como MAE (Mean Absolute Error), RMSE (Root Mean Squared Error) e MAPE (Mean Absolute Percentage Error).

## Arquitetura

- O projeto é composto por scripts Python que treinam o modelo, avaliam seu desempenho e expõem uma API FastAPI para consumo de previsões.
- O modelo LSTM é treinado utilizando dados históricos de preços de ações obtidos por meio da API `yfinance`.
- A API oferece endpoints para carregar o modelo, realizar previsões e acompanhar a performance do modelo em tempo real.

## Como Usar

1. Instale as dependências listadas no `requirements.txt`:
   pip install -r requirements.txt
   
3. Treine o modelo utilizando o script train_model.py.

4. Inicie a API FastAPI rodando o arquivo Main.py:
    uvicorn Main:app --reload
   
5. Acesse a API e utilize os endpoints disponíveis para realizar previsões e consultar métricas.


## Tecnologias Utilizadas
- FastAPI: Framework moderno e rápido para construir APIs.
- TensorFlow/Keras: Biblioteca utilizada para construir e treinar o modelo LSTM.
- yfinance: Biblioteca para obter dados financeiros históricos de ações.
- scikit-learn: Para o uso de técnicas de pré-processamento e avaliação do modelo.
- pandas, numpy: Para manipulação e análise de dados.

## Estrutura de Arquivos
- Main.py: Script principal da API FastAPI.
- Model/: Diretório que contém o modelo treinado, scaler, e os scripts de treinamento, avaliação e teste do modelo.
- requirements.txt: Arquivo com as dependências do projeto.
- README.md: Documentação do projeto.

----------------------------------------

# PROJETO:

## 1. main.py - API FastAPI para Previsão de Preços Futuros
Descrição:
Este arquivo contém a implementação da API em FastAPI que fornece endpoints para carregar um modelo de previsão de preços, fazer previsões de preços futuros, monitorar a performance da API e expor métricas de desempenho para o Prometheus.

### Funcionalidades:
- POST /load_model: Carrega o modelo e o scaler para fazer previsões.
- POST /predict: Faz previsões com base nos preços históricos fornecidos.
- GET /metrics: Exibe as métricas de performance da API em formato Prometheus.
- GET /: Endpoint de boas-vindas que descreve os endpoints disponíveis.

### Dependências:
- FastAPI: Framework para criação de APIs.
- Prometheus Client: Biblioteca para coleta e exposição de métricas.
- TensorFlow: Usado para carregar o modelo e realizar previsões.
- Joblib: Para carregar o scaler serializado.
- Pydantic: Para validação de dados de entrada.

## EXEMPLOS DE USO
### Carregar o modelo
POST /load_model

### Fazer previsões
POST /predict
{
    "historical_prices": [30.5, 31.0, 31.2, ..., 32.0],
    "n_steps": 3
}

### Ver métricas
GET /metrics

-------------------------------------------

## 2. train_model.py - Treinamento do Modelo LSTM para Previsão de Preços

### Descrição:
Este arquivo contém o script para treinar um modelo LSTM para previsão de preços com base em dados históricos. Ele coleta os dados de um ticker específico da bolsa de valores, normaliza os preços e treina um modelo de previsão.

### Funcionalidades:
- Coleta de dados históricos de preços usando a API yfinance.
- Normalização dos dados usando StandardScaler.
- Treinamento de um modelo LSTM para previsão de preços futuros.
- Salvamento do modelo treinado e do scaler para uso posterior.
- Visualização das previsões comparadas com os valores reais.

### Dependências:
- yfinance: Para coleta de dados financeiros.
- TensorFlow: Para construir e treinar o modelo LSTM.
- Scikit-learn: Para normalização dos dados.
- Matplotlib: Para visualização das previsões.

### Exemplos de uso:
#### Treinar e salvar o modelo
python train_model.py

-------------------------------------------

## 3. avaliacao_modelo.py - Avaliação do Modelo Treinado

### Descrição:
Este arquivo contém o script para avaliar o desempenho do modelo LSTM usando métricas como MAE, RMSE e MAPE. Ele também registra o tempo de resposta de cada previsão no arquivo de log.

### Funcionalidades:
- Carrega o modelo treinado e o scaler.
- Realiza previsões sobre dados de teste.
- Calcula métricas de erro (MAE, RMSE, MAPE).
- Registra o tempo de resposta das previsões no log para monitoramento de performance.

### Dependências:
- Scikit-learn: Para cálculo das métricas de avaliação.
- TensorFlow: Para carregar o modelo e fazer previsões.
- Yfinance: Para coleta de dados históricos para avaliação.
- Logging: Para registrar os tempos de resposta das previsões.

### Exemplos de uso:
#### Avaliar o modelo
python avaliacao_modelo.py

-------------------------------------------

## 4. test_values_ticker.py - Testes de Validação de Valores de Ticker

### Descrição:
Este arquivo é utilizado para realizar testes de integridade e valididade dos valores coletados para o ticker escolhido, como PETR4.SA. Ele verifica a consistência dos dados e a integridade da coleta.

### Funcionalidades:
- Coleta de dados históricos de um ticker especificado.
- Validação de dados para garantir que os valores sejam corretos e consistentes.

###Dependências:
- Yfinance: Para coleta de dados financeiros.

#### Testar valores para um ticker específico
python test_values_ticker.py

