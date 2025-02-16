from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Summary, generate_latest, REGISTRY
from prometheus_client.exposition import basic_auth_handler
import time
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import traceback

app = FastAPI()

# Variáveis globais para o modelo e o scaler
model = None
scaler = None

# Definindo os contadores e sumários para monitoramento
REQUEST_COUNT = Counter('request_count', 'Total de requisições feitas', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Summary('request_latency_seconds', 'Tempo de resposta das requisições', ['method', 'endpoint'])

# Tamanho da sequência utilizado no treinamento
SEQ_LENGTH = 50

# Definição do modelo de entrada
class PredictionRequest(BaseModel):
    historical_prices: list[float]
    n_steps: int = 1  # Default: 1

@app.post("/load_model")
def load_model_endpoint():
    """
    Endpoint para carregar o modelo e o scaler.
    Deve ser chamado uma vez antes de usar o endpoint de previsão.
    """
    global model, scaler

    try:
        # Carregando o modelo do arquivo .h5
        model = load_model('Model/model.h5')
        print("Modelo carregado com sucesso.")

        # Carregando o scaler do arquivo .pkl
        with open('Model/scaler.pkl', 'rb') as f:
            scaler = joblib.load(f)
        
        return {"message": "Modelo e scaler carregados com sucesso."}
    
    except Exception as e:
        print(f"Erro ao carregar o modelo ou scaler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro ao carregar o modelo ou scaler.")

@app.post("/predict")
def predict(data: PredictionRequest):
    """
    Recebe um JSON com os dados históricos de preços e retorna as previsões.
    Exemplo de payload:
    {
        "historical_prices": [30.5, 31.0, 31.2, ..., 32.0],
        "n_steps": 3   # opcional, default 1
    }
    """
    try:
        # Verifica se o modelo e o scaler foram carregados
        if model is None or scaler is None:
            raise HTTPException(status_code=400, detail="Modelo não carregado. Chame o endpoint /load_model primeiro.")

        historical_prices = data.historical_prices

        if len(historical_prices) < SEQ_LENGTH:
            raise HTTPException(status_code=400, detail=f"É necessário pelo menos {SEQ_LENGTH} valores históricos.")

        # Número de passos a prever
        n_steps = data.n_steps

        # Converte para array numpy e reescala os valores
        historical_prices = np.array(historical_prices).reshape(-1, 1)
        historical_prices_scaled = scaler.transform(historical_prices)

        # Seleciona os últimos SEQ_LENGTH valores para formar a sequência de entrada
        input_sequence = historical_prices_scaled[-SEQ_LENGTH:]

        predictions = []
        current_sequence = input_sequence.copy()

        # Previsão recursiva para n_steps passos à frente
        for _ in range(n_steps):
            # O modelo espera entrada com shape (1, SEQ_LENGTH, 1)
            input_data = current_sequence.reshape(1, SEQ_LENGTH, 1)
            pred_scaled = model.predict(input_data)
            # Converte o valor previsto para a escala original
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(float(pred_price))  # Convertendo para float

            # Atualiza a sequência: remove o primeiro valor e acrescenta o valor previsto (em escala)
            pred_scaled_value = pred_scaled[0][0]
            pred_scaled_value = np.array([[pred_scaled_value]])
            current_sequence = np.concatenate([current_sequence[1:], pred_scaled_value], axis=0)

        return {"predictions": predictions}

    except Exception as e:
        print(f"Erro durante a previsão: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro durante a previsão.")

# Middleware para medir o tempo de resposta
@app.middleware("http")
async def monitoramento_resposta(request: Request, call_next):
    start_time = time.time()
    
    # A requisição é processada
    response = await call_next(request)
    
    # Medindo o tempo de resposta
    process_time = time.time() - start_time
    
    # Registrando as métricas
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(process_time)
    
    return response

# Expondo as métricas para o Prometheus
#@app.get("/metrics")
#def metrics():
#    return generate_latest(REGISTRY)
@app.get("/metrics")
def metrics():
    try:
        # Coletando as métricas brutas em formato Prometheus
        prometheus_metrics = generate_latest(REGISTRY).decode('utf-8')
        
        # Processando e estruturando as métricas para exibição
        metrics_list = prometheus_metrics.split('\n')
        formatted_metrics = ""
        
        for metric in metrics_list:
            if metric.startswith('#'):
                # Comentários ou descrições (HELP, TYPE)
                formatted_metrics += f"<p><em>{metric}</em></p>"
            elif metric.strip():
                # Dados de métrica (não vazios)
                formatted_metrics += f"<pre>{metric}</pre>"

        # Criando HTML simples para exibir as métricas de uma forma mais bonita
        html_content = f"""
        <html>
            <head>
                <title>Monitoramento da API</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        color: #333;
                    }}
                    h1 {{
                        color: #0066CC;
                    }}
                    pre {{
                        background-color: #f4f4f4;
                        padding: 10px;
                        border-radius: 5px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }}
                    em {{
                        color: #888;
                    }}
                </style>
            </head>
            <body>
                <h1>Dados de Monitoramento da API</h1>
                <div>
                    {formatted_metrics}
                </div>
            </body>
        </html>
        """

        return HTMLResponse(content=html_content)
    
    except Exception as e:
        print(f"Erro ao formatar as métricas: {e}")
        raise HTTPException(status_code=500, detail="Erro ao obter métricas.")

@app.get("/")
def index():
    return {"message": "API para previsão de preços futuros. Use o endpoint POST /load_model para carregar o modelo e /predict para fazer previsões e /metrics para trazer o monitoramento de perfomance da API"}

# Para rodar a API com Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
