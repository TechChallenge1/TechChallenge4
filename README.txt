# Utiliza uma imagem oficial do Python como base
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de requisitos
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta em que a API irá rodar
EXPOSE 5000

# Define o comando para iniciar a API
CMD ["python", "app.py"]

-------------------------------------------------------
------------------ Instruções de Uso ------------------
-------------------------------------------------------

Treinamento do Modelo
Certifique-se de que todas as dependências estão instaladas (pode ser via ambiente virtual ou instalando com pip).

Execute o script de treinamento:

bash
Copiar
python train_model.py
Isso gerará os arquivos model.pkl e scaler.pkl.

Testando a API Localmente
Execute a API:

bash
Copiar
python app.py
Para testar a previsão, envie uma requisição POST para http://localhost:5000/predict com um payload JSON, por exemplo:

json
Copiar
{
    "historical_prices": [30.5, 30.7, 30.9, ..., 32.1],
    "n_steps": 3
}
Obs.: Certifique-se de fornecer ao menos 50 valores históricos, conforme definido pela constante SEQ_LENGTH.

Rodando a API em um Container Docker
Construa a imagem Docker:

bash
Copiar
docker build -t flask-api-model .
Execute o container:

bash
Copiar
docker run -p 5000:5000 flask-api-model
Agora, a API estará acessível em http://localhost:5000.

Com essa estrutura, você tem:

Treinamento separado: O script train_model.py gera os artefatos do modelo.
API em Flask: O script app.py carrega o modelo e responde as previsões conforme os dados históricos informados.
Container Docker: O Dockerfile empacota a API para facilitar o deploy.