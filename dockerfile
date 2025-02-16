# Usando uma imagem base com Python 3.9
FROM python:3.9-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos necessários para dentro do container
COPY . /app/

# Instalar as dependências do projeto
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expõe a porta que o FastAPI usará
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "5000"]
