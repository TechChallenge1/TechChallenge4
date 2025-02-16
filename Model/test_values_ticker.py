import yfinance as yf

# Definir o ticker e período de coleta
ticker = 'PETR4.SA'
start_date = '2023-01-01'
end_date = '2024-07-20'

# Coleta dos dados históricos (apenas o fechamento)
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']]

# Pega os últimos 50 valores de fechamento e converte para uma lista simples
historical_prices = df['Close'].tail(50).values.flatten()

# Exibe os valores como uma lista
print(historical_prices.tolist())
