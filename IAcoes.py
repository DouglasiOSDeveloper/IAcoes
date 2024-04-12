import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import warnings
import numpy as np

# Definir o ticker da ação
ticker = 'AAPL'

# Obter os dados
data = yf.download(ticker)

# Selecionar dados a partir de 2020
data_2020 = data['2020-01-01':]

# Ajustar o modelo ARIMA(2, 1, 2) aos seus dados
model = ARIMA(data_2020['Close'], order=(2, 1, 2))
results = model.fit()

# Prever os próximos 30 dias com base nos valores ajustados
forecast_values = results.forecast(steps=30)

# Criar datas para os próximos 30 dias
last_date = data_2020.index[-1]
next_dates = pd.date_range(start=last_date, periods=31)  # Ignorar o último dia dos dados históricos

# Concatenar as séries de valores ajustados e previsões
extended_values = pd.concat([results.fittedvalues, forecast_values])

# Plotar os dados originais e as previsões
plt.figure(figsize=(12, 6))
plt.plot(data_2020.index, data_2020['Close'], label='Dados Históricos')
plt.plot(extended_values.index, extended_values, color='red', label='Dados Ajustados e Previsões')
plt.title('Previsões Futuras com Modelo ARIMA para os Próximos 30 Dias')
plt.xlabel('Data')
plt.ylabel('Close Price')
plt.legend()
plt.show()



# #Definir o ticker da ação e o intervalo de datas
# ticker = 'AAPL'
# #Obter os dados
# data = yf.download(ticker)
# # Selecionar dados a partir de 2020
# data_2020 = data['2020-01-01':]

# # Ajustar o modelo ARIMA(2, 1, 2) aos seus dados
# model = ARIMA(data_2020['Close'], order=(2, 1, 2))
# results = model.fit()
# # Prever os próximos 30 dias
# forecast_values = results.forecast(steps=30)

# # Criar datas para os próximos 30 dias
# last_date = data_2020.index[-1]
# next_dates = pd.date_range(start=last_date, periods=31)[1:]  # Ignorar o último dia dos dados históricos

# # Plotar os dados originais e as previsões
# plt.figure(figsize=(12, 6))
# plt.plot(data_2020.index, data_2020['Close'], label='Dados Históricos')
# plt.plot(next_dates, forecast_values, color='green', label='Previsões')
# plt.plot(results.fittedvalues, color='red', label='ARIMA Values')
# plt.title('Previsões Futuras com Modelo ARIMA para os Próximos 30 Dias')
# plt.xlabel('Data')
# plt.ylabel('Close Price')
# plt.legend()
# plt.show()


#------------------------------------------------#

# # Desativar os avisos (warnings) para manter a saída limpa
# warnings.filterwarnings("ignore")

# # Definir os valores possíveis para p, d e q
# p_values = range(4)
# d_values = [1]  # Já diferenciamos os dados uma vez
# q_values = range(4)

# # Criar uma lista de todas as combinações de p, d e q
# pdq = list(itertools.product(p_values, d_values, q_values))

# # Ajustar o modelo ARIMA para cada combinação de parâmetros
# best_aic = np.inf
# best_params = None

# for param in pdq:
#     try:
#         # Ajustar o modelo ARIMA com os parâmetros dados
#         model = ARIMA(data_2020['Close'], order=param)
#         results = model.fit()

#         # Calcular o AIC do modelo ajustado
#         aic = results.aic

#         # Atualizar o melhor modelo se o AIC for menor
#         if aic < best_aic:
#             best_aic = aic
#             best_params = param

#         print(f'ARIMA{param} - AIC: {aic}')
#     except:
#         continue

# print(f'Best ARIMA Model: ARIMA{best_params} - AIC: {best_aic}')



#------------------------------------------------#

# # Diferenciar os dados para torná-los estacionários
# diff_data = data_2020['Close'].diff().dropna()

# # Plotar os dados diferenciados
# plt.figure(figsize=(12, 6))
# plt.plot(diff_data)
# plt.title('Differenced Close Prices')
# plt.xlabel('Date')
# plt.ylabel('Price Difference')
# plt.show()

# # Verificar a autocorrelação dos dados diferenciados
# plot_acf(diff_data, lags=20)
# plt.title('Autocorrelation Function (ACF) of Differenced Close Prices')
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.show()

# plot_pacf(diff_data, lags=20)
# plt.title('Partial Autocorrelation Function (PACF) of Differenced Close Prices')
# plt.xlabel('Lag')
# plt.ylabel('Partial Autocorrelation')
# plt.show()



#------------------------------------------------#

# def calculate_sma(data, window=14):
#     # Calcular a Média Móvel Simples (SMA)
#     sma = data['Close'].rolling(window=window).mean()
#     return sma

# def calculate_rsi(data, window=14):
#     # Calcular as mudanças diárias nos preços de fechamento
#     delta = data['Close'].diff()

#     # Separar as mudanças positivas e negativas
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

#     # Calcular o RSI
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))

#     return rsi

# # Definir o valor da janela
# window = 14

# # Calcular a SMA para os dados históricos da ação da Apple
# sma = calculate_sma(data.loc['2024-02-17':'2024-04-01'], window=window)

# # Calcular o RSI para os dados históricos da ação da Apple
# rsi = calculate_rsi(data.loc['2024-02-17':'2024-04-01'])

# # Plotar os preços de fechamento, SMA e RSI
# plt.figure(figsize=(12, 6))
# # Plotar os preços de fechamento
# plt.plot(data.loc['2024-02-17':'2024-04-01'].index, data.loc['2024-02-17':'2024-04-01']['Close'], label='Close Price', color='blue')
# # Plotar o SMA
# plt.plot(sma.index, sma, label=f'{window}-Day SMA', color='red')
# # Plotar o RSI
# plt.plot(rsi.index, rsi, label=f'{window}-Day RSI', color='green')
# # Ajustar a escala dos eixos y para reduzir a distância entre as linhas
# plt.ylim(data['Close'].min(), data['Close'].max())

# plt.title('Close Price, SMA and RSI of AAPL')
# plt.xlabel('Date')
# plt.ylabel('Price / RSI')
# plt.legend()
# plt.show()


