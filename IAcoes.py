import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from matplotlib.dates import DateFormatter


#----------------------------INÍCIO - Implementação Modelo LSTM----------------------------#
# Baixando dados históricos
today = pd.Timestamp.now()
tickers = ['PETR4.SA', 'CL=F']
data = yf.download(tickers, start=today - pd.DateOffset(years=5), end=today)['Close']
data = data.resample('M').last().dropna()  # Usando o último preço de cada mês

#----------------------------INÍCIO - Implementação da Decomposição de Séries Temporais (Ruídos)----------------------------#
# Escolhendo uma coluna para decomposição, e.g., 'PETR4.SA'
series = data['PETR4.SA']
# Decomposição de séries temporais
result = seasonal_decompose(series, model='additive', period=12)  # Supondo que 'M' representa mensal e um ciclo anual
# Plotando os componentes
decomposed = result.plot()
plt.show()
#----------------------------FIM - Implementação da Decomposição de Séries Temporais (Ruídos)----------------------------#

# Verificar os dados
print(data.head())  # Isso ajudará a confirmar que estamos acessando os dados corretamente

# Normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['PETR4.SA', 'CL=F']])

# Função para criar o conjunto de dados
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])  # Previsão do preço da PETR4
    return np.array(X), np.array(Y)

look_back = 3
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 2))  # Ajustando a quantidade de features

#----------------------------INÍCIO - Validação Cruzada em Séries Temporais----------------------------#
n_splits = 5  # Número de dobras
tscv = TimeSeriesSplit(n_splits=n_splits)
# Listas para armazenar os resultados de cada dobra
train_scores, test_scores = [], []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Construção do modelo LSTM
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Treinamento do modelo
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)
    # Avaliação do modelo
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_score = mean_squared_error(y_train, train_pred)
    test_score = mean_squared_error(y_test, test_pred)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"Train Score for fold {len(train_scores)}: {train_score}")
    print(f"Test Score for fold {len(test_scores)}: {test_score}")
# Média dos scores
print(f"Average Train Score: {np.mean(train_scores)}")
print(f"Average Test Score: {np.mean(test_scores)}")
#----------------------------FIM - Validação Cruzada em Séries Temporais----------------------------#

# Modelo LSTM
model = Sequential([
    InputLayer(input_shape=(look_back, 2)),  # Usando InputLayer para definir explicitamente a forma de entrada
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

# Previsões
predictions = model.predict(X)
predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], 1)))))

# Últimos dados de look_back meses usados para fazer a primeira previsão futura
last_samples = scaled_data[-look_back:]
last_samples = np.expand_dims(last_samples, axis=0)

# Gerando previsões futuras
num_months = 3  # Exemplo: prever os próximos 6 meses
future_predictions = []
current_input = last_samples

for _ in range(num_months):
    next_prediction = model.predict(current_input)
    future_predictions.append(next_prediction[0, 0])
    next_prediction = np.expand_dims(next_prediction, axis=0)
    next_prediction = np.repeat(next_prediction, 2, axis=2)  # Garantindo duas features
    current_input = np.append(current_input[:, 1:, :], next_prediction, axis=1)

future_predictions = np.array(future_predictions)
future_predictions = scaler.inverse_transform(np.column_stack((future_predictions, np.zeros_like(future_predictions))))

# Índices futuros para plotagem
future_dates = pd.date_range(data.index[-1], periods=num_months + 1, freq='M')[1:]
# Plotando os resultados
plt.figure(figsize=(12, 6))
plt.plot(data.index[look_back+1:], data['PETR4.SA'][look_back+1:], label='Original PETR4 Data')
plt.plot(data.index[look_back+1:], predictions[:, 0], label='Predicted PETR4 Price', linestyle='-')
plt.plot(future_dates, future_predictions[:, 0], label='Forecasted PETR4 Price', color='red')
plt.plot(data.index, data['CL=F'], label='Oil Price', color='green')
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()  # Auto-rotaciona as datas para melhor visualização
plt.title('PETR4 Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#----------------------------FIM - Implementação Modelo LSTM----------------------------#
