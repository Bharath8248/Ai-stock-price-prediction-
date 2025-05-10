
# Ai-stock-price-prediction-
# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Step 1: Data Collection
data = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
data.reset_index(inplace=True)

# Step 2: Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.fillna(method='ffill', inplace=True)

# Normalize data for LSTM
scaler = MinMaxScaler()
data['Close_Scaled'] = scaler.fit_transform(data[['Close']])

# Step 3: EDA
plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.title("Historical Stock Prices")
plt.show()

# Step 4: Feature Engineering
data['Close_t-1'] = data['Close'].shift(1)
data['Rolling_Mean_10'] = data['Close'].rolling(window=10).mean()
data.dropna(inplace=True)

# Step 5: Model 1 - ARIMA
arima_model = ARIMA(data['Close'], order=(5,1,0))
arima_result = arima_model.fit()
data['ARIMA_Pred'] = arima_result.predict(start=100, end=len(data)-1, typ='levels')

# Step 6: Model 2 - Prophet
df_prophet = data.reset_index()[['Date', 'Close']]
df_prophet.columns = ['ds', 'y']
prophet_model = Prophet()
prophet_model.fit(df_prophet)
future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)
prophet_model.plot(forecast)

# Step 7: Model 3 - LSTM
def create_dataset(series, time_step=10):
    X, y = [], []
    for i in range(len(series)-time_step-1):
        X.append(series[i:(i+time_step), 0])
        y.append(series[i + time_step, 0])
    return np.array(X), np.array(y)

series = data[['Close_Scaled']].values
X, y = create_dataset(series)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

preds = model.predict(X)
data = data.iloc[-len(preds):]
data['LSTM_Pred'] = scaler.inverse_transform(preds)

# Step 8: Model Comparison
def evaluate(true, pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "MAE": mean_absolute_error(true, pred)
    }

arima_eval = evaluate(data['Close'], data['ARIMA_Pred'])
lstm_eval = evaluate(data['Close'], data['LSTM_Pred'])

print("ARIMA:", arima_eval)
print("LSTM:", lstm_eval)

# Prophet evaluation skipped due to different output format

