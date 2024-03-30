import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import requests
import json


# Function to get historical cryptocurrency data
def get_historical_data(symbol, interval, start_time, end_time):
    url = f"https://api.binance.com/api/v1/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}"
    response = requests.get(url)
    data = json.loads(response.text)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)

    return df

# Function to preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']])
    return scaled_data, scaler


# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)


# Define parameters
symbol = 'BTCUSDT'  # Symbol for Bitcoin-USDT pair
interval = '1h'     # Interval for historical data (1 hour)
start_time = 1614556800000  # March 1st, 2021 in milliseconds
end_time = 1617148799000    # March 31st, 2021 in milliseconds
seq_length = 24     # Sequence length for LSTM
lookahead = 1       # Lookahead period for predicting the future

# Get historical data
data = get_historical_data(symbol, interval, start_time, end_time)

# Preprocess data
scaled_data, scaler = preprocess_data(data)

# Create sequences
x, y = create_sequences(scaled_data, seq_length + lookahead)

# Split data into train and test sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for LSTM
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate model
mse = model.evaluate(x_test, y_test)
print(f"Mean Squared Error: {mse}")

# Make predictions
predictions = model.predict(x_test)

# Inverse scaling for predictions
predictions = scaler.inverse_transform(predictions)

# Save model and scaler
model.save('cryptocurrency_lstm_model.keras')
joblib.dump(scaler, 'cryptocurrency_scaler.pkl')
