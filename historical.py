import joblib
from tensorflow.keras.models import load_model
import numpy as np
from neural_network import get_historical_data


# Function to preprocess data for prediction
def preprocess_for_prediction(data, seq_length, scaler):
    scaled_data = scaler.transform(data[['close']])
    x = []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:i+seq_length])

    return np.array(x)


# Function to make trading decision
def make_buy_decision(model, current_data, scaler):
    prediction = model.predict(np.array([current_data]))
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    return predicted_price


# Define parameters
symbol = 'BTCUSDT'  # Symbol for Bitcoin-USDT pair
interval = '1h'     # Interval for historical data (1 hour)
start_time = 1614556800000  # March 1st, 2021 in milliseconds
end_time = 1617148799000    # March 31st, 2021 in milliseconds
seq_length = 20  # Sequence length for LSTM

# Load the retrained model and scaler
model = load_model('cryptocurrency_lstm_model.keras')
scaler = joblib.load('cryptocurrency_scaler.pkl')

# Fetch historical cryptocurrency data
historical_data = get_historical_data(symbol, interval, start_time, end_time)

# Preprocess the historical data
preprocessed_data = preprocess_for_prediction(historical_data, seq_length, scaler)

# Evaluate the trading strategy on historical data
profit_loss = []
trades = []

for i in range(len(historical_data) - seq_length):
    current_data = historical_data.iloc[i:i+seq_length]
    predicted_price = make_buy_decision(model, preprocessed_data[i], scaler)
    current_price = historical_data.iloc[i + seq_length]['close']
    percentage_change = ((predicted_price - current_price) / current_price) * 100
    profit_loss.append(percentage_change)
    trades.append((historical_data.index[i + seq_length], predicted_price, current_price))

# Calculate overall profit/loss
total_profit_loss = sum(profit_loss)
print("Total Profit/Loss: {:.2f}%".format(total_profit_loss))

# Print trades
print("Trades:")
for trade in trades:
    print("Timestamp:", trade[0], "| Predicted Price:", trade[1], "| Actual Price:", trade[2])
