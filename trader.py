import joblib
import keras.models
import numpy as np
import requests
import json
import time

from sklearn.preprocessing import MinMaxScaler

scaler = joblib.load('cryptocurrency_scaler.pkl')

# Load the pre-trained model
model = keras.models.load_model('cryptocurrency_lstm_model.keras')


# Function to get real-time data
def get_realtime_data(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    data = json.loads(response.text)
    
    return float(data['price'])


def preprocess_data(data, seq_length, scaler):
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    x = []
    y = []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])

    return np.array(x), np.array(y)


def make_buy_decision(model, current_data, scaler, threshold):
    scaled_data = scaler.transform(np.array(current_data).reshape(-1, 1))
    prediction = model.predict(np.array([scaled_data]))
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    percentage_change = ((predicted_price - current_data[-1]) / current_data[-1]) * 100
    if percentage_change > threshold:
        return True, predicted_price
    else:
        return False, predicted_price


def trading():
    symbol = 'BTCUSDT'  # Symbol for Bitcoin-USDT pair
    threshold = 0.01  # 0.1% change threshold for buying
    seq_length = 20  # Length of input sequences

    previous_data = [get_realtime_data(symbol)]

    print("Initial Price:", previous_data[-1])

    while True:
        try:
            current_price = get_realtime_data(symbol)
            previous_data.append(current_price)

            if len(previous_data) > seq_length:
                current_data = previous_data[-seq_length:]
                buy_decision, predicted_price = make_buy_decision(model, current_data, scaler, threshold)
                print("Current Price:", current_price)
                print("Predicted Price:", predicted_price)

                if buy_decision:
                    print("Buy decision: Buy")
                else:
                    print("Buy decision: Do not buy")

            time.sleep(1)  # Adjust the interval based on your requirements
        except Exception as e:
            print("Error occurred:", e)
            time.sleep(5)  # Retry after 5 seconds in case of an error
