from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
# hyper param in just grid search not bayseian for now :)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import tensorflow as tfx

def fetch_stock_data(ticker, start_date, end_date):
    # Litterally just a shorthand but almost pointless
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def prepare_lstm_data(data, target_column, look_back=30):
# Just grab some data and chuck them in X and y 
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def AutoML2(stock_ticker):
    # Fetch stock data
    ticker = stock_ticker
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']].values)

    look_back = 30  # Number of days to look back

    # Prepare training and testing data
    train_end_index = len(scaled_data) - look_back - 30  # Training data excludes the last 30 days
    X_train, y_train = prepare_lstm_data(scaled_data[:train_end_index], target_column='Close', look_back=look_back)
    X_test, y_test = prepare_lstm_data(scaled_data[train_end_index:], target_column='Close', look_back=look_back)

    # Reshape for LSTM
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train Both Models
    lstm_model, lstm_predictions = train_lstm(X_train_lstm, y_train, X_test_lstm, y_test, units=50, dropout_rate=0.2, epochs=20)
    xgb_model, xgb_predictions = train_xgboost(X_train, y_train, X_test, y_test)

    # Inverse transform predictions
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    xgb_predictions = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE for both models
    lstm_rmse = np.sqrt(mean_squared_error(y_test_unscaled, lstm_predictions))
    xgb_rmse = np.sqrt(mean_squared_error(y_test_unscaled, xgb_predictions))

    # Determine the best model
    if lstm_rmse < xgb_rmse:
        print(f"LSTM performed better with RMSE: {lstm_rmse}")
        best_model = "LSTM"
        best_predictions = lstm_predictions
        model = lstm_model
    else:
        print(f"XGBoost performed better with RMSE: {xgb_rmse}")
        best_model = "XGBoost"
        best_predictions = xgb_predictions
        model = xgb_model

    # Train the best model on the entire dataset
    X, y = prepare_lstm_data(scaled_data, target_column='Close', look_back=look_back)
    if best_model == "LSTM":
        X_train_full = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM
        model.fit(X_train_full, y, epochs=20, verbose=1)
    else:
        model.fit(X, y)

    # Predict the next 30 days
    predictions_future = []
    last_sequence = X[-1]  # Use the last sequence from the training set

    for _ in range(30):
        if best_model == "LSTM":
            next_input = last_sequence.reshape(1, look_back, 1)
            next_pred = model.predict(next_input)
        else:
            next_input = last_sequence.reshape(1, -1)
            next_pred = model.predict(next_input)

        predictions_future.append(next_pred[0, 0] if best_model == "LSTM" else next_pred[0])

        # Update the sequence
        last_sequence = np.append(last_sequence[1:], next_pred).reshape(look_back, 1)

    predictions_future = scaler.inverse_transform(np.array(predictions_future).reshape(-1, 1)).flatten()

    # Plot Results
    plt.figure(figsize=(14, 8))

    # Historical data
    plt.plot(range(len(scaled_data)), scaler.inverse_transform(scaled_data), label="Historical Prices", color='black')

    # Training region
    plt.axvspan(0, train_end_index, color='lightblue', alpha=0.2, label='Training Data')

    # Testing region
    plt.axvspan(train_end_index, len(scaled_data), color='lightgreen', alpha=0.2, label='Testing Data')

    # Future predictions
    plt.plot(range(len(scaled_data), len(scaled_data) + 30), predictions_future, label=f"{best_model} Predictions (Future)", color='orange', linestyle='dashed')

    # Add labels, title, and legend
    plt.title("Historical and Future Predictions with Best Model")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()

    return predictions_future

def classify_prediction_outcome(predictions, time_range="day", threshold=0.005):

    # Day, Week Month Classification

    # Define time range mapping
    time_range_map = {
        "day": 1,
        "week": 7,
        "month": 30
    }

    # Ensure time_range is valid
    if time_range not in time_range_map:
        raise ValueError(f"Invalid time_range. Choose from {list(time_range_map.keys())}.")

    # Determine the time step for the specified range
    steps = time_range_map[time_range]
    if steps > len(predictions):
        raise ValueError(f"Time range ({time_range}) exceeds prediction range ({len(predictions)} days).")

    # Calculate the percentage change over the time range
    start_price = predictions[0]
    end_price = predictions[steps - 1]
    percentage_change = (end_price - start_price) / start_price

    # Classify the outcome
    if percentage_change > threshold:
        outcome = "Gain"
    elif percentage_change < -threshold:
        outcome = "Loss"
    else:
        outcome = "No Changes"

    return {
        "time_range": time_range,
        "percentage_change": percentage_change * 100,  # Convert to percentage
        "outcome": outcome
    }

def classify_preds(future_predictions):
        
    # Classify for a day
    result_day = classify_prediction_outcome(future_predictions, time_range="day")
    print(f"Day Classification: {result_day}")

    # Classify for a week
    result_week = classify_prediction_outcome(future_predictions, time_range="week")
    print(f"Week Classification: {result_week}")

    # Classify for a month
    result_month = classify_prediction_outcome(future_predictions, time_range="month")
    print(f"Month Classification: {result_month}")
    
    return(result_day, result_week, result_month)

app = FastAPI()

# Request Models
class PredictionRequest(BaseModel):
    stock_ticker: str

class ClassificationRequest(BaseModel):
    predictions: List[float]
    time_range: str
    threshold: float = 0.005  # Default threshold for classification

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    stock_ticker = request.stock_ticker
    predictions = AutoML2(stock_ticker)  # Replace with your function
    return {"stock_ticker": stock_ticker, "predictions": predictions}

# Define the classification endpoint
@app.post("/classify")
def classify(request: ClassificationRequest):
    classifications = classify_prediction_outcome(
        predictions=np.array(request.predictions),
        time_range=request.time_range,
        threshold=request.threshold
    )
    return {
        "time_range": request.time_range,
        "classification": classifications["outcome"],
        "percentage_change": classifications["percentage_change"]
    }

# Root endpoint to check the API status
@app.get("/")
def root():
    return {"message": "Stock Prediction API is running!"}
