from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import random

app = FastAPI()

# Request Models
class PredictionRequest(BaseModel):
    stock_ticker: str

class ClassificationRequest(BaseModel):
    predictions: List[float]
    time_range: str

# Placeholder function for predictions
def placeholder_predict(stock_ticker):
    # Generate random predictions as a placeholder
    return [round(random.uniform(100, 200), 2) for _ in range(30)]

# Placeholder function for classification
def placeholder_classify(predictions, time_range):
    # Fake logic to classify predictions as Gain, Loss, or No Changes
    if predictions[-1] > predictions[0]:
        return {"time_range": time_range, "outcome": "Gain", "percentage_change": 5.0}
    elif predictions[-1] < predictions[0]:
        return {"time_range": time_range, "outcome": "Loss", "percentage_change": -3.0}
    else:
        return {"time_range": time_range, "outcome": "No Changes", "percentage_change": 0.0}

# Prediction Endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    predictions = placeholder_predict(request.stock_ticker)
    return {"stock_ticker": request.stock_ticker, "predictions": predictions}

# Classification Endpoint
@app.post("/classify")
def classify(request: ClassificationRequest):
    classifications = placeholder_classify(request.predictions, request.time_range)
    return classifications

# Root Endpoint
@app.get("/")
def root():
    return {"message": "Stock Prediction API is running!"}
