from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Load model and column names
model = joblib.load("stock_model.pkl")
columns = joblib.load("feature_columns.pkl")

app = FastAPI()

# Define input schema (use same order as columns)
class StockInput(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi_7: float
    rsi_14: float
    cci_7: float
    cci_14: float
    sma_50: float
    ema_50: float
    sma_100: float
    ema_100: float
    macd: float
    bollinger: float
    TrueRange: float
    atr_7: float
    atr_14: float

@app.post("/predict")
def predict_trend(data: StockInput):
    input_data = np.array([[getattr(data, col) for col in columns]])
    prediction = model.predict(input_data)[0]
    label = "bullish" if prediction == 1 else "bearish"
    return {"prediction": label}
