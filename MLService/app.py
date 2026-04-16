from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
import datetime

app = FastAPI(title="Pharmacy Inventory Forecasting Service")

# 1. Define Request Schema
class InventoryRequest(BaseModel):
    product_id: int
    current_stock: int
    min_stock_level: int
    lead_time_days: int
    sales_history: List[int]
    is_holiday: int
    is_promotion: int
    days_until_next_event: int

# 2. Centralized Model Loading
try:
    demand_model = joblib.load('pharmacy_demand_model.joblib')
    safety_model = joblib.load('safety_stock_model.joblib')
    scaler = joblib.load('feature_scaler.joblib')
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

@app.get('/')
def health_check():
    return {"status": "running", "message": "Pharmacy Inventory API is operational"}

@app.post('/predict')
async def predict_inventory(data: InventoryRequest):
    if len(data.sales_history) < 7:
        raise HTTPException(status_code=400, detail="Sales history must contain at least 7 days.")

    sales_series = pd.Series(data.sales_history)
    today = datetime.date.today()

    # 3. Feature Engineering
    features = {
        'Current_Stock': data.current_stock,
        'Min_Stock_Level': data.min_stock_level,
        'Lead_Time_Days': data.lead_time_days,
        'Year': today.year,
        'Month': today.month,
        'Day': today.day,
        'DayOfWeek': today.weekday(),
        'WeekOfYear': today.isocalendar()[1],
        'Sales_MA_7_Days': sales_series.tail(7).mean(),
        'Sales_MA_30_Days': sales_series.mean(),
        'Sales_EWMA': sales_series.ewm(span=7, adjust=False).mean().iloc[-1],
        'is_holiday': data.is_holiday,
        'is_promotion': data.is_promotion,
        'days_until_next_event': data.days_until_next_event,
        'Sales_Lag_1': data.sales_history[-1],
        'Sales_Lag_2': data.sales_history[-2] if len(data.sales_history) >= 2 else 0,
        'Sales_Lag_3': data.sales_history[-3] if len(data.sales_history) >= 3 else 0,
        'Sales_Lag_5': data.sales_history[-5] if len(data.sales_history) >= 5 else 0,
        'Sales_Lag_7': data.sales_history[-7] if len(data.sales_history) >= 7 else 0,
        'Sales_Volatility_7D': sales_series.tail(7).std() if len(sales_series) >= 7 else 0
    }

    # 4. Map Product ID to One-Hot Columns
    for i in range(2, 6):
        features[f'Product_Name_Product_{i}'] = 1 if data.product_id == i else 0

    # 5. Convert to DataFrame and Scale
    input_df = pd.DataFrame([features])
    scaling_features = scaler.feature_names_in_
    input_scaled = input_df.copy()
    input_scaled[scaling_features] = scaler.transform(input_df[scaling_features])

    # 6. Inference
    demand_features = demand_model.feature_names_in_
    safety_features = safety_model.feature_names_in_

    forecast = demand_model.predict(input_scaled[demand_features])[0]
    safety_threshold = safety_model.predict(input_scaled[safety_features])[0]

    # 7. Replenishment Logic
    order_quantity = 0
    status = "Stock Level Adequate"

    if data.current_stock < safety_threshold:
        status = "REPLENISHMENT REQUIRED"
        needed = (safety_threshold - data.current_stock) + (forecast * data.lead_time_days)
        order_quantity = int(np.ceil(max(0, needed)))

    return {
        "predicted_demand": round(float(forecast), 2),
        "safety_threshold": round(float(safety_threshold), 2),
        "status": status,
        "suggested_order_quantity": order_quantity
    }
