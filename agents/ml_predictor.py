"""
ml_predictor.py
───────────────
ML agent that predicts 14-day price direction (UP/DOWN) using an XGBoost
classifier trained on technical indicators.

Loads model artifacts once at import time, then provides fast predictions
on live stock data from data_fetcher.
"""

import joblib
import json
import numpy as np
import pandas as pd
import os

# ── Load model artifacts once at module import ───────────────────────────────
MODEL_PATH   = "models/price_direction_model.pkl"
SCALER_PATH  = "models/feature_scaler.pkl"
COLUMNS_PATH = "models/feature_columns.json"

MODEL_AVAILABLE = False
model  = None
scaler = None
feature_cols = []

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH):
        model        = joblib.load(MODEL_PATH)
        scaler       = joblib.load(SCALER_PATH)
        feature_cols = json.load(open(COLUMNS_PATH))
        MODEL_AVAILABLE = True
        print(f"[ML_PREDICTOR] Model loaded successfully ({len(feature_cols)} features)")
    else:
        print("[ML_PREDICTOR] Model artifacts not found — predictions will return UNKNOWN")
except Exception as e:
    print(f"[ML_PREDICTOR] Failed to load model: {e}")


def predict_direction(metrics: dict) -> dict:
    """
    Predicts 14-day price direction using the trained XGBoost model.
    
    Args:
        metrics: dict from data_fetcher containing price_history and other data
    
    Returns:
        {
          "direction":  "UP" | "DOWN" | "UNKNOWN",
          "confidence": 0.0 to 1.0,
          "horizon":    "14d"
        }
    """
    # If model not available, return gracefully
    if not MODEL_AVAILABLE:
        return {"direction": "UNKNOWN", "confidence": 0.0, "horizon": "14d"}
    
    price_history = metrics.get("price_history", [])
    ticker        = metrics.get("ticker", "")

    # data_fetcher only returns 30 days — not enough for SMA_50
    # Fetch 6 months directly for ML features
    extended_history = []
    if ticker:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="6mo")
            if not hist.empty:
                extended_history = hist["Close"].tolist()
        except Exception:
            pass

    # Use extended history if available, fall back to price_history
    prices = extended_history if len(extended_history) >= 60 else price_history
    print(f"[ML_PREDICTOR] {ticker} — using {len(prices)} price points")

    # Need at least 60 data points to compute all indicators reliably
    if len(prices) < 60:
        print(f"[ML_PREDICTOR] {ticker} — insufficient data ({len(prices)} < 60)")
        return {"direction": "UNKNOWN", "confidence": 0.0, "horizon": "14d"}
    
    try:
        # Import pandas_ta here — only needed if model is available
        import pandas_ta as ta
        
        # Build a small DataFrame from extended price history
        df = pd.DataFrame({"close": prices})
        
        # Compute the same features as training — must match exactly
        # (These are the 14 features from engineer_features.py)
        
        df["RSI_14"] = ta.rsi(df["close"], length=14)
        
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["MACD"]        = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"]   = macd["MACDh_12_26_9"]
        
        bb = ta.bbands(df["close"], length=20, std=2)
        bb_upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
        bb_lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
        bb_upper = bb[bb_upper_col]
        bb_lower = bb[bb_lower_col]
        df["BB_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)
        
        df["SMA_20"] = ta.sma(df["close"], length=20)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
        
        df["EMA_12"] = ta.ema(df["close"], length=12)
        df["EMA_26"] = ta.ema(df["close"], length=26)
        
        # Volume ratio — use latest_volume from metrics if available
        latest_vol = metrics.get("latest_volume", 0)
        avg_vol    = metrics.get("avg_volume_30d", 1)
        df["volume_ratio"] = latest_vol / (avg_vol + 1e-9)
        
        df["price_momentum_5d"]  = df["close"].pct_change(5)
        df["price_momentum_10d"] = df["close"].pct_change(10)
        df["price_momentum_20d"] = df["close"].pct_change(20)
        
        daily_returns = df["close"].pct_change()
        df["volatility_20d"] = daily_returns.rolling(20).std() * np.sqrt(252)
        
        # high_low_range — use current_price as proxy for close
        current_price = metrics.get("current_price", df["close"].iloc[-1])
        high_52w = metrics.get("52w_high", current_price)
        low_52w  = metrics.get("52w_low", current_price)
        df["high_low_range"] = (high_52w - low_52w) / (current_price + 1e-9)
        
        # Take the last row — most recent values
        row = df.iloc[-1]
        
        # Build feature vector in exact training order
        features = [row.get(col, 0) for col in feature_cols]
        features = np.array(features).reshape(1, -1)
        
        # Check for NaN — if any feature is NaN, return UNKNOWN
        if np.isnan(features).any():
            nan_cols = [feature_cols[i] for i, v in enumerate(features[0]) if np.isnan(v)]
            print(f"[ML_PREDICTOR] NaN in features: {nan_cols}")
            return {"direction": "UNKNOWN", "confidence": 0.0, "horizon": "14d"}
        
        # Scale with the saved scaler
        features = scaler.transform(features)
        
        # Predict probabilities — [prob_down, prob_up]
        probs    = model.predict_proba(features)[0]
        prob_up  = probs[1]
        
        direction  = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = prob_up if direction == "UP" else (1 - prob_up)
        
        return {
            "direction":  direction,
            "confidence": round(float(confidence), 2),
            "horizon":    "14d"
        }
    
    except Exception as e:
        print(f"[ML_PREDICTOR] Prediction failed: {e}")
        return {"direction": "UNKNOWN", "confidence": 0.0, "horizon": "14d"}
