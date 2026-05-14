"""
engineer_features.py
─────────────────────
Loads datasets/training_raw.csv, computes technical indicators and
the 14-day forward label, then saves to datasets/training_features.csv.

Run after collect_data.py:  python engineer_features.py
Takes ~2-5 minutes.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "datasets/training_raw.csv"
OUTPUT_PATH = "datasets/training_features.csv"

# ── Load raw data ─────────────────────────────────────────────────────────────
print(f"Loading {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
df.sort_values(["ticker", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Loaded {len(df):,} rows across {df['ticker'].nunique()} tickers")

# ── Feature engineering per ticker ───────────────────────────────────────────
print("\nComputing features per ticker...")

all_frames = []
tickers    = df["ticker"].unique()

for i, ticker in enumerate(tickers, 1):
    sub = df[df["ticker"] == ticker].copy()
    sub.reset_index(drop=True, inplace=True)

    if len(sub) < 60:
        # Not enough history to compute meaningful indicators — skip
        print(f"  [{i}/{len(tickers)}] {ticker:12s} — too few rows ({len(sub)}), skipping")
        continue

    try:
        # ── Technical indicators ──────────────────────────────────────────────

        # RSI — momentum oscillator (14-day)
        sub["RSI_14"] = ta.rsi(sub["close"], length=14)

        # MACD — trend following
        macd = ta.macd(sub["close"], fast=12, slow=26, signal=9)
        sub["MACD"]        = macd["MACD_12_26_9"]
        sub["MACD_signal"] = macd["MACDs_12_26_9"]
        sub["MACD_hist"]   = macd["MACDh_12_26_9"]

        # Bollinger Bands — volatility bands
        bb = ta.bbands(sub["close"], length=20, std=2)
        # Dynamically find upper/lower band columns (name varies by pandas_ta version)
        bb_upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
        bb_lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
        bb_upper = bb[bb_upper_col]
        bb_lower = bb[bb_lower_col]
        # BB position: 0 = at lower band, 1 = at upper band
        sub["BB_position"] = (sub["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # Simple moving averages
        sub["SMA_20"] = ta.sma(sub["close"], length=20)
        sub["SMA_50"] = ta.sma(sub["close"], length=50)
        # SMA crossover signal: 1 if SMA_20 > SMA_50 (bullish), else 0
        sub["SMA_cross"] = (sub["SMA_20"] > sub["SMA_50"]).astype(int)

        # Exponential moving averages
        sub["EMA_12"] = ta.ema(sub["close"], length=12)
        sub["EMA_26"] = ta.ema(sub["close"], length=26)

        # Volume ratio: today's volume vs 20-day average
        sub["volume_ratio"] = sub["volume"] / (sub["volume"].rolling(20).mean() + 1e-9)

        # Price momentum: percentage return over N days
        sub["price_momentum_5d"]  = sub["close"].pct_change(5)
        sub["price_momentum_10d"] = sub["close"].pct_change(10)
        sub["price_momentum_20d"] = sub["close"].pct_change(20)

        # Annualized volatility: rolling 20-day std of daily returns × √252
        daily_returns = sub["close"].pct_change()
        sub["volatility_20d"] = daily_returns.rolling(20).std() * np.sqrt(252)

        # Daily high-low range as fraction of close
        sub["high_low_range"] = (sub["high"] - sub["low"]) / (sub["close"] + 1e-9)

        # ── Label: 14-day forward direction ──────────────────────────────────
        # 1 = price higher in 14 days (UP), 0 = price lower or equal (DOWN)
        sub["future_close"] = sub["close"].shift(-14)
        sub["label"] = (sub["future_close"] > sub["close"]).astype(int)

        # Drop last 14 rows — label is NaN (future not available)
        sub = sub.iloc[:-14]

        all_frames.append(sub)
        print(f"  [{i}/{len(tickers)}] {ticker:12s} — {len(sub)} rows")

    except Exception as e:
        import traceback
        print(f"  [{i}/{len(tickers)}] {ticker:12s} — ERROR: {e}")
        traceback.print_exc()
        continue

# ── Combine all tickers ───────────────────────────────────────────────────────
print(f"\nCombining {len(all_frames)} tickers...")
result = pd.concat(all_frames, ignore_index=True)

# ── Keep only feature columns + metadata ─────────────────────────────────────
FEATURE_COLS = [
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_position",
    "SMA_cross",
    "EMA_12", "EMA_26",
    "volume_ratio",
    "price_momentum_5d", "price_momentum_10d", "price_momentum_20d",
    "volatility_20d",
    "high_low_range",
]

keep_cols = ["ticker", "date", "close"] + FEATURE_COLS + ["label"]
result    = result[keep_cols]

# ── Drop rows with any NaN in feature columns ─────────────────────────────────
before = len(result)
result.dropna(subset=FEATURE_COLS + ["label"], inplace=True)
after  = len(result)
print(f"Dropped {before - after:,} rows with NaN — {after:,} rows remaining")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("datasets", exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Done.")
print(f"Total rows    : {len(result):,}")
print(f"Tickers       : {result['ticker'].nunique()}")
print(f"Date range    : {result['date'].min()} → {result['date'].max()}")
print(f"Label balance : {result['label'].mean()*100:.1f}% UP  /  {(1-result['label'].mean())*100:.1f}% DOWN")
print(f"Features      : {len(FEATURE_COLS)}")
print(f"Output file   : {OUTPUT_PATH}")
print(f"{'─'*50}")
print("\nNext step: run  python train_model.py")
