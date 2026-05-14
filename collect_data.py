"""
collect_data.py
───────────────
Fetches 5 years of daily OHLCV data for S&P 500 stocks from Yahoo Finance
and saves it to datasets/training_raw.csv.

Run once:  python collect_data.py
Takes ~20-30 minutes depending on network speed.
"""

import pandas as pd
import yfinance as yf
import time
import os

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_PATH = "datasets/training_raw.csv"

# ── Step 1: Fetch S&P 500 ticker list from Wikipedia ─────────────────────────
print("Fetching S&P 500 ticker list from Wikipedia...")
try:
    # Wikipedia blocks requests without a browser user-agent — spoof it
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    tables  = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        storage_options={"User-Agent": headers["User-Agent"]}
    )
    sp500_df = tables[0]  # first table on the page is the constituent list
    tickers  = sp500_df["Symbol"].tolist()
    # Wikipedia uses dots (BRK.B) but yfinance needs hyphens (BRK-B)
    tickers  = [t.replace(".", "-") for t in tickers]
    print(f"Found {len(tickers)} tickers")
except Exception as e:
    print(f"Failed to fetch S&P 500 list: {e}")
    raise SystemExit(1)

# ── Step 2: Download OHLCV data for each ticker ───────────────────────────────
all_frames = []
failed     = []

print(f"\nDownloading 5 years of daily data for {len(tickers)} tickers...")
print("This will take 20-30 minutes. Progress is shown below.\n")

for i, ticker in enumerate(tickers, 1):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="5y")

        if hist.empty:
            print(f"  [{i}/{len(tickers)}] {ticker:12s} — EMPTY, skipping")
            failed.append(ticker)
            continue

        # Keep only OHLCV columns, add ticker column
        hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        hist.columns = ["open", "high", "low", "close", "volume"]
        hist.index.name = "date"
        hist.reset_index(inplace=True)
        hist["ticker"] = ticker

        all_frames.append(hist)
        print(f"  [{i}/{len(tickers)}] {ticker:12s} — {len(hist)} rows")

    except Exception as e:
        print(f"  [{i}/{len(tickers)}] {ticker:12s} — ERROR: {e}")
        failed.append(ticker)

    # Small delay to avoid hammering Yahoo Finance
    time.sleep(0.3)

# ── Step 3: Combine and save ──────────────────────────────────────────────────
if not all_frames:
    print("\nNo data collected. Exiting.")
    raise SystemExit(1)

print(f"\nCombining data from {len(all_frames)} tickers...")
df = pd.concat(all_frames, ignore_index=True)

# Ensure date column is clean
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

# Sort by ticker then date
df.sort_values(["ticker", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Save
os.makedirs("datasets", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Done.")
print(f"Total rows   : {len(df):,}")
print(f"Tickers saved: {df['ticker'].nunique()}")
print(f"Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Output file  : {OUTPUT_PATH}")
if failed:
    print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")
print(f"{'─'*50}")
print("\nNext step: run  python engineer_features.py")
