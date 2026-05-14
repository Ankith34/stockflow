from groq import Groq
import os, asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

from agents.data_fetcher import fetch_stock_data
from agents.sentiment_agent import analyze_sentiment
from agents.anomaly_detector import detect_anomalies
from agents.memo_writer import write_memo
from agents.ml_predictor import predict_direction

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
executor = ThreadPoolExecutor(max_workers=8)


def get_currency(ticker): return "₹" if ticker.endswith((".NS",".BO")) else "$"

def resolve_ticker(user_input: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""Convert to a valid yfinance ticker.
Only add .NS for stocks EXCLUSIVELY listed in India with no US listing.
Reply with ONLY the ticker, nothing else.

Examples:
"reliance" → RELIANCE.NS
"tata motors" → TATAMOTORS.NS
"infosys" → INFY.NS
"hdfc bank" → HDFCBANK.NS
"wipro" → WIPRO.NS
"amazon" → AMZN
"google" → GOOGL
"microsoft" → MSFT
"apple" → AAPL
"tesla" → TSLA
"nvidia" → NVDA
"meta" → META
"netflix" → NFLX
"nokia" → NOK
"sony" → SONY
"toyota" → TM
"alibaba" → BABA

Input: {user_input}
Ticker:"""}],
        temperature=0, max_tokens=10
    )
    return response.choices[0].message.content.strip().upper()


# ── ASYNC PIPELINE ──
async def run_analysis_async(user_input: str) -> str:
    loop = asyncio.get_event_loop()

    ticker = await loop.run_in_executor(executor, resolve_ticker, user_input)
    print(f"Resolved: {ticker}")

    is_indian = ticker.endswith((".NS", ".BO"))
    currency  = get_currency(ticker)

    # metrics first — sentiment + anomaly both need it
    metrics = await loop.run_in_executor(executor, fetch_stock_data, ticker)

    # sentiment + anomaly + ml prediction IN PARALLEL
    sentiment, anomalies, prediction = await asyncio.gather(
        loop.run_in_executor(executor, analyze_sentiment, ticker, metrics.get("company_name", ticker)),
        loop.run_in_executor(executor, detect_anomalies, metrics),
        loop.run_in_executor(executor, predict_direction, metrics)
    )

    # memo_writer agent produces the final memo
    memo = await loop.run_in_executor(
        executor, write_memo, ticker, metrics, sentiment, anomalies, prediction, is_indian, currency
    )

    # Build chart data from price history
    import yfinance as yf
    from datetime import datetime, timedelta
    chart_data = []
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if not hist.empty:
            for dt, close in zip(hist.index, hist["Close"]):
                # Lightweight Charts expects {time: "YYYY-MM-DD", value: float}
                chart_data.append({
                    "time":  dt.strftime("%Y-%m-%d"),
                    "value": round(float(close), 2)
                })
    except Exception:
        # Fall back to price_history from metrics if yfinance fails
        price_history = metrics.get("price_history", [])
        if price_history:
            base_date = datetime.utcnow() - timedelta(days=len(price_history))
            for i, price in enumerate(price_history):
                chart_data.append({
                    "time":  (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "value": round(float(price), 2)
                })

    return {
        "memo":          memo,
        "chart_data":    chart_data,
        "ticker":        ticker,
        "company_name":  metrics.get("company_name", ticker),
        "currency":      currency,
        "price_change":  metrics.get("price_change_6m"),
    }

# sync wrapper
def run_analysis(user_input: str) -> dict:
    return asyncio.run(run_analysis_async(user_input))