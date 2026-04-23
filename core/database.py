import sqlite3
import json
from datetime import datetime

DB_PATH = "investment_analyst.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metrics TEXT,
            sentiment TEXT,
            anomalies TEXT,
            memo TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(ticker, metrics, sentiment, anomalies, memo):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses (ticker, timestamp, metrics, sentiment, anomalies, memo)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        datetime.now().isoformat(),
        json.dumps(metrics),
        json.dumps(sentiment),
        json.dumps(anomalies),
        memo
    ))
    conn.commit()
    conn.close()

def get_history(ticker: str, limit: int = 5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, memo FROM analyses
        WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?
    """, (ticker, limit))
    rows = c.fetchall()
    conn.close()
    return rows

def get_cached(ticker: str, max_age_minutes: int = 60):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT memo FROM analyses 
        WHERE ticker = ? 
        AND timestamp > datetime('now', ? )
        ORDER BY timestamp DESC LIMIT 1
    """, (ticker, f'-{max_age_minutes} minutes'))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None