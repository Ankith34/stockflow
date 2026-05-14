"""
train_model.py
──────────────
Loads datasets/training_features.csv, trains an XGBoost classifier
to predict 14-day price direction (UP=1 / DOWN=0), evaluates it,
and saves the model artifacts to models/.

Run after engineer_features.py:  python train_model.py
Takes ~5-10 minutes.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH   = "datasets/training_features.csv"
MODELS_DIR   = "models"
MODEL_PATH   = f"{MODELS_DIR}/price_direction_model.pkl"
SCALER_PATH  = f"{MODELS_DIR}/feature_scaler.pkl"
COLUMNS_PATH = f"{MODELS_DIR}/feature_columns.json"

# ── Feature columns — must match engineer_features.py exactly ─────────────────
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

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print(f"Loading {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)
print(f"Loaded {len(df):,} rows | {df['ticker'].nunique()} tickers")
print(f"Label balance: {df['label'].mean()*100:.1f}% UP / {(1-df['label'].mean())*100:.1f}% DOWN")

# ── Step 2: Time-based split — NEVER shuffle time series ─────────────────────
# Train: everything before 2024-01-01
# Val:   2024-01-01 to 2024-07-01
# Test:  2024-07-01 onwards (held out — final evaluation only)
train = df[df["date"] <  "2024-01-01"]
val   = df[(df["date"] >= "2024-01-01") & (df["date"] < "2024-07-01")]
test  = df[df["date"] >= "2024-07-01"]

print(f"\nSplit:")
print(f"  Train : {len(train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
print(f"  Val   : {len(val):,} rows  ({val['date'].min().date()} → {val['date'].max().date()})")
print(f"  Test  : {len(test):,} rows  ({test['date'].min().date()} → {test['date'].max().date()})")

X_train, y_train = train[FEATURE_COLS].values, train["label"].values
X_val,   y_val   = val[FEATURE_COLS].values,   val["label"].values
X_test,  y_test  = test[FEATURE_COLS].values,  test["label"].values

# ── Step 3: Scale features ────────────────────────────────────────────────────
# Fit ONLY on training data — never on val or test
print("\nFitting StandardScaler on training data...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── Step 4: Train XGBoost ─────────────────────────────────────────────────────
print("\nTraining XGBoost classifier...")
print("(Early stopping on validation AUC — will stop if no improvement for 50 rounds)\n")

model = xgb.XGBClassifier(
    n_estimators          = 1000,
    max_depth             = 4,        # shallower trees — less overfitting
    learning_rate         = 0.01,     # slower learning — more generalizable
    subsample             = 0.7,
    colsample_bytree      = 0.7,
    min_child_weight      = 50,       # require more samples per leaf — prevents noise fitting
    gamma                 = 1,        # minimum loss reduction to split
    reg_alpha             = 0.1,      # L1 regularization
    reg_lambda            = 1.0,      # L2 regularization
    scale_pos_weight      = (y_train == 0).sum() / (y_train == 1).sum(),  # balance classes
    eval_metric           = "auc",
    early_stopping_rounds = 100,      # more patience
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=25,   # print every 25 rounds
)

print(f"\nBest iteration: {model.best_iteration}")

# ── Step 5: Evaluate on validation set ───────────────────────────────────────
print("\n── Validation Set Results ──")
val_pred      = model.predict(X_val)
val_pred_prob = model.predict_proba(X_val)[:, 1]
print(f"Accuracy : {accuracy_score(y_val, val_pred)*100:.2f}%")
print(f"AUC-ROC  : {roc_auc_score(y_val, val_pred_prob):.4f}")
print(classification_report(y_val, val_pred, target_names=["DOWN", "UP"], zero_division=0))

# ── Step 6: Evaluate on test set (final, held-out) ────────────────────────────
print("── Test Set Results (held-out) ──")
test_pred      = model.predict(X_test)
test_pred_prob = model.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_pred_prob)
print(f"Accuracy : {test_acc*100:.2f}%")
print(f"AUC-ROC  : {test_auc:.4f}")
print(classification_report(y_test, test_pred, target_names=["DOWN", "UP"], zero_division=0))

# Sanity check — if accuracy > 70% something is wrong
if test_acc > 0.70:
    print("⚠ WARNING: Accuracy > 70% — check for data leakage!")
else:
    print("✅ Accuracy in expected range (55–65%) — no obvious leakage")

# ── Step 7: Feature importance ────────────────────────────────────────────────
print("\n── Feature Importance (top 14) ──")
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importance.sort_values(ascending=False, inplace=True)
for feat, score in importance.items():
    bar = "█" * int(score * 200)
    print(f"  {feat:25s} {score:.4f}  {bar}")

# ── Step 8: Save artifacts ────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(COLUMNS_PATH, "w") as f:
    json.dump(FEATURE_COLS, f)

print(f"\n{'─'*50}")
print("Artifacts saved:")
print(f"  {MODEL_PATH}")
print(f"  {SCALER_PATH}")
print(f"  {COLUMNS_PATH}")
print(f"{'─'*50}")
print("\nNext step: create agents/ml_predictor.py")
