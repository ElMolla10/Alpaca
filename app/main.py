#!/usr/bin/env python3
import os, sys, time, math, json, traceback
import datetime as dt
from datetime import datetime, timezone
import pytz
import numpy as np
import pandas as pd
from alpaca_trade_api import REST, TimeFrame
import xgboost as xgb
from ta.trend import MACD
from ta.volatility import BollingerBands

# =================== CONFIG / ENV ===================
TZ_NY = pytz.timezone("America/New_York")

BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
KEY_ID   = os.environ["APCA_API_KEY_ID"]
SECRET   = os.environ["APCA_API_SECRET_KEY"]

# Trading session: 10:00 → 16:00 ET in 1h blocks
SESSION_START_H = int(os.environ.get("SESSION_START_H", "10"))
SESSION_BLOCKS  = int(os.environ.get("SESSION_BLOCKS", "6"))  # 6 blocks → 10..16

# Live policy params (match backtest)
# =================== CONFIG (TUNABLE) ===================
PRIMARY_H      = int(os.environ.get("PRIMARY_H", "3"))        # model trained per 3h block
BAND_R         = float(os.environ.get("BAND_R", "0.8"))       # smaller = more sensitive
EMA_HALF_LIFE  = int(os.environ.get("EMA_HL", "1"))           # 1 = no smoothing
DPOS_CAP       = float(os.environ.get("DPOS_CAP", "0.5"))     # allow faster re-hedge
LEVERAGE       = float(os.environ.get("LEVERAGE", "3.0"))
STOP_LOSS_PCT  = float(os.environ.get("STOP_LOSS_PCT", "2.0"))  # unchanged

# new optional aggressiveness switches
FORCE_TRADE   = os.environ.get("FORCE_TRADE", "1") == "1"
FORCE_MIN_POS = float(os.environ.get("FORCE_MIN_POS", "0.20"))


TRADE_COST_BPS = float(os.environ.get("TRADE_COST_BPS", "8.0"))
SLIP_BPS       = float(os.environ.get("SLIPPAGE_BPS", "4.0"))

# Order sizing
MAX_NOTIONAL_PER_SYM = float(os.environ.get("MAX_NOTIONAL", "5000"))
FORCE_MIN_POS = float(os.environ.get("FORCE_MIN_POS", "0.20"))  # 20% exposure floor
FORCE_TRADE   = os.environ.get("FORCE_TRADE", "1") == "1"       # enable floor

# Symbols
TEST_MODE  = os.environ.get("TEST_MODE", "0") == "1"
SYMBOLS    = [os.environ.get("TEST_SYMBOL", "AAPL")] if TEST_MODE else \
             os.environ.get("SYMBOLS", "AAPL,MSFT,PG,AMD,JPM,XOM").split(",")

# Model & features
MODEL_PATH = os.environ.get("MODEL_PATH", "app/model/XGboost_model.json")
FEATS_PATH = os.environ.get("FEATS_PATH", "app/feat_cols.json")

# Internal constants for feature build
CLIP_HOURLY_RET = 8.0
HRS_LOOKBACK    = 500

# Persist EMA state across blocks
STATE_PATH = os.environ.get("STATE_PATH", "/tmp/live_state.json")

# =================== UTILS ===================
def utc_ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def now_ny():  return dt.datetime.now(TZ_NY)

def load_state():
    try:
        with open(STATE_PATH, "r") as f: return json.load(f)
    except Exception:
        return {"pos_ema": {}, "last_day": ""}

def save_state(st):
    try:
        with open(STATE_PATH, "w") as f: json.dump(st, f)
    except Exception:
        pass

# =================== ALPACA ===================
def latest_price(api, sym):
    try:
        tr = api.get_latest_trade(sym)
        return float(tr.price)
    except Exception:
        bars = api.get_bars(sym, TimeFrame.Minute, limit=1)
        return float(bars[-1].c) if bars else math.nan

def account_equity(api) -> float:
    try:
        return float(api.get_account().equity)
    except Exception:
        return 0.0

def flatten(api, sym):
    try:
        p = api.get_position(sym)
        qty = float(p.qty)
        if abs(qty) > 0:
            side = "sell" if qty > 0 else "buy"
            print(f"[FLATTEN] {sym} qty={qty}")
            api.submit_order(symbol=sym, qty=abs(qty), side=side, type="market", time_in_force="day")
    except Exception:
        pass

def submit_target(api, sym, target_pos_frac, eq, px):
    if not (px > 0 and eq > 0): return
    if abs(target_pos_frac) < MIN_ABS_POS:
        flatten(api, sym)
        return
    notional = min(MAX_NOTIONAL_PER_SYM, eq * 0.15) * float(target_pos_frac)  # 15% per symbol cap
    if abs(notional) < max(50.0, px):
        print(f"[SKIP] tiny notional for {sym}: {notional:.2f}")
        return
    try:
        if USE_NOTIONAL_ORDERS:
            side = "buy" if notional > 0 else "sell"
            print(f"[ORDER] {sym} {side.upper()} notional ${abs(notional):.2f}")
            api.submit_order(
                symbol=sym, notional=abs(notional), side=side,
                type="market", time_in_force="day",
                order_class="bracket",
                take_profit={"limit_price": round(px * (1 + 0.01 * max(0.5, abs(target_pos_frac))), 2)},
                stop_loss={"stop_price": round(px * (1 - STOP_LOSS_PCT/100.0), 2)}
            )
        else:
            qty = int(abs(notional) // px)
            if qty < 1:
                print(f"[SKIP] qty<1 for {sym}")
                return
            side = "buy" if notional > 0 else "sell"
            print(f"[ORDER] {sym} {side.upper()} qty {qty}")
            api.submit_order(
                symbol=sym, qty=qty, side=side,
                type="market", time_in_force="day",
                order_class="bracket",
                take_profit={"limit_price": round(px * (1 + 0.01 * max(0.5, abs(target_pos_frac))), 2)},
                stop_loss={"stop_price": round(px * (1 - STOP_LOSS_PCT/100.0), 2)}
            )
    except Exception as e:
        print(f"[ORDER_ERR] {sym}: {e}")

# =================== LIVE FEATURE PIPELINE (identical to training) ===================
with open(FEATS_PATH, "r") as f:
    FEAT_COLS = json.load(f)

def _fetch_hour_bars(api, sym, end_utc=None, hours=HRS_LOOKBACK) -> pd.DataFrame:
    if end_utc is None:
        end_utc = dt.datetime.now(dt.timezone.utc)
    start_utc = end_utc - dt.timedelta(hours=hours)
    bars = api.get_bars(sym, TimeFrame.Hour, start=start_utc, end=end_utc, adjustment='raw', limit=10000)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "timestamp": pd.Timestamp(b.t, tz="UTC"),
        "open": float(b.o),
        "high": float(b.h),
        "low": float(b.l),
        "close": float(b.c),
        "volume": float(b.v),
        "vwap": float(getattr(b, "vw", b.c))
    } for b in bars])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _regular_hours_filter(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    hours_et = ts.dt.tz_convert("America/New_York").dt.hour
    keep = hours_et.between(10, 15)  # 10:00–15:59 ET
    return df.loc[keep].reset_index(drop=True)

def _compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["price_change_pct"] = df["close"].pct_change() * 100.0
    macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["MACDh_12_26_9"] = macd.macd_diff()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2.0)
    df["BBM_20_2.0"] = bb.bollinger_mavg()
    return df

def _apply_training_pipeline_live(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1h_pct"] = df["close"].pct_change().shift(-1) * 100.0
    df["sigma20_pct"] = df["ret1h_pct"].rolling(20).std().shift(1)
    df["sigma20_pct"] = df["sigma20_pct"].replace(0.0, np.nan)
    base = ["open","high","low","close","volume","vwap",
            "price_change_pct","MACDh_12_26_9","BBM_20_2.0"]
    for f in base:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].shift(1)
    for f in ["price_change_pct","MACDh_12_26_9","BBM_20_2.0"]:
        df[f+"_lag2"] = df[f].shift(2)
        df[f+"_lag3"] = df[f].shift(3)
    df["vol20"] = df["ret1h_pct"].rolling(20).std().shift(1)
    df["ret5"]  = df["close"].pct_change(5).shift(1) * 100.0
    for f in ["price_change_pct","ret5","vol20","MACDh_12_26_9","BBM_20_2.0"]:
        if f in df.columns:
            df[f+"_N"] = df[f] / df["sigma20_pct"]
    df["ret1h_pct"] = df["ret1h_pct"].fillna(0.0).clip(lower=-CLIP_HOURLY_RET, upper=CLIP_HOURLY_RET)
    df = df.dropna().reset_index(drop=True)
    return df

def fetch_recent_features(api, sym: str) -> np.ndarray | None:
    df = _fetch_hour_bars(api, sym)
    if df.empty:
        print(f"[WARN] no hourly bars for {sym}")
        return None
    df = _regular_hours_filter(df)
    if len(df) < 60:
        print(f"[WAIT] need more regular-hour bars for {sym}: have {len(df)}")
        return None
    df = _compute_base_indicators(df)
    df = _apply_training_pipeline_live(df)
    if df.empty:
        print(f"[WAIT] no fully-formed row for {sym} after pipeline")
        return None
    row = df.iloc[-1]
    for col in FEAT_COLS:
        if col not in df.columns:
            row[col] = 0.0
    X = row[FEAT_COLS].astype(float).values.reshape(1, -1)
    return X

# =================== MODEL (Booster matching training) ===================
booster = xgb.Booster()
booster.load_model(MODEL_PATH)

def predict_block_return_pct(api, sym: str) -> float:
    """
    Predict % return for NEXT 1h, scaled from PRIMARY_H block as trained.
    """
    try:
        X = fetch_recent_features(api, sym)
        if X is None:
            return 0.0
        dmat = xgb.DMatrix(X)
        pred_pct_primary = float(booster.predict(dmat)[0])  # % per PRIMARY_H block
        # variance scaling 3h -> 1h
        scale = math.sqrt(1.0 / float(PRIMARY_H))
        pred_1h_pct = pred_pct_primary * scale
        print(f"[PRED] {sym}: {pred_pct_primary:.3f}% per {PRIMARY_H}h → {pred_1h_pct:.3f}% per 1h")
        return pred_1h_pct
    except Exception as e:
        print(f"[ERR] predict {sym}: {e}")
        return 0.0

# =================== POSITION POLICY (continuous) ===================
def update_pos_ema(prev, new, hl):
    if hl <= 1: return new
    alpha = 1.0 - 0.5 ** (1.0 / float(hl))
    return alpha * new + (1.0 - alpha) * prev

def target_position_from_pred(pred_pct_live, band_R, hl, sym, state):
    """
    Map live predicted block return (%) -> target position in [-1, +1].
    Applies EMA smoothing, Δ-position cap, and optional force-min exposure.
    """
    # 1) raw signal -> clipped
    s = float(pred_pct_live) / max(1e-9, band_R)   # e.g., pred 0.8%, band_R=0.8 -> s=1.0
    s = max(-1.0, min(1.0, s))

    # 2) EMA smoothing
    prev = state["pos_ema"].get(sym, 0.0)
    pos_smooth = update_pos_ema(prev, s, hl)

    # 3) Δ-position cap
    delta = pos_smooth - prev
    if   delta >  DPOS_CAP: pos_smooth = prev + DPOS_CAP
    elif delta < -DPOS_CAP: pos_smooth = prev - DPOS_CAP

    # 4) Optional exposure floor (ensures a trade this block)
    if FORCE_TRADE:
        if abs(pos_smooth) < FORCE_MIN_POS:
            sgn = 1.0 if s >= 0.0 else -1.0   # fall back to long if s==0
            pos_smooth = sgn * FORCE_MIN_POS

    # 5) persist & return
    state["pos_ema"][sym] = pos_smooth
    return pos_smooth


# =================== MAIN LOOP ===================
def run_session(api):
    state = load_state()
    today = now_ny().strftime("%Y-%m-%d")
    if state.get("last_day") != today:
        state = {"pos_ema": {}, "last_day": today}
        save_state(state)

    acct = api.get_account()
    print(f"[ACCT] status={acct.status} equity=${acct.equity} bp=${acct.buying_power}")

    # anchor at 10:00 ET (or next whole hour)
    block_start = now_ny().replace(hour=SESSION_START_H, minute=0, second=0, microsecond=0)
    if now_ny() > block_start:
        t = now_ny()
        block_start = t.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)

    for b in range(SESSION_BLOCKS):
        block_end = block_start + dt.timedelta(hours=1)

        # wait until block_start
        while now_ny() < block_start:
            time.sleep(5)

        print(f"\n=== BLOCK {b+1}/{SESSION_BLOCKS} {block_start.strftime('%H:%M')}→{block_end.strftime('%H:%M')} ET ===")
        eq = account_equity(api)

        # enter/update per symbol
        for sym in SYMBOLS:
            px   = latest_price(api, sym)
            pred = predict_block_return_pct(api, sym)     # % for next 1h
            target_frac = target_position_from_pred(pred, BAND_R, EMA_HALF_LIFE, sym, state)
            print(f"[PLAN] {sym}: px={px:.2f} pred_1h={pred:.3f}% target_pos={target_frac:+.3f}")
            submit_target(api, sym, target_frac, eq, px)

        save_state(state)

        # heartbeat until end
        last_hb = 0
        while now_ny() < block_end:
            time.sleep(10)
            if time.time() - last_hb >= 60:
                print(f"[HB] {now_ny().strftime('%H:%M:%S %Z')} → {block_end.strftime('%H:%M:%S %Z')}")
                last_hb = time.time()

        # flatten at hour end
        for sym in SYMBOLS:
            flatten(api, sym)

        block_start = block_end

    print("[DONE] session finished.")

# =================== ENTRY ===================
if __name__ == "__main__":
    print("=== START ===", utc_ts())
    try:
        api = REST(KEY_ID, SECRET, BASE_URL, api_version="v2")
        clock = api.get_clock()
        print(f"[CLOCK] open={clock.is_open} next_open={clock.next_open} next_close={clock.next_close}")
        if not clock.is_open:
            print("[INFO] Market closed. Exiting.")
            sys.exit(0)
        run_session(api)
        sys.exit(0)
    except KeyError as e:
        print(f"[FATAL] missing env var: {e}"); sys.exit(2)
    except Exception:
        traceback.print_exc(); sys.exit(2)

