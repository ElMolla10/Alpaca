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
from datetime import datetime, timezone, timedelta

def rfc3339(dtobj: datetime) -> str:
    return dtobj.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# =================== CONFIG / ENV ===================
ALPACA_DATA_FEED = os.environ.get("ALPACA_DATA_FEED", "iex")  # 'iex' for free tiers; 'sip' requires subscription
TZ_NY = pytz.timezone("America/New_York")

BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
KEY_ID   = os.environ["APCA_API_KEY_ID"]
SECRET   = os.environ["APCA_API_SECRET_KEY"]

# Trading session: 10:00 → 16:00 ET in 1h blocks
SESSION_START_H = int(os.environ.get("SESSION_START_H", "10"))
SESSION_BLOCKS  = int(os.environ.get("SESSION_BLOCKS", "6"))  # 6 blocks → 10..16

# Live policy params (match backtest)
# =================== CONFIG (TUNABLE) ===================
PRIMARY_H      = int(os.environ.get("PRIMARY_H", "1"))        # model trained per 3h block
BAND_R         = float(os.environ.get("BAND_R", "0.35"))       # smaller = more sensitive
EMA_HALF_LIFE  = int(os.environ.get("EMA_HL", "1"))           # 1 = no smoothing
DPOS_CAP       = float(os.environ.get("DPOS_CAP", "0.5"))     # allow faster re-hedge
LEVERAGE       = float(os.environ.get("LEVERAGE", "3.0"))
STOP_LOSS_PCT  = float(os.environ.get("STOP_LOSS_PCT", "2.0"))  # unchanged
# --- execution controls / sizing ---
MIN_ABS_POS = float(os.environ.get("MIN_ABS_POS", "0.01"))
MAX_NOTIONAL = float(os.environ.get("MAX_NOTIONAL", "8000"))
USE_NOTIONAL_ORDERS = os.environ.get("USE_NOTIONAL_ORDERS", "1") == "1"


# new optional aggressiveness switches
FORCE_TRADE   = os.environ.get("FORCE_TRADE", "0") == "0"
FORCE_MIN_POS = float(os.environ.get("FORCE_MIN_POS", "0.20"))


TRADE_COST_BPS = float(os.environ.get("TRADE_COST_BPS", "8.0"))
SLIP_BPS       = float(os.environ.get("SLIPPAGE_BPS", "4.0"))

# Order sizing
MAX_NOTIONAL_PER_SYM = float(os.environ.get("MAX_NOTIONAL", "8000"))

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
        tr = api.get_latest_trade(sym, feed=ALPACA_DATA_FEED)
        return float(tr.price)
    except Exception as e:
        print(f"[WARN] latest_trade {sym} feed={ALPACA_DATA_FEED} failed: {e}")
        # Fallback: use last bar close (IEX)
        try:
            now_utc = datetime.now(timezone.utc)
            start_utc = now_utc - timedelta(days=3)
            bars = api.get_bars(
                sym,
                TimeFrame.Minute,
                start=rfc3339(start_utc),
                end=rfc3339(now_utc),
                limit=1,
                adjustment="raw",
                feed=ALPACA_DATA_FEED,
            )
            if bars:
                b = bars[-1]
                return float(getattr(b, "c", np.nan))
        except Exception as e2:
            print(f"[ERR] latest_price fallback {sym}: {e2}")
        return math.nan


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

def submit_target(api, sym, target_pos_frac, equity, px):
    if abs(target_pos_frac) < MIN_ABS_POS:
        print(f"[SKIP] {sym}: |pos|<{MIN_ABS_POS:.3f} (tiny target)")
        return

    notional = min(MAX_NOTIONAL, abs(target_pos_frac) * float(equity))
    side = "buy" if target_pos_frac > 0 else "sell"

    try:
        if USE_NOTIONAL_ORDERS:
            # SIMPLE FRACTIONAL NOTIONAL MARKET ORDER
            api.submit_order(
                symbol=sym,
                notional=round(notional, 2),
                side=side,
                type="market",
                time_in_force="day"
            )
            print(f"[ORDER] {sym} {side.upper()} notional ${notional:,.2f} (simple)")
        else:
            # SIMPLE WHOLE-SHARE MARKET ORDER
            qty = max(1, int(notional // max(px, 1e-6)))
            api.submit_order(
                symbol=sym,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
            print(f"[ORDER] {sym} {side.upper()} qty {qty} (simple)")

    except Exception as e:
        print(f"[ORDER_ERR] {sym}: {e}")


# =================== LIVE FEATURE PIPELINE (identical to training) ===================
with open(FEATS_PATH, "r") as f:
    FEAT_COLS = json.load(f)

def _fetch_hour_bars(api, sym, end_utc=None, hours=HRS_LOOKBACK) -> pd.DataFrame:
    if end_utc is None:
        end_utc = dt.datetime.now(dt.timezone.utc)
    start_utc = end_utc - dt.timedelta(hours=hours)
    bars = api.get_bars(
    sym,
    TimeFrame.Hour,
    start=rfc3339(start_utc),
    end=rfc3339(now_utc),
    adjustment="raw",
    limit=1000,
)
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

def fetch_recent_features(api, sym, lookback_days: int = 30):
    # pull last ~30 days of hourly bars
    try:
        bars = api.get_bars(
            sym,
            TimeFrame.Hour,
            start=rfc3339(start_utc),
            end=rfc3339(now_utc),
            adjustment="raw",
            limit=1000,
            feed=ALPACA_DATA_FEED,
        )
    except Exception as e:
        print(f"[WARN] {sym}: get_bars feed={ALPACA_DATA_FEED} failed: {e}")
        if ALPACA_DATA_FEED.lower() != "iex":
            # fallback to iex automatically
            try:
                bars = api.get_bars(
                    sym,
                    TimeFrame.Hour,
                    start=rfc3339(start_utc),
                    end=rfc3339(now_utc),
                    adjustment="raw",
                    limit=1000,
                    feed="iex",
                )
                print(f"[INFO] {sym}: fell back to feed=iex")
            except Exception as e2:
                print(f"[ERR] {sym}: get_bars failed even with iex: {e2}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

# =================== MODEL (Booster matching training) ===================
booster = xgb.Booster()
booster.load_model(MODEL_PATH)
import json
FEATS_PATH = os.environ.get("FEATS_PATH", "app/feat_cols.json")
with open(FEATS_PATH, "r") as f:
    FEAT_COLS = json.load(f)
print(f"[INIT] loaded {len(FEAT_COLS)} feat cols")

def predict_block_return_pct(api, sym):
    """
    Returns predicted next-1h return in percent.
    Works with XGBClassifier (probability) or XGBRegressor (direct).
    Prints strong debug to diagnose feature flow.
    """
    try:
        df = fetch_recent_features(api, sym)  # must return a DataFrame
        if df is None or df.empty:
            print(f"[WARN] {sym}: no features returned")
            return 0.0

        # Ensure columns exist & align to training order
        missing = [c for c in FEAT_COLS if c not in df.columns]
        if missing:
            print(f"[ERR] {sym}: missing features: {missing[:8]}{'...' if len(missing)>8 else ''}")
            return 0.0

        X = df[FEAT_COLS].astype(float).iloc[-1:].values
        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else "n/a"
        print(f"[DBG] {sym}: feature row OK shape={X.shape} last_ts={last_ts}")

        # Inference
        if hasattr(model, "predict_proba"):
            p_up = float(model.predict_proba(X)[0, 1])
            pred_pct = (p_up - 0.5) * 200.0   # map prob to signed %
            print(f"[DBG] {sym}: p_up={p_up:.3f} -> pred={pred_pct:.3f}%")
            return pred_pct
        else:
            raw = float(model.predict(X)[0])
            # If your model outputs normalized return, you’d rescale here.
            print(f"[DBG] {sym}: reg_pred={raw:.4f} -> pred={raw:.3f}%")
            return raw

    except Exception as e:
        print(f"[ERR] predict {sym}: {e}")
        return 0.0


# =================== POSITION POLICY (continuous) ===================
def update_pos_ema(prev, new, hl):
    if hl <= 1: return new
    alpha = 1.0 - 0.5 ** (1.0 / float(hl))
    return alpha * new + (1.0 - alpha) * prev

def target_position_from_pred(pred_pct_live, band_R, hl, sym, state):
    s = float(pred_pct_live) / max(1e-9, band_R)
    s = max(-1.0, min(1.0, s))

    prev = state["pos_ema"].get(sym, 0.0)
    pos_smooth = update_pos_ema(prev, s, hl)

    delta = pos_smooth - prev
    if   delta >  DPOS_CAP: pos_smooth = prev + DPOS_CAP
    elif delta < -DPOS_CAP: pos_smooth = prev - DPOS_CAP

    if FORCE_TRADE and abs(pos_smooth) < FORCE_MIN_POS:
        sgn = 1.0 if s >= 0.0 else -1.0
        pos_smooth = sgn * FORCE_MIN_POS

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

    # session window today
    t_now = now_ny()
    session_open  = t_now.replace(hour=SESSION_START_H, minute=0, second=0, microsecond=0)
    session_close = t_now.replace(hour=16, minute=0, second=0, microsecond=0)

    # start immediately if already past session_open; else start at session_open
    block_start = t_now if t_now >= session_open else session_open
    block_end   = block_start + dt.timedelta(hours=1)

    for b in range(SESSION_BLOCKS):
        # stop if we'd pass the close
        if block_start >= session_close:
            print("[INFO] Reached session close window; stopping.")
            break

        # clamp last block to end at session_close if needed
        if block_end > session_close:
            block_end = session_close

        print(f"\n=== BLOCK {b+1}/{SESSION_BLOCKS} {block_start.strftime('%H:%M')}→{block_end.strftime('%H:%M')} ET ===")
        print(f"[TIMECHK] now={now_ny().strftime('%H:%M:%S %Z')} block_end={block_end.strftime('%H:%M:%S %Z')}", flush=True)

        eq = account_equity(api)

        # trade immediately at block_start
        for sym in SYMBOLS:
            try:
                print(f"[DBG] fetching & predicting {sym} ...", flush=True)
                px   = latest_price(api, sym)
                pred = predict_block_return_pct(api, sym)  # % for next 1h
                if pred == 0.0:
                    print(f"[WARN] {sym}: prediction is 0.0 (check features)")
                target_frac = target_position_from_pred(pred, BAND_R, EMA_HALF_LIFE, sym, state)
                print(f"[PLAN] {sym}: px={px:.2f} pred_1h={pred:.3f}% target_pos={target_frac:+.3f}")
                submit_target(api, sym, target_frac, eq, px)
            except Exception as e:
                print(f"[ERR] symbol {sym}: {e}", flush=True)

        save_state(state)

        # wait until block_end with a heartbeat
        last_hb = 0
        while now_ny() < block_end:
            time.sleep(10)
            if time.time() - last_hb >= 60:
                print(f"[HB] {now_ny().strftime('%H:%M:%S %Z')} → {block_end.strftime('%H:%M:%S %Z')}", flush=True)
                last_hb = time.time()

        # flatten at the end of the block
        print("[EXIT] flattening hour positions...", flush=True)
        for sym in SYMBOLS:
            flatten(api, sym)

        # advance to next block
        block_start = block_end
        block_end   = block_start + dt.timedelta(hours=1)


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

