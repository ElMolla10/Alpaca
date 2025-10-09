#!/usr/bin/env python3
import os, sys, time, math, json, traceback
import datetime as dt
from datetime import datetime, timezone
import pytz
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
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
# --- dynamic sizing knobs ---
BAND_R            = float(os.environ.get("BAND_R", "1.10"))   # % per 1h that maps to |pos|=1
POS_EXP           = float(os.environ.get("POS_EXP", "1.5"))   # convexity: 1=linear, >1 emphasizes strong signals
EMA_HALF_LIFE     = int(os.environ.get("EMA_HL", "8"))        # smoothing of target position
DPOS_CAP          = float(os.environ.get("DPOS_CAP", "0.10")) # max change per block
MAX_POS           = float(os.environ.get("MAX_POS", "0.75"))  # absolute cap on |position|
MIN_ABS_POS       = float(os.environ.get("MIN_ABS_POS", "0.02"))  # below this → 0
USE_EQUITY_SIZING = os.environ.get("USE_EQUITY_SIZING", "1") == "1"  # size from equity
PER_SYM_GROSS_CAP = float(os.environ.get("PER_SYM_GROSS_CAP", "0.05"))  # 5% of equity at |pos|=1
BASE_NOTIONAL_PER_TRADE  = float(os.environ.get("BASE_NOTIONAL_PER_TRADE", "3000"))  # baseline $ per trade at |pos|=1.0
MAX_NOTIONAL      = float(os.environ.get("MAX_NOTIONAL", "10000"))  # hard $ cap per symbol
PER_TRADE_NOTIONAL_CAP_F = float(os.environ.get("PER_TRADE_NOTIONAL_CAP_F", "0.75")) # cap per name as fraction of equity (e.g. 0.25 = 25%)
USE_NOTIONAL_ORDERS = os.environ.get("USE_NOTIONAL_ORDERS", "1") == "1"  # 1 → use notional $, 0 → use share qty
SHORTS_ENABLED = os.environ.get("SHORTS_ENABLED", "1") == "1"  # 1=allow shorts





# new optional aggressiveness switches
FORCE_TRADE   = os.environ.get("FORCE_TRADE", "0") == "1"
FORCE_MIN_POS = float(os.environ.get("FORCE_MIN_POS", "0.20"))


TRADE_COST_BPS = float(os.environ.get("TRADE_COST_BPS", "8.0"))
SLIP_BPS       = float(os.environ.get("SLIPPAGE_BPS", "4.0"))

# Order sizing
MAX_NOTIONAL_PER_SYM = float(os.environ.get("MAX_NOTIONAL", "10000"))

# Symbols
TEST_MODE  = os.environ.get("TEST_MODE", "0") == "1"
symbols_env = os.environ.get("SYMBOLS", "AAPL,MSFT,NVDA,AMD,JPM,GS,XOM,CVX,PG,KO")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
print(f"[INIT] SYMBOLS={SYMBOLS}")

# Model & features
MODEL_PATH = os.environ.get("MODEL_PATH", "app/model/XGboost_model.json")
FEATS_PATH = os.environ.get("FEATS_PATH", "app/feat_cols.json")

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

# (optional) confirm feed choice in logs
print(f"[INIT] using data feed={ALPACA_DATA_FEED}")

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

# ---------- add this helper (near top-level utils) ----------
def ensure_market_open_or_wait(api):
    """Start immediately if market is open; otherwise wait until next open."""
    clock = api.get_clock()
    if clock.is_open:
        print("[INFO] Market is open. Starting trading immediately.")
        return

    ny = pytz.timezone("America/New_York")
    open_time_ny = clock.next_open.astimezone(ny)
    now_ny = dt.datetime.now(ny)
    wait_sec = max(0, (open_time_ny - now_ny).total_seconds())
    hrs = wait_sec / 3600.0
    print(f"[WAIT] Market closed. Waiting {hrs:.2f} hours until next open ({open_time_ny}).")

    # Sleep in chunks to avoid very long single sleeps
    while wait_sec > 0:
        chunk = min(900, wait_sec)  # 15-minute chunks
        time.sleep(chunk)
        wait_sec -= chunk

    print("[INFO] Market is now open. Starting session.")


# =================== ALPACA ===================
def latest_price(api, sym):
    try:
        tr = api.get_latest_trade(sym, feed=ALPACA_DATA_FEED)
        return float(tr.price)
    except Exception as e:
        print(f"[WARN] latest_trade {sym} feed={ALPACA_DATA_FEED} failed: {e}")
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
def linear_notional_from_posfrac(pos_frac: float) -> float:
    """
    Linear map: BASE + (MAX-BASE)*|pos|, clipped to [BASE, MAX].
    Guarantees no rounding down below the $3,000 floor when a trade is taken.
    """
    mag = min(max(abs(float(pos_frac)), 0.0), 1.0)
    raw = BASE_NOTIONAL_PER_TRADE + (MAX_NOTIONAL - BASE_NOTIONAL_PER_TRADE) * mag
    return float(min(MAX_NOTIONAL, max(BASE_NOTIONAL_PER_TRADE, raw)))

def gross_exposure_and_limit(api) -> tuple[float, float]:
    acct = api.get_account()
    equity = float(getattr(acct, "last_equity", acct.equity))
    limit = 5.0 * equity    # LEVERAGE = 5
    gross = 0.0
    try:
        for p in api.list_positions():
            gross += abs(float(p.market_value))
    except Exception:
        pass
    return gross, limit

def submit_target(api, sym, target_pos_frac, equity, last_px):
    """
    target_pos_frac: [-1..+1] desired fraction per name (our “position”), signed.
    equity: current account equity in dollars
    last_px: latest trade price
    """
    try:
        # --- tiny-position gate (floor only if FORCE_TRADE) ---
        pos_frac = float(target_pos_frac)
        if abs(pos_frac) < MIN_ABS_POS:
            if FORCE_TRADE and pos_frac != 0.0:
                pos_frac = math.copysign(max(MIN_ABS_POS, FORCE_MIN_POS), pos_frac)
                print(f"[FORCE] {sym}: bumping |pos| to {abs(pos_frac):.3f}")
            else:
                print(f"[SKIP] {sym}: |pos|={abs(pos_frac):.3f} < MIN_ABS_POS={MIN_ABS_POS:.3f}")
                return

        # --- shorting policy ---
        if pos_frac < 0 and not SHORTS_ENABLED:
            print(f"[SKIP] {sym}: shorts disabled")
            return

        # --- linear sizing: BASE → MAX by |pos_frac|, no rounding down ---
        tgt_notional = linear_notional_from_posfrac(pos_frac)  # uses BASE_NOTIONAL_PER_TRADE & MAX_NOTIONAL

        # floor enforcement only when forced & nonzero signal (redundant with linear map, kept for clarity)
        if tgt_notional < BASE_NOTIONAL_PER_TRADE:
            if FORCE_TRADE and pos_frac != 0.0:
                print(f"[FORCE] {sym}: lifting notional → {BASE_NOTIONAL_PER_TRADE:.2f}")
                tgt_notional = BASE_NOTIONAL_PER_TRADE
            else:
                print(f"[SKIP] {sym}: notional {tgt_notional:.2f} < BASE_NOTIONAL_PER_TRADE {BASE_NOTIONAL_PER_TRADE:.2f}")
                return

        side = "buy" if pos_frac > 0 else "sell"
        signed_notional = tgt_notional if pos_frac > 0 else -tgt_notional

        # --- portfolio gross-exposure guard (≤ LEVERAGE × equity) ---
        gross, limit = gross_exposure_and_limit(api)
        headroom = max(0.0, limit - gross)
        if headroom <= 1.0:
            print(f"[GUARD] gross {gross:,.0f} ≈ limit {limit:,.0f}; skip {sym}")
            return
        if abs(signed_notional) > headroom:
            print(f"[CLIP] {sym}: clip {abs(signed_notional):,.0f} → {headroom:,.0f} by leverage guard")
            signed_notional = math.copysign(headroom, signed_notional)

        # --- place orders ---
        if side == "sell" and SHORTS_ENABLED:
            # Block fractional shorts: convert to integer qty
            if last_px is None or last_px <= 0 or math.isnan(last_px):
                print(f"[SKIP] {sym}: invalid price for short")
                return
            qty_int = int(abs(signed_notional) / last_px)
            if qty_int <= 0:
                print(f"[SKIP] {sym}: short qty rounds to 0 (fractional short blocked)")
                return
            est_val = qty_int * last_px
            print(f"[ORDER] {sym} SELL qty={qty_int} (~${est_val:,.2f}) |pos|={abs(pos_frac):.3f}")
            api.submit_order(
                symbol=sym,
                qty=qty_int,
                side="sell",
                type="market",
                time_in_force="day"
            )
        else:
            # Longs use notional → fractional shares allowed
            notional_abs = abs(signed_notional)
            if notional_abs < 1.0:
                print(f"[SKIP] {sym}: notional too small")
                return
            print(f"[ORDER] {sym} {side.upper()} notional ${notional_abs:,.2f} |pos|={abs(pos_frac):.3f}")
            api.submit_order(
                symbol=sym,
                notional=notional_abs,
                side=side,
                type="market",
                time_in_force="day"
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

def fetch_recent_features(api: REST, sym: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Pull last ~30 days of HOURLY bars from Alpaca (IEX feed) and compute the same
    features as training. Returns DataFrame with 'timestamp' (UTC) + FEAT_COLS.
    """
    # --- define window first ---
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)

    # --- bars fetch with IEX feed (avoid SIP) ---
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

    if not bars:
        print(f"[WARN] {sym}: no hourly bars")
        return pd.DataFrame()

    # --- build DataFrame ---
    rows = []
    for b in bars:
        ts = b.t if isinstance(b.t, datetime) else pd.to_datetime(b.t, utc=True)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        rows.append({
            "timestamp": ts,
            "open": float(getattr(b, "o", np.nan)),
            "high": float(getattr(b, "h", np.nan)),
            "low":  float(getattr(b, "l", np.nan)),
            "close": float(getattr(b, "c", np.nan)),
            "volume": float(getattr(b, "v", np.nan)),
            "vwap": float(getattr(b, "vw", np.nan)) if hasattr(b, "vw") else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # --- filter to 10:00–15:59 ET (match training) ---
    ts_et = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.loc[ts_et.dt.hour.between(10, 15)].reset_index(drop=True)
    if df.empty:
        print(f"[WARN] {sym}: bars after RTH filter are empty")
        return df

    # --- base features ---
    df["price_change_pct"] = df["close"].pct_change() * 100.0
    macd_ind = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACDh_12_26_9"] = macd_ind.macd_diff()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2.0)
    df["BBM_20_2.0"] = bb.bollinger_mavg()

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

    df = df.dropna().reset_index(drop=True)

    # ensure all FEAT_COLS exist (fill missing with 0.0 to avoid crashes)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df = df[["timestamp"] + [c for c in FEAT_COLS if c in df.columns]]
    return df


def update_pos_ema(prev: float, target: float, half_life: int) -> float:
    if half_life <= 1:
        return target
    alpha = 1.0 - 0.5 ** (1.0 / float(half_life))
    return alpha * target + (1.0 - alpha) * prev

def map_signal_to_pos(pred_pct: float, band_R: float, exp_: float) -> float:
    """
    Convert predicted % to target raw position in [-1,1].
    - Normalize by band_R (percent at which |pos|=1 before convexity)
    - Apply convexity: |s|**exp (keeps sign)
    - Clip to [-1,1]
    """
    s = float(pred_pct) / max(1e-12, band_R)     # normalize
    s = max(-10.0, min(10.0, s))                 # guard
    mag = abs(s) ** exp_
    pos = (1.0 if s >= 0 else -1.0) * min(1.0, mag)
    return pos

def compute_dynamic_notional(equity: float, pos_frac: float) -> float:
    """
    Compute dynamic notional size with a minimum floor and cap.
    - Starts at BASE_NOTIONAL_PER_TRADE even for weak signals.
    - Scales up with |pos_frac| (prediction strength).
    - Never exceeds MAX_NOTIONAL.
    """
    if USE_EQUITY_SIZING:
        dynamic_size = float(equity) * PER_SYM_GROSS_CAP * abs(pos_frac)
    else:
        dynamic_size = BASE_NOTIONAL_PER_TRADE * abs(pos_frac)

    # enforce min = BASE_NOTIONAL_PER_TRADE and cap = MAX_NOTIONAL
    notional = max(BASE_NOTIONAL_PER_TRADE, dynamic_size)
    notional = min(MAX_NOTIONAL, notional)

    return float(notional)


def asset_caps(api, sym):
    try:
        a = api.get_asset(sym)
        return bool(a.tradable), bool(a.fractionable), bool(a.shortable), bool(getattr(a, "easy_to_borrow", False))
    except Exception:
        return True, True, False, False  # be conservative if lookup fails




# =================== MODEL (Booster matching training) ===================
booster = xgb.Booster()
booster.load_model(MODEL_PATH)
import json
FEATS_PATH = os.environ.get("FEATS_PATH", "app/feat_cols.json")
with open(FEATS_PATH, "r") as f:
    FEAT_COLS = json.load(f)
print(f"[INIT] loaded {len(FEAT_COLS)} feat cols")

def predict_block_return_pct(api: REST, sym: str) -> float:
    """
    Predict next-1h return (%) using xgboost.Booster + live features.
    """
    try:
        df = fetch_recent_features(api, sym)
        if df is None or df.empty:
            print(f"[WARN] {sym}: no features (empty df)")
            return 0.0

        # build latest row in training order
        X = df[FEAT_COLS].astype(float).iloc[-1:].values
        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else "n/a"
        print(f"[DBG] {sym}: feature row OK shape={X.shape} last_ts={last_ts}")

        # booster inference
        dmat = xgb.DMatrix(X, feature_names=FEAT_COLS)
        pred = float(booster.predict(dmat)[0])   # assumes model outputs % per 1h
        print(f"[DBG] {sym}: booster_pred={pred:.4f}%")
        return pred

    except Exception as e:
        print(f"[ERR] predict {sym}: {e}")
        return 0.0


# =================== POSITION POLICY (continuous) ===================
def update_pos_ema(prev, new, hl):
    if hl <= 1: return new
    alpha = 1.0 - 0.5 ** (1.0 / float(hl))
    return alpha * new + (1.0 - alpha) * prev

def target_position_from_pred(pred_pct_live: float, band_R: float, hl: int, sym: str, state: dict) -> float:
    # map prediction → raw desired position
    raw = map_signal_to_pos(pred_pct_live, band_R, POS_EXP)

    # EMA smooth + delta cap
    prev = state["pos_ema"].get(sym, 0.0)
    pos  = update_pos_ema(prev, raw, hl)
    delta = pos - prev
    if   delta >  DPOS_CAP: pos = prev + DPOS_CAP
    elif delta < -DPOS_CAP: pos = prev - DPOS_CAP

    # global cap & tiny cutoff
    pos = max(-MAX_POS, min(MAX_POS, pos))
    if abs(pos) < MIN_ABS_POS:
        pos = 0.0

    state["pos_ema"][sym] = pos
    return pos




# =================== MAIN LOOP ===================
def run_session(api):
    state = load_state()
    today = now_ny().strftime("%Y-%m-%d")
    if state.get("last_day") != today:
        state = {"pos_ema": {}, "last_day": today}
        save_state(state)

    acct = api.get_account()
    print(f"[ACCT] status={acct.status} equity=${acct.equity} bp=${acct.buying_power}")

    ensure_market_open_or_wait(api)

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

        if clock.is_open:
            print("[INFO] Market is open. Starting trading immediately.")
            run_session(api)
            sys.exit(0)
        else:
            # Wait until the official next market open from Alpaca, then start.
            now_utc = dt.datetime.now(dt.timezone.utc)
            open_utc = clock.next_open.astimezone(dt.timezone.utc)
            wait_sec = max(0, (open_utc - now_utc).total_seconds())
            hrs = wait_sec / 3600.0
            print(f"[WAIT] Market closed. Waiting {hrs:.2f} hours until next open ({open_utc.isoformat()}).")

            # Sleep in chunks to keep logs alive during long waits
            remaining = wait_sec
            while remaining > 0:
                chunk = min(300, remaining)  # 5-minute chunks
                time.sleep(chunk)
                remaining -= chunk
                left_hrs = remaining / 3600.0
                print(f"[WAIT] ... {left_hrs:.2f} hours remaining until market open.")

            print("[INFO] Market is now open. Starting session.")
            run_session(api)
            sys.exit(0)

    except KeyError as e:
        print(f"[FATAL] missing env var: {e}")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(2)

