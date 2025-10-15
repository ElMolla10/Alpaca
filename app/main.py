#!/usr/bin/env python3
import os, sys, time, math, json, traceback
import datetime as dt
from datetime import datetime, timezone, timedelta
import pytz
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import xgboost as xgb
from ta.trend import MACD
from ta.volatility import BollingerBands
from decimal import Decimal, ROUND_DOWN
import re

# ==================================================
# Helpers
# ==================================================
def rfc3339(dtobj: datetime) -> str:
    return dtobj.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

TZ_NY = pytz.timezone("America/New_York")
def now_ny() -> dt.datetime:
    return dt.datetime.now(TZ_NY)

def is_friday_ny(dt_obj: datetime | None = None):
    """Return tuple (is_friday_bool, ny_hour_int)."""
    _now = (dt_obj or dt.datetime.now(dt.timezone.utc)).astimezone(TZ_NY)
    return (_now.weekday() == 4, _now.hour)

# ==================================================
# CONFIG / ENV
# ==================================================
ALPACA_DATA_FEED = os.environ.get("ALPACA_DATA_FEED", "iex")  # 'iex' for free tiers; 'sip' requires subscription
BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
KEY_ID   = os.environ["APCA_API_KEY_ID"]
SECRET   = os.environ["APCA_API_SECRET_KEY"]

# Trading session: 10:00 → 16:00 ET in 1h blocks
SESSION_START_H = int(os.environ.get("SESSION_START_H", "10"))
SESSION_BLOCKS  = int(os.environ.get("SESSION_BLOCKS", "6"))  # 6 blocks → 10..16

# Live policy params (match backtest)
PRIMARY_H          = int(os.environ.get("PRIMARY_H", "1"))
BAND_R             = float(os.environ.get("BAND_R", "1.10"))
POS_EXP            = float(os.environ.get("POS_EXP", "1.5"))
EMA_HALF_LIFE      = int(os.environ.get("EMA_HL", "12"))
DPOS_CAP           = float(os.environ.get("DPOS_CAP", "0.05"))
SIGN_MULT          = float(os.environ.get("SIGN_MULT", "-1.0"))     # flip sign if CV showed inversion
REBALANCE_BAND     = float(os.environ.get("REBALANCE_BAND", "0.02"))

# Per-symbol multipliers
PER_SYMBOL_SIZE_MULT = {
    "GS": 1.00, "PG": 1.00, "JPM": 1.00, "CVX": 1.00, "AAPL": 1.00,
    "MSFT": 0.90,
    "NVDA": 0.50, "AMD": 0.50, "KO": 0.50, "XOM": 0.50,
}

# End-of-week & risk caps
FRIDAY_FORCE_FLATTEN      = os.environ.get("FRIDAY_FORCE_FLATTEN", "0") == "1"
FRIDAY_FLATTEN_AFTER_H    = int(os.environ.get("FRIDAY_FLATTEN_AFTER_H", "15"))
FRIDAY_EXPOSURE_CAP       = float(os.environ.get("FRIDAY_EXPOSURE_CAP", "0.60"))
FRIDAY_VOL_LOOKBACK_MIN   = int(os.environ.get("FRIDAY_VOL_LOOKBACK_MIN", "90"))
MAX_POS                   = float(os.environ.get("MAX_POS", "0.75"))
MIN_ABS_POS               = float(os.environ.get("MIN_ABS_POS", "0.02"))
USE_EQUITY_SIZING         = os.environ.get("USE_EQUITY_SIZING", "1") == "1"
PER_SYM_GROSS_CAP         = float(os.environ.get("PER_SYM_GROSS_CAP", "0.05"))
BASE_NOTIONAL_PER_TRADE   = float(os.environ.get("BASE_NOTIONAL_PER_TRADE", "3000"))
MAX_NOTIONAL              = float(os.environ.get("MAX_NOTIONAL", "10000"))
PER_TRADE_NOTIONAL_CAP_F  = float(os.environ.get("PER_TRADE_NOTIONAL_CAP_F", "0.75"))
USE_NOTIONAL_ORDERS       = os.environ.get("USE_NOTIONAL_ORDERS", "1") == "1"
SHORTS_ENABLED            = os.environ.get("SHORTS_ENABLED", "1") == "1"

# Friday de-risking knobs
FRIDAY_SIZE_MULT_DAY        = float(os.environ.get("FRIDAY_SIZE_MULT_DAY", "0.75"))
FRIDAY_SIZE_MULT_LATE       = float(os.environ.get("FRIDAY_SIZE_MULT_LATE", "0.50"))
FRIDAY_LATE_CUTOFF_H        = int(os.environ.get("FRIDAY_LATE_CUTOFF_H", "14"))
FRIDAY_BLOCK_NEW_AFTER_LATE = os.environ.get("FRIDAY_BLOCK_NEW_AFTER_LATE", "1") == "1"
FRIDAY_MIN_POS              = float(os.environ.get("FRIDAY_MIN_POS", "0.03"))

# New-trade cutoff before close
TRADE_CUTOFF_MIN_BEFORE_CLOSE = int(os.environ.get("TRADE_CUTOFF_MIN_BEFORE_CLOSE", "30"))

# Optional aggressiveness
FORCE_TRADE   = os.environ.get("FORCE_TRADE", "0") == "1"
FORCE_MIN_POS = float(os.environ.get("FORCE_MIN_POS", "0.20"))

TRADE_COST_BPS = float(os.environ.get("TRADE_COST_BPS", "8.0"))
SLIP_BPS       = float(os.environ.get("SLIPPAGE_BPS", "4.0"))
MAX_NOTIONAL_PER_SYM = float(os.environ.get("MAX_NOTIONAL", "10000"))

# Symbols
TEST_MODE  = os.environ.get("TEST_MODE", "0") == "1"
symbols_env = os.environ.get("SYMBOLS", "AAPL,MSFT,NVDA,AMD,JPM,GS,XOM,CVX,PG,KO")
SYMBOLS = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
print(f"[INIT] SYMBOLS={SYMBOLS}")

# Model & features
MODEL_PATH = os.environ.get("MODEL_PATH", "app/model/XGboost_model.json")
FEATS_PATH = os.environ.get("FEATS_PATH", "app/feat_cols.json")

# (optional) confirm feed choice in logs
print(f"[INIT] using data feed={ALPACA_DATA_FEED}")

# Internal constants for feature build
CLIP_HOURLY_RET = 8.0
HRS_LOOKBACK    = 500

# Persist EMA state across blocks
STATE_PATH = os.environ.get("STATE_PATH", "/tmp/live_state.json")

# ==================================================
# State utils
# ==================================================
def load_state():
    try:
        with open(STATE_PATH, "r") as f:
            st = json.load(f)
    except Exception:
        st = {"pos_ema": {}, "hold_timer": {}, "last_day": ""}

    if "pos_ema" not in st:      st["pos_ema"] = {}
    if "hold_timer" not in st:   st["hold_timer"] = {}
    if "last_day" not in st:     st["last_day"] = ""
    return st

def save_state(st):
    try:
        with open(STATE_PATH, "w") as f: json.dump(st, f)
    except Exception:
        pass

# ==================================================
# Market open handling
# ==================================================
def ensure_market_open_or_wait(api):
    """Start immediately if market is open; otherwise wait until next open."""
    clock = api.get_clock()
    if clock.is_open:
        print("[INFO] Market is open. Starting trading immediately.")
        return

    ny = pytz.timezone("America/New_York")
    open_time_ny = clock.next_open.astimezone(ny)
    now_ny_dt = dt.datetime.now(ny)
    wait_sec = max(0, (open_time_ny - now_ny_dt).total_seconds())
    hrs = wait_sec / 3600.0
    print(f"[WAIT] Market closed. Waiting {hrs:.2f} hours until next open ({open_time_ny}).")

    while wait_sec > 0:
        chunk = min(900, wait_sec)  # 15-minute chunks
        time.sleep(chunk)
        wait_sec -= chunk

    print("[INFO] Market is now open. Starting session.")

# ==================================================
# Alpaca helpers
# ==================================================
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

def realized_vol_last_minutes(api, minutes=90) -> float:
    try:
        end_utc = dt.datetime.now(dt.timezone.utc)
        start_utc = end_utc - dt.timedelta(minutes=minutes+10)
        bars = api.get_bars("SPY", TimeFrame.Minute, start=rfc3339(start_utc), end=rfc3339(end_utc),
                            limit=minutes+10, feed=ALPACA_DATA_FEED)
        if not bars: return 0.0
        px = np.array([float(b.c) for b in bars], float)
        r = np.diff(np.log(px)) * 100.0  # % log-returns
        if r.size < 10: return 0.0
        return float(np.std(r, ddof=1))  # % per minute (rough)
    except Exception:
        return 0.0

def should_flatten_eow(api, equity: float, is_fri: bool, ny_hour: int) -> bool:
    if not is_fri or ny_hour < FRIDAY_FLATTEN_AFTER_H:
        return False
    if FRIDAY_FORCE_FLATTEN:
        return True
    gross, _limit = gross_exposure_and_limit(api)
    gross_frac = gross / max(1e-9, equity)
    if gross_frac > FRIDAY_EXPOSURE_CAP:
        print(f"[EOW] gross {gross_frac:.2f} > cap {FRIDAY_EXPOSURE_CAP:.2f}")
        return True
    vol_pm = realized_vol_last_minutes(api, FRIDAY_VOL_LOOKBACK_MIN)
    if vol_pm > 0.20:
        print(f"[EOW] realized vol {vol_pm:.3f}%/min high → flatten")
        return True
    return False

# ==================================================
# Order placement with robust fallback
# ==================================================
def submit_target(api, sym, target_pos_frac, equity, last_px):
    """
    target_pos_frac: [-1..+1] desired fraction per name (our “position”), signed.
    equity: current account equity in dollars
    last_px: latest trade price
    """
    def _round_to_cents(x: float) -> float:
        return float(Decimal(str(max(0.0, x))).quantize(Decimal("0.01"), rounding=ROUND_DOWN))

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

        if pos_frac < 0 and not SHORTS_ENABLED:
            print(f"[SKIP] {sym}: shorts disabled")
            return

        # --- linear sizing: BASE → MAX by |pos_frac| ---
        tgt_notional = linear_notional_from_posfrac(pos_frac)

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
        headroom = _round_to_cents(headroom)
        if headroom <= 1.00:
            print(f"[GUARD] gross {gross:,.2f} ≈ limit {limit:,.2f}; skip {sym}")
            return
        if abs(signed_notional) > headroom:
            print(f"[CLIP] {sym}: clip {abs(signed_notional):,.2f} → {headroom:,.2f} by leverage guard")
            signed_notional = math.copysign(headroom, signed_notional)

        # --- place orders ---
        if side == "sell" and SHORTS_ENABLED:
            if last_px is None or last_px <= 0 or math.isnan(last_px):
                print(f"[SKIP] {sym}: invalid price for short")
                return
            qty_int = int(abs(signed_notional) / last_px)
            if qty_int <= 0:
                print(f"[SKIP] {sym}: short qty rounds to 0 (fractional short blocked)")
                return
            est_val = _round_to_cents(qty_int * last_px)
            print(f"[ORDER] {sym} SELL qty={qty_int} (~${est_val:,.2f}) |pos|={abs(pos_frac):.3f}")
            api.submit_order(
                symbol=sym,
                qty=qty_int,
                side="sell",
                type="market",
                time_in_force="day"
            )
        else:
            notional_abs = _round_to_cents(abs(signed_notional))
            if notional_abs < 1.00:
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
        msg = str(e)
        print(f"[ORDER_ERR] {sym}: {msg}")
        # Fallback: if broker says "insufficient qty available", cap and retry once
        if "insufficient qty available" in msg.lower():
            m = re.search(r"available:\s*([0-9.]+)", msg)
            if m:
                avail = float(m.group(1))
                if side == "sell":
                    retry_qty = max(0, int(math.floor(avail)))
                    if retry_qty > 0:
                        print(f"[RETRY] {sym}: SELL qty→{retry_qty} (cap by available)")
                        try:
                            api.submit_order(symbol=sym, qty=retry_qty, side="sell",
                                             type="market", time_in_force="day")
                        except Exception as e2:
                            print(f"[RETRY_ERR] {sym}: {e2}")
                else:
                    if last_px and last_px > 0:
                        capped_notional = float(Decimal(str(avail * last_px)).quantize(Decimal("0.01"), rounding=ROUND_DOWN))
                        if capped_notional >= 1.00:
                            print(f"[RETRY] {sym}: BUY notional→${capped_notional:,.2f} (cap by available)")
                            try:
                                api.submit_order(symbol=sym, notional=capped_notional, side="buy",
                                                 type="market", time_in_force="day")
                            except Exception as e2:
                                print(f"[RETRY_ERR] {sym}: {e2}")

# ==================================================
# LIVE FEATURE PIPELINE (identical to training)
# ==================================================
with open(FEATS_PATH, "r") as f:
    FEAT_COLS = json.load(f)
print(f"[INIT] loaded {len(FEAT_COLS)} feat cols")

def _fetch_hour_bars(api, sym, end_utc=None, hours=HRS_LOOKBACK) -> pd.DataFrame:
    if end_utc is None:
        end_utc = dt.datetime.now(dt.timezone.utc)
    start_utc = end_utc - dt.timedelta(hours=hours)
    bars = api.get_bars(
        sym,
        TimeFrame.Hour,
        start=rfc3339(start_utc),
        end=rfc3339(end_utc),
        adjustment="raw",
        limit=1000,
        feed=ALPACA_DATA_FEED,
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

def fetch_recent_features(api: REST, sym: str, lookback_days: int = 30) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)
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

    ts_et = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.loc[ts_et.dt.hour.between(10, 15)].reset_index(drop=True)
    if df.empty:
        print(f"[WARN] {sym}: bars after RTH filter are empty")
        return df

    # base features
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

    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df = df[["timestamp"] + [c for c in FEAT_COLS if c in df.columns]]
    return df

# ==================================================
# Model + mapping
# ==================================================
booster = xgb.Booster()
booster.load_model(MODEL_PATH)

def update_pos_ema(prev: float, target: float, half_life: int) -> float:
    if half_life <= 1:
        return target
    alpha = 1.0 - 0.5 ** (1.0 / float(half_life))
    return alpha * target + (1.0 - alpha) * prev

def map_signal_to_pos(pred_pct: float, band_R: float, exp_: float) -> float:
    s = float(pred_pct) / max(1e-12, band_R)
    s = max(-10.0, min(10.0, s))
    mag = abs(s) ** exp_
    pos = (1.0 if s >= 0 else -1.0) * min(1.0, mag)
    return pos

def target_position_from_pred(pred_pct_live: float, band_R: float, hl: int, sym: str, state: dict) -> float:
    raw = map_signal_to_pos(pred_pct_live, band_R, POS_EXP)
    prev = state["pos_ema"].get(sym, 0.0)
    pos  = update_pos_ema(prev, raw, hl)
    delta = pos - prev
    if   delta >  DPOS_CAP: pos = prev + DPOS_CAP
    elif delta < -DPOS_CAP: pos = prev - DPOS_CAP
    pos = max(-MAX_POS, min(MAX_POS, pos))
    if abs(pos) < MIN_ABS_POS:
        pos = 0.0
    state["pos_ema"][sym] = pos
    return pos

def predict_block_return_pct(api: REST, sym: str) -> float:
    try:
        df = fetch_recent_features(api, sym)
        if df is None or df.empty:
            print(f"[WARN] {sym}: no features (empty df)")
            return 0.0
        X = df[FEAT_COLS].astype(float).iloc[-1:].values
        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else "n/a"
        print(f"[DBG] {sym}: feature row OK shape={X.shape} last_ts={last_ts}")
        dmat = xgb.DMatrix(X, feature_names=FEAT_COLS)
        pred = float(booster.predict(dmat)[0])   # % per 1h
        print(f"[DBG] {sym}: booster_pred={pred:.4f}%")
        return pred
    except Exception as e:
        print(f"[ERR] predict {sym}: {e}")
        return 0.0

# ==================================================
# MAIN LOOP
# ==================================================
def run_session(api):
    state = load_state()
    today = now_ny().strftime("%Y-%m-%d")
    if state.get("last_day") != today:
        state = {"pos_ema": {}, "hold_timer": {}, "last_day": today}
        save_state(state)

    acct = api.get_account()
    print(f"[ACCT] status={acct.status} equity=${acct.equity} bp=${acct.buying_power}")

    ensure_market_open_or_wait(api)

    # Build initial window
    t_now = now_ny()
    session_open  = t_now.replace(hour=SESSION_START_H, minute=0, second=0, microsecond=0)
    session_close = t_now.replace(hour=16, minute=0, second=0, microsecond=0)
    block_start   = t_now if t_now >= session_open else session_open

    trade_cutoff = session_close - dt.timedelta(minutes=TRADE_CUTOFF_MIN_BEFORE_CLOSE)
    print(f"[CUTOFF] New trades stop at {trade_cutoff.strftime('%H:%M:%S %Z')} "
          f"({TRADE_CUTOFF_MIN_BEFORE_CLOSE} min before close).")

    b = 0
    while True:
        # Stop scheduling NEW trade blocks once we hit the cutoff window
        if block_start >= trade_cutoff:
            print("[INFO] Reached trade cutoff window; no further new trades will be placed.")
            break

        # Set current block window (clamped to cutoff)
        unclamped_end = block_start + dt.timedelta(hours=1)
        block_end = min(unclamped_end, trade_cutoff)

        secs_left = (trade_cutoff - block_start).total_seconds()
        blocks_left = math.ceil(secs_left / 3600.0)
        b += 1
        print(f"\n=== BLOCK {b}/{blocks_left} {block_start.strftime('%H:%M')}→{block_end.strftime('%H:%M')} ET ===")
        print(f"[TIMECHK] now={now_ny().strftime('%H:%M:%S %Z')} block_end={block_end.strftime('%H:%M:%S %Z')}", flush=True)

        eq = account_equity(api)

        # ---- Friday context (computed once per block) ----
        is_fri, ny_hour = is_friday_ny()
        is_late = is_fri and (ny_hour >= FRIDAY_LATE_CUTOFF_H)
        friday_mult = 1.0
        if is_fri:
            friday_mult = FRIDAY_SIZE_MULT_LATE if is_late else FRIDAY_SIZE_MULT_DAY
            print(f"[FRIDAY] context: hour={ny_hour}, late={is_late}, size_mult={friday_mult:.2f}, "
                  f"block_new_after_late={FRIDAY_BLOCK_NEW_AFTER_LATE}")

        for sym in SYMBOLS:
            try:
                # --- HOLD/CUTOFF GUARDS ---
                if now_ny() >= trade_cutoff:
                    print(f"[CUTOFF] {sym}: within last {TRADE_CUTOFF_MIN_BEFORE_CLOSE} min → blocking NEW entry.")
                    continue
                rem = int(state.get("hold_timer", {}).get(sym, 0))
                if rem > 0:
                    print(f"[HOLD] {sym}: already open; {rem} block(s) remaining. Skipping new order.")
                    continue
                if is_fri and is_late and FRIDAY_BLOCK_NEW_AFTER_LATE and rem == 0:
                    print(f"[FRIDAY] {sym}: after {FRIDAY_LATE_CUTOFF_H}:00 ET → blocking NEW entry.")
                    continue

                print(f"[DBG] fetching & predicting {sym} ...", flush=True)
                px = latest_price(api, sym)
                pred = predict_block_return_pct(api, sym)
                if pred == 0.0:
                    print(f"[WARN] {sym}: prediction is 0.0 (check features)")
                pred *= SIGN_MULT

                # --- Plan position size ---
                target_frac = target_position_from_pred(pred, BAND_R, EMA_HALF_LIFE, sym, state)

                # Per-symbol multiplier
                sym_mult = PER_SYMBOL_SIZE_MULT.get(sym, 1.0)
                if sym_mult != 1.0 and target_frac != 0.0:
                    target_frac *= sym_mult
                    print(f"[SIZE]  {sym}: per-symbol×={sym_mult:.2f} → target_pos={target_frac:+.3f}")

                # Friday reduction
                if friday_mult < 1.0 and target_frac != 0.0:
                    target_frac *= friday_mult
                    print(f"[FRIDAY] {sym}: size×={friday_mult:.2f} → target_pos={target_frac:+.3f}")

                # Friday micro floor
                if is_fri and abs(target_frac) < FRIDAY_MIN_POS:
                    print(f"[FRIDAY] {sym}: |target_pos|={abs(target_frac):.3f} < Friday floor {FRIDAY_MIN_POS:.3f} → skip")
                    continue

                # Skip micro rebalances before submit
                prev_pos = float(state.get("pos_ema", {}).get(sym, 0.0))
                if abs(target_frac - prev_pos) < REBALANCE_BAND:
                    print(f"[SKIP] {sym}: |Δtarget|={abs(target_frac - prev_pos):.3f} < REBALANCE_BAND={REBALANCE_BAND:.3f}")
                    continue

                print(f"[PLAN] {sym}: px={px:.2f} pred_1h={pred:.3f}% target_pos={target_frac:+.3f}")
                submit_target(api, sym, target_frac, eq, px)

                # Determine hold duration by confidence size
                absf = abs(target_frac)
                if absf < 0.25:
                    hold_blocks = 1
                elif absf < 0.50:
                    hold_blocks = 2
                else:
                    hold_blocks = 3
                state.setdefault("hold_timer", {})[sym] = hold_blocks

            except Exception as e:
                print(f"[ERR] {sym}: {e}")
                traceback.print_exc(limit=1)

        save_state(state)

        # Heartbeats until block end
        last_hb = 0
        while now_ny() < block_end:
            time.sleep(10)
            if time.time() - last_hb >= 60:
                print(f"[HB] {now_ny().strftime('%H:%M:%S %Z')} → {block_end.strftime('%H:%M:%S %Z')}", flush=True)
                last_hb = time.time()

        # EOW flatten decision
        if is_fri and should_flatten_eow(api, eq, is_fri, ny_hour):
            print("[EOW] Friday policy active → flattening ALL open positions")
            for sym in SYMBOLS:
                flatten(api, sym)
                state.setdefault("hold_timer", {})[sym] = 0
            save_state(state)
            block_start = block_end
            continue

        # Timer-based flatten at block end
        print("[EXIT] flattening hour positions...", flush=True)
        for sym in SYMBOLS:
            rem = int(state.get("hold_timer", {}).get(sym, 0))
            if rem <= 1:
                print(f"[FLATTEN] {sym}: closing position (timer expired)")
                flatten(api, sym)
                state["hold_timer"][sym] = 0
            else:
                state["hold_timer"][sym] = rem - 1
                print(f"[HOLD] {sym}: keeping open for {state['hold_timer'][sym]} more block(s)")

        save_state(state)

        # Advance to next block
        block_start = block_end

    # After loop: ensure final close flatten at session close
    if now_ny() < session_close:
        print(f"[POST] Waiting until official close {session_close.strftime('%H:%M:%S %Z')} to flatten.")
        while now_ny() < session_close:
            time.sleep(5)

    print("[CLOSE] Market close reached. Final flatten of ALL symbols.")
    for sym in SYMBOLS:
        try:
            flatten(api, sym)
            state.setdefault("hold_timer", {})[sym] = 0
        except Exception as e:
            print(f"[CLOSE_ERR] {sym}: {e}")
    save_state(state)

# ==================================================
# ENTRY
# ==================================================
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
            now_utc = dt.datetime.now(dt.timezone.utc)
            open_utc = clock.next_open.astimezone(dt.timezone.utc)
            wait_sec = max(0, (open_utc - now_utc).total_seconds())
            hrs = wait_sec / 3600.0
            print(f"[WAIT] Market closed. Waiting {hrs:.2f} hours until next open ({open_utc.isoformat()}).")

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
        print(f("[FATAL] missing env var: {e}"))
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
