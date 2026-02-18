#!/usr/bin/env python3
# app/main.py

import os, sys, time, math, json, traceback
import datetime as dt
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
import pathlib
import re

import pytz
import numpy as np
import pandas as pd
import xgboost as xgb
try:
    import joblib  # for ElasticNet / sklearn pipelines
except ImportError:
    joblib = None
    print("[INIT] joblib not available; ElasticNet model will be disabled (XGBoost only).")
from ta.trend import MACD
from ta.volatility import BollingerBands
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv


# Load .env variables early
load_dotenv()

# === Agentic RL layer ===
from app.agent.arl_agent import ARLAgent, UserStyle, default_style, BlockContext, Decision
from app.execution import BlockLedger

# === DriftWatch ===
from app.driftwatch_client import DriftWatchClient

DW_MODEL_ID = "trading_ensemble_ret_1h"
DW_MODEL_VERSION = "mock-v1"
DW_TIMEFRAME = "1h"

DW_REQUIRED_FEATURE_KEYS = [
    "pred_pct", "p_up", "sigma20_pct", "price_change_pct", "ret5", "vol20",
    "MACDh_12_26_9", "BBM_20_2.0",
    "price_change_pct_lag2", "price_change_pct_lag3",
    "MACDh_12_26_9_lag2", "MACDh_12_26_9_lag3",
    "BBM_20_2.0_lag2", "BBM_20_2.0_lag3",
    "px", "target_frac"
]

def dw_env_from_base_url(base_url: str) -> str:
    return "paper" if ("paper" in (base_url or "").lower()) else "live"

def _to_float_or_none(x):
    if x is None:
        return None
    try:
        # numpy scalars -> python scalars
        if isinstance(x, np.generic):
            x = x.item()
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

# =================== CONFIG / ENV ===================
ALPACA_DATA_FEED = os.environ.get("ALPACA_DATA_FEED", "iex")  # 'iex' for free; 'sip' if subscribed
TZ_NY = pytz.timezone("America/New_York")

BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
KEY_ID   = os.environ["APCA_API_KEY_ID"]
SECRET   = os.environ["APCA_API_SECRET_KEY"]

# Market session
SESSION_START_H = int(os.environ.get("SESSION_START_H", "10"))   # 10 ET
TRADE_CUTOFF_MIN_BEFORE_CLOSE = int(os.environ.get("TRADE_CUTOFF_MIN_BEFORE_CLOSE", "30"))

# Block duration (minutes) — set BLOCK_MINUTES=30 in .env for half-hour blocks
BLOCK_MINUTES = int(os.environ.get("BLOCK_MINUTES", "60"))
BLOCK_SECONDS = max(60, BLOCK_MINUTES * 60)

# Model / policy parameters
PRIMARY_H          = int(os.environ.get("PRIMARY_H", "1"))   # models trained for 1h horizon
BAND_R             = float(os.environ.get("BAND_R", "0.35"))
POS_EXP            = float(os.environ.get("POS_EXP", "1.5"))
EMA_HALF_LIFE      = int(os.environ.get("EMA_HL", "6"))
DPOS_CAP           = float(os.environ.get("DPOS_CAP", "0.10"))
MAX_POS            = float(os.environ.get("MAX_POS", "0.75"))
MIN_ABS_POS        = float(os.environ.get("MIN_ABS_POS", "0.02"))
USE_EQUITY_SIZING  = os.environ.get("USE_EQUITY_SIZING", "1") == "1"
PER_SYM_GROSS_CAP  = float(os.environ.get("PER_SYM_GROSS_CAP", "0.05"))
BASE_NOTIONAL_PER_TRADE = float(os.environ.get("BASE_NOTIONAL_PER_TRADE", "3000"))
MAX_NOTIONAL       = float(os.environ.get("MAX_NOTIONAL", "10000"))
PER_TRADE_NOTIONAL_CAP_F = float(os.environ.get("PER_TRADE_NOTIONAL_CAP_F", "0.75"))
USE_NOTIONAL_ORDERS = os.environ.get("USE_NOTIONAL_ORDERS", "1") == "1"
SHORTS_ENABLED     = os.environ.get("SHORTS_ENABLED", "1") == "1"
LONGS_ONLY         = os.environ.get("LONGS_ONLY", "0") == "1"
SIGN_MULT          = float(os.environ.get("SIGN_MULT", "1.0"))
REBALANCE_BAND     = float(os.environ.get("REBALANCE_BAND", "0.01"))

# Friday rules
FRIDAY_LATE_CUTOFF_H         = int(os.environ.get("FRIDAY_LATE_CUTOFF_H", "14"))
FRIDAY_SIZE_MULT_DAY         = float(os.environ.get("FRIDAY_SIZE_MULT_DAY", "0.85"))
FRIDAY_SIZE_MULT_LATE        = float(os.environ.get("FRIDAY_SIZE_MULT_LATE", "0.60"))
FRIDAY_BLOCK_NEW_AFTER_LATE  = os.environ.get("FRIDAY_BLOCK_NEW_AFTER_LATE", "1") == "1"
FRIDAY_MIN_POS               = float(os.environ.get("FRIDAY_MIN_POS", "0.05"))

# End-of-day hard exit N minutes before close
EOD_FLATTEN_MIN_BEFORE_CLOSE = float(os.environ.get("EOD_FLATTEN_MIN_BEFORE_CLOSE", "2"))

DAY_KILL_DD_PCT      = float(os.environ.get("DAY_KILL_DD_PCT", "1.0"))   # flatten+stop for the day at -1.0%
DAY_THROTTLE_DD_PCT  = float(os.environ.get("DAY_THROTTLE_DD_PCT", "0.7"))  # start throttling at -0.7%
THROTTLE_SIZE_MULT   = float(os.environ.get("THROTTLE_SIZE_MULT", "0.70"))  # scale positions when throttled

# Agentic switches
AGENTIC_MODE      = os.environ.get("AGENTIC_MODE", "1") == "1"
USER_STYLE        = os.environ.get("USER_STYLE", "high_risk_short_term")
AGENT_MAX_SYMBOLS = int(os.environ.get("AGENT_MAX_SYMBOLS", "20"))

# Trading costs (BPS)
TRADE_COST_BPS = float(os.environ.get("TRADE_COST_BPS", "8.0"))
SLIP_BPS       = float(os.environ.get("SLIPPAGE_BPS", "4.0"))

# Per-symbol position size multipliers (for future fine-tuning)
PER_SYMBOL_SIZE_MULT = {}

# =================== SYMBOL UNIVERSE ===================
universe_path = pathlib.Path("app/universe.csv")
if universe_path.exists():
    SYMBOLS = [l.strip().upper() for l in universe_path.read_text().splitlines()[1:] if l.strip()]
    print(f"[INIT] Loaded {len(SYMBOLS)} symbols from universe.csv")
else:
    symbols_env = os.environ.get("SYMBOLS", "AAPL,MSFT,NVDA,AMD,JPM,GS,XOM,CVX,PG,KO")
    SYMBOLS = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    print(f"[INIT] Fallback SYMBOLS={SYMBOLS}")

print(f"[INIT] using data feed={ALPACA_DATA_FEED}")

# =================== MODELS & FEATURES ===================
MODEL_PATH_XGB   = os.environ.get("MODEL_PATH_XGB", "app/model/XGboost_model.json")
MODEL_PATH_ENET  = os.environ.get("MODEL_PATH_ENET", "app/model/elasticnet_ret1h.pkl")
FEATS_PATH       = os.environ.get("FEATS_PATH", "app/feat_cols.json")

# XGBoost (Booster trained on 1h returns)
xgb_booster = xgb.Booster()
xgb_booster.load_model(MODEL_PATH_XGB)
print(f"[INIT] Loaded XGBoost model from {MODEL_PATH_XGB}")

# ElasticNet (sklearn pipeline or estimator)
enet_model = None
if joblib is not None and os.path.exists(MODEL_PATH_ENET):
    try:
        enet_model = joblib.load(MODEL_PATH_ENET)
        print(f"[INIT] Loaded ElasticNet model from {MODEL_PATH_ENET}")
    except Exception as e:
        print(f"[WARN] Failed to load ElasticNet from {MODEL_PATH_ENET}: {e}")
elif joblib is None:
    print(f"[INIT] joblib missing → ElasticNet disabled, running with XGBoost only.")
else:
    print(f"[INIT] ElasticNet path {MODEL_PATH_ENET} not found; running with XGBoost only.")

# Persisted state
STATE_PATH = os.environ.get("STATE_PATH", "/tmp/live_state.json")

# =================== UTILS ===================
def rfc3339(dtobj: datetime) -> str:
    return dtobj.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def utc_ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def now_ny():  return dt.datetime.now(TZ_NY)

def is_friday_ny() -> tuple[bool, int]:
    n = now_ny()
    return (n.weekday() == 4), n.hour

def load_state():
    try:
        with open(STATE_PATH, "r") as f:
            st = json.load(f)
    except Exception:
        st = {"pos_ema": {}, "hold_timer": {}, "last_day": "", "agent": {}}
    if "pos_ema" not in st:    st["pos_ema"] = {}
    if "hold_timer" not in st: st["hold_timer"] = {}
    if "last_day" not in st:   st["last_day"] = ""
    if "agent" not in st:      st["agent"] = {}
    return st

def save_state(st):
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(st, f)
    except Exception:
        pass

def ensure_market_open_or_wait(api):
    try:
        clock = api.get_clock()
        if getattr(clock, "is_open", False):
            print("[INFO] Market is open. Starting trading immediately.")
        else:
            print("[WAIT] Market closed. main() will handle wait.")
    except Exception as e:
        print(f"[WARN] ensure_market_open_or_wait: {e}")

# =================== ALPACA HELPERS ===================
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
                sym, TimeFrame.Minute,
                start=rfc3339(start_utc), end=rfc3339(now_utc),
                limit=1, adjustment="raw", feed=ALPACA_DATA_FEED
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

def flatten(api, sym, ledger: BlockLedger | None = None):
    try:
        p = api.get_position(sym)
        qty = float(p.qty)
        if abs(qty) > 0:
            side = "sell" if qty > 0 else "buy"
            print(f"[FLATTEN] {sym} qty={qty}")
            api.submit_order(symbol=sym, qty=abs(qty), side=side, type="market", time_in_force="day")
            if ledger is not None:
                px = latest_price(api, sym)
                ledger.record_fill(sym, side=side, qty=abs(qty), price=px)
    except Exception:
        pass

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

def linear_notional_from_posfrac(pos_frac: float) -> float:
    mag = min(max(abs(float(pos_frac)), 0.0), 1.0)
    raw = BASE_NOTIONAL_PER_TRADE + (MAX_NOTIONAL - BASE_NOTIONAL_PER_TRADE) * mag
    return float(min(MAX_NOTIONAL, max(BASE_NOTIONAL_PER_TRADE, raw)))

# === DELTA-AWARE TARGETING (replaces old submit_target) ===
def submit_target(api, sym, target_pos_frac, equity, last_px, ledger: BlockLedger | None = None):
    """
    target_pos_frac: [-1..+1] desired fraction per name (our “position”), signed.
    Records an approximate fill to the ledger (fees+slippage accounted).
    Retries shorts on borrow/availability limits; falls back qty if notional fails.
    """
    def _round_to_cents(x: float) -> float:
        return float(Decimal(str(max(0.0, x))).quantize(Decimal("0.01"), rounding=ROUND_DOWN))

    try:
        # Policy gates
        if LONGS_ONLY and target_pos_frac < 0:
            print(f"[LONGS_ONLY] {sym}: short target blocked.")
            return

        pos_frac = float(target_pos_frac)
        if abs(pos_frac) < MIN_ABS_POS:
            print(f"[SKIP] {sym}: |pos|={abs(pos_frac):.3f} < MIN_ABS_POS={MIN_ABS_POS:.3f}")
            return

        if pos_frac < 0 and not SHORTS_ENABLED:
            print(f"[SKIP] {sym}: shorts disabled")
            return

        # Linear sizing from |pos_frac|
        tgt_notional = linear_notional_from_posfrac(pos_frac)

        side = "buy" if pos_frac > 0 else "sell"
        signed_notional = tgt_notional if pos_frac > 0 else -tgt_notional

        # Leverage headroom guard
        gross, limit = gross_exposure_and_limit(api)
        headroom = _round_to_cents(max(0.0, limit - gross))
        if headroom <= 1.00:
            print(f"[GUARD] gross {gross:,.2f} ≈ limit {limit:,.2f}; skip {sym}")
            return
        if abs(signed_notional) > headroom:
            print(f"[CLIP] {sym}: clip {abs(signed_notional):,.2f} → {headroom:,.2f} by leverage guard")
            signed_notional = math.copysign(headroom, signed_notional)

        # ----- SHORT / SELL (qty path with availability-aware retry) -----
        if side == "sell" and SHORTS_ENABLED:
            if last_px is None or last_px <= 0 or math.isnan(last_px):
                print(f"[SKIP] {sym}: invalid price for short")
                return

            qty_int = int(abs(signed_notional) / last_px)
            if qty_int <= 0:
                print(f"[SKIP] {sym}: short qty rounds to 0")
                return

            print(f"[ORDER] {sym} SELL qty={qty_int} (~${qty_int*last_px:,.2f}) |pos|={abs(pos_frac):.3f}")
            try:
                api.submit_order(symbol=sym, qty=qty_int, side="sell", type="market", time_in_force="day")
                if ledger is not None:
                    ledger.record_fill(sym, side="sell", qty=qty_int, price=last_px)
                return
            except Exception as e:
                msg = str(e)
                # Try to parse "available: N" and clip
                m = re.search(r"available:\s*([\d\.]+)", msg.lower())
                if m:
                    avail = int(float(m.group(1)))
                    if avail > 0:
                        print(f"[RETRY] {sym}: clipping short qty {qty_int}→{avail} due to availability")
                        api.submit_order(symbol=sym, qty=avail, side="sell", type="market", time_in_force="day")
                        if ledger is not None:
                            ledger.record_fill(sym, side="sell", qty=avail, price=last_px)
                        return
                    else:
                        print(f"[SKIP] {sym}: no borrow available")
                        return
                # Fallback: halve and retry once
                clipped = max(1, qty_int // 2)
                if clipped < qty_int:
                    print(f"[RETRY] {sym}: halve qty {qty_int}→{clipped}")
                    try:
                        api.submit_order(symbol=sym, qty=clipped, side="sell", type="market", time_in_force="day")
                        if ledger is not None:
                            ledger.record_fill(sym, side="sell", qty=clipped, price=last_px)
                        return
                    except Exception as e2:
                        print(f"[ORDER_ERR] {sym}: retry failed: {e2}")
                        return
                print(f"[ORDER_ERR] {sym}: {e}")
                return

        # ----- LONG / BUY (prefer notional; fall back to qty if needed) -----
        notional_abs = _round_to_cents(abs(signed_notional))
        if notional_abs < 1.00:
            print(f"[SKIP] {sym}: notional too small")
            return

        print(f"[ORDER] {sym} {side.upper()} notional ${notional_abs:,.2f} |pos|={abs(pos_frac):.3f}")

        if USE_NOTIONAL_ORDERS:
            try:
                api.submit_order(symbol=sym, notional=notional_abs, side=side, type="market", time_in_force="day")
                if ledger is not None and last_px and last_px > 0 and not math.isnan(last_px):
                    qty_est = notional_abs / last_px
                    ledger.record_fill(sym, side=side, qty=qty_est, price=last_px)
                return
            except Exception as e:
                # Fallback to integer qty if broker rejects notional
                if last_px and last_px > 0 and not math.isnan(last_px):
                    qty_int = max(1, int(notional_abs / last_px))
                    print(f"[FALLBACK] {sym}: switching to qty order qty={qty_int} after notional error: {e}")
                    try:
                        api.submit_order(symbol=sym, qty=qty_int, side=side, type="market", time_in_force="day")
                        if ledger is not None:
                            ledger.record_fill(sym, side=side, qty=qty_int, price=last_px)
                        return
                    except Exception as e2:
                        print(f"[ORDER_ERR] {sym}: qty fallback failed: {e2}")
                        return
                else:
                    print(f"[ORDER_ERR] {sym}: notional submit failed and no valid price: {e}")
                    return
        else:
            # Direct qty route if not using notionals
            if last_px and last_px > 0 and not math.isnan(last_px):
                qty_int = max(1, int(notional_abs / last_px))
                print(f"[ORDER] {sym} {side.upper()} qty={qty_int} (~${qty_int*last_px:,.2f})")
                try:
                    api.submit_order(symbol=sym, qty=qty_int, side=side, type="market", time_in_force="day")
                    if ledger is not None:
                        ledger.record_fill(sym, side=side, qty=qty_int, price=last_px)
                except Exception as e:
                    print(f"[ORDER_ERR] {sym}: {e}")
            else:
                print(f"[SKIP] {sym}: invalid price for qty path")

    except Exception as e:
        print(f"[ORDER_ERR] {sym}: {e}")

# =================== FEATURE PIPELINE (LIVE) ===================
with open(FEATS_PATH, "r") as f:
    FEAT_COLS = json.load(f)
print(f"[INIT] loaded {len(FEAT_COLS)} feat cols")
print(f"[PARITY] FEAT_COLS={len(FEAT_COLS)} window=10:00–15:59 ET, model_horizon={PRIMARY_H}h, block_minutes={BLOCK_MINUTES}")

def _ts_to_utc(ts_any) -> pd.Timestamp:
    """Robustly coerce any alpaca timestamp-like to UTC-aware pandas Timestamp."""
    ts = pd.Timestamp(getattr(ts_any, "t", None) or getattr(ts_any, "Timestamp", None) or ts_any)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def fetch_recent_features(api: REST, sym: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Pull last ~30 days of HOURLY bars and (if between 10:00 and <11:00 ET)
    synthesize a partial 10:00→now bar from minute data so the first block
    is tradable without waiting for 11:00.

    Features are aligned to the 1h-trained models; trading loop may run at 30-minute blocks.
    """

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)

    # --- fetch hourly bars ---
    try:
        bars = api.get_bars(
            sym, TimeFrame.Hour,
            start=rfc3339(start_utc), end=rfc3339(now_utc),
            adjustment="raw", limit=1000, feed=ALPACA_DATA_FEED
        )
    except Exception as e:
        print(f"[WARN] {sym}: get_bars feed={ALPACA_DATA_FEED} failed: {e}")
        if ALPACA_DATA_FEED.lower() != "iex":
            try:
                bars = api.get_bars(
                    sym, TimeFrame.Hour,
                    start=rfc3339(start_utc), end=rfc3339(now_utc),
                    adjustment="raw", limit=1000, feed="iex"
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
        ts = _ts_to_utc(getattr(b, "t", None))
        rows.append({
            "timestamp": ts,
            "open": float(getattr(b, "o", np.nan)),
            "high": float(getattr(b, "h", np.nan)),
            "low": float(getattr(b, "l", np.nan)),
            "close": float(getattr(b, "c", np.nan)),
            "volume": float(getattr(b, "v", np.nan)),
            "vwap": float(getattr(b, "vw", np.nan)) if hasattr(b, "vw") else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # --- synthesize partial 10:00→now bar (only 10:00–<11:00 ET) ---
    try:
        ny_now = datetime.now(TZ_NY)
        if 10 <= ny_now.hour < 11:
            start_ny = ny_now.replace(hour=10, minute=0, second=0, microsecond=0)
            start_utc_min = start_ny.astimezone(timezone.utc)

            mins = api.get_bars(
                sym, TimeFrame.Minute,
                start=rfc3339(start_utc_min), end=rfc3339(now_utc),
                adjustment="raw", limit=1000, feed=ALPACA_DATA_FEED
            )

            if mins:
                rows_m = []
                for m in mins:
                    ts_m = _ts_to_utc(getattr(m, "t", None))
                    rows_m.append({
                        "t": ts_m,
                        "o": float(getattr(m, "o", np.nan)),
                        "h": float(getattr(m, "h", np.nan)),
                        "l": float(getattr(m, "l", np.nan)),
                        "c": float(getattr(m, "c", np.nan)),
                        "v": float(getattr(m, "v", np.nan)),
                        "vw": float(getattr(m, "vw", np.nan)) if hasattr(m, "vw") else np.nan,
                    })
                mdf = pd.DataFrame(rows_m)
                if not mdf.empty:
                    # clip to [10:00, now] UTC
                    mdf = mdf[(mdf["t"] >= pd.Timestamp(start_utc_min)) &
                              (mdf["t"] <= pd.Timestamp(now_utc))]
                    if not mdf.empty:
                        open_px = float(mdf["o"].iloc[0])
                        high_px = float(mdf["h"].max())
                        low_px  = float(mdf["l"].min())
                        close_px = float(mdf["c"].iloc[-1])
                        vol_sum = float(mdf["v"].sum())
                        if "vw" in mdf:
                            vwap_w = float(np.average(
                                mdf["vw"].fillna(mdf["c"]),
                                weights=np.clip(mdf["v"], 1e-12, None)
                            ))
                        else:
                            vwap_w = float(mdf["c"].mean())

                        synth_row = {
                            "timestamp": _ts_to_utc(start_utc_min),
                            "open": open_px, "high": high_px, "low": low_px,
                            "close": close_px, "volume": vol_sum, "vwap": vwap_w
                        }

                        if not ((df["timestamp"] == synth_row["timestamp"]).any()):
                            df = pd.concat([df, pd.DataFrame([synth_row])], ignore_index=True)
                            df = df.sort_values("timestamp").reset_index(drop=True)
                            print(f"[SYNTH] {sym}: appended partial 10:00→now pseudo-hour bar.")
    except Exception as _e:
        print(f"[WARN] {sym}: failed to synthesize partial hour: {_e}")

    # --- Filter to regular hours 10:00–15:59 ET ---
    ts_et = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.loc[ts_et.dt.hour.between(10, 15)].reset_index(drop=True)
    if df.empty:
        print(f"[WARN] {sym}: bars after RTH filter are empty")
        return df

    # --- Indicators ---
    df["price_change_pct"] = df["close"].pct_change() * 100.0
    df["price_change_pct_lag1"] = df["price_change_pct"].shift(1)
    df["price_change_pct_lag2"] = df["price_change_pct"].shift(2)
    df["price_change_pct_lag3"] = df["price_change_pct"].shift(3)

    macd_ind = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACDh_12_26_9"] = macd_ind.macd_diff()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2.0)
    df["BBM_20_2.0"] = bb.bollinger_mavg()

    df["ret1h_pct"] = df["close"].pct_change().shift(-1) * 100.0
    df["sigma20_pct"] = df["ret1h_pct"].rolling(20).std().shift(1)
    df["sigma20_pct"] = df["sigma20_pct"].replace(0.0, np.nan)

    base = ["open","high","low","close","volume","vwap",
        "MACDh_12_26_9","BBM_20_2.0"]
    for f in base:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].shift(1)

    for f in ["MACDh_12_26_9","BBM_20_2.0"]:
        df[f+"_lag2"] = df[f].shift(2)
        df[f+"_lag3"] = df[f].shift(3)

    df["vol20"] = df["ret1h_pct"].rolling(20).std().shift(1)
    df["ret5"]  = df["close"].pct_change(5).shift(1) * 100.0

    for f in ["price_change_pct","ret5","vol20","MACDh_12_26_9","BBM_20_2.0"]:
        if f in df.columns:
            df[f+"_N"] = df[f] / df["sigma20_pct"]

    df = df.dropna().reset_index(drop=True)

    # Ensure all FEAT_COLS exist
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    cols = ["timestamp"] + [c for c in FEAT_COLS if c in df.columns]
    # Keep sigma20_pct alongside features for vol_est, but not as a model feature
    if "sigma20_pct" in df.columns and "sigma20_pct" not in cols:
        cols.append("sigma20_pct")

        # --- Ensure DriftWatch raw feature columns are present in returned df ---
    DW_RAW_FROM_DF = [
        "sigma20_pct", "price_change_pct", "ret5", "vol20",
        "MACDh_12_26_9", "BBM_20_2.0",
        "price_change_pct_lag2", "price_change_pct_lag3",
        "MACDh_12_26_9_lag2", "MACDh_12_26_9_lag3",
        "BBM_20_2.0_lag2", "BBM_20_2.0_lag3"
    ]
    for c in DW_RAW_FROM_DF:
        if c in df.columns and c not in cols:
            cols.append(c)


    df = df[cols]
    return df

def map_signal_to_pos(pred_pct: float, band_R: float, exp_: float) -> float:
    s = float(pred_pct) / max(1e-12, band_R)
    s = max(-10.0, min(10.0, s))
    mag = abs(s) ** exp_
    return (1.0 if s >= 0 else -1.0) * min(1.0, mag)

def update_pos_ema(prev: float, target: float, half_life: int) -> float:
    if half_life <= 1:
        return target
    alpha = 1.0 - 0.5 ** (1.0 / float(half_life))
    return alpha * target + (1.0 - alpha) * prev

def target_position_from_pred(pred_pct_live: float, band_R: float, hl: int, sym: str, state: dict) -> float:
    raw = map_signal_to_pos(pred_pct_live, band_R, POS_EXP)
    prev = state["pos_ema"].get(sym, 0.0)
    pos  = update_pos_ema(prev, raw, hl)
    delta = pos - prev
    if   delta >  DPOS_CAP: pos = prev + DPOS_CAP
    elif delta < -DPOS_CAP: pos = prev - DPOS_CAP
    pos = max(-MAX_POS, min(MAX_POS, pos))
    state["pos_ema"][sym] = pos
    return pos

def predict_block_signal(api: REST, sym: str, df_override: pd.DataFrame = None) -> dict:
    """
    Returns an enriched prediction dict:
      - pred_pct: ensemble mean predicted 1h return in %
      - p_up: pseudo probability of up move (0..1)
      - vol_est: latest sigma20_pct
      - per_model: individual model outputs
    Trading loop currently only uses pred_pct; p_up/vol_est are for future RL logic.
    """
    out = {
        "pred_pct": 0.0,
        "p_up": 0.5,
        "vol_est": float("nan"),
        "per_model": {"xgb": None, "elastic": None}
    }
    try:
        df = df_override if df_override is not None else fetch_recent_features(api, sym)
        if df is None or df.empty:

        X = df[FEAT_COLS].astype(float).iloc[-1:].values
        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else "n/a"
        vol_est = float(df["sigma20_pct"].iloc[-1]) if "sigma20_pct" in df.columns else float("nan")
        out["vol_est"] = vol_est

        print(f"[DBG] {sym}: feature row OK shape={X.shape} last_ts={last_ts}")

        # XGBoost prediction
        pred_xgb = None
        try:
            dmat = xgb.DMatrix(X, feature_names=FEAT_COLS)
            pred_xgb = float(xgb_booster.predict(dmat)[0])
            out["per_model"]["xgb"] = pred_xgb
        except Exception as e:
            print(f"[WARN] {sym}: XGBoost predict failed: {e}")

        # ElasticNet prediction (if available)
        pred_enet = None
        if enet_model is not None:
            try:
                pred_enet = float(enet_model.predict(X)[0])
                out["per_model"]["elastic"] = pred_enet
            except Exception as e:
                print(f"[WARN] {sym}: ElasticNet predict failed: {e}")

        preds = [p for p in (pred_xgb, pred_enet) if p is not None and np.isfinite(p)]
        if not preds:
            print(f"[WARN] {sym}: no valid ensemble preds → returning 0")
            return out

        pred_mean = float(np.mean(preds))
        out["pred_pct"] = pred_mean

        # Map mean prediction to a pseudo probability p_up using vol_est as a scale
        if np.isfinite(pred_mean):
            if np.isfinite(vol_est) and vol_est > 1e-6:
                z = pred_mean / max(1.0, vol_est)
            else:
                z = pred_mean / 1.0
            z = float(np.clip(z, -5.0, 5.0))
            p_up = 1.0 / (1.0 + math.exp(-z))
            out["p_up"] = float(p_up)

        print(f"[DBG] {sym}: ensemble_pred={pred_mean:.4f}% p_up={out['p_up']:.3f} vol_est={out['vol_est']}")
        return out

    except Exception as e:
        print(f"[ERR] predict_block_signal {sym}: {e}")
        return out

# =================== MAIN LOOP ===================
def run_session(api):
    from app.driftwatch_client import DriftWatchClient
    dw = DriftWatchClient()
    try:
        state = load_state()
        state.setdefault("open_req_id", {})
        today = now_ny().strftime("%Y-%m-%d")

        if state.get("last_day") != today:
            state = {
                "pos_ema": {},
                "hold_timer": {},
                "open_req_id": {},              # ✅ keep inside state on new day
                "last_day": today,
                "agent": state.get("agent", {}) # persist agent learning across days
            }
            save_state(state)
        else:
            state.setdefault("open_req_id", {})  # ✅ ensure exists on same day

        acct = api.get_account()
        print(f"[ACCT] status={acct.status} equity=${acct.equity} bp=${acct.buying_power}")

        # ... keep the rest of your existing run_session code here ...

    finally:
        dw.close()


    # === Daily drawdown tracking ===
    day_start_equity = account_equity(api)
    throttled = False

    ensure_market_open_or_wait(api)

    if LONGS_ONLY:
        try:
            for p in api.list_positions():
                qty = float(getattr(p, "qty", 0.0))
                sym = getattr(p, "symbol", "")
                if qty < 0:
                    print(f"[LONGS_ONLY] Covering pre-existing short: {sym} qty={qty}")
                    api.submit_order(symbol=sym, qty=abs(qty), side="buy", type="market", time_in_force="day")
                    state.setdefault("hold_timer", {})[sym] = 0
            save_state(state)
        except Exception as e:
            print(f"[LONGS_ONLY_ERR] pre-open cover: {e}")

    agent_style = default_style(USER_STYLE)
    agent_style.max_symbols = AGENT_MAX_SYMBOLS
    agent = ARLAgent(agent_style, persisted=state.get("agent", {}))
    print(f"[AGENT] style={agent_style.name} max_symbols={agent_style.max_symbols}")
    
    ledger = BlockLedger(TRADE_COST_BPS, SLIP_BPS)


    t_now = now_ny()
    session_open  = t_now.replace(hour=SESSION_START_H, minute=0, second=0, microsecond=0)
    session_close = t_now.replace(hour=16, minute=0, second=0, microsecond=0)
    block_start   = t_now if t_now >= session_open else session_open

    b = 0
    while True:

        # 1) Always check EOD first based on *actual time*
        mins_to_close_now = (session_close - now_ny()).total_seconds() / 60.0
        if mins_to_close_now <= EOD_FLATTEN_MIN_BEFORE_CLOSE:
            print(f"[EOD] {EOD_FLATTEN_MIN_BEFORE_CLOSE:.0f} min to close → flattening all open positions.")
            try:
                for p in api.list_positions():
                    sym = getattr(p, "symbol", None)
                    if sym:
                        flatten(api, sym, ledger=None)
            except Exception as e:
                print(f"[EOD_WARN] list_positions/flatten: {e}")

            # Reset timers
            for sym in SYMBOLS:
                state.setdefault("hold_timer", {})[sym] = 0
            save_state(state)

            print("[EOD] All positions flattened. Ending session loop.")
            break

        # 2) Then check if we’re beyond session_close
        if block_start >= session_close:
            print("[INFO] Reached session close window; stopping.")
            break


        # Use BLOCK_MINUTES instead of fixed 1h
        unclamped_end = block_start + dt.timedelta(minutes=BLOCK_MINUTES)
        block_end = min(unclamped_end, session_close)
        
        block_id = block_start.isoformat()  # ET offset included
        env = dw_env_from_base_url(BASE_URL)

        secs_left = (session_close - block_start).total_seconds()
        blocks_left = max(1, math.ceil(secs_left / float(BLOCK_SECONDS)))
        b += 1

        print(f"\n=== BLOCK {b}/{blocks_left} {block_start.strftime('%H:%M')}→{block_end.strftime('%H:%M')} ET (block={BLOCK_MINUTES}min) ===")
        print(f"[TIMECHK] now={now_ny().strftime('%H:%M:%S %Z')} block_end={block_end.strftime('%H:%M:%S %Z')}", flush=True)

        eq = account_equity(api)
        eq_now = account_equity(api)
        if day_start_equity > 0:
            dd_day = 100.0 * (eq_now - day_start_equity) / day_start_equity

            # Kill switch
            if dd_day <= -abs(DAY_KILL_DD_PCT):
                print(f"[KILL] Daily PnL {dd_day:.2f}% ≤ -{DAY_KILL_DD_PCT:.2f}% → flatten and stop.")
                for p in api.list_positions():
                    sym = getattr(p, "symbol", None)
                    if sym:
                        flatten(api, sym, ledger=None)
                break

            # Throttle
            if (not throttled) and dd_day <= -abs(DAY_THROTTLE_DD_PCT):
                throttled = True
                print(f"[THROTTLE] Daily PnL {dd_day:.2f}% ≤ -{DAY_THROTTLE_DD_PCT:.2f}% → reducing size ×{THROTTLE_SIZE_MULT:.2f}")

        is_fri, ny_hour = is_friday_ny()
        is_late = is_fri and (ny_hour >= FRIDAY_LATE_CUTOFF_H)
        friday_mult = 1.0
        if is_fri:
            friday_mult = FRIDAY_SIZE_MULT_LATE if is_late else FRIDAY_SIZE_MULT_DAY
            print(f"[FRIDAY] context: hour={ny_hour}, late={is_late}, size_mult={friday_mult:.2f}, block_new_after_late={FRIDAY_BLOCK_NEW_AFTER_LATE}")

        # >>> AGENT 3.3 BEGIN
        sym_to_df = {}
        for s in SYMBOLS:
            try:
                sym_to_df[s] = fetch_recent_features(api, s, lookback_days=agent_style.lookback_days)
            except Exception:
                sym_to_df[s] = None
        feats = agent.build_selection_frame(sym_to_df)
        trade_universe = set(agent.select_universe(feats)) if AGENTIC_MODE else set(SYMBOLS)
        print(f"[AGENT] universe this block: {sorted(list(trade_universe))}")
        # >>> AGENT 3.3 END

        price_cache = {}

        for sym in SYMBOLS:
            try:
                minutes_to_close = (session_close - now_ny()).total_seconds() / 60.0
                cutoff_active = minutes_to_close <= TRADE_CUTOFF_MIN_BEFORE_CLOSE

                rem = int(state.get("hold_timer", {}).get(sym, 0))
                if rem > 0:
                    print(f"[HOLD] {sym}: already open; {rem} block(s) remaining. Skipping new order.")
                    continue

                if AGENTIC_MODE and (sym not in trade_universe):
                    print(f"[AGENT] {sym}: not selected this block → skip")
                    continue

                ctx = BlockContext(
                    is_friday=is_fri,
                    is_late=is_late,
                    minutes_to_close=minutes_to_close,
                    equity=eq
                )
                dec = agent.decide_for_symbol(sym, ctx)
                if not dec.allow:
                    print(f"[AGENT] {sym}: decision=blocked")
                    continue

                if cutoff_active and rem == 0:
                    print(f"[CUTOFF] {sym}: {TRADE_CUTOFF_MIN_BEFORE_CLOSE} min to close → blocking NEW entry.")
                    continue

                print(f"[DBG] fetching & predicting {sym} ...", flush=True)
                px = price_cache.get(sym)
                if px is None:
                    px = latest_price(api, sym)
                    price_cache[sym] = px

                sig = predict_block_signal(api, sym)
                pred = sig["pred_pct"] * SIGN_MULT  # still 1h prediction, used every BLOCK_MINUTES

                if pred == 0.0:
                    print(f"[WARN] {sym}: prediction is 0.0 (check features/models)")

                band_R_used = dec.band_R_override if dec.band_R_override is not None else BAND_R
                ema_hl_used = dec.ema_hl_override if dec.ema_hl_override is not None else EMA_HALF_LIFE

                target_frac = target_position_from_pred(pred, band_R_used, ema_hl_used, sym, state)
                if dec.size_mult and target_frac != 0.0:
                    target_frac *= float(dec.size_mult)
                    print(f"[AGENT] {sym}: size×={dec.size_mult:.2f} → target_pos={target_frac:+.3f}")

                sym_mult = PER_SYMBOL_SIZE_MULT.get(sym, 1.0)
                if sym_mult != 1.0 and target_frac != 0.0:
                    target_frac *= sym_mult
                    print(f"[SIZE]  {sym}: per-symbol×={sym_mult:.2f} → target_pos={target_frac:+.3f}")

                if is_fri and friday_mult < 1.0 and target_frac != 0.0:
                    target_frac *= friday_mult
                    print(f"[FRIDAY] {sym}: size×={friday_mult:.2f} → target_pos={target_frac:+.3f}")

                per_sym_cap_frac = max(1e-9, PER_SYM_GROSS_CAP)
                if abs(target_frac) > per_sym_cap_frac:
                    target_frac = math.copysign(per_sym_cap_frac, target_frac)
                    print(f"[CAP] {sym}: target clipped to per-sym cap frac={per_sym_cap_frac:.3f}")

                prev_pos = float(state.get("pos_ema", {}).get(sym, 0.0))
                is_new_entry = (abs(prev_pos) < MIN_ABS_POS) and (abs(target_frac) >= MIN_ABS_POS)
                if is_fri and is_late and FRIDAY_BLOCK_NEW_AFTER_LATE and is_new_entry:
                    print(f"[FRIDAY] {sym}: blocking NEW entry after {FRIDAY_LATE_CUTOFF_H}:00 ET.")
                    continue

                if throttled and target_frac != 0.0:
                    target_frac *= THROTTLE_SIZE_MULT
                    print(f"[THROTTLE] {sym}: size×={THROTTLE_SIZE_MULT:.2f} → target_pos={target_frac:+.3f}")

                                # --- DriftWatch logging (inference + decision latency only) ---
                t0 = time.perf_counter()

                # reuse the df already fetched for agent selection
                df_feat = sym_to_df.get(sym)
                sig = predict_block_signal(api, sym, df_override=df_feat)

                pred = sig["pred_pct"] * SIGN_MULT

                # (your existing target_frac logic continues here...)
                # target_frac = target_position_from_pred(...)

                latency_ms = max(1, int(round((time.perf_counter() - t0) * 1000)))

                # Feature snapshot from the same df row used for inference
                feat_row = None
                if df_feat is not None and not df_feat.empty:
                    feat_row = df_feat.iloc[-1]

                features_json = {k: None for k in DW_REQUIRED_FEATURE_KEYS}
                features_json["pred_pct"] = _to_float_or_none(pred)
                features_json["p_up"] = _to_float_or_none(sig.get("p_up"))
                features_json["sigma20_pct"] = _to_float_or_none(sig.get("vol_est"))
                features_json["px"] = _to_float_or_none(px)
                features_json["target_frac"] = _to_float_or_none(target_frac)

                # fill raw feature cols if available
                if feat_row is not None:
                    for k in features_json.keys():
                        if k in ("pred_pct", "p_up", "sigma20_pct", "px", "target_frac"):
                            continue
                
                        # Map model's lag1 feature into DriftWatch's canonical "price_change_pct"
                        if k == "price_change_pct":
                            if "price_change_pct_lag1" in feat_row.index:
                                features_json["price_change_pct"] = _to_float_or_none(feat_row["price_change_pct_lag1"])
                            continue
                
                        if k in feat_row.index:
                            features_json[k] = _to_float_or_none(feat_row[k])

                segment_json = {
                    "sym": sym,
                    "timeframe": DW_TIMEFRAME,
                    "env": env,
                    "session": "regular",
                    "block_id": block_id,
                    "block_minutes": BLOCK_MINUTES,
                }

                request_id = f"{block_id}|{sym}"

                dw.log_inference(
                    model_id=DW_MODEL_ID,
                    model_version=DW_MODEL_VERSION,
                    ts=datetime.now(timezone.utc),
                    pred_type="regression",
                    y_pred_num=_to_float_or_none(pred),
                    y_pred_text=None,
                    latency_ms=latency_ms,
                    features_json=features_json,
                    segment_json=segment_json,
                    request_id=request_id
                )

                px_print = px if (px and np.isfinite(px)) else float("nan")
                print(f"[PLAN] {sym}: px={px_print:.2f} pred_1h={pred:.3f}% p_up={sig['p_up']:.3f} vol_est={sig['vol_est']:.4f} target_pos={target_frac:+.3f}")

                before_n = ledger.fills_count(sym)
                submit_target(api, sym, target_frac, eq, px, ledger=ledger)
                after_n = ledger.fills_count(sym)

                submitted = after_n > before_n


                # store request_id only for a real new entry
                if submitted and is_new_entry:
                    state["open_req_id"][sym] = request_id


                absf = abs(target_frac)
                if dec.hold_min_blocks is not None:
                    hold_blocks = int(max(1, dec.hold_min_blocks))
                else:
                    if absf < 0.25:   hold_blocks = 1
                    elif absf < 0.50: hold_blocks = 2
                    else:             hold_blocks = 3
                state.setdefault("hold_timer", {})[sym] = hold_blocks

            except Exception as e:
                print(f"[ERR] {sym}: {e}")
                traceback.print_exc(limit=1)

        save_state(state)

        last_hb = 0
        while now_ny() < block_end:
            time.sleep(10)
            if time.time() - last_hb >= 60:
                print(f"[HB] {now_ny().strftime('%H:%M:%S %Z')} → {block_end.strftime('%H:%M:%S %Z')}", flush=True)
                last_hb = time.time()

        print("[EXIT] flattening block positions...", flush=True)

        closed_pnl = {}  # symbol -> pnl_pct for rewards
        for sym in SYMBOLS:
            rem = int(state.get("hold_timer", {}).get(sym, 0))
            if rem <= 1:
                print(f"[FLATTEN] {sym}: closing position (timer expired)")

                # close + record exit fill into persistent ledger
                flatten(api, sym, ledger=ledger)
                state["hold_timer"][sym] = 0

                # compute realized pnl% for this symbol (entry+exit across blocks)
                pnl_pct = ledger.compute_symbol_pnl_pct(sym)
                ledger.clear_symbol(sym)

                if pnl_pct is not None:
                    closed_pnl[sym] = float(pnl_pct)

                # DriftWatch label: join via request_id saved at entry time
                req_id = state.get("open_req_id", {}).pop(sym, None)
                if req_id is not None and pnl_pct is not None:
                    dw.log_label(
                        ts=datetime.now(timezone.utc),
                        request_id=req_id,
                        y_true_num=float(pnl_pct),
                        label_type="regression",
                        extra_json={"label_name": "realized_pnl_pct"}
                    )

            else:
                state["hold_timer"][sym] = rem - 1
                print(f"[HOLD] {sym}: keeping open for {state['hold_timer'][sym]} more block(s)")

        # update RL rewards only for symbols that actually closed
        if closed_pnl:
            agent.update_rewards(closed_pnl)

        state["agent"] = agent.export_state()
        save_state(state)

        # DriftWatch: end-of-block flush
        dw.flush()

        block_start = block_end



# =================== ENTRY ===================
if __name__ == "__main__":
    print("=== START ===", utc_ts())
    try:
        api = REST(KEY_ID, SECRET, BASE_URL, api_version="v2")
        clock = api.get_clock()
        print(f"[CLOCK] open={clock.is_open} next_open={clock.next_open} next_close={clock.next_close}")

        if getattr(clock, "is_open", False):
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
                chunk = min(300, remaining)
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
