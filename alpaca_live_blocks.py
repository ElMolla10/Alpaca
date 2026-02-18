#!/usr/bin/env python3
import os, sys, time, traceback, math
import datetime as dt
import pytz
from datetime import datetime, timezone
from alpaca_trade_api.rest import REST, TimeFrame  # ✅ safer import

from app.driftwatch_client import DriftWatchClient  # ✅ DriftWatch

# --- helpers ---
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def dw_env_from_base_url(base_url: str) -> str:
    return "paper" if ("paper" in (base_url or "").lower()) else "live"

def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

# =================== DEBUG HEADER ===================
print("=== DEBUG START ===", flush=True)
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("=== END DEBUG HEADER ===", flush=True)

# =================== CONFIG ===================
TZ_NY = pytz.timezone("America/New_York")

POLL_SECONDS   = 15
HEARTBEAT_SEC  = 60
HARD_LIMIT_SEC = 3*3600+90

TEST_MODE     = os.environ.get("TEST_MODE", "0") == "1"
TEST_SYMBOL   = os.environ.get("TEST_SYMBOL", "AAPL")
TEST_HOLD_MIN = int(os.environ.get("TEST_HOLD_MIN", "0"))

# DriftWatch constants (match main setup)
DW_MODEL_ID = "trading_ensemble_ret_1h"
DW_MODEL_VERSION = "liveblocks-v1"
DW_TIMEFRAME = "1h"
DW_REQUIRED_FEATURE_KEYS = [
    "pred_pct", "p_up", "sigma20_pct", "price_change_pct", "ret5", "vol20",
    "MACDh_12_26_9", "BBM_20_2.0",
    "price_change_pct_lag2", "price_change_pct_lag3",
    "MACDh_12_26_9_lag2", "MACDh_12_26_9_lag3",
    "BBM_20_2.0_lag2", "BBM_20_2.0_lag3",
    "px", "target_frac"
]

def now_ny():
    return dt.datetime.now(TZ_NY)

# ================== ALPACA SETUP ==================
print("=== [START] container up ===", utc_ts())
try:
    BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    KEY_ID   = os.environ["APCA_API_KEY_ID"]
    SECRET   = os.environ["APCA_API_SECRET_KEY"]
    print(f"[ENV] BASE_URL={BASE_URL}  KEY? {bool(KEY_ID)}  SECRET? {bool(SECRET)}")

    api = REST(KEY_ID, SECRET, BASE_URL, api_version="v2")
    clock = api.get_clock()
    print(f"[CLOCK] open={clock.is_open} next_open={clock.next_open} next_close={clock.next_close}")
    acct = api.get_account()
    print(f"[ACCT] status={acct.status} equity=${acct.equity} bp=${acct.buying_power}")
except KeyError as e:
    print(f"[FATAL] Missing env var: {e}"); sys.exit(2)
except Exception:
    traceback.print_exc(); sys.exit(2)

# DriftWatch client
dw = DriftWatchClient()
DW_ENV = dw_env_from_base_url(BASE_URL)

# ================== CORE ACTIONS ==================
def latest_price(sym):
    try:
        tr = api.get_latest_trade(sym)
        return float(tr.price)
    except Exception:
        bars = api.get_bars(sym, TimeFrame.Minute, limit=1)
        return float(bars[-1].c) if bars else math.nan

def pos_qty(sym):
    try:
        p = api.get_position(sym)
        q = int(p.qty)
        return q if p.side == "long" else -q
    except Exception:
        return 0

def market(sym, qty, tag):
    if qty == 0:
        return
    side = "buy" if qty > 0 else "sell"
    print(f"[ORDER] {side.upper()} {abs(qty)} {sym} @ MARKET  tag={tag}")
    api.submit_order(symbol=sym, qty=abs(qty), side=side, type="market", time_in_force="day")

def flatten(sym):
    q = pos_qty(sym)
    if q != 0:
        market(sym, -q, "flatten")

# ========= STRATEGY HOOK (replace with your model) =========
def predict_block_return_pct(sym, dt_block_start_et):
    if TEST_MODE and sym == TEST_SYMBOL:
        return 0.3
    return 0.0

# ================== RUN ONE SESSION (10→16 ET) ==================
def run_session_6_hours(symbols):
    """Run six consecutive 1h trades from 10:00 ET to 16:00 ET."""
    dt_start = now_ny().replace(hour=10, minute=0, second=0, microsecond=0)
    dt_end = dt_start + dt.timedelta(hours=1)

    print(f"=== SESSION start {dt_start.strftime('%Y-%m-%d %H:%M ET')} → 16:00 ET ===")

    # Track open trades for labeling
    open_req_id = {}
    open_entry_px = {}
    open_entry_qty = {}

    for h in range(6):
        if h > 0:
            dt_start = dt_end
            dt_end = dt_start + dt.timedelta(hours=1)

        block_id = dt_start.isoformat()
        print(f"--- 1h BLOCK {h+1}/6: {dt_start.strftime('%H:%M')}→{dt_end.strftime('%H:%M')} ET ---")

        # enter at block start
        for sym in symbols:
            px = latest_price(sym)

            t0 = time.perf_counter()
            pred = predict_block_return_pct(sym, dt_start)  # predict next 1h
            latency_ms = max(1, int(round((time.perf_counter() - t0) * 1000)))

            qty_target = 1 if pred > 0 else (-1 if pred < 0 else 0)
            cur = pos_qty(sym)
            delta = qty_target - cur

            print(f"[PLAN] {sym}: px={px} pred_1h={pred:.3f}% cur={cur} target={qty_target} delta={delta}")

            request_id = f"{block_id}|{sym}"

            # DriftWatch inference event (always logs a decision attempt)
            features_json = {k: None for k in DW_REQUIRED_FEATURE_KEYS}
            features_json["pred_pct"] = _safe_float(pred)
            features_json["p_up"] = 0.5 if pred == 0 else (0.6 if pred > 0 else 0.4)
            features_json["px"] = _safe_float(px)
            features_json["target_frac"] = _safe_float(qty_target)

            segment_json = {
                "sym": sym,
                "timeframe": DW_TIMEFRAME,
                "env": DW_ENV,
                "session": "regular",
                "block_id": block_id,
                "block_minutes": 60,
                "script": "Alpaca_live_blocks.py",
            }

            dw.log_inference(
                model_id=DW_MODEL_ID,
                model_version=DW_MODEL_VERSION,
                ts=now_utc(),
                pred_type="regression",
                y_pred_num=_safe_float(pred),
                y_pred_text=None,
                latency_ms=latency_ms,
                features_json=features_json,
                segment_json=segment_json,
                request_id=request_id,
            )

            # Place order if needed
            if delta != 0:
                market(sym, delta, "enter_1h")

                # Track only if we now have a position
                new_pos = pos_qty(sym)
                if new_pos != 0:
                    open_req_id[sym] = request_id
                    open_entry_px[sym] = px
                    open_entry_qty[sym] = new_pos

        dw.flush()

        # monitor until hour end
        last_hb = 0
        while now_ny() < dt_end:
            time.sleep(POLL_SECONDS)
            if time.time() - last_hb >= HEARTBEAT_SEC:
                print(f"[HB] {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')} → {dt_end.strftime('%H:%M:%S %Z')}")
                last_hb = time.time()

        # flatten all at end of each hour + log labels
        print("[EXIT] flattening hour positions...")
        for sym in symbols:
            if pos_qty(sym) != 0:
                exit_px = latest_price(sym)  # proxy
                flatten(sym)

                req_id = open_req_id.pop(sym, None)
                entry_px = open_entry_px.pop(sym, None)
                entry_qty = open_entry_qty.pop(sym, None)

                if req_id and entry_px and entry_qty:
                    # realized pnl% proxy
                    if entry_qty > 0:
                        pnl_pct = ((exit_px - entry_px) / entry_px) * 100.0
                    else:
                        pnl_pct = ((entry_px - exit_px) / entry_px) * 100.0

                    dw.log_label(
                        ts=now_utc(),
                        request_id=req_id,
                        y_true_num=_safe_float(pnl_pct),
                        label_type="regression",
                        extra_json={"label_name": "realized_pnl_pct", "entry_px": _safe_float(entry_px), "exit_px": _safe_float(exit_px)},
                    )

        dw.flush()

    print("[DONE] session complete (10→16 ET).")

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    try:
        print("[MAIN] starting", utc_ts())
        symbols = [TEST_SYMBOL] if TEST_MODE else ["AAPL", "MSFT", "PG", "AMD", "JPM", "XOM"]
        run_session_6_hours(symbols)
        print("[MAIN] exit", utc_ts())
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    finally:
        dw.close()
