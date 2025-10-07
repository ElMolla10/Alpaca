#!/usr/bin/env python3
import os, sys, time, traceback, math
import datetime as dt
import pytz
from alpaca_trade_api import REST, TimeFrame

# =================== CONFIG ===================
TZ_NY = pytz.timezone("America/New_York")

# Block schedule (normal run)
BLOCK_STARTS_ET = [(10,0), (13,0)]
BLOCK_ENDS_ET   = [(13,0), (16,0)]

POLL_SECONDS   = 15        # status/stop check cadence
HEARTBEAT_SEC  = 60        # heartbeat print cadence
HARD_LIMIT_SEC = 3*3600+90 # safety stop for any run

# Test mode flags (set in Render → Environment)
TEST_MODE     = os.environ.get("TEST_MODE", "0") == "1"      # 1 = force a visible trade
TEST_SYMBOL   = os.environ.get("TEST_SYMBOL", "AAPL")
TEST_HOLD_MIN = int(os.environ.get("TEST_HOLD_MIN", "0"))    # when >0, run NOW for N minutes

# ================== UTILITIES ==================
def now_ny():
    return dt.datetime.now(TZ_NY)

def utc_stamp():
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

def next_block_window():
    """Pick next block [start, end] in ET. If TEST_HOLD_MIN>0: run now for N minutes."""
    t = now_ny().replace(second=0, microsecond=0)
    if TEST_HOLD_MIN > 0:
        return t, t + dt.timedelta(minutes=TEST_HOLD_MIN)

    starts = [t.replace(hour=h, minute=m) for (h,m) in BLOCK_STARTS_ET]
    ends   = [t.replace(hour=h, minute=m) for (h,m) in BLOCK_ENDS_ET]
    for i, s in enumerate(starts):
        if t <= s:
            return s, ends[i]
    # past last start → tomorrow first block
    tomorrow = t + dt.timedelta(days=1)
    s = tomorrow.replace(hour=BLOCK_STARTS_ET[0][0], minute=BLOCK_STARTS_ET[0][1], second=0, microsecond=0)
    e = tomorrow.replace(hour=BLOCK_ENDS_ET[0][0],   minute=BLOCK_ENDS_ET[0][1],   second=0, microsecond=0)
    return s, e

def sleep_until(dt_target, ping_sec=15):
    while True:
        now = now_ny()
        if now >= dt_target: break
        rem = int((dt_target - now).total_seconds())
        print(f"[WAIT] {now.strftime('%H:%M:%S %Z')} → start {dt_target.strftime('%H:%M:%S %Z')}  ETA={rem}s")
        time.sleep(min(ping_sec, max(1, rem)))

# ================== ALPACA SETUP ==================
print("=== [START] container up ===", utc_stamp())
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
        q = int(p.qty);  return q if p.side=="long" else -q
    except Exception:
        return 0

def market(sym, qty, tag):
    if qty == 0: return
    side = "buy" if qty>0 else "sell"
    print(f"[ORDER] {side.upper()} {abs(qty)} {sym} @ MARKET  tag={tag}")
    api.submit_order(symbol=sym, qty=abs(qty), side=side, type="market", time_in_force="day")

def flatten(sym):
    q = pos_qty(sym)
    if q != 0:
        market(sym, -q, "flatten")

# ========= STRATEGY HOOK (replace with your model) =========
def predict_block_return_pct(sym, dt_block_start_et):
    # Default returns 0 → no trade. In TEST_MODE we force a tiny signal.
    if TEST_MODE and sym == TEST_SYMBOL:
        return 0.3   # pretend +0.3% over block → small long
    return 0.0

# ================== RUN ONE BLOCK ==================
# --- 1h block start/end anchored on the hour in ET ---
def hour_window_after(t_et):
    t = t_et.replace(minute=0, second=0, microsecond=0)
    start = t if t_et.minute == 0 else (t + dt.timedelta(hours=1))
    end   = start + dt.timedelta(hours=1)
    return start, end

def run_session_6_hours(symbols):
    """Run six consecutive 1h trades from 10:00 ET to 16:00 ET."""
    dt_start = now_ny().replace(hour=10, minute=0, second=0, microsecond=0)
    dt_end   = dt_start + datetime.timedelta(hours=1)

    print(f"=== SESSION start {dt_start.strftime('%Y-%m-%d %H:%M ET')} → 16:00 ET ===")

    for h in range(6):
        if h > 0:
            dt_start = dt_end
            dt_end   = dt_start + datetime.timedelta(hours=1)

        print(f"--- 1h BLOCK {h+1}/6: {dt_start.strftime('%H:%M')}→{dt_end.strftime('%H:%M')} ET ---")

        # enter at block start
        entries = {}
        for sym in symbols:
            px = latest_price(sym)
            pred = predict_block_return_pct(sym, dt_start)   # predict next 1h
            qty = 1 if pred > 0 else (-1 if pred < 0 else 0)
            cur = pos_qty(sym)
            delta = qty - cur
            print(f"[PLAN] {sym}: px={px} pred_1h={pred:.3f}% cur={cur} target={qty} delta={delta}")
            if delta != 0:
                market(sym, delta, "enter_1h")
            entries[sym] = px

        # monitor until hour end
        last_hb = 0
        while now_ny() < dt_end:
            time.sleep(POLL_SECONDS)
            if time.time() - last_hb >= HEARTBEAT_SEC:
                print(f"[HB] {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')} → {dt_end.strftime('%H:%M:%S %Z')}")
                last_hb = time.time()

        # flatten all at end of each hour
        print("[EXIT] flattening hour positions...")
        for sym in symbols:
            flatten(sym)

    print("[DONE] session complete (10→16 ET).")



# ================== ENTRY POINT ==================
if __name__ == "__main__":
    try:
        print("[MAIN] starting", utc_ts())
        symbols = [TEST_SYMBOL] if TEST_MODE else ["AAPL","MSFT","PG","AMD","JPM","XOM"]
        run_session_6_hours(symbols)
        print("[MAIN] exit", utc_ts())
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(2)

