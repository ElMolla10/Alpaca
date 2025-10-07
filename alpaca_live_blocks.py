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
def run_one_block():
    t0 = time.time()
    dt_start, dt_end = next_block_window()
    print(f"=== BLOCK {dt_start.strftime('%Y-%m-%d %H:%M ET')} → {dt_end.strftime('%H:%M ET')} ===")
    now = now_ny()
    if now < dt_start:
        sleep_until(dt_start)
    else:
        lag = (now - dt_start).total_seconds()/60
        print(f"[INFO] launching mid-block: +{lag:.1f} min after open")

    # 1) enter / size once at block start
    symbols = [TEST_SYMBOL] if TEST_MODE else ["AAPL","MSFT","PG","AMD","JPM","XOM"]
    entries = {}
    for sym in symbols:
        px = latest_price(sym); entries[sym] = px
        pred = predict_block_return_pct(sym, dt_start)   # % for the whole block
        # Scale to a tiny trade in TEST_MODE; otherwise leave sizing=0 when pred=0
        qty = 1 if (TEST_MODE and sym==TEST_SYMBOL) else 0
        if pred > 0 and not TEST_MODE:
            qty = 1  # example: 1 share long if positive prediction
        elif pred < 0 and not TEST_MODE:
            qty = -1 # example: 1 share short if negative prediction
        cur = pos_qty(sym)
        delta = qty - cur
        print(f"[PLAN] {sym}: px={px} pred={pred:.3f}% cur={cur} target={qty} delta={delta}")
        if delta != 0:
            try: market(sym, delta, "enter")
            except Exception: traceback.print_exc()

    # 2) monitor until end, with heartbeats and simple stop
    last_hb = 0
    while now_ny() < dt_end:
        time.sleep(POLL_SECONDS)
        if time.time() - last_hb >= HEARTBEAT_SEC:
            print(f"[HB] {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')} watching until {dt_end.strftime('%H:%M:%S %Z')}")
            last_hb = time.time()

        # simple stop-loss demo (10% adverse unlevered) for TEST symbol
        if TEST_MODE:
            sym = TEST_SYMBOL
            q = pos_qty(sym)
            if q != 0:
                side = 1 if q>0 else -1
                px0 = entries.get(sym, latest_price(sym))
                px1 = latest_price(sym)
                pnl_pct = side*((px1/px0)-1.0)*100.0
                if pnl_pct <= -10.0:
                    print(f"[STOP] {sym} pnl={pnl_pct:.2f}% → flatten")
                    flatten(sym)

        if time.time() - t0 >= HARD_LIMIT_SEC:
            print("[SAFEGUARD] hard 3h limit hit → exiting loop"); break

    # 3) flatten at block end
    print("[EXIT] flattening all")
    for sym in symbols:
        try: flatten(sym)
        except Exception: traceback.print_exc()
    print("[DONE] block finished", utc_stamp())

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    try:
        print("[MAIN] starting", utc_stamp())
        run_one_block()
        print("[MAIN] exit", utc_stamp())
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
