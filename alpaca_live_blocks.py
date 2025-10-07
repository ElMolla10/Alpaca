#!/usr/bin/env python3
import os, sys, time, traceback, datetime, pytz
from alpaca_trade_api import REST, TimeFrame

# ============ CONFIG ============
SYMBOL = "AAPL"
HORIZON_H = 3                   # predictive horizon
HEARTBEAT_SEC = 120             # how often to print “alive” message
TIMEZONE = pytz.timezone("America/New_York")

# ============ HELPERS ============
def now_ny():
    return datetime.datetime.now(TIMEZONE)

def utc_ts():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ============ INIT ============
print("=== [START] Alpaca bot container running ===", utc_ts())

try:
    print("[DEBUG] Import check done at", now_ny())
    BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    KEY_ID = os.environ.get("APCA_API_KEY_ID")
    SECRET = os.environ.get("APCA_API_SECRET_KEY")

    if not KEY_ID or not SECRET:
        print("[ERROR] Missing Alpaca credentials in environment!")
        sys.exit(2)

    print(f"[DEBUG] BASE_URL={BASE_URL}")
    print(f"[DEBUG] KEY_ID loaded={bool(KEY_ID)}, SECRET loaded={bool(SECRET)}")

    api = REST(KEY_ID, SECRET, BASE_URL, api_version="v2")
    clock = api.get_clock()
    print(f"[DEBUG] Alpaca clock: is_open={clock.is_open}, next_close={clock.next_close}, next_open={clock.next_open}")

    account = api.get_account()
    print(f"[DEBUG] Account status={account.status}, equity={account.equity}, buying_power={account.buying_power}")

except Exception:
    print("[ERROR] Exception during setup:")
    traceback.print_exc()
    sys.exit(2)

print("[DEBUG] Setup complete, entering main()...", utc_ts())

# ============ MAIN LOOP ============
def run_one_block():
    """Main block of trading — replace with your strategy logic."""
    start_t = now_ny()
    print(f"[INFO] Starting block at {start_t.strftime('%Y-%m-%d %H:%M:%S')} NY time")

    try:
        # --- Example: read last 10 bars for symbol
        bars = api.get_bars(SYMBOL, TimeFrame.Hour, limit=10).df
        print(f"[DATA] Retrieved {len(bars)} hourly bars for {SYMBOL}")
        if not bars.empty:
            print(bars.tail(2))

        # --- Placeholder trade condition (replace later)
        last_close = bars['close'].iloc[-1]
        prev_close = bars['close'].iloc[-2]
        change = (last_close - prev_close) / prev_close * 100
        print(f"[DEBUG] Price change {change:.2f}% over last hour")

        if abs(change) > 0.3:
            side = "buy" if change > 0 else "sell"
            print(f"[ACTION] Would {side.upper()} {SYMBOL} based on +{change:.2f}% move")
        else:
            print("[ACTION] No trade triggered.")

    except Exception:
        print("[ERROR] Exception in run_one_block:")
        traceback.print_exc()

    print("[INFO] Finished block", utc_ts())

# ============ HEARTBEAT ============
if __name__ == "__main__":
    print("[DEBUG] Entering main() section...", utc_ts())
    hb_timer = time.time()

    while True:
        run_one_block()
        print("[SLEEP] Sleeping for 3 hours before next block...")
        for _ in range(int(3*3600 / HEARTBEAT_SEC)):
            time.sleep(HEARTBEAT_SEC)
            print(f"[HB] Alive at {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Optional safety refresh:
        try:
            clock = api.get_clock()
            print(f"[CLOCK] Market open? {clock.is_open}, next_close={clock.next_close}")
        except Exception:
            print("[WARN] Could not refresh clock:")
            traceback.print_exc()
