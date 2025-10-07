#!/usr/bin/env python3
import os, time, math, pathlib
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
from alpaca_trade_api import REST

# ===== USER CONFIG (edit tickers & params as you like) =====
TICKERS = ["AAPL","MSFT","PG","AMD","JPM","XOM"]
PRIMARY_H = 3                                      # 3-hour blocks
BLOCK_STARTS_ET = [(10,0), (13,0)]                 # Launch at 10:00 ET via Render Cron
BLOCK_ENDS_ET   = [(13,0), (16,0)]

LEVERAGE = 3.0
BAND_R_PCT = {"AAPL": 1.80, "MSFT": 1.10, "PG": 1.30, "AMD": 2.00, "JPM": 0.80, "XOM": 0.80}
EMA_HALF_LIFE_BLOCKS = 6
DPOS_CAP = 0.10
STOP_LOSS_PCT = 2.0
MAX_TICKER_GROSS_ALLOC = 0.30
LOG_FILE = "alpaca_live_trades.csv"
POLL_SECONDS = 15

# ===== ALPACA CREDS (provided by Render env vars) =====
BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
KEY_ID   = os.environ["APCA_API_KEY_ID"]
SECRET   = os.environ["APCA_API_SECRET_KEY"]
api = REST(KEY_ID, SECRET, BASE_URL, api_version="v2")
from datetime import datetime
import pytz, sys, traceback
TZ_NY = pytz.timezone("America/New_York")
try:
    acct = api.get_account()
    print(f"[START] Account={acct.status} equity=${acct.equity} buying_power=${acct.buying_power}")
    print(f"[START] Now ET: {datetime.now(TZ_NY).strftime('%Y-%m-%d %H:%M:%S')}")
except Exception:
    traceback.print_exc(); sys.exit(2)


TZ_NY = pytz.timezone("America/New_York")

def now_ny(): return datetime.now(tz=TZ_NY)

def next_block_start_and_end():
    """Pick the next start >= now and its matching end; if launched right at 10:00 ET it picks 10→13."""
    t = now_ny().replace(second=0, microsecond=0)
    starts = [t.replace(hour=h, minute=m) for (h,m) in BLOCK_STARTS_ET]
    ends   = [t.replace(hour=h, minute=m) for (h,m) in BLOCK_ENDS_ET]
    for i, dt in enumerate(starts):
        if t <= dt:
            return dt, ends[i]
    # past last start → schedule tomorrow first block
    tomorrow = t + timedelta(days=1)
    return tomorrow.replace(hour=BLOCK_STARTS_ET[0][0], minute=BLOCK_STARTS_ET[0][1]), \
           tomorrow.replace(hour=BLOCK_ENDS_ET[0][0],   minute=BLOCK_ENDS_ET[0][1])

def load_price(symbol):
    try:
        tr = api.get_latest_trade(symbol)
        return float(tr.price)
    except Exception:
        bars = api.get_bars(symbol, "1Min", limit=1)
        return float(bars[-1].c) if bars else np.nan

def account_equity(): return float(api.get_account().equity)

def get_position_qty(symbol):
    try:
        p = api.get_position(symbol)
        return int(p.qty) if p.side == "long" else -int(p.qty)
    except Exception:
        return 0

def place_market(symbol, qty, side, tag):
    if qty == 0: return
    api.submit_order(symbol=symbol, qty=abs(int(qty)), side=("buy" if side=="buy" else "sell"),
                     type="market", time_in_force="day", client_order_id=f"{symbol}-{tag}-{int(time.time())}")

def flatten(symbol):
    q = get_position_qty(symbol)
    if q != 0:
        place_market(symbol, abs(q), ("sell" if q>0 else "buy"), tag="flatten")

def write_log(row):
    df = pd.DataFrame([row])
    p  = pathlib.Path(LOG_FILE)
    df.to_csv(p, index=False, mode=("a" if p.exists() else "w"), header=not p.exists())

# === your model hook (replace with real inference) ===
def predict_block_return_pct(symbol, dt_block_start_et):
    # TODO: call your trained model to get predicted % return for [start, start+3h)
    return 0.0

def ema_alpha_from_half_life(hl):
    return 1.0 if hl<=1 else 1.0 - 0.5**(1.0/hl)

class Sizer:
    def __init__(self, band_R_pct, hl_blocks=EMA_HALF_LIFE_BLOCKS, dpos_cap=DPOS_CAP):
        self.band_R = max(1e-6, band_R_pct)  # % per 3h
        self.alpha  = ema_alpha_from_half_life(hl_blocks)
        self.dcap   = dpos_cap
        self.prev   = 0.0
    def step(self, pred_pct):
        raw = np.clip(pred_pct / self.band_R, -1.0, 1.0)
        smooth = self.alpha*raw + (1-self.alpha)*self.prev
        delta  = smooth - self.prev
        if   delta >  self.dcap: pos = self.prev + self.dcap
        elif delta < -self.dcap: pos = self.prev - self.dcap
        else: pos = smooth
        self.prev = pos
        return float(pos)

def run_one_block():
    dt_start, dt_end = next_block_start_and_end()
    print(f"=== BLOCK {dt_start.strftime('%Y-%m-%d %H:%M ET')} → {dt_end.strftime('%H:%M ET')} ===")
    if now_ny() > dt_start:
    lag_min = (now_ny() - dt_start).total_seconds() / 60.0
    print(f"[INFO] Launching mid-block (started {lag_min:.1f} minutes after block open).")

    # If we launched before market open and the next start is not today 10:00 ET, just exit.
    if dt_start.date() != now_ny().date():
        print("Not a trading window yet; exiting.")
        return

    print(f"=== BLOCK {dt_start.strftime('%Y-%m-%d %H:%M ET')} → {dt_end.strftime('%H:%M ET')} ===")
    sizers = {s: Sizer(BAND_R_PCT.get(s, 1.0)) for s in TICKERS}
    eq0 = account_equity()
    gross = eq0 * LEVERAGE
    per_sym_cap = MAX_TICKER_GROSS_ALLOC * gross

    # Enter/resize at block start
    entry_prices = {}
    for sym in TICKERS:
        price = load_price(sym)
        entry_prices[sym] = price
        pred = predict_block_return_pct(sym, dt_start)
        pos  = sizers[sym].step(pred)
        target_notional = min(abs(pos)*gross/len(TICKERS), per_sym_cap)
        qty  = int((target_notional / price) * (1 if pos>=0 else -1))
        cur  = get_position_qty(sym)
        delta = qty - cur
        if delta != 0: place_market(sym, abs(delta), ("buy" if delta>0 else "sell"), "enter")
        write_log({"ts_start": dt_start.isoformat(), "symbol": sym, "entry": price, "pred_pct": pred, "pos_frac": pos, "qty": qty, "equity_start": eq0})

    # Monitor stop until end
    while now_ny() < dt_end:
        print(f"[HB] {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')} watching until {dt_end.strftime('%H:%M:%S %Z')}")
        time.sleep(POLL_SECONDS)
        for sym in TICKERS:
            q = get_position_qty(sym)
            if q == 0: continue
            side = 1 if q>0 else -1
            px0  = entry_prices.get(sym, load_price(sym))
            px1  = load_price(sym)
            pnl_pct_lev = side * ((px1/px0)-1.0) * 100.0 * LEVERAGE
            if pnl_pct_lev <= -STOP_LOSS_PCT:
                print(f"[{sym}] STOP {pnl_pct_lev:.2f}% → flatten")
                flatten(sym)

    # Block end: flatten all
    for sym in TICKERS: flatten(sym)
    print("Block done, exiting.")

if __name__ == "__main__":
    run_one_block()
