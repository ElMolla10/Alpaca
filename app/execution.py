# app/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict
import time

@dataclass
class Fill:
    ts: float
    symbol: str
    side: str           # "buy" | "sell"
    qty: float
    price: float
    notional: float     # signed (buy +, sell -)
    fees_abs: float     # absolute $ costs (fees + slippage)

class BlockLedger:
    """
    Per-block trade ledger that computes **net PnL% after fees+slippage** per symbol.
    If you don't have real-time fills, it's valid to record at the last trade price as an approximation.
    """
    def __init__(self, trade_cost_bps: float, slip_bps: float):
        self.trade_cost_bps = float(trade_cost_bps)
        self.slip_bps = float(slip_bps)
        self._fills: Dict[str, List[Fill]] = defaultdict(list)

    def record_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        notional_signed = (qty * price) * (1.0 if side == "buy" else -1.0)
        bps = (self.trade_cost_bps + self.slip_bps) / 10_000.0
        fees_abs = abs(notional_signed) * bps
        self._fills[symbol].append(
            Fill(ts=time.time(), symbol=symbol, side=side, qty=qty, price=price,
                 notional=notional_signed, fees_abs=fees_abs)
        )

    def compute_block_pnl_pct(self) -> Dict[str, float]:
        """
        Returns per-symbol net PnL% normalized by gross entry notional in this block.
        Assumes end-of-block flattening (your main loop does this).
        """
        out: Dict[str, float] = {}
        for sym, fills in self._fills.items():
            if not fills:
                continue

            # Exposure basis: count only entries/extends (abs of buys when opening or sells when shorting)
            # Simple proxy that works with hourly-open/flatten workflow:
            gross_exposed = sum(abs(f.notional) for f in fills if f.notional > 0)

            # Realized PnL ≈ sum(sell_notional) - sum(buy_notional) - fees
            buy_notional  = sum(f.qty * f.price for f in fills if f.side == "buy")
            sell_notional = sum(f.qty * f.price for f in fills if f.side == "sell")
            pnl_abs = sell_notional - buy_notional
            costs = sum(f.fees_abs for f in fills)
            pnl_abs -= costs

            out[sym] = 0.0 if gross_exposed <= 0 else (pnl_abs / gross_exposed) * 100.0
        return out

    def reset(self) -> None:
        self._fills.clear()
