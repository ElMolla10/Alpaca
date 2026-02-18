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

        def compute_symbol_pnl_pct(self, sym: str) -> float | None:
        fills = self._fills.get(sym, [])
        if not fills:
            return None

        buy_notional  = sum(f.qty * f.price for f in fills if f.side == "buy")
        sell_notional = sum(f.qty * f.price for f in fills if f.side == "sell")

        # Entry basis: use the side of the first fill as the "entry side"
        first_side = fills[0].side
        gross_exposed = buy_notional if first_side == "buy" else sell_notional

        pnl_abs = sell_notional - buy_notional
        costs = sum(f.fees_abs for f in fills)
        pnl_abs -= costs

        return 0.0 if gross_exposed <= 0 else (pnl_abs / gross_exposed) * 100.0

    def clear_symbol(self, sym: str) -> None:
        if sym in self._fills:
            del self._fills[sym]

    def fills_count(self, sym: str) -> int:
        return len(self._fills.get(sym, []))
