# app/agent/arl_agent.py
import math, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------- User style ----------
@dataclass
class UserStyle:
    name: str  # "high_risk_short_term" | "low_risk_long_term"
    target_dd_pct: float           # max acceptable drawdown (for reward penalty)
    prefer_vol_rank: bool          # prefers high or low volatility
    prefer_momentum: bool          # favors trending names
    min_liquidity_usd: float       # skip illiquid
    lookback_days: int             # feature window for selection
    max_symbols: int               # cap trade universe per block

def default_style(style_name: str) -> UserStyle:
    style_name = (style_name or "").lower()
    if style_name in ("high_risk_short_term", "high", "day_trading", "scalper"):
        return UserStyle(
            name="high_risk_short_term",
            target_dd_pct=6.0,
            prefer_vol_rank=True,
            prefer_momentum=True,
            min_liquidity_usd=5_000_000,
            lookback_days=60,
            max_symbols=8
        )
    # default low-risk
    return UserStyle(
        name="low_risk_long_term",
        target_dd_pct=3.0,
        prefer_vol_rank=False,
        prefer_momentum=True,
        min_liquidity_usd=3_000_000,
        lookback_days=60,
        max_symbols=8
    )

# ---------- Small helpers ----------
def _pct_change(a):
    a = np.asarray(a, float)
    if a.size < 2: return 0.0
    return (a[-1] / a[0] - 1.0) * 100.0

def _safe_std(a):
    a = np.asarray(a, float)
    return float(np.nanstd(a)) if a.size else 0.0

# ---------- Agent core ----------
@dataclass
class Decision:
    # per-symbol overrides decided by the agent
    allow: bool
    size_mult: float           # multiply target position fraction
    band_R_override: Optional[float] = None
    ema_hl_override: Optional[int] = None
    hold_min_blocks: Optional[int] = None

@dataclass
class BlockContext:
    is_friday: bool
    is_late: bool
    minutes_to_close: float
    equity: float

@dataclass
class PolicyMemory:
    # Moving rewards per symbol; use EMA to stabilize
    ema_reward: Dict[str, float] = field(default_factory=dict)
    ema_alpha: float = 0.2
    # Per-symbol size multipliers adapted by reward
    size_mult: Dict[str, float] = field(default_factory=dict)

class ARLAgent:
    """
    Lightweight, online-updating 'policy' that:
      - selects symbols based on last ~2 months features,
      - returns per-symbol decision (allow, size multiplier, risk knobs),
      - updates rewards after each block using realized P&L signal you pass in.
    """
    def __init__(self, style: UserStyle):
        self.style = style
        self.mem = PolicyMemory()
        # Base risk knobs; adapted slightly with reward
        self.base_band_R = 1.10
        self.base_ema_hl = 8
        self.base_hold_min_blocks = 2

    # ---------- feature builder for selection ----------
    def build_selection_frame(
        self,
        sym_to_df: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Input: map symbol -> hourly DataFrame with ['timestamp','close','volume','vwap'] (or more)
        Returns: DataFrame with per-symbol features for selection.
        """
        rows = []
        for sym, df in sym_to_df.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            # restrict to last N days
            ts = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=self.style.lookback_days)
            df2 = df.loc[ts.tz_convert("UTC").tz_localize(None) >= cutoff].copy()
            if df2.empty: continue

            px = df2["close"].to_numpy(float)
            # hourly returns in %
            r1 = (pd.Series(px).pct_change() * 100.0).fillna(0.0).to_numpy()
            vol_std = _safe_std(r1)
            mom_10h = _pct_change(px[-10:]) if px.size >= 10 else 0.0
            mom_40h = _pct_change(px[-40:]) if px.size >= 40 else 0.0
            # proxy liquidity: last price * median volume (very rough)
            if "volume" in df2.columns:
                med_vol = float(np.nanmedian(df2["volume"].to_numpy(float)))
            else:
                med_vol = 0.0
            last_px = float(px[-1])
            liquidity_proxy = last_px * med_vol

            rows.append({
                "symbol": sym,
                "vol_std": vol_std,
                "mom_10h": mom_10h,
                "mom_40h": mom_40h,
                "liquidity_proxy": liquidity_proxy,
            })
        return pd.DataFrame(rows)

    def select_universe(self, feats: pd.DataFrame) -> List[str]:
        """
        Choose symbols for this block, following user style.
        """
        if feats is None or feats.empty:
            return []
        # liquidity filter
        f = feats.loc[feats["liquidity_proxy"] >= self.style.min_liquidity_usd].copy()
        if f.empty:  # fallback
            f = feats.copy()

        # score = weighted combination
        # high risk: favor higher vol + positive momentum
        # low  risk: favor lower vol but still positive momentum
        if self.style.prefer_vol_rank:
            # higher vol better
            f["vol_score"] = (f["vol_std"] - f["vol_std"].min()) / max(1e-9, (f["vol_std"].max()-f["vol_std"].min()))
        else:
            # lower vol better
            vmax, vmin = f["vol_std"].max(), f["vol_std"].min()
            f["vol_score"] = 1.0 - (f["vol_std"] - vmin) / max(1e-9, (vmax - vmin))
        # momentum (blend short and medium)
        f["mom_score"] = 0.6 * np.tanh(f["mom_10h"]/5.0) + 0.4 * np.tanh(f["mom_40h"]/10.0)
        f["score"] = 0.6 * f["mom_score"] + 0.4 * f["vol_score"]
        f = f.sort_values("score", ascending=False)
        return list(f["symbol"].head(self.style.max_symbols).values)

    # ---------- decision per symbol ----------
    def decide_for_symbol(
        self,
        sym: str,
        ctx: BlockContext
    ) -> Decision:
        # base size multiplier from reward memory
        sz = self.mem.size_mult.get(sym, 1.0)
        reward_ema = self.mem.ema_reward.get(sym, 0.0)
        # nudge size by reward (cap within [0.5, 1.5])
        if reward_ema > 0.0:
            sz = min(1.5, 1.0 + 0.5 * np.tanh(reward_ema / 0.02))
        elif reward_ema < 0.0:
            sz = max(0.5, 1.0 - 0.5 * np.tanh(abs(reward_ema) / 0.02))
        self.mem.size_mult[sym] = sz

        # adapt hold time slightly: longer when reward rising, shorter when falling
        hold = self.base_hold_min_blocks
        if reward_ema > 0.0: hold = min(3, hold + 1)
        if reward_ema < 0.0: hold = max(1, hold - 1)

        # small adapt of band_R / ema_hl
        band_R = self.base_band_R * (1.1 if self.style.name == "low_risk_long_term" else 1.0)
        ema_hl = self.base_ema_hl

        return Decision(
            allow=True,
            size_mult=float(sz),
            band_R_override=float(band_R),
            ema_hl_override=int(ema_hl),
            hold_min_blocks=int(hold),
        )

    # ---------- reward update ----------
    def update_rewards(
        self,
        sym_to_realized_ret_pct: Dict[str, float]
    ):
        """
        Call once per block with realized % returns by symbol (unlevered or net—be consistent).
        We keep a per-symbol EMA of reward and adjust size multipliers next block.
        """
        a = self.mem.ema_alpha
        for sym, r in (sym_to_realized_ret_pct or {}).items():
            prev = self.mem.ema_reward.get(sym, 0.0)
            cur = (1 - a) * prev + a * float(r)
            self.mem.ema_reward[sym] = cur
