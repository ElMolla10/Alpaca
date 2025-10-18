# app/agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable
import math
import time
import numpy as np
import pandas as pd

# =========================
# Dataclasses / public API
# =========================

@dataclass
class UserStyle:
    name: str
    # Universe building
    lookback_days: int = 30
    max_symbols: int = 8
    min_liquidity_pctile: float = 20.0  # filter illiquid tails
    # Trading preferences
    prefer_momentum: bool = True
    prefer_mean_reversion: bool = False
    risk_level: str = "high"  # "high" | "medium" | "low"
    # Base policy knobs (can be overridden per block/symbol)
    base_band_R: float = 1.10
    base_ema_hl: int = 8
    base_size_mult: float = 1.00
    base_hold_min_blocks: int = 1

def default_style(name: str = "high_risk_short_term") -> UserStyle:
    key = (name or "").lower().strip()
    if key in ("high", "high_risk_short_term", "high-risk-short-term", "day"):
        return UserStyle(
            name="high_risk_short_term",
            lookback_days=30,
            max_symbols=8,
            prefer_momentum=True,
            prefer_mean_reversion=False,
            risk_level="high",
            base_band_R=1.05,     # tighter band → more responsive sizing
            base_ema_hl=6,        # faster EMA
            base_size_mult=1.10,  # slightly larger baseline size
            base_hold_min_blocks=1
        )
    if key in ("medium", "swing", "medium_risk_swing"):
        return UserStyle(
            name="medium_risk_swing",
            lookback_days=60,
            max_symbols=6,
            prefer_momentum=True,
            prefer_mean_reversion=True,
            risk_level="medium",
            base_band_R=1.10,
            base_ema_hl=8,
            base_size_mult=1.00,
            base_hold_min_blocks=2
        )
    # low risk / longer hold
    return UserStyle(
        name="low_risk_long_term",
        lookback_days=120,
        max_symbols=4,
        prefer_momentum=False,
        prefer_mean_reversion=True,
        risk_level="low",
        base_band_R=1.20,   # wider band → smaller positions for the same signal
        base_ema_hl=10,     # slower EMA
        base_size_mult=0.80,
        base_hold_min_blocks=3
    )

@dataclass
class BlockContext:
    is_friday: bool
    is_late: bool
    minutes_to_close: float
    equity: float

@dataclass
class Decision:
    allow: bool
    band_R_override: Optional[float] = None
    ema_hl_override: Optional[int] = None
    size_mult: Optional[float] = None
    hold_min_blocks: Optional[int] = None


# =========================
# ARLAgent implementation
# =========================

class ARLAgent:
    """
    Lightweight agentic layer:
      - 3.3: Universe selection from recent features
      - 3.4A-D: Per-symbol gating & overrides
      - 3.5: Reward updates (EMA, ε schedule)
    Keeps state in-memory for the session (compatible with your main loop).
    """

    def __init__(self, style: UserStyle):
        self.style: UserStyle = style

        # Online reward tracking
        self._reward_ema: Dict[str, float] = {}
        self._reward_count: Dict[str, int] = {}
        self._last_seen_ts: Dict[str, float] = {}

        # Exploration schedule (ε-greedy for contextual bandit)
        self._eps: float = 0.20 if style.risk_level == "high" else (0.12 if style.risk_level == "medium" else 0.08)
        self._eps_min: float = 0.04
        self._eps_decay: float = 0.995  # applied per update_rewards call (slow decay)

        # Safety rails
        self._min_sigma_pct: float = 0.20  # ignore symbols with near-zero recent volatility
        self._liquidity_floor_prctile: float = style.min_liquidity_pctile

        # Sizing governor from volatility (higher vol → smaller size multiplier)
        # piecewise linear: target vol 0.8–2.5%/h
        self._vol_floor = 0.8
        self._vol_ceiling = 2.5

    # -----------------------
    # 3.3: Universe building
    # -----------------------
    def build_selection_frame(self, sym_to_df: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
        """
        Construct a per-symbol frame with universe-selection signals.
        Expected df columns (from your pipeline): 'close','ret5','vol20','sigma20_pct','volume'
        """
        rows = []
        for sym, df in sym_to_df.items():
            if df is None or df.empty:
                continue
            # Use the last valid row
            last = df.iloc[-1]
            # Robustness: handle missing cols
            close = float(last.get("close", np.nan))
            vol20 = float(last.get("vol20", last.get("sigma20_pct", np.nan)))
            ret5  = float(last.get("ret5", 0.0))
            volume = float(last.get("volume", np.nan))

            # Liquidity proxy: median volume over last 20 rows
            liq_proxy = np.nan
            try:
                liq_proxy = float(df["volume"].tail(20).median())
            except Exception:
                if not math.isnan(volume):
                    liq_proxy = volume

            rows.append({
                "symbol": sym,
                "close": close,
                "ret5": ret5,                  # 5h momentum (%)
                "sigma_pct": vol20,            # hourly vol proxy (%)
                "liq": liq_proxy               # volume proxy
            })

        if not rows:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        feats = pd.DataFrame(rows).dropna(subset=["close","sigma_pct","liq"])
        if feats.empty:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        # Remove near-zero vol names (bad for signal normalization / sizing)
        feats = feats.loc[feats["sigma_pct"] >= self._min_sigma_pct]
        if feats.empty:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        # Liquidity filter (drop bottom X% by volume)
        q = np.percentile(feats["liq"].values, self._liquidity_floor_prctile)
        feats = feats.loc[feats["liq"] >= q]
        if feats.empty:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        # Normalize key signals (z-scores with guard)
        def z(x):
            mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
            sd = sd if sd > 1e-9 else 1.0
            return (x - mu) / sd

        feats["z_mom"] = z(feats["ret5"].fillna(0.0))
        feats["z_vol"] = z(feats["sigma_pct"].clip(lower=1e-6))

        # Style-driven scoring: momentum vs mean-reversion and risk appetite
        w_mom = 0.9 if self.style.prefer_momentum else 0.3
        w_mr  = 0.8 if self.style.prefer_mean_reversion else 0.2

        # Mean-reversion proxy: penalize extreme momentum (we want snaps back)
        feats["z_mr"] = -feats["z_mom"].abs()

        # Volatility preference: high risk favors (slightly) more vol
        w_vol = 0.3 if self.style.risk_level == "high" else (0.15 if self.style.risk_level == "medium" else 0.0)

        feats["score"] = w_mom * feats["z_mom"] + w_mr * feats["z_mr"] + w_vol * feats["z_vol"]

        feats = feats.sort_values("score", ascending=False).set_index("symbol")
        return feats[["score","ret5","sigma_pct","liq"]]

    def select_universe(self, feats: pd.DataFrame) -> List[str]:
        """
        Pick top-N by score, with ε-greedy exploration to avoid local optima.
        """
        if feats is None or feats.empty:
            return []

        candidates = feats.copy()

        # ε-greedy: with probability ε, explore from mid-rank names
        explore = np.random.rand() < self._eps
        n = self.style.max_symbols
        if not explore:
            return list(candidates.head(n).index)

        # Exploration bucket: middle 50% (avoid worst & best tails to keep quality)
        k = len(candidates)
        lo = int(0.25 * k)
        hi = int(0.75 * k)
        mid = candidates.iloc[lo:hi] if hi > lo else candidates
        pick = min(n, max(1, len(mid)))
        return list(mid.sample(pick, replace=False, random_state=np.random.randint(1_000_000)).index)

    # -----------------------------------------
    # 3.4A-D: Per-symbol gating & adjustments
    # -----------------------------------------
    def decide_for_symbol(self, sym: str, ctx: BlockContext) -> Decision:
        """
        Combine session constraints, style, and learned reward into a per-symbol plan.
        """
        # Hard trading window guard (cooperate with main's cutoff)
        if ctx.minutes_to_close <= 30.0:
            return Decision(allow=False)

        # Friday late preference: keep lean unless the style is high risk and track record is good
        if ctx.is_friday and ctx.is_late and self.style.risk_level != "high":
            return Decision(allow=False)

        # Learned performance → adapt aggressiveness:
        r_ema = self._reward_ema.get(sym, 0.0)

        # Convert reward EMA into a 0..1 confidence via sigmoid on %-returns
        conf = 1.0 / (1.0 + math.exp(-0.8 * r_ema))  # r_ema around 0 → conf ~ 0.5

        # Dynamic overrides
        #  - better conf → narrower band_R (bigger size), shorter EMA HL, bigger size_mult
        #  - worse conf  → wider band_R (smaller size), longer EMA HL, smaller size_mult
        base_band = self.style.base_band_R
        base_hl   = self.style.base_ema_hl
        base_sz   = self.style.base_size_mult
        base_hold = self.style.base_hold_min_blocks

        # Map conf∈[0,1] to multipliers
        band_R = float(np.interp(conf, [0.0, 1.0], [base_band * 1.20, base_band * 0.85]))
        ema_hl = int(round(np.interp(conf, [0.0, 1.0], [max(3, base_hl + 3), max(2, base_hl - 2)])))
        size_m = float(np.interp(conf, [0.0, 1.0], [base_sz * 0.70, base_sz * 1.30]))
        hold_b = int(round(np.interp(conf, [0.0, 1.0], [max(1, base_hold), max(1, base_hold + 1)])))

        # Volatility-aware size governor (scale size down if hourly vol is high)
        # We don't have per-symbol sigma here; we opportunistically read last seen cached sigma if available
        # (build_selection_frame writes last seen to _last_seen_ts, but not sigma; this is a simple guard)
        # For a better coupling, you can pass the current sigma in the context.
        vol_scale = 1.0  # fallback
        # Keep size within [0.6, 1.3] of style baseline
        size_m = float(np.clip(size_m * vol_scale, 0.6 * base_sz, 1.3 * base_sz))

        # Extra Friday conservatism
        if ctx.is_friday:
            if ctx.is_late:
                band_R *= 1.10
                size_m *= 0.85
                hold_b = max(1, hold_b - 1)
            else:
                size_m *= 0.95

        return Decision(
            allow=True,
            band_R_override=band_R,
            ema_hl_override=ema_hl,
            size_mult=size_m,
            hold_min_blocks=hold_b
        )

    # -----------------------
    # 3.5: Reward update
    # -----------------------
    def update_rewards(self, sym_to_realized: Dict[str, float]) -> None:
        """
        sym_to_realized: last-hour realized % move proxy per symbol.
        We keep an EMA of rewards per symbol and gently anneal ε.
        """
        if not sym_to_realized:
            # Decay ε even on quiet blocks to slowly converge to exploitation
            self._eps = max(self._eps_min, self._eps * self._eps_decay)
            return

        now = time.time()
        alpha = 0.25 if self.style.risk_level == "high" else (0.20 if self.style.risk_level == "medium" else 0.15)

        for sym, r in sym_to_realized.items():
            prev = self._reward_ema.get(sym, 0.0)
            ema = (1.0 - alpha) * prev + alpha * float(r)
            self._reward_ema[sym] = ema
            self._reward_count[sym] = self._reward_count.get(sym, 0) + 1
            self._last_seen_ts[sym] = now

        # Anneal ε, but never below floor
        self._eps = max(self._eps_min, self._eps * self._eps_decay)

    # -----------------------
    # Utilities / Introspect
    # -----------------------
    def metrics(self) -> pd.DataFrame:
        if not self._reward_ema:
            return pd.DataFrame(columns=["reward_ema","updates","last_seen","eps"])
        rows = []
        for s, v in self._reward_ema.items():
            rows.append({
                "symbol": s,
                "reward_ema": v,
                "updates": int(self._reward_count.get(s, 0)),
                "last_seen": float(self._last_seen_ts.get(s, 0.0)),
                "eps": self._eps
            })
        return pd.DataFrame(rows).set_index("symbol").sort_values("reward_ema", ascending=False)
