# app/arl_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import time
import numpy as np
import pandas as pd


# =========================
# Public dataclasses / API
# =========================

@dataclass
class UserStyle:
    name: str
    # Universe building
    lookback_days: int = 30
    max_symbols: int = 8
    min_liquidity_pctile: float = 20.0  # filter out illiquid tails
    # Trading preferences
    prefer_momentum: bool = True
    prefer_mean_reversion: bool = False
    risk_level: str = "high"  # "high" | "medium" | "low"
    # Base policy knobs (overridable per block/symbol)
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
            base_band_R=1.05,
            base_ema_hl=6,
            base_size_mult=1.10,
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
        base_band_R=1.20,
        base_ema_hl=10,
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
    Agentic bandit:
      - Universe selection (ε-greedy)
      - Per-symbol gating & dynamic overrides
      - Reward = caller-provided % (now: net PnL% after fees+slippage)
    Persistence supported via export/import of internal state.
    """

    def __init__(self, style: UserStyle, persisted: Optional[dict] = None):
        self.style: UserStyle = style

        # Online reward tracking
        self._reward_ema: Dict[str, float] = {}
        self._reward_count: Dict[str, int] = {}
        self._last_seen_ts: Dict[str, float] = {}

        # Exploration schedule (ε-greedy)
        self._eps: float = 0.20 if style.risk_level == "high" else (0.12 if style.risk_level == "medium" else 0.08)
        self._eps_min: float = 0.04
        self._eps_decay: float = 0.995

        # Filters
        self._min_sigma_pct: float = 0.20
        self._liquidity_floor_prctile: float = style.min_liquidity_pctile

        if isinstance(persisted, dict) and persisted:
            try:
                re = persisted.get("reward_ema", {})
                rc = persisted.get("reward_count", {})
                ls = persisted.get("last_seen_ts", {})
                eps = persisted.get("eps", None)
                if isinstance(re, dict): self._reward_ema = {k: float(v) for k, v in re.items()}
                if isinstance(rc, dict): self._reward_count = {k: int(v) for k, v in rc.items()}
                if isinstance(ls, dict): self._last_seen_ts = {k: float(v) for k, v in ls.items()}
                if isinstance(eps, (int, float)): self._eps = float(eps)
            except Exception:
                pass

    # -------- 3.3: Universe building --------
    def build_selection_frame(self, sym_to_df: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
        rows = []
        for sym, df in sym_to_df.items():
            if df is None or df.empty:
                continue
            last = df.iloc[-1]
            close = float(last.get("close", np.nan))
            vol20 = float(last.get("vol20", last.get("sigma20_pct", np.nan)))
            ret5  = float(last.get("ret5", 0.0))
            volume = float(last.get("volume", np.nan))
            try:
                liq_proxy = float(df["volume"].tail(20).median())
            except Exception:
                liq_proxy = volume
            rows.append({"symbol": sym, "close": close, "ret5": ret5, "sigma_pct": vol20, "liq": liq_proxy})

        if not rows:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        feats = pd.DataFrame(rows).dropna(subset=["close","sigma_pct","liq"])
        if feats.empty:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        feats = feats.loc[feats["sigma_pct"] >= self._min_sigma_pct]
        if feats.empty:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        q = np.percentile(feats["liq"].values, self._liquidity_floor_prctile)
        feats = feats.loc[feats["liq"] >= q]
        if feats.empty:
            return pd.DataFrame(columns=["symbol","score","ret5","sigma_pct","liq"]).set_index("symbol")

        def z(x):
            mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
            sd = sd if sd > 1e-9 else 1.0
            return (x - mu) / sd

        feats["z_mom"] = z(feats["ret5"].fillna(0.0))
        feats["z_vol"] = z(feats["sigma_pct"].clip(lower=1e-6))
        feats["z_mr"]  = -feats["z_mom"].abs()

        w_mom = 0.9 if self.style.prefer_momentum else 0.3
        w_mr  = 0.8 if self.style.prefer_mean_reversion else 0.2
        w_vol = 0.3 if self.style.risk_level == "high" else (0.15 if self.style.risk_level == "medium" else 0.0)

        feats["score"] = w_mom * feats["z_mom"] + w_mr * feats["z_mr"] + w_vol * feats["z_vol"]
        feats = feats.sort_values("score", ascending=False).set_index("symbol")
        return feats[["score","ret5","sigma_pct","liq"]]

    def select_universe(self, feats: pd.DataFrame) -> List[str]:
        if feats is None or feats.empty:
            return []
        n = self.style.max_symbols
        if np.random.rand() >= self._eps:
            return list(feats.head(n).index)
        k = len(feats)
        lo, hi = int(0.25*k), int(0.75*k)
        mid = feats.iloc[lo:hi] if hi > lo else feats
        pick = min(n, max(1, len(mid)))
        return list(mid.sample(pick, replace=False, random_state=np.random.randint(1_000_000)).index)

    # -------- 3.4A–D: Decisions / overrides --------
    def decide_for_symbol(self, sym: str, ctx: BlockContext) -> Decision:
        if ctx.minutes_to_close <= 30.0:
            return Decision(allow=False)
        if ctx.is_friday and ctx.is_late and self.style.risk_level != "high":
            return Decision(allow=False)

        r_ema = self._reward_ema.get(sym, 0.0)
        conf = 1.0 / (1.0 + math.exp(-0.8 * r_ema))  # sigmoid on %-PnL

        base_band, base_hl, base_sz, base_hold = (
            self.style.base_band_R, self.style.base_ema_hl, self.style.base_size_mult, self.style.base_hold_min_blocks
        )

        band_R = float(np.interp(conf, [0.0, 1.0], [base_band * 1.20, base_band * 0.85]))
        ema_hl = int(round(np.interp(conf, [0.0, 1.0], [max(3, base_hl + 3), max(2, base_hl - 2)])))
        size_m = float(np.interp(conf, [0.0, 1.0], [base_sz * 0.70, base_sz * 1.30]))
        hold_b = int(round(np.interp(conf, [0.0, 1.0], [max(1, base_hold), max(1, base_hold + 1)])))

        if ctx.is_friday:
            if ctx.is_late:
                band_R *= 1.10
                size_m *= 0.85
                hold_b = max(1, hold_b - 1)
            else:
                size_m *= 0.95

        size_m = float(np.clip(size_m, 0.6 * base_sz, 1.3 * base_sz))

        return Decision(True, band_R_override=band_R, ema_hl_override=ema_hl, size_mult=size_m, hold_min_blocks=hold_b)

    # -------- 3.5: Rewards / annealing --------
    def update_rewards(self, sym_to_reward_pct: Dict[str, float]) -> None:
        """
        Reward units are percent (%), caller supplies **net PnL%** per symbol for the block.
        """
        if not sym_to_reward_pct:
            self._eps = max(self._eps_min, self._eps * self._eps_decay)
            return

        now = time.time()
        alpha = 0.25 if self.style.risk_level == "high" else (0.20 if self.style.risk_level == "medium" else 0.15)

        for sym, r in sym_to_reward_pct.items():
            prev = self._reward_ema.get(sym, 0.0)
            ema = (1.0 - alpha) * prev + alpha * float(r)
            self._reward_ema[sym] = ema
            self._reward_count[sym] = self._reward_count.get(sym, 0) + 1
            self._last_seen_ts[sym] = now

        self._eps = max(self._eps_min, self._eps * self._eps_decay)

    # -------- Persistence I/O --------
    def export_state(self) -> dict:
        return {
            "reward_ema": self._reward_ema,
            "reward_count": self._reward_count,
            "last_seen_ts": self._last_seen_ts,
            "eps": float(self._eps),
        }
