<div align="center">

```
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝

███████╗      ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗
██╔════╝      ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝
█████╗    ███    ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
██╔══╝   ╚═══╝   ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
███████╗         ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
╚══════╝         ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝
```

### *A Production-Grade Agentic RL Trading System*

---

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14+-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Prisma_ORM-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1200+_Trees-FF6600?style=for-the-badge)
![Alpaca](https://img.shields.io/badge/Alpaca-Paper_&_Live-FECC00?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-00C853?style=for-the-badge)

</div>

---

## What Is This?

The **Agentic E-Trading Engine** is a full-stack automated trading platform that combines supervised machine learning with online reinforcement learning to make, execute, and monitor trading decisions in real time — without human intervention.

It is not just a trading bot. It is a complete system: from raw market data ingestion, through ensemble ML inference, through an RL agent that dynamically sizes positions based on its own performance history, through a cascade of hard risk guardrails, all the way to a live web dashboard with real-time monitoring and full control over the trading engine.

The architecture answers a question that most trading systems avoid: **what happens when the market turns against you?** Not just does it trade, but can it protect capital under extreme conditions? Does it adapt dynamically to changing performance? Does it fail gracefully?

This system does all of that.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Intelligence Stack](#the-intelligence-stack)
   - [Feature Engineering](#feature-engineering)
   - [Ensemble Prediction Layer](#ensemble-prediction-layer)
   - [The RL Agent (ARLAgent)](#the-rl-agent-arlagent)
   - [Hybrid Architecture](#hybrid-architecture)
3. [Position Sizing & Execution](#position-sizing--execution)
4. [Risk Management: 7-Layer Guardrails](#risk-management-7-layer-guardrails)
5. [Web Dashboard](#web-dashboard)
6. [Full Project Structure](#full-project-structure)
7. [Configuration Reference](#configuration-reference)
8. [Running the System](#running-the-system)
9. [Environment Variables](#environment-variables)
10. [Research Foundations](#research-foundations)
11. [Contributors](#contributors)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BROWSER CLIENT                              │
│              Next.js Dashboard  ·  JWT Auth  ·  REST                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS / JSON
┌──────────────────────────────▼──────────────────────────────────────┐
│                         API GATEWAY (Node.js)                        │
│         Agent Controller  ·  Auth Service  ·  REST Routes            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          ┌──────▼──────┐           ┌────────▼────────┐
          │  PostgreSQL │           │  Trading Engine  │
          │  (Prisma)   │           │    (Python)      │
          │             │           │                  │
          │ • Users     │           │ • Data Feed      │
          │ • Trades    │           │ • ML Models      │
          │ • Metrics   │           │ • RL Agent       │
          │ • AgentState│           │ • Risk Guards    │
          └─────────────┘           │ • Execution      │
                                    └────────┬─────────┘
                                             │ HTTPS / JSON
                                    ┌────────▼─────────┐
                                    │   Alpaca API      │
                                    │  Paper / Live     │
                                    │  Market Data      │
                                    │  Order Routing    │
                                    └──────────────────┘
```

The system follows a **client-server architecture** with a layered backend and a standalone trading engine. The trading engine is the only component that communicates with the broker. The web dashboard never touches the broker directly. All components share a centralized PostgreSQL database (via Prisma ORM).

---

## The Intelligence Stack

### Feature Engineering

Every trading decision starts with a 43-feature vector computed from 1-hour OHLCV bars. These features are designed to give the model a complete picture of current price action, trend state, and volatility regime:

| Category | Features |
|---|---|
| **Price dynamics** | `price_change_pct`, `price_change_pct_lag1/2/3`, `ret5` |
| **Volatility** | `vol20`, `sigma20_pct` (20-period rolling σ) |
| **Momentum (MACD)** | `MACDh_12_26_9`, `MACDh_12_26_9_lag2/3` |
| **Band position (BB)** | `BBM_20_2.0`, `BBM_20_2.0_lag2/3` |
| **Extended lags** | All key features are lagged 1–3 blocks back |

Features are computed using the `ta` library (MACD via `ta.trend.MACD`, Bollinger via `ta.volatility.BollingerBands`) from live Alpaca bar data. Lookahead bias is prevented by construction: features at decision time `t` only use data up to `t-1`, enforced by daily state resets and the block-boundary timestamp protocol.

---

### Ensemble Prediction Layer

Two complementary models run in parallel and their outputs are averaged into a single `pred_pct` signal — a predicted 1-hour return percentage:

```python
# XGBoost — 1200+ trees, 43 features, gradient-boosted ensemble
dmat = xgb.DMatrix(X, feature_names=FEAT_COLS)
pred_xgb = float(xgb_booster.predict(dmat)[0])

# ElasticNet — L1+L2 regularized linear regression (sklearn)
pred_enet = float(enet_model.predict(X)[0])

# Ensemble mean → directional bias signal
pred_pct = mean([pred_xgb, pred_enet])
```

**XGBoost** captures non-linear interactions between features — the dominant signal in volatile, momentum-driven regimes. **ElasticNet** provides a regularized linear anchor that resists overfitting to noise. Together, they hedge each other's failure modes.

The ensemble also outputs:
- `p_up` — pseudo-probability of upward move (sigmoid of pred_pct)
- `vol_est` — current volatility estimate (sigma20_pct)
- `per_model` — individual model outputs for debugging

> ⚠️ **Critical Design Choice**: These models provide a *directional bias hint* — not a hard buy/sell signal. The RL agent decides whether and how much to trust that hint based on its own performance history.

---

### The RL Agent (ARLAgent)

The core of the system is a **contextual bandit with ε-greedy exploration and EMA reward tracking**. It does not make buy/sell decisions — it *modulates* the supervised signal by learning which symbols and market contexts are currently profitable, then scaling positions up or down accordingly.

```
ARLAgent
│
├── Universe Building (ε-greedy symbol selection)
│   ├── build_selection_frame()  — scores all symbols by vol, liquidity, momentum
│   ├── ε-greedy selection       — explores new symbols with prob ε
│   └── ε-decay schedule         — eps * 0.995 per block, floor at 0.04
│
├── Per-Symbol Policy (decide_for_symbol)
│   ├── r_ema = EMA of past PnL rewards for this symbol
│   ├── conf = sigmoid(0.8 × r_ema)    — [0, 1] confidence score
│   ├── band_R    ← interp(conf, [1.20×base, 0.85×base])
│   ├── ema_hl    ← interp(conf, [base+3, base-2])
│   ├── size_mult ← interp(conf, [0.70×base, 1.30×base])
│   └── hold_min_blocks ← interp(conf, [base, base+1])
│
└── Reward Update (update_rewards)
    ├── alpha = 0.25 / 0.20 / 0.15  (high / medium / low risk)
    ├── EMA[sym] = (1-α) × EMA[sym] + α × net_pnl_pct
    └── ε decay applied after every reward batch
```

**The reward signal** that drives all learning:

$$\text{Reward} = \frac{\text{PnL} - (\text{Fees} + \text{Slippage})}{\text{Exposure}}$$

This is a risk-adjusted PnL percentage. Fees (`8 BPS`) and slippage (`4 BPS`) are deducted before computing the reward — the agent only gets credit for *real* profit after all costs.

**Three configurable risk profiles** (UserStyle):

| Style | ε initial | α (reward EMA) | max_symbols | hold_min | band_R |
|---|---|---|---|---|---|
| `high_risk_short_term` | 0.20 | 0.25 | 8 | 1 block | 1.05 |
| `medium_risk_swing` | 0.12 | 0.20 | 6 | 2 blocks | 1.10 |
| `low_risk_long_term` | 0.08 | 0.15 | 4 | 3 blocks | 1.20 |

---

### Hybrid Architecture

The system implements a **Predict-then-Modulate** pattern — a practical realization of the Smart Predict-then-Optimize (SPO) paradigm:

```
Raw Market Data
      │
      ▼
Feature Engineering (43 features)
      │
      ▼
┌─────────────────────┐
│  Supervised Models  │  ← XGBoost + ElasticNet ensemble
│  → pred_pct         │     (direction + magnitude)
│  → p_up             │
│  → vol_est          │
└──────────┬──────────┘
           │  directional bias signal
           ▼
┌─────────────────────┐
│    RL Agent         │  ← learns from realized PnL
│  → size_mult        │     (amplifies / dampens signal)
│  → band_R           │     based on past performance
│  → ema_hl           │
└──────────┬──────────┘
           │  confidence-scaled signal
           ▼
┌─────────────────────┐
│  Position Sizing    │  ← EMA smoothing + delta cap
│  → target_frac      │     + absolute cap
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Risk Guardrails    │  ← 7 independent hard gates
│  (may veto any      │     (override everything above)
│   trade)            │
└──────────┬──────────┘
           │
           ▼
      Alpaca API Order
```

The RL agent never picks direction. The supervised models do. The RL agent only asks: *"Given what I know about how well these models have been performing on this symbol lately — should I trust them more or less right now?"*

---

## Position Sizing & Execution

### Continuous Fractional Positions

Rather than discrete buy/sell/hold, the system maintains a **continuous fractional position** in [-1.0, +1.0] for each symbol:
- `+1.0` = fully long at max notional
- `-1.0` = fully short at max notional
- `0.0` = flat

Position targets pass through four smoothing stages:

```python
# 1. Signal → raw position (power function, volatility-normalized)
s = pred_pct / band_R          # normalize by current vol regime
raw = sign(s) × |s|^POS_EXP   # non-linear mapping

# 2. EMA smoothing (reduces turnover / order churn)
pos = α × raw + (1-α) × prev_pos    # half-life configurable per symbol

# 3. Delta capping (no sudden large position jumps)
delta = clip(pos - prev, -DPOS_CAP, +DPOS_CAP)

# 4. Absolute capping
pos = clip(pos, -MAX_POS, +MAX_POS)
```

### Execution Layer

The `BlockLedger` tracks all order lifecycle events with realistic cost modeling:

```python
TRADE_COST_BPS = 8.0    # commission (basis points)
SLIP_BPS       = 4.0    # slippage (basis points)
```

Key execution features:
- **Idempotent order placement** — safe retries, no duplicate fills
- **Short availability retry** — detects borrow limits and clips quantity
- **Partial fill detection and correction**
- **Leverage headroom guard** — checks `gross_exposure / limit` before every order
- **Notional sizing** — `linear_notional_from_posfrac(pos_frac)` → dollar amount

---

## Risk Management: 7-Layer Guardrails

The system implements a **hierarchical, cascading risk control** architecture. Every layer is independent. Any layer can veto a trade. They activate in escalating order of severity:

```
─────────────────────────────────────────────────────────────────
Layer   Condition                    Action
─────────────────────────────────────────────────────────────────
  1     Daily DD > 0.7%              Scale all positions × 0.70
  2     Daily DD > 1.0%              Flatten all + halt for day
  3     Minutes to close ≤ 30        Block all new entries
  4     Friday after 14:00 ET        Reduce size × 0.85, no new
  5     |pos| > PER_SYM_GROSS_CAP   Clip to per-symbol limit
  6     Gross exposure ≈ limit       Skip order (headroom guard)
  7     2 min before market close    Flatten ALL positions (EOD)
─────────────────────────────────────────────────────────────────
```

**Why external guardrails instead of reward penalties?**

Embedding drawdown constraints into the RL reward causes over-conservative policy convergence — a known failure mode in risk-aware DRL research. Instead, the reward signal stays clean (pure PnL-based), and hard constraints are enforced *externally* as a runtime shield that overrides the agent's decisions. The agent learns to generate alpha; the guardrails prevent catastrophic loss.

**The two-threshold drawdown cascade:**

```python
# Soft throttle — early warning, reduce exposure
if daily_drawdown > DAY_THROTTLE_DD_PCT:   # default: -0.7%
    target_frac *= THROTTLE_SIZE_MULT       # default: 0.70×

# Hard kill — stop trading for the day
if daily_drawdown > DAY_KILL_DD_PCT:       # default: -1.0%
    flatten_all_positions()
    halt_session()
```

**Friday rules:**

Research shows holding positions over the weekend consistently produces higher drawdowns. The system treats Friday afternoon as a high-risk regime:

```python
FRIDAY_LATE_CUTOFF_H        = 14     # 14:00 ET
FRIDAY_SIZE_MULT_LATE       = 0.60   # 60% of normal size after cutoff
FRIDAY_BLOCK_NEW_AFTER_LATE = True   # no new entries after 14:00
```

---

## Web Dashboard

Built in **Next.js 14**, the dashboard provides real-time visibility into all trading activity:

| Screen | What It Shows |
|---|---|
| **Login** | JWT-authenticated access |
| **Dashboard** | Live equity curve, PnL, drawdown %, engine status |
| **Positions** | Current open positions with unrealized PnL |
| **Trades** | Paginated history of all closed trades |
| **Control Panel** | Start / Pause / Stop engine; update UserStyle parameters |

Configuration changes (UserStyle, max_symbols, risk_level) take effect at the **next block boundary** — never mid-block — preserving decision atomicity (FR-09).

---

## Full Project Structure

```
.
├── app/                          # Trading Engine (Python)
│   ├── agent/
│   │   ├── __init__.py
│   │   └── arl_agent.py          # ARLAgent: ε-greedy bandit, EMA rewards, UserStyle
│   ├── model/
│   │   ├── __init__.py
│   │   ├── XGboost_model.json    # Trained XGBoost (1200+ trees, 43 features)
│   │   └── elasticnet_ret1h.pkl  # ElasticNet sklearn pipeline
│   ├── execution.py              # BlockLedger, idempotent orders, slippage
│   ├── feat_cols.json            # Feature column definitions
│   ├── main.py                   # Main trading loop (data→inference→RL→risk→order)
│   ├── universe.csv              # Symbol universe (loaded at startup)
│   ├── render.yaml               # Cloud deployment config
│   ├── requirements.txt
│   └── runtime.txt
│
├── web/                          # Next.js Dashboard (Frontend)
│   ├── app/
│   │   ├── dashboard/            # Main trading overview
│   │   ├── positions/            # Open positions view
│   │   ├── trades/               # Trade history (paginated)
│   │   └── control/              # Engine control panel
│   ├── components/
│   └── prisma/
│       └── schema.prisma         # Database ORM schema
│
├── docker-compose.yml            # Local PostgreSQL + Metabase stack
└── .env.example                  # Environment variable template
```

---

## Configuration Reference

### UserStyle Parameters

Edit `USER_STYLE` in `.env` to select a risk profile, or set parameters directly:

| Parameter | Type | Description | high | medium | low |
|---|---|---|---|---|---|
| `risk_level` | string | Risk mode | `"high"` | `"medium"` | `"low"` |
| `max_symbols` | int | Max symbols to trade simultaneously | 8 | 6 | 4 |
| `lookback_days` | int | Historical data window for universe building | 30 | 60 | 120 |
| `prefer_momentum` | bool | Favor uptrend breakouts | true | true | false |
| `prefer_mean_reversion` | bool | Favor reversal setups | false | true | true |
| `base_band_R` | float | Volatility normalization factor | 1.05 | 1.10 | 1.20 |
| `base_ema_hl` | int | Position EMA half-life (blocks) | 6 | 8 | 10 |
| `base_size_mult` | float | Base position size multiplier | 1.10 | 1.00 | 0.80 |
| `base_hold_min_blocks` | int | Minimum hold duration before exit | 1 | 2 | 3 |

### Risk Guardrail Parameters

| Variable | Default | Description |
|---|---|---|
| `DAY_THROTTLE_DD_PCT` | `0.7` | Soft drawdown throttle trigger (%) |
| `DAY_KILL_DD_PCT` | `1.0` | Hard daily kill switch trigger (%) |
| `THROTTLE_SIZE_MULT` | `0.70` | Position scale factor when throttled |
| `EOD_FLATTEN_MIN_BEFORE_CLOSE` | `2` | Minutes before close to flatten all |
| `FRIDAY_LATE_CUTOFF_H` | `14` | Hour (ET) after which Friday rules apply |
| `FRIDAY_SIZE_MULT_LATE` | `0.60` | Position scale on late Friday |
| `PER_SYM_GROSS_CAP` | `0.25` | Max fractional exposure per symbol |
| `MAX_NOTIONAL` | `10000` | Maximum total notional (USD) |

### Trading Cost Parameters

| Variable | Default | Description |
|---|---|---|
| `TRADE_COST_BPS` | `8.0` | Commission in basis points |
| `SLIPPAGE_BPS` | `4.0` | Slippage estimate in basis points |
| `REBALANCE_BAND` | `0.01` | Min position change to trigger rebalance |
| `MIN_ABS_POS` | `0.03` | Minimum position fraction to trade |

---

## Running the System

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop (for local PostgreSQL)
- Alpaca account (paper trading is free)

### 1. Clone and install

```bash
git clone <repo_url>
cd project_root

# Python trading engine
pip install -r app/requirements.txt

# Node.js dashboard
cd web && npm install && cd ..
```

### 2. Set up environment

```bash
cp .env.example .env
# Fill in your Alpaca keys and database URL
```

Required `.env` variables:

```bash
APCA_API_KEY_ID=<your_alpaca_key>
APCA_API_SECRET_KEY=<your_alpaca_secret>
APCA_API_BASE_URL=https://paper-api.alpaca.markets   # or live
DATABASE_URL=postgresql://user:pass@localhost:5432/trading
```

### 3. Start the database stack

```bash
docker-compose up -d
# PostgreSQL on :5432
```

### 4. Run database migrations

```bash
cd web && npx prisma migrate deploy && cd ..
```

### 5. Start the trading engine

```bash
# Paper trading (recommended first)
python -m app.main
```

The engine will:
1. Wait for market open (if pre-market)
2. Build symbol universe
3. Start the block loop (data → inference → RL → risk → order)
4. Flatten all positions at EOD
5. Persist agent state to `/tmp/live_state.json`

### 6. Start the dashboard

```bash
cd web && npm run dev
# Dashboard at http://localhost:3000
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `APCA_API_KEY_ID` | ✅ | Alpaca API key |
| `APCA_API_SECRET_KEY` | ✅ | Alpaca secret key |
| `APCA_API_BASE_URL` | ✅ | Paper or live base URL |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `USER_STYLE` | ⬜ | `high_risk_short_term` / `medium_risk_swing` / `low_risk_long_term` |
| `AGENT_MAX_SYMBOLS` | ⬜ | Override max symbols (default: 20) |
| `AGENTIC_MODE` | ⬜ | `"1"` to enable RL modulation (default: 1) |
| `SHORTS_ENABLED` | ⬜ | `"1"` to allow short positions (default: 1) |
| `LONGS_ONLY` | ⬜ | `"1"` to disable shorts (default: 0) |
| `DAY_KILL_DD_PCT` | ⬜ | Daily drawdown kill threshold (default: 1.0) |
| `DAY_THROTTLE_DD_PCT` | ⬜ | Drawdown throttle threshold (default: 0.7) |
| `MAX_NOTIONAL` | ⬜ | Max portfolio notional USD (default: 10000) |
| `TRADE_COST_BPS` | ⬜ | Commission in bps (default: 8.0) |
| `SLIPPAGE_BPS` | ⬜ | Slippage in bps (default: 4.0) |
| `STATE_PATH` | ⬜ | Path to persist agent state JSON (default: /tmp/live_state.json) |
| `MODEL_PATH_XGB` | ⬜ | Path to XGBoost model JSON |
| `MODEL_PATH_ENET` | ⬜ | Path to ElasticNet `.pkl` |

---

## Research Foundations

This system was designed in alignment with — and validated against — the following body of academic literature:

### Reinforcement Learning & Bandit Framework
- Huo & Fu (2017) — *Risk-Aware Multi-Armed Bandit Problem with Application to Portfolio Selection*, Royal Society Open Science
- Ni et al. (2023) — *Contextual Combinatorial Bandit on Portfolio Management*, Expert Systems with Applications
- Pricope (2021) — *Deep Reinforcement Learning for Trading: A Critical Survey*, MDPI Data
- Systematic Review (2025) — *RL in Financial Decision Making: 167 Studies 2017–2025*, arXiv

### Hybrid Supervised + RL Architecture
- Elmachtoub & Grigas (2022) — *Smart Predict-then-Optimize*, Management Science
- Sun, Wang & An (2023) — *RL for Quantitative Trading*, ACM Transactions on Intelligent Systems
- Théate & Ernst (2021) — *Deep RL for Trading*, Neural Networks

### Feature Engineering
- Kumbure et al. (2022) — *ML Techniques and Data for Stock Market Forecasting: A Literature Review*, Expert Systems with Applications
- Htun et al. (2023) — *Survey of Feature Selection and Extraction for Stock Market Prediction*, Financial Innovation, Springer

### Continuous Position Sizing
- Théate & Ernst (2019) — *Deep Reinforcement Learning for Trading*, arXiv:1911.10107 (volatility scaling)
- Lim et al. (2019) — *Enhancing Time Series Momentum with Deep Neural Networks*, JFDS (EMA smoothing)
- Maclean et al. (2010) — *Fractional Kelly Strategies* (growth-volatility trade-off)

### Risk Management Architecture
- FIA (2024) — *Best Practices for Automated Trading Risk Controls and System Safeguards*
- Buehler et al. (2019) — *Deep Hedging* (hard constraint separation)
- Stavros et al. (2018) — *Take Profit and Stop Loss in Combination with MACD*, JRFM (weekend flattening)

---

## Non-Functional Requirements

| ID | Requirement | Implementation |
|---|---|---|
| NFR-01 | All API endpoints authenticated | JWT middleware on every route |
| NFR-02 | Secrets via environment variables | `.env` file, never hardcoded |
| NFR-03 | Encrypted internal communications | HTTPS throughout |
| NFR-04 | Internal errors hidden from UI | Sanitized error responses |
| NFR-05 | Retry transient broker failures | Exponential backoff on Alpaca calls |
| NFR-06 | Safe state on repeated failures | Flattens positions, halts session |
| NFR-07 | API P99 latency ≤ 300ms | Lightweight stateless REST handlers |
| NFR-08 | Decision cycle fits in block time | Async-compatible block loop design |
| NFR-09 | Stateless, horizontally scalable | All state in PostgreSQL |
| NFR-10 | Audit records of critical events | Trade and order logs persisted to PostgreSQL |
| NFR-11 | Crash-consistent state | `save_state()` after every block |

---

## Important Disclaimers

> **This system trades real financial markets.** Always start with paper trading and validate behavior thoroughly before switching to live mode.

> **Past performance of ML models does not guarantee future returns.** Market regimes change. Monitor model behavior and equity curves regularly.

> **The drawdown guardrails are designed to protect capital, not guarantee profit.** Set `DAY_KILL_DD_PCT` conservatively.

> **High-frequency trading and options trading are explicitly out of scope.** This system is designed for 1-hour bar intraday equity trading only.

---

## Contributors

<table>
<tr>
<td align="center"><b>Mohamed Ehab</b></td>
<td align="center"><b>Abdelrahman Tamer</b></td>
<td align="center"><b>Mohamed Atef</b></td>
<td align="center"><b>Moataz Kamal</b></td>
<td align="center"><b>Yahia Abdelmonaem</b></td>
</tr>
</table>

---

<div align="center">

*Built with precision. Deployed with conviction. Protected without compromise.*

```
Reward = (PnL − Fees − Slippage) / Exposure
```

*The only signal that matters.*

</div>
