# 🧠 Agentic Reinforcement Learning Trading Bot

An **autonomous, agentic trading system** that uses numerical prediction, risk management, and adaptive decision-making to trade stocks through **MetaTrader 5 / Alpaca API**.

The bot continuously observes markets, makes trading decisions based on multiple predictive models, executes orders automatically, and learns from results over time.

---

## ⚙️ System Overview

### Core Loop
1. **Observe**
   - Fetch hourly OHLCV data and features (e.g., volatility, momentum, moving averages).
   - Preprocess into model-ready format.

2. **Predict**
   - Gradient-boosted regression model (`XGBoost`) forecasts next-hour return %.
   - Output normalized for volatility.

3. **Decide + Act**
   - Position sizing proportional to predicted strength (BAND_R scaling).
   - Apply Friday risk reduction, kill-switches, and hold-timers.
   - Send orders to Alpaca / MetaTrader 5 via API.

4. **Learn (Reinforcement)**
   - Record realized P/L, Sharpe ratio, hit-rate, and drawdown.
   - Adjust leverage, hold-time, and risk bands based on reward feedback.

---

## 🧩 Key Features

| Category | Description |
|-----------|-------------|
| **Agentic Control** | Autonomous observe–decide–act–learn cycle. |
| **Multi-Ticker Training** | Pre-trained across 10 tickers (AAPL, AMD, NVDA, MSFT, JPM, GS, CVX, XOM, PG, KO). |
| **Risk Management** | Confidence thresholds, volatility sizing, kill-switches, stop-loss, drawdown limits. |
| **Friday Logic** | Reduces exposure and blocks new entries near session close. |
| **Backtesting** | 5-year historical replay with transaction cost, slippage, and Sharpe analysis. |
| **Paper → Live Bridge** | Runs in paper trading first, then connects to MT5 / Alpaca for live execution. |

---

## 🧮 Performance Snapshot

| Symbol | CV Sharpe | Hold Sharpe | Direction |
|:-------|-----------:|------------:|:----------|
| GS | 0.56 | **1.04** | Short |
| PG | 0.25 | 0.87 | Short |
| JPM | 0.73 | 0.75 | Short |
| CVX | 0.51 | 0.69 | Short |
| MSFT | 0.95 | 0.54 | Short |
| AAPL | 0.12 | 0.51 | Short |
| NVDA | 0.41 | -0.04 | Short |
| AMD | 0.16 | -0.14 | Short |
| KO | -0.01 | -1.28 | Short |
| XOM | **1.41** | -2.12 | Short |

---

## 🧠 Agentic Reinforcement Learning Extension

Upcoming reinforcement layer adds dynamic **user-style adaptation**:

- **Input:** User chooses trading style (e.g., *high-risk short-term* or *low-risk long-term*).
- **Action:** Agent selects volatile or stable stocks accordingly, recalibrates leverage / holding time.
- **Reward:** Based on Sharpe, drawdown, and realized P/L.
- **Learning:** Policy updated through reinforcement to maximize long-term reward.

---

## 🏗️ Architecture

```
data/
 ├── *_hourly_last5y_selected_features.csv
models/
 ├── xgb_boosters/
app/
 ├── main.py                # live trading loop
 ├── model_train.py         # training & backtest pipeline
 ├── utils.py               # indicators, volatility, I/O
state/
 └── bot_state.json         # persistent timers & positions
```

---

## 🖥️ Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train multi-ticker model
python app/model_train.py

# 3. Run live session (paper mode)
python app/main.py
```

---

## ⚙️ Configuration

Edit `config.py` or environment variables before running:

| Variable | Description |
|-----------|-------------|
| `ALPACA_API_KEY` | Your Alpaca API key |
| `ALPACA_SECRET_KEY` | Your Alpaca secret key |
| `SESSION_START_H` | Market session start (ET hour) |
| `FRIDAY_LATE_CUTOFF_H` | Hour to block new Friday entries |
| `FRIDAY_SIZE_MULT_DAY` | Friday position size multiplier |
| `FRIDAY_BLOCK_NEW_AFTER_LATE` | True = block new Friday orders |

---

## 🛡️ Safety & Risk Rules

- Trades only above confidence threshold.
- Avoids halts / earnings / wide spreads.
- Size scales down on volatility or Fridays.
- Auto-flatten at day end; kill-switch on abnormal equity drop.

---

## 📈 Example Session

- **Portfolio:** $99 585.80  (+0.62%)
- **Daily P/L:** +$611.61
- **Buying Power:** $314 172.75
- **Cash:** $65 051.10

---

## 📅 Roadmap

1. **Integrate RL policy** for adaptive ticker & parameter selection.
2. **Expand agentic layer** to learn user style preferences.
3. **Add CNN chart vision & headline NLP signals.**
4. **Deploy dashboard** for metrics & alerts.

---

## 📜 License
MIT License – Free for research and non-commercial use.
