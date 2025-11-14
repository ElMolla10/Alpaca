# 🧠 Agentic Reinforcement Learning Trading System

A modular, adaptive trading engine built on **Reinforcement Learning (RL)**, contextual decision logic, and predictive modeling.  
The system learns from trading performance, adjusts behavior based on reward feedback, and executes trades with strict risk and safety constraints.

---

## 📁 Project Structure

```
app/
 ├── agent/
 │    ├── __init__.py
 │    └── arl_agent.py          # Core RL logic, state updates, and policy methods
 │
 ├── model/
 │    ├── __init__.py
 │    └── XGboost_model.json    # Predictive model used for directional bias
 │
 ├── execution.py               # Order execution, fills, slippage, and PnL ledger
 ├── feat_cols.json             # List of engineered features used by the model
 ├── main.py                    # Main loop orchestrating data → decision → execution
 │
 ├── render.yaml                # Cloud deployment configuration
 ├── requirements.txt           # Python dependencies
 ├── runtime.txt                # Python runtime version
 └── README.md
```

---

## ⚙️ Core Components

### 🔹 1. RL Agent (`arl_agent.py`)
The RL agent is responsible for:

- Learning from **net PnL% per block**
- Adjusting **position size, timing, and hold duration**
- Reacting to performance and volatility changes
- Resetting daily to prevent leakage
- Shaping its behavior using the `UserStyle` dataclass:
  - Momentum or mean-reversion preference  
  - Risk levels (low / medium / high)

---

### 🔹 2. Execution Layer (`execution.py`)
Handles the trading pipeline:

- Idempotent order placement (safe retries)
- Partial-fill detection and correction
- Slippage + fee modeling
- Block-based ledger tracking:
  - Realized PnL  
  - Unrealized PnL  
  - Exposure  
  - Transaction cost impact  

---

### 🔹 3. Predictive Model (`model/XGboost_model.json`)
Provides directional bias for the RL agent:

- Encodes OHLCV patterns, volatility, and trend signals  
- Serves as a **non-deterministic hint**, not a hard decision  
- Supports hybrid learning (model signal + RL adjustment)

---

## 📊 Key Features

- Adaptive RL policy updated every trading block  
- Context-aware sizing and entry timing  
- Realistic execution via slippage + fee modeling  
- Daily state resets  
- Deterministic reward loops  
- Easy integration with other agents (sentiment, macro, LSTM, etc.)

---

## 🧠 Reward Function

\[
Reward = \frac{PnL - (Fees + Slippage)}{Exposure}
\]

Rewards are directly tied to profitability and risk efficiency.

---

## 🚀 Running the System

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the engine
```bash
python -m app.main
```

Ensure you're running from the project root so module imports resolve correctly.

---

## 🧩 Configuration (UserStyle)

| Parameter | Description | Example |
|----------|-------------|---------|
| `max_symbols` | Max symbols to trade | 8 |
| `prefer_momentum` | Trend-following bias | True |
| `risk_level` | User risk mode | "high" |
| `base_size_mult` | Base sizing | 1.0 |
| `base_hold_min_blocks` | Minimum holding time | 2 |

---

## 🛡️ QA & Safety

- Detects missing or stale data  
- Ensures feature completeness (no leakage)  
- Prevents accidental duplicate orders  
- Tracks success rate and exposure consistency  
- Validates environment before each execution loop  

---

## 👥 Contributors

- Mohamed Ehab  
- Abdelrahman Tamer  
- Mohamed Atef  
- Moataz Kamal  
- Yahia Abdelmonaem  
