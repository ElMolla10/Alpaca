Agentic E-Trading Engine

A modular, adaptive trading engine built on Reinforcement Learning (RL), contextual decision logic, and predictive modeling.
The system learns from real trading performance, adjusts behavior based on reward feedback, and executes trades with strict safety and risk controls.

This repository powers an agent that:

Adapts position sizing and timing automatically

Learns from PnL-based rewards

Uses XGBoost predictions as structured initial signals

Models slippage, commissions, and execution uncertainty

Supports user-specific trading styles

📁 Project Structure
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

⚙️ Core Components
🔹 1. RL Agent (arl_agent.py)

The RL agent is the heart of the system:

Learns from net PnL% per block

Adjusts size multiplier, timing, hold duration, and directional bias

Responds dynamically to live performance and volatility

Resets internal state daily to avoid leakage

Uses the UserStyle dataclass to shape personality:

momentum-focused

mean-reversion

conservative / aggressive risk levels

The architecture allows the agent to change behavior without hardcoding strategies.

🔹 2. Execution Layer (execution.py)

Handles all interaction with the broker (simulated or real):

Idempotent order placement (safe retries, no duplicates)

Partial fill handling

Slippage and commission injection for realistic reward signals

Maintains a block-based ledger tracking:

realized PnL

unrealized PnL

exposures

transaction cost impact

🔹 3. Predictive Model (XGboost_model.json)

An XGBoost model provides directional and volatility information:

Encodes recent OHLCV patterns

Provides a soft signal that guides the RL agent

Never acts alone — the agent decides when to trust or ignore predictions

This hybrid approach gives the system structured signals + adaptive learning.

📊 Key Features

Adaptive RL policy updated every trading block

Context-aware trade sizing and timing

Realistic slippage and fee modeling

Directional bias from XGBoost predictions

Daily state reset to avoid time leakage

Extensible architecture (plug in sentiment, macro data, alternative agents)

Deterministic reward loops to stabilize learning

🧠 Reward Function

The agent’s reward aligns directly with profitability:

𝑅
𝑒
𝑤
𝑎
𝑟
𝑑
=
𝑃
𝑛
𝐿
−
(
𝐹
𝑒
𝑒
𝑠
+
𝑆
𝑙
𝑖
𝑝
𝑝
𝑎
𝑔
𝑒
)
𝐸
𝑥
𝑝
𝑜
𝑠
𝑢
𝑟
𝑒
Reward=
Exposure
PnL−(Fees+Slippage)
	​


This ensures the system optimizes for risk-adjusted real returns, not just raw price movement.

🚀 Running the System
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the engine
python -m app.main

3️⃣ (Optional) Fix path issues

Make sure you run the command from the project root so imports resolve correctly.

🧩 Configuration (UserStyle)

Customize trading behavior in arl_agent.py:

Parameter	Description	Example
max_symbols	Maximum number of symbols to trade	8
prefer_momentum	Bias toward trend continuation	True
risk_level	Risk appetite	"high"
base_size_mult	Starting size multiplier	1.0
base_hold_min_blocks	Minimum holding duration	2

You can define additional styles or override defaults per session.

🛡️ QA & Safety Checks

The system validates its environment continuously:

Detects missing or stale market data

Rejects orders with inconsistent size or invalid state

Prevents duplicate executions via idempotent logic

Ensures feature strictness (no future leakage or incomplete inputs)

Logs execution success rates and exposure behavior for analysis

👥 Contributors

Mohamed Ehab

Abdelrahman Tamer

Mohamed Atef

Moataz Kamal

Yahia Abdelmonaem
