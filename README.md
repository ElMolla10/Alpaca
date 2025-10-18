🧠 Agentic Reinforcement Learning Trading System

This repository implements an agentic reinforcement learning (RL) trading system that adapts to user trading styles, learns from performance, and makes risk-aware trading decisions.
It integrates data from multiple sources, applies cost and slippage modeling, and uses XGBoost for predictive features.

📁 Project Structure
app/
 ├── agent/
 │    ├── __init__.py          # Agent exports and optional modules
 │    └── arl_agent.py         # Main RL agent logic (core learning, actions)
 │
 ├── model/
 │    ├── __init__.py
 │    └── XGboost_model.json   # Trained predictive model
 │
 ├── execution.py              # Handles order placement, fills, and ledger updates
 ├── feat_cols.json            # Feature column definitions
 ├── main.py                   # Entry point (system loop)
 │
 ├── render.yaml               # Deployment configuration
 ├── requirements.txt          # Dependencies
 ├── runtime.txt               # Python runtime version
 └── README.md                 # Project documentation

⚙️ Core Functionality
🔹 Reinforcement Learning Agent (arl_agent.py)

Learns from PnL-based rewards (realized profit minus fees and slippage).

Adapts position sizing and trade direction dynamically.

Resets state daily to avoid data leakage.

Configurable user styles through the UserStyle dataclass (momentum, mean-reversion, etc.).

🔹 Execution Layer (execution.py)

Manages idempotent orders (safe retries, no duplicates).

Tracks slippage, commission, and partial fills.

Updates a block-based ledger for PnL tracking.

🔹 Model Layer (model/XGboost_model.json)

Predictive model used to guide the RL agent’s initial bias.

Encodes recent price and volatility patterns for symbol selection.

📊 Key Features

Dynamic policy learning per trading block

Fee & slippage modeling for realistic rewards

Risk-adjusted trade sizing

Agent resets per session to avoid lookahead bias

Extendable structure for multi-agent integration (news, sentiment, etc.)

🧠 Reward Mechanism

The agent’s reward is proportional to the block’s net PnL%, ensuring learning aligns with actual profitability:

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
𝐹
𝑒
𝑒
𝑠
−
𝑆
𝑙
𝑖
𝑝
𝑝
𝑎
𝑔
𝑒
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
PnL−Fees−Slippage
	​

🚀 Running the Project
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run main entry
python -m app.main


If your module path causes errors, make sure your working directory is set to the project root.

🧩 Configuration

Edit parameters in UserStyle (inside arl_agent.py) to change behavior:

Parameter	Description	Example
max_symbols	number of tradable stocks	8
prefer_momentum	favor uptrend breakouts	True
risk_level	risk mode	"high"
base_size_mult	base position multiplier	1.0
✅ QA & Validation

Checks for missing data, invalid features, or lookahead bias.

Ensures all orders are unique and consistent across retries.

Tracks execution success rate and exposure control.

👥 Contributors

Mohamed Ehab

Abdelrahman Tamer

Mohamed Atef

Moataz Kamal

Yahia Abdelmonaem
