# App Modules

This directory contains the live Python trading engine and the local DriftWatch command center.

## Core Modules

- `main.py` - Alpaca paper-trading loop.
- `execution.py` - order execution, fills, and PnL ledger.
- `driftwatch_client.py` - PostgreSQL persistence for inference and label events.
- `driftwatch_dashboard.py` - localhost project command center.
- `agent/arl_agent.py` - adaptive agent policy and user-style handling.

## Data And Schema

- `feat_cols.json` - model feature manifest.
- `model/` - predictive model artifacts.
- `universe.csv` - trading universe.
- `schema/driftwatch.sql` - shared PostgreSQL schema for DriftWatch.

## Running

```bash
python -m app.main
python -m app.driftwatch_dashboard --host 127.0.0.1 --port 8765
```

The dashboard is local/demo-only and should not be exposed publicly without authentication.
