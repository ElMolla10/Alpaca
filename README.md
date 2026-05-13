# Alpaca

Python agentic trading system with a local DriftWatch Project Command Center.

## What This Repo Runs

- `app/main.py` - the Alpaca paper-trading engine.
- `app/driftwatch_client.py` - writes inference and label events to PostgreSQL.
- `app/driftwatch_dashboard.py` - the local Python command center for trading status, DriftWatch observability, architecture, risk, validation, config, and roadmap.
- `schema/driftwatch.sql` - the PostgreSQL schema for DriftWatch tables.

The project now presents one dashboard story: the local Python command center.

## Run The Engine

```bash
python -m app.main
```

The engine uses Alpaca paper trading and writes DriftWatch events to PostgreSQL.

## Run The Command Center

```bash
python -m app.driftwatch_dashboard --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

This command center is intended for local or demo use only. Do not expose it publicly without authentication.

## Environment

The main runtime expects these secrets in `.env`:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`
- `APCA_API_BASE_URL`
- `DRIFTWATCH_DATABASE_URL`

## Repository Layout

```text
app/
  agent/               ARL agent implementation
  execution.py         order execution and PnL ledger
  driftwatch_client.py DriftWatch persistence client
  driftwatch_dashboard.py local command center
  main.py              trading engine entry point
schema/
  driftwatch.sql        DriftWatch PostgreSQL schema
tests/                 pytest coverage for engine and dashboard
```

## Notes

- Trading executes against Alpaca paper trading.
- DriftWatch stores inference and label events in PostgreSQL.
- The command center reads live local data and should stay behind local access or auth.
