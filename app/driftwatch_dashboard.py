#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from alpaca_trade_api.rest import REST
except Exception:  # pragma: no cover - optional runtime dependency guard
    REST = None


try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    load_dotenv = None


APP_TITLE = "Alpaca Project Command Center"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_PATHS = {
    "/",
    "/overview",
    "/trading",
    "/architecture",
    "/agent",
    "/risk",
    "/driftwatch",
    "/validation",
    "/config",
    "/roadmap",
}

_ENV_LOADED = False


PROJECT_SPEC: Dict[str, Any] = {
    "headline": "End-to-end agentic electronic trading engine",
    "thesis": (
        "Alpaca separates prediction, online modulation, position sizing, and hard risk constraints. "
        "The website mirrors the report: requirements, architecture, agent behavior, guardrails, "
        "observability, validation, and future work."
    ),
    "architecture": [
        {
            "name": "Data ingestion",
            "detail": "Hourly Alpaca OHLCV bars with IEX/SIP feed selection, fallback handling, and partial first-hour synthesis.",
            "requirements": ["FR-01", "FR-02", "FR-03", "FR-04"],
        },
        {
            "name": "Feature engineering",
            "detail": "Lagged returns, rolling volatility, five-bar momentum, MACD, Bollinger middle band, and VWAP alignment.",
            "requirements": ["FR-05", "FR-06", "FR-07"],
        },
        {
            "name": "Prediction layer",
            "detail": "XGBoost plus optional ElasticNet ensemble producing pred_pct, p_up, and volatility estimate.",
            "requirements": ["FR-08", "FR-09", "FR-10"],
        },
        {
            "name": "ARLAgent modulation",
            "detail": "Contextual epsilon-greedy bandit that selects symbols and adapts trust from realised net PnL.",
            "requirements": ["FR-11", "FR-12", "FR-13", "FR-14", "FR-15", "FR-16", "FR-17", "FR-18"],
        },
        {
            "name": "Sizing and execution",
            "detail": "Volatility-normalised signal mapping, EMA smoothing, delta caps, notional conversion, and broker routing.",
            "requirements": ["FR-19", "FR-20", "FR-21", "FR-22", "FR-23", "FR-24", "FR-25", "FR-26"],
        },
        {
            "name": "Risk shield",
            "detail": "Seven external guardrails can reduce, clip, block, flatten, or halt trading without changing the agent.",
            "requirements": ["FR-27", "FR-28", "FR-29", "FR-30", "FR-31", "FR-32"],
        },
        {
            "name": "Persistence and observability",
            "detail": "State snapshots, stdout decision traces, DriftWatch inference events, and realised PnL labels.",
            "requirements": ["FR-33", "FR-34", "FR-35", "FR-36"],
        },
        {
            "name": "Dashboard",
            "detail": "Authenticated trading dashboard in the main app, plus this localhost project command center.",
            "requirements": ["FR-37", "FR-38", "FR-39", "FR-40", "FR-41", "FR-42"],
        },
    ],
    "risk_guardrails": [
        {"layer": 1, "name": "Daily drawdown throttle", "default": "-0.7%", "action": "Scale subsequent targets by 0.70."},
        {"layer": 2, "name": "Daily drawdown kill", "default": "-1.0%", "action": "Flatten open positions and stop the session."},
        {"layer": 3, "name": "Trading cut-off", "default": "30 min before close", "action": "Block new entries near close."},
        {"layer": 4, "name": "Friday late-day rule", "default": "14:00 ET", "action": "Reduce size and block new weekend-spanning entries."},
        {"layer": 5, "name": "Per-symbol gross cap", "default": "5% equity", "action": "Clip absolute target fraction per symbol."},
        {"layer": 6, "name": "Leverage headroom", "default": "5x equity model", "action": "Skip or clip orders when gross exposure is high."},
        {"layer": 7, "name": "End-of-day flatten", "default": "2 min before close", "action": "Flatten all open positions and exit cleanly."},
    ],
    "user_styles": [
        {"name": "high_risk_short_term", "risk": "high", "lookback": 30, "max_symbols": 8, "band": 1.05, "ema": 6, "size": 1.10, "hold": 1, "epsilon": 0.20, "alpha": 0.25},
        {"name": "medium_risk_swing", "risk": "medium", "lookback": 60, "max_symbols": 6, "band": 1.10, "ema": 8, "size": 1.00, "hold": 2, "epsilon": 0.12, "alpha": 0.20},
        {"name": "low_risk_long_term", "risk": "low", "lookback": 120, "max_symbols": 4, "band": 1.20, "ema": 10, "size": 0.80, "hold": 3, "epsilon": 0.08, "alpha": 0.15},
    ],
    "tests": [
        "Agent universe filtering, scoring, epsilon-greedy selection, epsilon decay, confidence interpolation, temporal gates, reward EMA, and persistence.",
        "Ledger long and short round trips, profit after costs, empty state, open-fill cost component, clear, and reset.",
        "DriftWatch disabled state, table creation, sanitization, upserts, retry behavior, and failure drop behavior.",
        "Localhost project dashboard page and JSON API fallback behavior.",
    ],
    "future_work": [
        "Walk-forward backtest with cost sensitivity, bootstrap confidence intervals, and ablation tests.",
        "Explicit market-regime detection with per-symbol-per-regime reward memory.",
        "Options support with strike, expiry, gamma, and vega-aware guardrails.",
        "Full drift monitoring dashboard with feature distributions and alert thresholds.",
        "Smart predict-then-optimise retraining from realised decision loss.",
        "Other asset classes and brokers.",
    ],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def load_project_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED or load_dotenv is None:
        return
    # app/.env may contain runtime defaults, but the repo-root .env is the
    # project authority for broker and DriftWatch credentials.
    for env_path, override in ((REPO_ROOT / "app" / ".env", False), (REPO_ROOT / ".env", True)):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=override)
    _ENV_LOADED = True


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 3) -> float | None:
    number = _safe_float(value)
    return None if number is None else round(number, digits)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _money(value: Any) -> float:
    number = _safe_float(value)
    return 0.0 if number is None else round(number, 2)


def _percent(value: Any) -> float:
    number = _safe_float(value)
    return 0.0 if number is None else round(number * 100.0, 3)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _read_universe() -> List[str]:
    path = REPO_ROOT / "app" / "universe.csv"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    symbols = []
    for line in lines[1:]:
        symbol = line.strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def _test_inventory() -> Dict[str, Any]:
    tests_dir = REPO_ROOT / "tests"
    files = sorted(tests_dir.glob("test_*.py"))
    count = 0
    names = []
    for path in files:
        text = path.read_text()
        found = re.findall(r"^def (test_[a-zA-Z0-9_]+)", text, flags=re.MULTILINE)
        count += len(found)
        names.extend(found)
    return {"files": [path.name for path in files], "count": count, "names": names}


def repo_snapshot() -> Dict[str, Any]:
    load_project_env()
    feature_cols = _read_json(REPO_ROOT / "app" / "feat_cols.json", [])
    universe = _read_universe()
    model_paths = {
        "xgboost_json": REPO_ROOT / "app" / "model" / "XGboost_model.json",
        "elasticnet_joblib": REPO_ROOT / "app" / "model" / "elasticnet_ret1h.pkl",
        "legacy_xgboost": REPO_ROOT / "XGboost_model",
    }
    env_checks = {
        "APCA_API_KEY_ID": bool(os.getenv("APCA_API_KEY_ID")),
        "APCA_API_SECRET_KEY": bool(os.getenv("APCA_API_SECRET_KEY")),
        "APCA_API_BASE_URL": bool(os.getenv("APCA_API_BASE_URL")),
        "DRIFTWATCH_DATABASE_URL": bool(os.getenv("DRIFTWATCH_DATABASE_URL")),
        "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
        "NEXTAUTH_SECRET": bool(os.getenv("NEXTAUTH_SECRET")),
    }
    return {
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "universe_count": len(universe),
        "universe_preview": universe[:24],
        "models": {
            name: {"exists": path.exists(), "bytes": _file_size(path)}
            for name, path in model_paths.items()
        },
        "env": env_checks,
        "tests": _test_inventory(),
        "important_paths": [
            "app/main.py",
            "app/agent/arl_agent.py",
            "app/execution.py",
            "app/driftwatch_client.py",
            "app/driftwatch_dashboard.py",
            "Website/prisma/schema.prisma",
            "schema/driftwatch.sql",
            "tests/",
        ],
    }


def _sample_trading(reason: str) -> Dict[str, Any]:
    return {
        "mode": "offline",
        "status": "Alpaca API data is unavailable.",
        "detail": reason,
        "account": {
            "status": "unavailable",
            "account_number": "",
            "currency": "USD",
            "equity": None,
            "last_equity": None,
            "cash": None,
            "buying_power": None,
            "portfolio_value": None,
            "daily_pl": None,
            "daily_plpc": None,
            "gross_exposure": None,
            "gross_exposure_pct": None,
        },
        "positions": [],
        "orders": [],
        "equity_series": [],
        "position_pnl_series": [],
        "order_status_counts": {},
    }


def fetch_trading_data() -> Dict[str, Any]:
    load_project_env()
    if REST is None:
        return _sample_trading("alpaca-trade-api is not importable in this environment.")

    key_id = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    if not key_id or not secret:
        return _sample_trading("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env for live paper-account data.")

    try:
        api = REST(key_id, secret, base_url, api_version="v2")
        account = api.get_account()
        positions_raw = api.list_positions()
        try:
            orders_raw = api.list_orders(status="all", limit=50, direction="desc")
        except TypeError:
            orders_raw = api.list_orders(status="all", limit=50)

        positions = []
        for p in positions_raw:
            qty = _safe_float(_get_attr(p, "qty")) or 0.0
            market_value = _money(_get_attr(p, "market_value"))
            positions.append(
                {
                    "symbol": _get_attr(p, "symbol", "-"),
                    "side": "long" if qty >= 0 else "short",
                    "qty": qty,
                    "price": _money(_get_attr(p, "current_price")),
                    "market_value": market_value,
                    "unrealized_pl": _money(_get_attr(p, "unrealized_pl")),
                    "unrealized_plpc": _percent(_get_attr(p, "unrealized_plpc")),
                    "avg_entry_price": _money(_get_attr(p, "avg_entry_price")),
                }
            )
        positions.sort(key=lambda row: abs(row["market_value"]), reverse=True)

        orders = []
        status_counts: Dict[str, int] = {}
        for o in orders_raw:
            status = str(_get_attr(o, "status", "-"))
            status_counts[status] = status_counts.get(status, 0) + 1
            orders.append(
                {
                    "submitted_at": _get_attr(o, "submitted_at"),
                    "symbol": _get_attr(o, "symbol", "-"),
                    "type": _get_attr(o, "type", "-"),
                    "side": _get_attr(o, "side", "-"),
                    "qty": _safe_float(_get_attr(o, "qty")),
                    "notional": _safe_float(_get_attr(o, "notional")),
                    "filled_qty": _safe_float(_get_attr(o, "filled_qty")),
                    "filled_avg_price": _safe_float(_get_attr(o, "filled_avg_price")),
                    "status": status,
                }
            )

        equity = _money(_get_attr(account, "equity"))
        last_equity = _money(_get_attr(account, "last_equity"))
        daily_pl = round(equity - last_equity, 2)
        daily_plpc = round((daily_pl / last_equity) * 100.0, 3) if last_equity else 0.0
        gross_exposure = round(sum(abs(p["market_value"]) for p in positions), 2)
        gross_exposure_pct = round((gross_exposure / equity) * 100.0, 2) if equity else 0.0

        equity_series = []
        try:
            history = api.get_portfolio_history(period="1D", timeframe="5Min", extended_hours=True)
            timestamps = list(getattr(history, "timestamp", []) or [])
            equities = list(getattr(history, "equity", []) or [])
            for ts, value in zip(timestamps, equities):
                if value is None:
                    continue
                try:
                    ts_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                except Exception:
                    ts_dt = ts
                equity_series.append({"ts": ts_dt, "value": _money(value)})
        except Exception:
            equity_series = []

        return {
            "mode": "live",
            "status": "Connected to Alpaca paper account using local .env credentials.",
            "detail": "Read-only account, positions, orders, and portfolio-history endpoints. API secrets are not exposed.",
            "account": {
                "status": _get_attr(account, "status", "-"),
                "account_number": str(_get_attr(account, "account_number", "paper"))[-4:].rjust(8, "*"),
                "currency": _get_attr(account, "currency", "USD"),
                "equity": equity,
                "last_equity": last_equity,
                "cash": _money(_get_attr(account, "cash")),
                "buying_power": _money(_get_attr(account, "buying_power")),
                "portfolio_value": _money(_get_attr(account, "portfolio_value")),
                "daily_pl": daily_pl,
                "daily_plpc": daily_plpc,
                "gross_exposure": gross_exposure,
                "gross_exposure_pct": gross_exposure_pct,
            },
            "positions": positions,
            "orders": orders,
            "equity_series": equity_series,
            "position_pnl_series": [{"label": p["symbol"], "value": p["unrealized_pl"]} for p in positions[:16]],
            "order_status_counts": status_counts,
        }
    except Exception as exc:
        return _sample_trading(f"{exc.__class__.__name__}: {exc}")


def _sample_results(reason: str) -> Dict[str, Any]:
    return {
        "mode": "offline",
        "is_demo": False,
        "status": "DriftWatch database is unavailable.",
        "detail": reason,
        "summary": {
            "inference_count": 0,
            "label_count": 0,
            "avg_latency_ms": None,
            "avg_pred_pct": None,
            "avg_realized_pnl_pct": None,
            "win_rate_pct": None,
            "mean_abs_error_pct": None,
            "last_event_ts": None,
        },
        "recent": [],
        "latency_series": [],
        "pnl_series": [],
        "feature_keys": [],
    }


def fetch_results() -> Dict[str, Any]:
    load_project_env()
    dsn = os.getenv("DRIFTWATCH_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        return _sample_results("Set DRIFTWATCH_DATABASE_URL or DATABASE_URL to read live DriftWatch tables.")

    try:
        with psycopg2.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)::int AS inference_count,
                        COALESCE(AVG(latency_ms), 0)::float AS avg_latency_ms,
                        COALESCE(AVG(y_pred_num), 0)::float AS avg_pred_pct,
                        MAX(ts) AS last_event_ts
                    FROM inference_events
                    """
                )
                inference_summary = dict(cur.fetchone() or {})

                cur.execute(
                    """
                    SELECT
                        COUNT(*)::int AS label_count,
                        COALESCE(AVG(y_true_num), 0)::float AS avg_realized_pnl_pct,
                        COALESCE(AVG(CASE WHEN y_true_num > 0 THEN 1.0 ELSE 0.0 END) * 100.0, 0)::float AS win_rate_pct
                    FROM label_events
                    """
                )
                label_summary = dict(cur.fetchone() or {})

                cur.execute(
                    """
                    SELECT COALESCE(AVG(ABS(l.y_true_num - i.y_pred_num)), 0)::float AS mean_abs_error_pct
                    FROM label_events l
                    JOIN inference_events i ON i.request_id = l.request_id
                    WHERE l.y_true_num IS NOT NULL AND i.y_pred_num IS NOT NULL
                    """
                )
                error_summary = dict(cur.fetchone() or {})

                cur.execute(
                    """
                    SELECT
                        i.ts, i.request_id, i.model_id, i.model_version, i.y_pred_num,
                        i.latency_ms, i.features_json, i.segment_json, l.y_true_num
                    FROM inference_events i
                    LEFT JOIN label_events l ON l.request_id = i.request_id
                    ORDER BY i.ts DESC
                    LIMIT 80
                    """
                )
                recent = []
                feature_keys = set()
                for row in cur.fetchall():
                    features = row.get("features_json") or {}
                    segment = row.get("segment_json") or {}
                    if isinstance(features, dict):
                        feature_keys.update(features.keys())
                    recent.append(
                        {
                            "ts": row.get("ts"),
                            "request_id": row.get("request_id"),
                            "symbol": segment.get("sym") or segment.get("symbol") or "-",
                            "model_id": row.get("model_id"),
                            "model_version": row.get("model_version"),
                            "y_pred_num": _round(row.get("y_pred_num"), 5),
                            "y_true_num": _round(row.get("y_true_num"), 5),
                            "latency_ms": row.get("latency_ms"),
                            "target_frac": _round(features.get("target_frac"), 4) if isinstance(features, dict) else None,
                            "p_up": _round(features.get("p_up"), 4) if isinstance(features, dict) else None,
                        }
                    )

                cur.execute(
                    """
                    SELECT ts, latency_ms AS value
                    FROM inference_events
                    WHERE latency_ms IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT 50
                    """
                )
                latency_series = list(reversed([dict(row) for row in cur.fetchall()]))

                cur.execute(
                    """
                    SELECT ts, y_true_num AS value
                    FROM label_events
                    WHERE y_true_num IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT 50
                    """
                )
                pnl_series = list(reversed([dict(row) for row in cur.fetchall()]))

        summary = {**inference_summary, **label_summary, **error_summary}
        if not summary.get("inference_count"):
            summary["avg_latency_ms"] = None
            summary["avg_pred_pct"] = None
            summary["last_event_ts"] = None
        if not summary.get("label_count"):
            summary["avg_realized_pnl_pct"] = None
            summary["win_rate_pct"] = None
            summary["mean_abs_error_pct"] = None
        return {
            "mode": "live",
            "is_demo": False,
            "status": "Connected to DriftWatch database.",
            "detail": "Rendering inference_events and label_events from PostgreSQL.",
            "summary": {k: _round(v, 3) if isinstance(v, float) else v for k, v in summary.items()},
            "recent": recent,
            "latency_series": latency_series,
            "pnl_series": pnl_series,
            "feature_keys": sorted(feature_keys),
        }
    except Exception as exc:
        return _sample_results(f"{exc.__class__.__name__}: {exc}")


def fetch_project() -> Dict[str, Any]:
    return {
        "project": PROJECT_SPEC,
        "repo": repo_snapshot(),
        "trading": fetch_trading_data(),
        "drift": fetch_results(),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Alpaca Project Command Center</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #04070d;
      --bg-2: #07101c;
      --panel: #09111f;
      --panel-2: #0c1729;
      --line: #17365f;
      --blue: #2f8cff;
      --blue-2: #62b7ff;
      --blue-3: #9fd4ff;
      --text: #e8f2ff;
      --muted: #85a4c5;
      --danger: #ff5b7a;
      --good: #32d79b;
      --warn: #ffd166;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 10% -5%, rgba(47,140,255,.28), transparent 32rem),
        radial-gradient(circle at 95% 10%, rgba(98,183,255,.14), transparent 24rem),
        linear-gradient(180deg, var(--bg), var(--bg-2) 48%, var(--bg));
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: inherit; text-decoration: none; }
    .shell { max-width: 1440px; margin: 0 auto; padding: 26px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
      gap: 18px;
      align-items: stretch;
      min-height: 390px;
      padding-top: 8px;
    }
    .hero-main {
      border: 1px solid rgba(98,183,255,.18);
      background: linear-gradient(145deg, rgba(10,22,42,.72), rgba(4,7,13,.96));
      border-radius: 8px;
      padding: 28px;
      overflow: hidden;
      position: relative;
    }
    .hero-main:after {
      content: "";
      position: absolute;
      inset: auto 0 0 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--blue), transparent);
    }
    .eyebrow { color: var(--blue-2); text-transform: uppercase; letter-spacing: .12em; font-size: 12px; font-weight: 700; }
    h1 { margin: 16px 0 14px; font-size: clamp(34px, 5vw, 76px); line-height: .95; letter-spacing: 0; }
    .lead { color: #bdd4ee; max-width: 820px; font-size: 16px; margin: 0; }
    .nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 28px;
    }
    .nav a, .chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid rgba(98,183,255,.24);
      background: rgba(47,140,255,.08);
      color: #d9ecff;
      border-radius: 6px;
      padding: 8px 10px;
      min-height: 34px;
    }
    .status-card {
      border: 1px solid rgba(98,183,255,.18);
      background: rgba(7,13,24,.86);
      border-radius: 8px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 100%;
    }
    .mode { display: flex; align-items: center; gap: 9px; color: var(--blue-2); font-weight: 700; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--blue); box-shadow: 0 0 18px var(--blue); }
    .grid { display: grid; gap: 16px; margin-top: 18px; }
    .metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .two { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .wide-right { grid-template-columns: minmax(0, .78fr) minmax(0, 1.22fr); }
    .card {
      border: 1px solid rgba(98,183,255,.18);
      background: linear-gradient(180deg, rgba(12,22,41,.95), rgba(6,12,23,.96));
      border-radius: 8px;
      box-shadow: 0 18px 60px rgba(0,0,0,.34);
      overflow: hidden;
    }
    .metric { padding: 18px; min-height: 116px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 11px; font-size: 31px; font-weight: 780; letter-spacing: 0; }
    .meta { margin-top: 8px; color: var(--muted); font-size: 12px; }
    .good { color: var(--good); }
    .bad { color: var(--danger); }
    .warn { color: var(--warn); }
    .section { margin-top: 26px; scroll-margin-top: 18px; }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(98,183,255,.14);
      align-items: center;
    }
    .section-head h2 { margin: 0; font-size: 17px; letter-spacing: 0; }
    .section-head span { color: var(--muted); font-size: 12px; }
    .body { padding: 18px; }
    .flow { display: grid; gap: 10px; }
    .flow-step {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 12px;
      padding: 13px;
      border: 1px solid rgba(98,183,255,.15);
      background: rgba(47,140,255,.05);
      border-radius: 8px;
    }
    .num {
      width: 34px; height: 34px; display: grid; place-items: center;
      border-radius: 50%; background: rgba(47,140,255,.18); color: var(--blue-2); font-weight: 800;
    }
    .flow-step h3, .mini h3 { margin: 0 0 6px; font-size: 15px; }
    .flow-step p, .mini p { margin: 0; color: var(--muted); }
    .reqs { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .reqs span { color: var(--blue-3); border: 1px solid rgba(98,183,255,.2); border-radius: 5px; padding: 3px 6px; font-size: 11px; }
    .mini-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .mini {
      padding: 14px;
      border: 1px solid rgba(98,183,255,.14);
      background: rgba(4,8,15,.35);
      border-radius: 8px;
    }
    canvas { display: block; width: 100%; height: 300px; }
    .features, .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .features span, .chips span {
      border: 1px solid rgba(98,183,255,.2);
      background: rgba(47,140,255,.08);
      padding: 6px 9px;
      border-radius: 6px;
      color: #cfe5ff;
    }
    .risk-row {
      display: grid;
      grid-template-columns: 38px 1fr 86px;
      gap: 12px;
      align-items: center;
      padding: 11px 0;
      border-bottom: 1px solid rgba(98,183,255,.1);
    }
    .risk-row:last-child { border-bottom: 0; }
    .bar { height: 7px; border-radius: 999px; background: rgba(98,183,255,.12); overflow: hidden; margin-top: 8px; }
    .bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--blue), var(--blue-2)); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid rgba(98,183,255,.1); white-space: nowrap; }
    th { color: var(--muted); font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
    td { color: #dcecff; }
    .table-wrap { overflow: auto; max-height: 520px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .footer { color: var(--muted); padding: 28px 0 8px; }
    @media (max-width: 980px) {
      .shell { padding: 16px; }
      .hero, .metrics, .two, .three, .wide-right, .mini-grid { grid-template-columns: 1fr; }
      canvas { height: 240px; }
      h1 { font-size: 42px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-main">
        <div class="eyebrow">Alpaca trading system</div>
        <h1>Project Command Center</h1>
        <p class="lead" id="thesis">Loading project model.</p>
        <div class="nav">
          <a href="/overview#overview">Overview</a>
          <a href="/trading#trading">Trading</a>
          <a href="/architecture#architecture">Architecture</a>
          <a href="/agent#agent">Agent</a>
          <a href="/risk#risk">Risk</a>
          <a href="/driftwatch#driftwatch">DriftWatch</a>
          <a href="/validation#validation">Validation</a>
          <a href="/config#config">Config</a>
          <a href="/roadmap#roadmap">Roadmap</a>
        </div>
      </div>
      <aside class="status-card">
        <div>
          <div class="mode"><span class="dot"></span><span id="mode">Connecting</span></div>
          <p class="lead" id="detail" style="margin-top:18px">Loading database and repository status.</p>
        </div>
        <div class="chips" id="envChips"></div>
      </aside>
    </section>

    <section id="overview" class="section">
      <div class="grid metrics" id="metrics"></div>
    </section>

    <section id="trading" class="section grid wide-right">
      <div class="card">
        <div class="section-head"><h2>Trading Account</h2><span id="tradingMode">loading</span></div>
        <div class="body">
          <div class="grid metrics" id="tradingMetrics" style="margin-top:0"></div>
        </div>
      </div>
      <div class="card">
        <div class="section-head"><h2>Equity Curve</h2><span>Alpaca portfolio history</span></div>
        <canvas id="equityChart"></canvas>
      </div>
    </section>

    <section class="section grid two">
      <div class="card">
        <div class="section-head"><h2>Open Positions</h2><span id="positionCount">0 positions</span></div>
        <div class="table-wrap"><table><thead><tr><th>Asset</th><th>Side</th><th>Qty</th><th>Price</th><th>Market Value</th><th>Avg Entry</th><th>Total P/L</th><th>P/L %</th></tr></thead><tbody id="positionRows"></tbody></table></div>
      </div>
      <div class="card">
        <div class="section-head"><h2>Position P/L</h2><span>dollars by symbol</span></div>
        <canvas id="positionPnlChart"></canvas>
      </div>
    </section>

    <section class="section card">
      <div class="section-head"><h2>Recent Orders</h2><span id="orderCount">0 orders</span></div>
      <div class="table-wrap"><table><thead><tr><th>Submitted</th><th>Asset</th><th>Type</th><th>Side</th><th>Qty</th><th>Notional</th><th>Filled Qty</th><th>Avg Fill</th><th>Status</th></tr></thead><tbody id="orderRows"></tbody></table></div>
    </section>

    <section id="architecture" class="section grid wide-right">
      <div class="card">
        <div class="section-head"><h2>System Architecture</h2><span>report FR-01 to FR-42</span></div>
        <div class="body flow" id="architectureFlow"></div>
      </div>
      <div class="grid" style="margin-top:0">
        <div class="card">
          <div class="section-head"><h2>Pipeline Map</h2><span>data to execution</span></div>
          <canvas id="pipelineChart"></canvas>
        </div>
        <div class="card">
          <div class="section-head"><h2>Repository Snapshot</h2><span>local files</span></div>
          <div class="body mini-grid" id="repoSnapshot"></div>
        </div>
      </div>
    </section>

    <section id="agent" class="section grid two">
      <div class="card">
        <div class="section-head"><h2>ARLAgent Profiles</h2><span>UserStyle defaults</span></div>
        <canvas id="agentChart"></canvas>
      </div>
      <div class="card">
        <div class="section-head"><h2>Profile Matrix</h2><span>risk and learning knobs</span></div>
        <div class="table-wrap"><table><thead><tr><th>Profile</th><th>Lookback</th><th>Symbols</th><th>Band</th><th>EMA</th><th>Size</th><th>Hold</th><th>epsilon</th><th>alpha</th></tr></thead><tbody id="profileRows"></tbody></table></div>
      </div>
    </section>

    <section id="risk" class="section grid two">
      <div class="card">
        <div class="section-head"><h2>Seven-Layer Guardrail Cascade</h2><span>external runtime shield</span></div>
        <div class="body" id="riskRows"></div>
      </div>
      <div class="card">
        <div class="section-head"><h2>Risk Thresholds</h2><span>default envelope</span></div>
        <canvas id="riskChart"></canvas>
      </div>
    </section>

    <section id="driftwatch" class="section card">
      <div class="section-head"><h2>DriftWatch Status</h2><span id="driftMode">loading</span></div>
      <div class="body" id="driftNotice"></div>
    </section>

    <section class="section grid two">
      <div class="card">
        <div class="section-head"><h2>DriftWatch Results</h2><span>prediction vs realised labels</span></div>
        <canvas id="pnlChart"></canvas>
      </div>
      <div class="card">
        <div class="section-head"><h2>Inference Latency</h2><span>latest events</span></div>
        <canvas id="latencyChart"></canvas>
      </div>
    </section>

    <section class="section card">
      <div class="section-head"><h2>Recent Inference Events</h2><span id="rowCount">0 rows</span></div>
      <div class="table-wrap"><table><thead><tr><th>Time</th><th>Symbol</th><th>Pred %</th><th>Actual %</th><th>P Up</th><th>Target</th><th>Latency</th><th>Request</th></tr></thead><tbody id="eventRows"></tbody></table></div>
    </section>

    <section id="validation" class="section grid two">
      <div class="card">
        <div class="section-head"><h2>Validation Coverage</h2><span>report Chapter 6</span></div>
        <canvas id="validationChart"></canvas>
      </div>
      <div class="card">
        <div class="section-head"><h2>Test Inventory</h2><span id="testCount">0 tests</span></div>
        <div class="body" id="tests"></div>
      </div>
    </section>

    <section id="config" class="section grid two">
      <div class="card">
        <div class="section-head"><h2>Feature Manifest</h2><span id="featureCount">0 features</span></div>
        <div class="body features" id="features"></div>
      </div>
      <div class="card">
        <div class="section-head"><h2>Paths and Models</h2><span>local artifacts</span></div>
        <div class="body mini-grid" id="paths"></div>
      </div>
    </section>

    <section id="roadmap" class="section card">
      <div class="section-head"><h2>Future Work</h2><span>from report Chapter 7</span></div>
      <div class="body mini-grid" id="future"></div>
    </section>

    <div class="footer">Localhost command center. Data refreshes every 15 seconds.</div>
  </main>

  <script>
    const fmt = (v, digits = 2) => v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(digits);
    const signed = (v, digits = 3) => v === null || v === undefined ? "-" : `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(digits)}`;
    const money = (v) => v === null || v === undefined ? "-" : `${Number(v) < 0 ? "-" : ""}$${Math.abs(Number(v)).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    const cls = (v) => v === null || v === undefined || Number.isNaN(Number(v)) ? "" : (Number(v) >= 0 ? "good" : "bad");
    const time = (v) => v ? new Date(v).toLocaleString() : "-";
    const bytes = (n) => !n ? "0 B" : n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${(n / 1024).toFixed(1)} KB`;

    function metric(label, value, meta, kind = "") {
      return `<div class="card metric"><div class="label">${label}</div><div class="value ${kind}">${value}</div><div class="meta">${meta || ""}</div></div>`;
    }

    function drawLine(canvas, data, color, zeroLine = false) {
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, rect.width * scale);
      canvas.height = Math.max(1, rect.height * scale);
      ctx.scale(scale, scale);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const pad = 28, w = rect.width - pad * 2, h = rect.height - pad * 2;
      ctx.strokeStyle = "rgba(98,183,255,.14)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i++) {
        const y = pad + (h / 3) * i;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + w, y); ctx.stroke();
      }
      const values = (data || []).map(d => Number(d.value)).filter(Number.isFinite);
      if (!values.length) return;
      let min = Math.min(...values), max = Math.max(...values);
      if (zeroLine) { min = Math.min(min, 0); max = Math.max(max, 0); }
      if (min === max) { min -= 1; max += 1; }
      const points = (data || []).map((d, i) => {
        const x = pad + (data.length === 1 ? w : (i / (data.length - 1)) * w);
        const y = pad + h - ((Number(d.value) - min) / (max - min)) * h;
        return [x, y, Number(d.value)];
      });
      if (zeroLine && min < 0 && max > 0) {
        const zy = pad + h - ((0 - min) / (max - min)) * h;
        ctx.strokeStyle = "rgba(232,242,255,.22)";
        ctx.beginPath(); ctx.moveTo(pad, zy); ctx.lineTo(pad + w, zy); ctx.stroke();
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.stroke();
      points.slice(-14).forEach(([x, y, v]) => {
        ctx.fillStyle = zeroLine ? (v >= 0 ? "#32d79b" : "#ff5b7a") : color;
        ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
      });
    }

    function drawBars(canvas, items, color = "#2f8cff") {
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = rect.width * scale; canvas.height = rect.height * scale; ctx.scale(scale, scale);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const pad = 34, w = rect.width - pad * 2, h = rect.height - pad * 2;
      const max = Math.max(...items.map(i => Number(i.value)), 1);
      const bw = w / items.length * .62;
      items.forEach((item, i) => {
        const x = pad + (i + .19) * (w / items.length);
        const bh = (Number(item.value) / max) * h;
        const y = pad + h - bh;
        ctx.fillStyle = "rgba(47,140,255,.16)";
        ctx.fillRect(x, pad, bw, h);
        ctx.fillStyle = color;
        ctx.fillRect(x, y, bw, bh);
        ctx.fillStyle = "#bdd4ee";
        ctx.font = "12px system-ui";
        ctx.fillText(item.label, x, rect.height - 10);
      });
    }

    function drawDivergingBars(canvas, items) {
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = rect.width * scale; canvas.height = rect.height * scale; ctx.scale(scale, scale);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const pad = 34, w = rect.width - pad * 2, h = rect.height - pad * 2;
      const values = (items || []).map(i => Number(i.value)).filter(Number.isFinite);
      if (!values.length) return;
      const maxAbs = Math.max(...values.map(Math.abs), 1);
      const zeroY = pad + h / 2;
      ctx.strokeStyle = "rgba(232,242,255,.22)";
      ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(pad + w, zeroY); ctx.stroke();
      const bw = w / items.length * .62;
      items.forEach((item, i) => {
        const value = Number(item.value);
        const bh = Math.abs(value) / maxAbs * (h / 2);
        const x = pad + (i + .19) * (w / items.length);
        const y = value >= 0 ? zeroY - bh : zeroY;
        ctx.fillStyle = value >= 0 ? "#32d79b" : "#ff5b7a";
        ctx.fillRect(x, y, bw, bh);
        ctx.fillStyle = "#bdd4ee";
        ctx.font = "12px system-ui";
        ctx.fillText(item.label, x, rect.height - 10);
      });
    }

    function render(data) {
      const project = data.project, repo = data.repo, drift = data.drift, trading = data.trading || {}, s = drift.summary || {};
      const account = trading.account || {};
      const liveTrading = trading.mode === "live";
      const liveDrift = drift.mode === "live";
      document.getElementById("thesis").textContent = project.thesis;
      document.getElementById("mode").textContent = liveTrading ? "Live Alpaca trading data" : (liveDrift ? "Live DriftWatch database" : "Operational sources offline");
      document.getElementById("detail").textContent = liveTrading
        ? `${trading.status} DriftWatch: ${drift.status}`
        : `${trading.status || ""} ${drift.status} ${drift.detail || ""}`;

      const env = repo.env || {};
      document.getElementById("envChips").innerHTML = Object.entries(env).map(([k, v]) =>
        `<span class="${v ? "good" : "warn"}">${k}: ${v ? "set" : "missing"}</span>`).join("");

      document.getElementById("metrics").innerHTML = [
        metric("Universe", repo.universe_count, "liquid US large-cap symbols"),
        metric("Features", repo.feature_count, "model input manifest"),
        metric("Equity", money(account.equity), liveTrading ? "live Alpaca paper account" : "Alpaca source offline", cls(account.daily_pl)),
        metric("Daily P/L", `${money(account.daily_pl)} (${signed(account.daily_plpc, 2)}%)`, "broker account change", cls(account.daily_pl)),
        metric("Buying Power", money(account.buying_power), "available notional"),
        metric("Open Positions", (trading.positions || []).length, "current holdings"),
        metric("Recent Orders", (trading.orders || []).length, "latest broker orders"),
        metric("Tests", repo.tests.count, "pytest functions in repo"),
      ].join("");

      document.getElementById("tradingMode").textContent = trading.mode === "live" ? "live paper account" : "offline";
      document.getElementById("tradingMetrics").innerHTML = [
        metric("Equity", money(account.equity), "portfolio value", cls(account.daily_pl)),
        metric("Daily P/L", `${money(account.daily_pl)} (${signed(account.daily_plpc, 2)}%)`, "from last equity", cls(account.daily_pl)),
        metric("Cash", money(account.cash), account.currency || "USD", Number(account.cash) >= 0 ? "good" : "bad"),
        metric("Buying Power", money(account.buying_power), "available notional"),
        metric("Gross Exposure", money(account.gross_exposure), `${fmt(account.gross_exposure_pct, 2)}% of equity`),
        metric("Account", account.status || "-", `paper ${account.account_number || ""}`),
        metric("Open Positions", (trading.positions || []).length, "current holdings"),
        metric("Recent Orders", (trading.orders || []).length, "latest broker orders"),
      ].join("");

      const positions = trading.positions || [];
      document.getElementById("positionCount").textContent = `${positions.length} positions`;
      document.getElementById("positionRows").innerHTML = positions.map(p => `
        <tr>
          <td class="mono">${p.symbol}</td><td>${p.side}</td><td>${fmt(p.qty, 4)}</td><td>${money(p.price)}</td>
          <td>${money(p.market_value)}</td><td>${money(p.avg_entry_price)}</td>
          <td class="${cls(p.unrealized_pl)}">${money(p.unrealized_pl)}</td><td class="${cls(p.unrealized_plpc)}">${signed(p.unrealized_plpc, 2)}%</td>
        </tr>`).join("") || `<tr><td colspan="8">No open positions.</td></tr>`;

      const orders = trading.orders || [];
      document.getElementById("orderCount").textContent = `${orders.length} orders`;
      document.getElementById("orderRows").innerHTML = orders.map(o => `
        <tr>
          <td>${time(o.submitted_at)}</td><td class="mono">${o.symbol}</td><td>${o.type || "-"}</td><td class="${o.side === "sell" ? "bad" : "good"}">${o.side || "-"}</td>
          <td>${fmt(o.qty, 4)}</td><td>${o.notional === null || o.notional === undefined ? "-" : money(o.notional)}</td>
          <td>${fmt(o.filled_qty, 4)}</td><td>${o.filled_avg_price === null || o.filled_avg_price === undefined ? "-" : money(o.filled_avg_price)}</td><td>${o.status || "-"}</td>
        </tr>`).join("") || `<tr><td colspan="9">No recent orders.</td></tr>`;

      document.getElementById("architectureFlow").innerHTML = project.architecture.map((step, i) => `
        <div class="flow-step">
          <div class="num">${i + 1}</div>
          <div><h3>${step.name}</h3><p>${step.detail}</p><div class="reqs">${step.requirements.map(r => `<span>${r}</span>`).join("")}</div></div>
        </div>`).join("");

      document.getElementById("repoSnapshot").innerHTML = [
        ["Models", Object.entries(repo.models).filter(([, m]) => m.exists).length + " present", "XGBoost and ElasticNet artifacts"],
        ["Universe preview", (repo.universe_preview || []).slice(0, 8).join(", "), "first symbols loaded from app/universe.csv"],
        ["Drift mode", drift.mode, drift.status],
        ["Last event", s.last_event_ts ? time(s.last_event_ts) : "-", "latest inference timestamp"],
      ].map(([a,b,c]) => `<div class="mini"><h3>${a}</h3><p class="value" style="font-size:20px">${b}</p><p>${c}</p></div>`).join("");

      document.getElementById("profileRows").innerHTML = project.user_styles.map(p => `
        <tr><td class="mono">${p.name}</td><td>${p.lookback}d</td><td>${p.max_symbols}</td><td>${p.band}</td><td>${p.ema}</td><td>${p.size}</td><td>${p.hold}</td><td>${p.epsilon}</td><td>${p.alpha}</td></tr>
      `).join("");

      document.getElementById("riskRows").innerHTML = project.risk_guardrails.map(g => `
        <div class="risk-row">
          <div class="num">${g.layer}</div>
          <div><strong>${g.name}</strong><p class="meta">${g.action}</p><div class="bar"><span style="width:${18 + g.layer * 10}%"></span></div></div>
          <div class="mono">${g.default}</div>
        </div>`).join("");

      const recent = drift.recent || [];
      document.getElementById("driftMode").textContent = liveDrift ? "live database" : "database unavailable";
      document.getElementById("driftNotice").innerHTML = liveDrift ? `
        <div class="mini-grid">
          <div class="mini"><h3>Inferences</h3><p class="value" style="font-size:20px">${s.inference_count ?? 0}</p><p>rows in inference_events</p></div>
          <div class="mini"><h3>Labels</h3><p class="value" style="font-size:20px">${s.label_count ?? 0}</p><p>rows in label_events</p></div>
          <div class="mini"><h3>Win Rate</h3><p class="value ${Number(s.win_rate_pct) >= 50 ? "good" : "bad"}" style="font-size:20px">${fmt(s.win_rate_pct, 1)}%</p><p>realised positive labels</p></div>
          <div class="mini"><h3>Avg Realised</h3><p class="value ${cls(s.avg_realized_pnl_pct)}" style="font-size:20px">${signed(s.avg_realized_pnl_pct, 3)}%</p><p>mean realised P/L label</p></div>
        </div>` : `
        <div class="mini">
          <h3>Live DriftWatch data is not connected</h3>
          <p>${drift.detail || drift.status}</p>
          <p>No DriftWatch rows or charts are rendered until PostgreSQL accepts the configured connection. Overview P/L and trading metrics come from Alpaca, not DriftWatch.</p>
        </div>`;
      document.getElementById("rowCount").textContent = `${recent.length} rows`;
      document.getElementById("eventRows").innerHTML = recent.map(r => `
        <tr>
          <td>${time(r.ts)}</td><td class="mono">${r.symbol || "-"}</td>
          <td class="${cls(r.y_pred_num)}">${signed(r.y_pred_num, 4)}</td>
          <td class="${r.y_true_num === null || r.y_true_num === undefined ? "" : cls(r.y_true_num)}">${signed(r.y_true_num, 4)}</td>
          <td>${fmt(r.p_up, 3)}</td><td>${signed(r.target_frac, 4)}</td><td>${r.latency_ms ?? "-"} ms</td><td class="mono">${r.request_id || "-"}</td>
        </tr>`).join("") || `<tr><td colspan="8">No live DriftWatch rows are available.</td></tr>`;

      document.getElementById("testCount").textContent = `${repo.tests.count} tests`;
      document.getElementById("tests").innerHTML = project.tests.map((t, i) => `<div class="mini" style="margin-bottom:10px"><h3>Test area ${i + 1}</h3><p>${t}</p></div>`).join("") +
        `<div class="chips">${repo.tests.files.map(f => `<span class="mono">${f}</span>`).join("")}</div>`;

      const features = repo.feature_cols || [];
      document.getElementById("featureCount").textContent = `${features.length} features`;
      document.getElementById("features").innerHTML = features.map(f => `<span class="mono">${f}</span>`).join("");

      document.getElementById("paths").innerHTML = [
        ...Object.entries(repo.models).map(([name, m]) => `<div class="mini"><h3>${name}</h3><p class="${m.exists ? "good" : "bad"}">${m.exists ? "present" : "missing"}</p><p>${bytes(m.bytes)}</p></div>`),
        ...repo.important_paths.slice(0, 4).map(p => `<div class="mini"><h3 class="mono">${p}</h3><p>Core project artifact</p></div>`),
      ].join("");

      document.getElementById("future").innerHTML = project.future_work.map((item, i) => `<div class="mini"><h3>${i + 1}. ${item.split(" with ")[0]}</h3><p>${item}</p></div>`).join("");

      drawBars(document.getElementById("pipelineChart"), project.architecture.map((x) => ({label: x.name.split(" ")[0], value: x.requirements.length})), "#62b7ff");
      drawBars(document.getElementById("agentChart"), project.user_styles.map(p => ({label: p.risk, value: p.lookback})), "#2f8cff");
      drawBars(document.getElementById("riskChart"), project.risk_guardrails.map(g => ({label: `L${g.layer}`, value: g.layer})), "#62b7ff");
      drawBars(document.getElementById("validationChart"), [
        {label: "Agent", value: 7}, {label: "Ledger", value: 6}, {label: "Drift", value: 4}, {label: "Site", value: 2}
      ], "#2f8cff");
      drawLine(document.getElementById("equityChart"), trading.equity_series || [], "#f6c945", false);
      drawDivergingBars(document.getElementById("positionPnlChart"), trading.position_pnl_series || []);
      drawLine(document.getElementById("pnlChart"), drift.pnl_series || [], "#32d79b", true);
      drawLine(document.getElementById("latencyChart"), drift.latency_series || [], "#62b7ff", false);
    }

    async function load() {
      const res = await fetch("/api/project", { cache: "no-store" });
      render(await res.json());
    }
    load();
    setInterval(load, 15000);
    addEventListener("resize", load);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def _write(self, status: HTTPStatus, content_type: str, body: bytes, include_body: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _handle(self, include_body: bool = True) -> None:
        path = urlparse(self.path).path
        if path in PAGE_PATHS:
            self._write(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode("utf-8"), include_body)
            return
        if path == "/api/results":
            body = json.dumps(fetch_results(), default=_json_default).encode("utf-8")
            self._write(HTTPStatus.OK, "application/json; charset=utf-8", body, include_body)
            return
        if path == "/api/trading":
            body = json.dumps(fetch_trading_data(), default=_json_default).encode("utf-8")
            self._write(HTTPStatus.OK, "application/json; charset=utf-8", body, include_body)
            return
        if path == "/api/project":
            body = json.dumps(fetch_project(), default=_json_default).encode("utf-8")
            self._write(HTTPStatus.OK, "application/json; charset=utf-8", body, include_body)
            return
        self._write(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found", include_body)

    def do_HEAD(self) -> None:
        self._handle(include_body=False)

    def do_GET(self) -> None:
        self._handle(include_body=True)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[AlpacaSite] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a localhost Alpaca project website.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=int(os.getenv("DRIFTWATCH_DASHBOARD_PORT", DEFAULT_PORT)))
    args = parser.parse_args()

    load_project_env()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"{APP_TITLE} running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
