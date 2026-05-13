from datetime import datetime, timezone

import pytest
import psycopg2

from app import driftwatch_client as drift
from app.driftwatch_client import DriftWatchClient


class FakeCursor:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql):
        self.calls.append(("execute", sql))

    def executemany(self, sql, params):
        self.calls.append(("executemany", sql, list(params)))


class FakeConnection:
    def __init__(self, calls, fail=False):
        self.calls = calls
        self.fail = fail

    def __enter__(self):
        if self.fail:
            raise psycopg2.OperationalError("database unavailable")
        return self

    def __exit__(self, *_exc):
        return False

    def cursor(self):
        return FakeCursor(self.calls)

    def commit(self):
        self.calls.append(("commit", None))


def fixed_now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


def test_client_disables_when_enabled_without_dsn(monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_ENABLED", "true")
    monkeypatch.delenv("DRIFTWATCH_DATABASE_URL", raising=False)

    client = DriftWatchClient()

    assert not client.enabled


def test_log_inference_and_label_sanitize_and_flush(monkeypatch):
    calls = []
    monkeypatch.setenv("DRIFTWATCH_ENABLED", "true")
    monkeypatch.setenv("DRIFTWATCH_DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("DRIFTWATCH_BATCH_SIZE", "10")
    monkeypatch.setattr(drift.psycopg2, "connect", lambda *_args, **_kwargs: FakeConnection(calls))

    client = DriftWatchClient()
    client.log_inference(
        model_id="model",
        model_version="v1",
        ts=fixed_now(),
        pred_type="regression",
        y_pred_num=float("nan"),
        y_pred_text=None,
        latency_ms=12,
        features_json={"x": float("inf"), "ok": 1.5},
        segment_json={"sym": "AAPL"},
        request_id="req-1",
    )
    client.log_label(
        ts=fixed_now(),
        request_id="req-1",
        y_true_num=float("-inf"),
        extra_json={"label_name": "realized_pnl_pct"},
    )

    assert len(client.buffer) == 1
    assert len(client.label_buffer) == 1

    client.flush()

    assert len(client.buffer) == 0
    assert len(client.label_buffer) == 0
    assert client.insert_success == 1
    assert client.label_insert_success == 1

    executed_sql = "\n".join(call[1] for call in calls if call[0] == "execute")
    assert "CREATE TABLE IF NOT EXISTS inference_events" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS label_events" in executed_sql

    insert_calls = [call for call in calls if call[0] == "executemany"]
    assert len(insert_calls) == 2
    assert "ON CONFLICT (request_id) DO UPDATE" in insert_calls[0][1]
    assert "ON CONFLICT (request_id) DO UPDATE" in insert_calls[1][1]

    inference_event = insert_calls[0][2][0]
    label_event = insert_calls[1][2][0]
    assert inference_event[7] is None
    assert label_event[2] is None


def test_auto_create_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setenv("DRIFTWATCH_ENABLED", "true")
    monkeypatch.setenv("DRIFTWATCH_DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("DRIFTWATCH_AUTO_CREATE", "false")
    monkeypatch.setattr(drift.psycopg2, "connect", lambda *_args, **_kwargs: FakeConnection(calls))

    client = DriftWatchClient()
    client.log_inference(
        model_id="model",
        model_version="v1",
        ts=fixed_now(),
        pred_type="regression",
        y_pred_num=0.1,
        y_pred_text=None,
        latency_ms=1,
        features_json={},
        segment_json={},
        request_id="req-2",
    )
    client.flush()

    executed_sql = "\n".join(call[1] for call in calls if call[0] == "execute")
    assert "CREATE TABLE IF NOT EXISTS" not in executed_sql


def test_failed_flush_retries_then_drops_batch(monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_ENABLED", "true")
    monkeypatch.setenv("DRIFTWATCH_DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("DRIFTWATCH_BATCH_SIZE", "10")
    monkeypatch.setattr(drift, "MAX_RETRIES", 2)
    monkeypatch.setattr(drift.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(drift.psycopg2, "connect", lambda *_args, **_kwargs: FakeConnection([], fail=True))

    client = DriftWatchClient()
    client.log_label(ts=fixed_now(), request_id="req-3", y_true_num=1.0)
    client.flush()

    assert client.flush_failures == 1
    assert len(client.label_buffer) == 0
    assert client.label_insert_success == 0
