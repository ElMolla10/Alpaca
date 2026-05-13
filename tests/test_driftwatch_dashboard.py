import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from app import driftwatch_dashboard as dash
from app.driftwatch_dashboard import DashboardHandler, fetch_project, fetch_results, fetch_trading_data


def test_fetch_results_returns_sample_without_database_url(monkeypatch):
    monkeypatch.setattr(dash, "load_project_env", lambda: None)
    monkeypatch.delenv("DRIFTWATCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    data = fetch_results()

    assert data["mode"] == "offline"
    assert data["is_demo"] is False
    assert data["summary"]["inference_count"] == 0
    assert data["summary"]["win_rate_pct"] is None
    assert data["recent"] == []
    assert data["feature_keys"] == []


def test_fetch_project_includes_report_sections(monkeypatch):
    monkeypatch.setattr(dash, "load_project_env", lambda: None)
    monkeypatch.delenv("DRIFTWATCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(dash, "REST", None)

    data = fetch_project()

    assert data["project"]["architecture"]
    assert len(data["project"]["risk_guardrails"]) == 7
    assert len(data["project"]["user_styles"]) == 3
    assert data["repo"]["feature_count"] > 0
    assert data["repo"]["tests"]["count"] > 0
    assert data["drift"]["mode"] == "offline"
    assert "trading" in data


def test_fetch_trading_data_has_sample_fallback(monkeypatch):
    monkeypatch.setattr(dash, "REST", None)
    data = fetch_trading_data()

    assert data["mode"] == "offline"
    assert data["account"]["equity"] is None
    assert data["positions"] == []
    assert data["orders"] == []
    assert data["equity_series"] == []


def test_dashboard_serves_page_and_api(monkeypatch):
    monkeypatch.setattr(dash, "load_project_env", lambda: None)
    monkeypatch.delenv("DRIFTWATCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(dash, "REST", None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    try:
        with urllib.request.urlopen(base, timeout=3) as response:
            html = response.read().decode("utf-8")
        assert "Alpaca Project Command Center" in html
        assert "System Architecture" in html
        assert "/risk#risk" in html
        assert "pnlChart" in html

        with urllib.request.urlopen(f"{base}/risk", timeout=3) as response:
            risk_page = response.read().decode("utf-8")
        assert "Seven-Layer Guardrail Cascade" in risk_page

        with urllib.request.urlopen(f"{base}/api/results", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["mode"] == "offline"
        assert payload["is_demo"] is False
        assert payload["summary"]["label_count"] == 0

        with urllib.request.urlopen(f"{base}/api/project", timeout=3) as response:
            project = json.loads(response.read().decode("utf-8"))
        assert project["project"]["headline"] == "End-to-end agentic electronic trading engine"
        assert len(project["project"]["architecture"]) >= 8
        assert "trading" in project

        with urllib.request.urlopen(f"{base}/api/trading", timeout=3) as response:
            trading = json.loads(response.read().decode("utf-8"))
        assert "account" in trading
        assert "positions" in trading
    finally:
        server.shutdown()
        server.server_close()
