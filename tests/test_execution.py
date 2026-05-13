import pytest

from app.execution import BlockLedger


TRADE_COST_BPS = 8.0
SLIP_BPS = 4.0
ONE_WAY_COST_PCT = (TRADE_COST_BPS + SLIP_BPS) / 100.0
ROUND_TRIP_COST_PCT = (TRADE_COST_BPS + SLIP_BPS) / 50.0


def ledger():
    return BlockLedger(TRADE_COST_BPS, SLIP_BPS)


def test_empty_ledger_returns_none():
    assert ledger().compute_symbol_pnl_pct("AAPL") is None


def test_round_trip_long_same_price_is_negative_round_trip_cost():
    book = ledger()
    book.record_fill("AAPL", "buy", qty=10, price=100.0)
    book.record_fill("AAPL", "sell", qty=10, price=100.0)

    assert book.compute_symbol_pnl_pct("AAPL") == pytest.approx(-ROUND_TRIP_COST_PCT)


def test_round_trip_long_with_profit_is_profit_less_costs():
    book = ledger()
    book.record_fill("AAPL", "buy", qty=10, price=100.0)
    book.record_fill("AAPL", "sell", qty=10, price=101.0)

    expected = ((1010.0 - 1000.0) - (1000.0 + 1010.0) * 0.0012) / 1000.0 * 100.0
    assert book.compute_symbol_pnl_pct("AAPL") == pytest.approx(expected)


def test_round_trip_short_same_price_is_negative_round_trip_cost():
    book = ledger()
    book.record_fill("AAPL", "sell", qty=10, price=100.0)
    book.record_fill("AAPL", "buy", qty=10, price=100.0)

    assert book.compute_symbol_pnl_pct("AAPL") == pytest.approx(-ROUND_TRIP_COST_PCT)


def test_single_open_fill_returns_cost_component_only():
    book = ledger()
    book.record_fill("AAPL", "buy", qty=10, price=100.0)

    assert book.compute_symbol_pnl_pct("AAPL") == pytest.approx(-ONE_WAY_COST_PCT)


def test_clear_symbol_and_reset():
    book = ledger()
    book.record_fill("AAPL", "buy", qty=10, price=100.0)
    book.record_fill("MSFT", "buy", qty=5, price=200.0)

    book.clear_symbol("AAPL")
    assert book.compute_symbol_pnl_pct("AAPL") is None
    assert book.compute_symbol_pnl_pct("MSFT") is not None

    book.reset()
    assert book.compute_symbol_pnl_pct("MSFT") is None
