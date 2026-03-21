-- ============================================================
-- Market Data Persistence Schema
-- Run once against your Supabase / PostgreSQL database.
-- All statements are idempotent (IF NOT EXISTS / DO NOTHING).
-- ============================================================

-- ------------------------------------------------------------
-- Table: market_data
-- Stores raw OHLCV bars fetched from Alpaca.
-- Primary key prevents duplicate ingestion for the same bar.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_data (
    symbol    TEXT             NOT NULL,
    ts        TIMESTAMPTZ      NOT NULL,
    timeframe TEXT             NOT NULL DEFAULT '1h',
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    DOUBLE PRECISION,
    vwap      DOUBLE PRECISION,

    PRIMARY KEY (symbol, ts, timeframe)
);

-- Fast time-series lookups per symbol
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_ts
    ON market_data (symbol, ts DESC);

-- ------------------------------------------------------------
-- Table: feature_snapshots
-- Stores the computed feature vector for each bar.
-- Used for offline model retraining without re-fetching Alpaca.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_snapshots (
    symbol     TEXT        NOT NULL,
    ts         TIMESTAMPTZ NOT NULL,
    features   JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_ts
    ON feature_snapshots (symbol, ts DESC);
