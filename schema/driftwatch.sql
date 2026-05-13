CREATE TABLE IF NOT EXISTS inference_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    request_id TEXT UNIQUE,
    pred_type TEXT NOT NULL,
    latency_ms INTEGER,
    features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    y_pred_num DOUBLE PRECISION,
    y_pred_text TEXT,
    segment_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_events_ts
    ON inference_events (ts DESC);

CREATE INDEX IF NOT EXISTS idx_inference_events_model_ts
    ON inference_events (model_id, model_version, ts DESC);

CREATE TABLE IF NOT EXISTS label_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    y_true_num DOUBLE PRECISION,
    y_true_text TEXT,
    label_type TEXT NOT NULL DEFAULT 'regression',
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_label_events_request_id
        FOREIGN KEY (request_id)
        REFERENCES inference_events (request_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_label_events_ts
    ON label_events (ts DESC);
