import os
import time
import math
import logging
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional, Deque, List

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

MAX_BUFFER = 5000
DEFAULT_BATCH_SIZE = 50
DEFAULT_FLUSH_SECONDS = 5
MAX_RETRIES = 5
CONNECT_TIMEOUT = 3
STATEMENT_TIMEOUT = "3000ms"

DRIFTWATCH_SCHEMA_SQL = """
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
"""

INFERENCE_UPSERT_SQL = """
INSERT INTO inference_events
(ts, model_id, model_version, request_id, pred_type, latency_ms,
 features_json, y_pred_num, y_pred_text, segment_json)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (request_id) DO UPDATE SET
    ts = EXCLUDED.ts,
    model_id = EXCLUDED.model_id,
    model_version = EXCLUDED.model_version,
    pred_type = EXCLUDED.pred_type,
    latency_ms = EXCLUDED.latency_ms,
    features_json = EXCLUDED.features_json,
    y_pred_num = EXCLUDED.y_pred_num,
    y_pred_text = EXCLUDED.y_pred_text,
    segment_json = EXCLUDED.segment_json
"""

LABEL_UPSERT_SQL = """
INSERT INTO label_events
(ts, request_id, y_true_num, y_true_text, label_type, extra_json)
VALUES (%s,%s,%s,%s,%s,%s)
ON CONFLICT (request_id) DO UPDATE SET
    ts = EXCLUDED.ts,
    y_true_num = EXCLUDED.y_true_num,
    y_true_text = EXCLUDED.y_true_text,
    label_type = EXCLUDED.label_type,
    extra_json = EXCLUDED.extra_json
"""


def _sanitize(val: Any) -> Any:
    """Sanitize numeric values: NaN/Inf -> None."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


class DriftWatchClient:
    def __init__(self):
        self.enabled = os.getenv("DRIFTWATCH_ENABLED", "true").lower() == "true"
        self.dsn = os.getenv("DRIFTWATCH_DATABASE_URL")

        try:
            self.batch_size = int(os.getenv("DRIFTWATCH_BATCH_SIZE", DEFAULT_BATCH_SIZE))
            self.flush_seconds = int(os.getenv("DRIFTWATCH_FLUSH_SECONDS", DEFAULT_FLUSH_SECONDS))
        except ValueError:
            self.batch_size = DEFAULT_BATCH_SIZE
            self.flush_seconds = DEFAULT_FLUSH_SECONDS

        self.buffer: Deque[tuple] = deque(maxlen=MAX_BUFFER)        # inference_events
        self.label_buffer: Deque[tuple] = deque(maxlen=MAX_BUFFER)  # label_events
        self.last_flush_time = time.time()
        self.auto_create = os.getenv("DRIFTWATCH_AUTO_CREATE", "true").lower() == "true"

        self.dropped_events = 0
        self.flush_failures = 0
        self.insert_success = 0
        self.label_insert_success = 0

        if self.enabled and not self.dsn:
            logger.warning("DriftWatch enabled but DRIFTWATCH_DATABASE_URL missing. Disabling.")
            self.enabled = False

        if self.enabled:
            logger.info("DriftWatchClient initialized. Batch=%d, Flush=%ds",
                        self.batch_size, self.flush_seconds)

    def log_inference(
        self,
        model_id: str,
        model_version: str,
        ts: datetime,
        pred_type: str,
        y_pred_num: Optional[float],
        y_pred_text: Optional[str],
        latency_ms: Optional[int],
        features_json: Dict[str, Any],
        segment_json: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        clean_features = {k: _sanitize(v) for k, v in (features_json or {}).items()}
        clean_segment = {k: _sanitize(v) for k, v in (segment_json or {}).items()}

        event = (
            ts,
            model_id,
            model_version,
            request_id,
            pred_type,
            latency_ms,
            Json(clean_features),
            _sanitize(y_pred_num),
            y_pred_text,
            Json(clean_segment),
        )

        if len(self.buffer) == MAX_BUFFER:
            self.dropped_events += 1
        self.buffer.append(event)

        if (len(self.buffer) >= self.batch_size) or ((time.time() - self.last_flush_time) >= self.flush_seconds):
            self.flush()

    def log_label(
        self,
        ts: datetime,
        request_id: str,
        y_true_num: Optional[float],
        y_true_text: Optional[str] = None,
        label_type: str = "regression",
        extra_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        event = (
            ts,
            request_id,
            _sanitize(y_true_num),
            y_true_text,
            label_type,
            Json({k: _sanitize(v) for k, v in (extra_json or {}).items()}),
        )

        if len(self.label_buffer) == MAX_BUFFER:
            self.dropped_events += 1
        self.label_buffer.append(event)

        if (len(self.label_buffer) >= self.batch_size) or ((time.time() - self.last_flush_time) >= self.flush_seconds):
            self.flush()

    def _pop_batch(self, q: Deque[tuple]) -> List[tuple]:
        batch: List[tuple] = []
        while q and len(batch) < self.batch_size:
            batch.append(q.popleft())
        return batch

    def _ensure_schema(self, cur) -> None:
        if self.auto_create:
            cur.execute(DRIFTWATCH_SCHEMA_SQL)

    def flush(self) -> None:
        if not self.enabled:
            return
        if not self.buffer and not self.label_buffer:
            return

        inf_batch = self._pop_batch(self.buffer)
        lab_batch = self._pop_batch(self.label_buffer)
        if not inf_batch and not lab_batch:
            return

        success = False
        wait = 0.5

        for attempt in range(MAX_RETRIES):
            try:
                with psycopg2.connect(self.dsn, connect_timeout=CONNECT_TIMEOUT) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                        self._ensure_schema(cur)

                        if inf_batch:
                            cur.executemany(INFERENCE_UPSERT_SQL, inf_batch)

                        if lab_batch:
                            cur.executemany(LABEL_UPSERT_SQL, lab_batch)
                    conn.commit()

                if inf_batch:
                    self.insert_success += len(inf_batch)
                if lab_batch:
                    self.label_insert_success += len(lab_batch)

                self.last_flush_time = time.time()
                success = True
                break

            except (psycopg2.Error, OSError) as e:
                logger.warning("DriftWatch flush attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                    wait *= 2

        if not success:
            self.flush_failures += 1
            logger.error("DriftWatch flush failed after retries. Dropping batches.")

    def close(self) -> None:
        if not self.enabled:
            return
        while self.buffer or self.label_buffer:
            self.flush()
