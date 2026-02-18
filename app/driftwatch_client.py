import os
import time
import math
import logging
import psycopg2
from psycopg2.extras import Json
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constants
MAX_BUFFER = 5000
DEFAULT_BATCH_SIZE = 50
DEFAULT_FLUSH_SECONDS = 5
MAX_RETRIES = 5
CONNECT_TIMEOUT = 3
STATEMENT_TIMEOUT = "3000ms"


class DriftWatchClient:
    def __init__(self):
        self.enabled = os.getenv("DRIFTWATCH_ENABLED", "true").lower() == "true"
        self.dsn = os.getenv("DRIFTWATCH_DATABASE_URL")

        # Buffer configs
        try:
            self.batch_size = int(os.getenv("DRIFTWATCH_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
            self.flush_seconds = int(os.getenv("DRIFTWATCH_FLUSH_SECONDS", str(DEFAULT_FLUSH_SECONDS)))
        except ValueError:
            self.batch_size = DEFAULT_BATCH_SIZE
            self.flush_seconds = DEFAULT_FLUSH_SECONDS

        # Buffers
        self.infer_buffer = deque(maxlen=MAX_BUFFER)
        self.label_buffer = deque(maxlen=MAX_BUFFER)

        # State
        self.last_flush_time = time.time()

        # Counters
        self.dropped_events = 0
        self.flush_failures = 0
        self.insert_success = 0
        self.label_insert_success = 0

        if self.enabled and not self.dsn:
            logger.warning("DriftWatch enabled but DRIFTWATCH_DATABASE_URL missing. Disabling.")
            self.enabled = False

        if self.enabled:
            logger.info(
                "DriftWatchClient initialized. Batch=%d, Flush=%ds",
                self.batch_size,
                self.flush_seconds
            )

    def _sanitize(self, val: Any) -> Any:
        """Sanitize numeric values: NaN/Inf -> None."""
        if val is None:
            return None
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return None
        return val

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
        request_id: Optional[str] = None
    ):
        if not self.enabled:
            return

        clean_features = {k: self._sanitize(v) for k, v in features_json.items()}
        clean_segment = {k: self._sanitize(v) for k, v in segment_json.items()}

        event = (
            ts,
            model_id,
            model_version,
            request_id,
            pred_type,
            latency_ms,
            Json(clean_features),
            self._sanitize(y_pred_num),
            y_pred_text,
            Json(clean_segment),
        )

        if len(self.infer_buffer) == MAX_BUFFER:
            self.dropped_events += 1

        self.infer_buffer.append(event)

        if (len(self.infer_buffer) >= self.batch_size) or \
           ((time.time() - self.last_flush_time) >= self.flush_seconds):
            self.flush()

    def log_label(
        self,
        ts: datetime,
        request_id: str,
        y_true_num: Optional[float],
        y_true_text: Optional[str] = None,
        label_type: str = "regression",
        extra_json: Optional[Dict[str, Any]] = None
    ):
        if not self.enabled:
            return

        clean_extra = {k: self._sanitize(v) for k, v in (extra_json or {}).items()}

        event = (
            ts,
            request_id,
            self._sanitize(y_true_num),
            y_true_text,
            label_type,
            Json(clean_extra),
        )

        if len(self.label_buffer) == MAX_BUFFER:
            self.dropped_events += 1

        self.label_buffer.append(event)

        if (len(self.label_buffer) >= self.batch_size) or \
           ((time.time() - self.last_flush_time) >= self.flush_seconds):
            self.flush()

    def flush(self):
        if not self.enabled:
            return

        # Build batches
        infer_batch = []
        while self.infer_buffer and len(infer_batch) < self.batch_size:
            infer_batch.append(self.infer_buffer.popleft())

        label_batch = []
        while self.label_buffer and len(label_batch) < self.batch_size:
            label_batch.append(self.label_buffer.popleft())

        if not infer_batch and not label_batch:
            return

        success = False
        wait = 0.5

        for attempt in range(MAX_RETRIES):
            try:
                with psycopg2.connect(self.dsn, connect_timeout=CONNECT_TIMEOUT) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")

                        if infer_batch:
                            sql_i = """
                                INSERT INTO inference_events
                                (ts, model_id, model_version, request_id, pred_type, latency_ms,
                                 features_json, y_pred_num, y_pred_text, segment_json)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            cur.executemany(sql_i, infer_batch)
                            self.insert_success += len(infer_batch)

                        if label_batch:
                            sql_l = """
                                INSERT INTO label_events
                                (ts, request_id, y_true_num, y_true_text, label_type, extra_json)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """
                            cur.executemany(sql_l, label_batch)
                            self.label_insert_success += len(label_batch)

                    conn.commit()

                success = True
                self.last_flush_time = time.time()
                break

            except (psycopg2.Error, OSError) as e:
                logger.warning("DriftWatch flush attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                    wait *= 2

        if not success:
            self.flush_failures += 1
            logger.error(
                "DriftWatch flush failed after retries. Dropping batches (infer=%d, label=%d).",
                len(infer_batch),
                len(label_batch),
            )
            # batches already popped -> dropped

    def close(self):
        """Force flush remaining events on exit."""
        if not self.enabled:
            return

        logger.info(
            "DriftWatch closing. Flushing remaining infer=%d label=%d ...",
            len(self.infer_buffer),
            len(self.label_buffer),
        )
        while self.infer_buffer or self.label_buffer:
            self.flush()
