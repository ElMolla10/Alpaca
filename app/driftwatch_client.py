import os
import time
import math
import json
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
            self.batch_size = int(os.getenv("DRIFTWATCH_BATCH_SIZE", DEFAULT_BATCH_SIZE))
            self.flush_seconds = int(os.getenv("DRIFTWATCH_FLUSH_SECONDS", DEFAULT_FLUSH_SECONDS))
        except ValueError:
            self.batch_size = DEFAULT_BATCH_SIZE
            self.flush_seconds = DEFAULT_FLUSH_SECONDS

        # State
        self.buffer = deque(maxlen=MAX_BUFFER)
        self.last_flush_time = time.time()
        
        # Counters
        self.dropped_events = 0
        self.flush_failures = 0
        self.insert_success = 0

        if self.enabled and not self.dsn:
            logger.warning("DriftWatch enabled but DRIFTWATCH_DATABASE_URL missing. Disabling.")
            self.enabled = False

        if self.enabled:
            logger.info("DriftWatchClient initialized. Batch=%d, Flush=%ds", 
                        self.batch_size, self.flush_seconds)
    
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

        # Sanitize features and segment numbers
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
            Json(clean_segment)
        )
        
        # Add to buffer (auto-drops oldest if full per deque maxlen)
        if len(self.buffer) == MAX_BUFFER:
            self.dropped_events += 1
            
        self.buffer.append(event)
        
        # Check for flush
        if (len(self.buffer) >= self.batch_size) or \
           ((time.time() - self.last_flush_time) >= self.flush_seconds):
            self.flush()

    def flush(self):
        if not self.enabled or not self.buffer:
            return

        batch = []
        # Pop a batch
        while self.buffer and len(batch) < self.batch_size:
            batch.append(self.buffer.popleft())
            
        if not batch:
            return
            
        success = False
        wait = 0.5
        
        # Retry loop
        for attempt in range(MAX_RETRIES):
            try:
                with psycopg2.connect(self.dsn, connect_timeout=CONNECT_TIMEOUT) as conn:
                    with conn.cursor() as cur:
                        # Transaction-scoped timeout
                        cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                        
                        sql = """
                        INSERT INTO inference_events 
                        (ts, model_id, model_version, request_id, pred_type, latency_ms, features_json, y_pred_num, y_pred_text, segment_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cur.executemany(sql, batch)
                    conn.commit()
                
                success = True
                self.insert_success += len(batch)
                self.last_flush_time = time.time()
                break  # Success
                
            except (psycopg2.Error, OSError) as e:
                logger.warning(f"DriftWatch flush attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                    wait *= 2
        
        if not success:
            self.flush_failures += 1
            logger.error("DriftWatch flush failed after retries. Dropping batch.")
            # Note: Batch is already popped from buffer, so it is dropped effectively.

    def close(self):
        """Force flush remaining events on exit."""
        if self.enabled:
            logger.info("DriftWatch closing. Flushing remaining %d events...", len(self.buffer))
            # Flush everything in batches
            while self.buffer:
                self.flush()
