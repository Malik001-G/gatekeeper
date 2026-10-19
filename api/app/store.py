"""Decision log in Postgres.

Every decision is queued in memory and written by a background thread in small batches,
so saving it never slows down the answer to the payment.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

log = logging.getLogger("gatekeeper.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_time          TIMESTAMPTZ,
    transaction_id      TEXT,
    source              TEXT,
    amount              DOUBLE PRECISION,
    product             TEXT,
    card_key            TEXT,
    fraud_score         DOUBLE PRECISION NOT NULL,
    decision            TEXT NOT NULL,
    threshold           DOUBLE PRECISION NOT NULL,
    latency_ms          DOUBLE PRECISION,
    card_tx_last_hour   INTEGER,
    card_avg_amount     DOUBLE PRECISION,
    amount_vs_card_avg  DOUBLE PRECISION,
    new_device          BOOLEAN,
    label               SMALLINT
);
CREATE INDEX IF NOT EXISTS decisions_created_at ON decisions (created_at);
"""

COLUMNS = ["event_time", "transaction_id", "source", "amount", "product", "card_key", "fraud_score",
           "decision", "threshold", "latency_ms", "card_tx_last_hour", "card_avg_amount",
           "amount_vs_card_avg", "new_device", "label"]
INSERT = f"INSERT INTO decisions ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))})"


class DecisionStore:
    def __init__(self, url: str, max_queue: int = 50_000, batch: int = 200):
        self.pool = ConnectionPool(url, min_size=1, max_size=4, open=False, kwargs={"autocommit": True})
        self.pool.open(wait=False)
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.batch = batch
        self.ready = False
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="decision-writer", daemon=True)
        self._thread.start()

    # ---- writing -------------------------------------------------------------------------
    def add(self, record: dict) -> None:
        try:
            self.q.put_nowait(tuple(record.get(c) for c in COLUMNS))
        except queue.Full:
            self.dropped += 1

    def _ensure_schema(self) -> None:
        while not self._stop.is_set() and not self.ready:
            try:
                with self.pool.connection(timeout=5) as conn:
                    conn.execute(SCHEMA)
                self.ready = True
            except Exception as e:  # database still starting up
                log.warning("decision store not ready yet: %s", e)
                time.sleep(2)

    def _run(self) -> None:
        self._ensure_schema()
        while not self._stop.is_set() or not self.q.empty():
            try:
                rows = [self.q.get(timeout=0.5)]
            except queue.Empty:
                continue
            while len(rows) < self.batch:
                try:
                    rows.append(self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                with self.pool.connection(timeout=5) as conn, conn.cursor() as cur:
                    cur.executemany(INSERT, rows)
                self.written += len(rows)
            except Exception as e:
                self.errors += 1
                log.error("could not save %d decisions: %s", len(rows), e)

    def flush(self, timeout: float = 10) -> None:
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.2)

    def close(self) -> None:
        self.flush()
        self._stop.set()
        self._thread.join(timeout=5)
        self.pool.close()

    # ---- reading -------------------------------------------------------------------------
    def status(self) -> dict:
        return {"ready": self.ready, "queued": self.q.qsize(), "written": self.written,
                "dropped": self.dropped, "write_errors": self.errors}

    def recent(self, limit: int = 50, decision: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM decisions"
        args: list = []
        if decision:
            sql += " WHERE decision = %s"
            args.append(decision)
        sql += " ORDER BY id DESC LIMIT %s"
        args.append(limit)
        with self.pool.connection(timeout=2) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def stats(self, minutes: Optional[int] = None) -> dict:
        where = "WHERE created_at > now() - make_interval(mins => %s)" if minutes else ""
        args = [minutes] if minutes else []
        sql = f"""
        SELECT count(*)                                                    AS decisions,
               count(*) FILTER (WHERE decision = 'suspicious')             AS flagged,
               avg(latency_ms)                                             AS avg_latency_ms,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)    AS p99_latency_ms,
               count(*) FILTER (WHERE label IS NOT NULL)                   AS labelled,
               count(*) FILTER (WHERE label = 1)                           AS frauds,
               count(*) FILTER (WHERE label = 1 AND decision = 'suspicious') AS frauds_caught,
               count(*) FILTER (WHERE label = 0 AND decision = 'suspicious') AS false_alarms,
               count(*) FILTER (WHERE label = 0)                           AS genuine,
               coalesce(sum(amount) FILTER (WHERE label = 1 AND decision = 'allow'), 0)      AS fraud_missed_usd,
               coalesce(sum(amount) FILTER (WHERE label = 1 AND decision = 'suspicious'), 0) AS fraud_stopped_usd
        FROM decisions {where}
        """
        with self.pool.connection(timeout=2) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, args)
            s = cur.fetchone()
        s["flag_rate"] = s["flagged"] / s["decisions"] if s["decisions"] else None
        s["catch_rate"] = s["frauds_caught"] / s["frauds"] if s["frauds"] else None
        s["false_alarm_rate"] = s["false_alarms"] / s["genuine"] if s["genuine"] else None
        return s
