"""SQLite storage: every measurement, with the context needed to reproduce it."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    ts REAL, session TEXT, chip TEXT, os TEXT, engine TEXT, engine_version TEXT,
    model TEXT, quant TEXT, bits REAL, bpw REAL, variant TEXT,
    prompt_id TEXT, trial INTEGER,
    prompt_tokens INTEGER, gen_tokens INTEGER,
    ttft_ms REAL, prefill_tps REAL, decode_tps REAL, wall_s REAL, timing_source TEXT,
    energy_j REAL, energy_dyn_j REAL, avg_power_w REAL, idle_power_w REAL,
    cpu_j REAL, gpu_j REAL, ane_j REAL, dram_j REAL, power_backend TEXT,
    j_per_token REAL, j_per_token_dyn REAL,
    peak_mem_mb REAL, thermal_state INTEGER
);
CREATE TABLE IF NOT EXISTS quality (
    id INTEGER PRIMARY KEY,
    ts REAL, session TEXT, engine TEXT, model TEXT, quant TEXT, bits REAL, variant TEXT,
    task TEXT, n INTEGER, score REAL, ci_low REAL, ci_high REAL,
    energy_j REAL, gen_tokens INTEGER, details TEXT
);
CREATE TABLE IF NOT EXISTS thermal (
    id INTEGER PRIMARY KEY,
    ts REAL, session TEXT, variant TEXT, t_s REAL,
    decode_tps REAL, avg_power_w REAL, thermal_state INTEGER
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _insert(self, table: str, row: dict) -> None:
        row = {"ts": time.time(), **row}
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        self.db.commit()

    def add_run(self, **row) -> None:
        self._insert("runs", row)

    def add_quality(self, details: dict | list | None = None, **row) -> None:
        self._insert("quality", {**row, "details": json.dumps(details) if details is not None else None})

    def add_thermal(self, **row) -> None:
        self._insert("thermal", row)

    def has_runs(self, session: str, variant: str) -> bool:
        return self.db.execute("SELECT 1 FROM runs WHERE session=? AND variant=? LIMIT 1", (session, variant)).fetchone() is not None

    def has_quality(self, session: str, variant: str, task: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM quality WHERE session=? AND variant=? AND task=? LIMIT 1", (session, variant, task)
        ).fetchone() is not None

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, params).fetchall()

    def close(self) -> None:
        self.db.close()
