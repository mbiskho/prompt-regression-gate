"""SQLite results store.

Runs are kept so that "did this get worse" can be answered against history
rather than against whatever happened to be in the working directory. Two
queries carry the tool:

* the latest run for a prompt id - the baseline a pull request is compared to
* runs over time for one case - which is how you spot a case that flaps rather
  than one that broke

SQLite because a regression gate that needs a database server will not get
installed.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .models import RunReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    prompt_id         TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    prompt_hash       TEXT NOT NULL,
    goldenset_version TEXT NOT NULL,
    started_at        REAL NOT NULL,
    duration_s        REAL NOT NULL,
    cost_usd          REAL NOT NULL,
    cases             INTEGER NOT NULL,
    passed            INTEGER NOT NULL,
    failed            INTEGER NOT NULL,
    stopped_early     TEXT NOT NULL DEFAULT '',
    label             TEXT NOT NULL DEFAULT '',
    payload           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_prompt ON runs (prompt_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_label ON runs (prompt_id, label);
"""


class ResultStore:
    def __init__(self, path: Path | str = ".promptgate/results.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, report: RunReport, *, label: str = "") -> str:
        """Persist a run. `label` marks a baseline, e.g. 'main'."""
        with closing(self._connect()) as conn:
            # Columns are named rather than positional: the schema has grown
            # once already, and a positional INSERT fails loudly only if you
            # are lucky about the column count.
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, prompt_id, prompt_version, prompt_hash, "
                "goldenset_version, started_at, duration_s, cost_usd, cases, passed, failed, "
                "stopped_early, label, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    report.run_id,
                    report.prompt_id,
                    report.prompt_version,
                    report.prompt_hash,
                    report.goldenset_version,
                    report.started_at,
                    report.duration_s,
                    report.cost_usd,
                    len(report.results),
                    report.passed,
                    len(report.failed),
                    report.stopped_early,
                    label,
                    json.dumps(report.to_dict()),
                ),
            )
            conn.commit()
        return report.run_id

    def get(self, run_id: str) -> RunReport | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunReport.from_dict(json.loads(row["payload"])) if row else None

    def latest(self, prompt_id: str, *, label: str | None = None) -> RunReport | None:
        """Most recent run, optionally restricted to a label.

        A labelled baseline beats "most recent" for CI: the last run might be
        a colleague's experiment on a branch.
        """
        query = "SELECT payload FROM runs WHERE prompt_id = ?"
        params: list = [prompt_id]
        if label:
            query += " AND label = ?"
            params.append(label)
        query += " ORDER BY started_at DESC LIMIT 1"
        with closing(self._connect()) as conn:
            row = conn.execute(query, params).fetchone()
        return RunReport.from_dict(json.loads(row["payload"])) if row else None

    def history(self, prompt_id: str, *, limit: int = 20) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT run_id, prompt_version, started_at, cases, passed, failed, cost_usd, label "
                "FROM runs WHERE prompt_id = ? ORDER BY started_at DESC LIMIT ?",
                (prompt_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def case_history(self, prompt_id: str, case_id: str, *, limit: int = 20) -> list[dict]:
        """One case across recent runs. Distinguishes 'broke' from 'flaps'."""
        out = []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT payload, started_at, prompt_version FROM runs "
                "WHERE prompt_id = ? ORDER BY started_at DESC LIMIT ?",
                (prompt_id, limit),
            ).fetchall()
        for row in rows:
            report = RunReport.from_dict(json.loads(row["payload"]))
            match = next((r for r in report.results if r.case_id == case_id), None)
            if match:
                out.append(
                    {
                        "started_at": row["started_at"],
                        "prompt_version": row["prompt_version"],
                        "status": match.status.value,
                        "judge_score": match.judge_score,
                    }
                )
        return out
