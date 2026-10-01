"""Minimal SQLite persistence and append-only dispatcher audit logging."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .schemas import Incident

PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
VALID_STATUSES = {"Reported", "Dispatched", "En route", "Resolved"}


def mask_phone_numbers(text: str) -> str:
    """Mask likely phone numbers before persisting report text."""
    return PHONE_PATTERN.sub("[PHONE REDACTED]", text)


def _redact_strings(value):
    if isinstance(value, str):
        return mask_phone_numbers(value)
    if isinstance(value, list):
        return [_redact_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_strings(item) for key, item in value.items()}
    return value


class IncidentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    report_text TEXT NOT NULL,
                    incident_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    incident_id TEXT,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);
                """
            )

    def save_incident(self, incident: Incident, report_text: str) -> None:
        redacted = mask_phone_numbers(report_text)
        incident_json = json.dumps(_redact_strings(incident.model_dump(mode="json")))
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO incidents VALUES (?, ?, ?, ?)",
                (incident.incident_id, incident.created_at.isoformat(), redacted, incident_json),
            )
            connection.execute(
                "INSERT INTO audit_log(created_at, actor, action, incident_id, details_json) VALUES (?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), "system", "report_created", incident.incident_id, "{}"),
            )

    def list_incidents(self, limit: int = 300) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT incident_id, created_at, report_text, incident_json FROM incidents ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "incident_id": row["incident_id"],
                "created_at": row["created_at"],
                "report_text": row["report_text"],
                "incident": Incident.model_validate_json(row["incident_json"]),
            }
            for row in rows
        ]

    def update_status(self, incident_id: str, status: str, actor: str = "dispatcher") -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unsupported incident status: {status}")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT incident_json FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if row is None:
                raise KeyError(incident_id)
            payload = json.loads(row["incident_json"])
            payload["status"] = status
            connection.execute(
                "UPDATE incidents SET incident_json = ? WHERE incident_id = ?",
                (json.dumps(payload), incident_id),
            )
            self._audit(connection, actor, "status_updated", incident_id, {"status": status})

    def record_override(self, incident_id: str, actor: str, action: str, details: dict[str, Any]) -> None:
        with self._connection() as connection:
            self._audit(connection, actor, action, incident_id, details)

    def merge_duplicate(self, incident_id: str, independent_reports: int, priority: str) -> None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT incident_json FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if row is None:
                return
            payload = json.loads(row["incident_json"])
            payload["independent_reports"] = independent_reports
            payload["priority"] = priority
            connection.execute(
                "UPDATE incidents SET incident_json = ? WHERE incident_id = ?",
                (json.dumps(payload), incident_id),
            )
            self._audit(connection, "system", "duplicate_merged", incident_id, {"reports": independent_reports})

    def set_priority(self, incident_id: str, priority: str, actor: str = "dispatcher") -> None:
        if priority not in {"Immediate", "High", "Elevated", "Routine"}:
            raise ValueError(f"Unsupported priority: {priority}")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT incident_json FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if row is None:
                raise KeyError(incident_id)
            payload = json.loads(row["incident_json"])
            previous = payload["priority"]
            payload["priority"] = priority
            connection.execute(
                "UPDATE incidents SET incident_json = ? WHERE incident_id = ?",
                (json.dumps(payload), incident_id),
            )
            self._audit(connection, actor, "priority_override", incident_id, {"from": previous, "to": priority})

    def cleanup_expired(self, retention_days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._connection() as connection:
            result = connection.execute("DELETE FROM incidents WHERE created_at < ?", (cutoff,))
            connection.execute("DELETE FROM audit_log WHERE created_at < ?", (cutoff,))
            return result.rowcount

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT created_at, actor, action, incident_id, details_json FROM audit_log ORDER BY audit_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {**dict(row), "details": json.loads(row["details_json"])}
            for row in rows
        ]

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        actor: str,
        action: str,
        incident_id: str,
        details: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO audit_log(created_at, actor, action, incident_id, details_json) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), actor, action, incident_id, json.dumps(details)),
        )