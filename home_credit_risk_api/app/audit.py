import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DB = PROJECT_ROOT / "underwriting_audit.sqlite"

def get_audit_connection() -> sqlite3.Connection:
    """
    Opens the SQLite database used to persist underwriting decision history.
    """

    connection = sqlite3.connect(AUDIT_DB)
    connection.row_factory = sqlite3.Row

    return connection

def initialize_audit_db() -> None:
    """
    Creates the underwriting audit table used to persist review-level decision history.
    """

    with get_audit_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS underwriting_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                review_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                recommendation TEXT,
                triggered_rules TEXT,
                human_decision TEXT,
                human_comment TEXT,
                decision_status TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def log_audit_event(
    *,
    case_id: str,
    review_id: str,
    event_type: str,
    recommendation: str | None = None,
    triggered_rules: list[str] | None = None,
    human_decision: str | None = None,
    human_comment: str | None = None,
    decision_status: str | None = None,
) -> None:
    """
    Stores one underwriting event together with its recommendation, status and supporting decision data.
    """

    initialize_audit_db()

    with get_audit_connection() as connection:
        connection.execute(
            """
            INSERT INTO underwriting_audit (
                case_id,
                review_id,
                event_type,
                recommendation,
                triggered_rules,
                human_decision,
                human_comment,
                decision_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                review_id,
                event_type,
                recommendation,
                json.dumps(triggered_rules or []),
                human_decision,
                human_comment,
                decision_status,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        
def get_case_audit(case_id: str,) -> list[dict[str, Any]]:
    """
    Returns all recorded underwriting events for a case in chronological order.
    """

    initialize_audit_db()

    with get_audit_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM underwriting_audit
            WHERE case_id = ?
            ORDER BY audit_id
            """,
            (case_id,)
        ).fetchall()

    events = []

    for row in rows:
        event = dict(row)

        event["triggered_rules"] = json.loads(
            event["triggered_rules"] or "[]"
        )

        events.append(event)

    return events

def get_review_audit(
    case_id: str,
    review_id: str,
) -> list[dict[str, Any]]:
    """
    Returns the chronological audit history for one
    specific underwriting review.
    """

    initialize_audit_db()

    with get_audit_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM underwriting_audit
            WHERE case_id = ?
              AND review_id = ?
            ORDER BY audit_id
            """,
            (
                case_id,
                review_id,
            ),
        ).fetchall()

    events = []

    for row in rows:
        event = dict(row)

        event["triggered_rules"] = json.loads(
            event["triggered_rules"] or "[]"
        )

        events.append(event)

    return events