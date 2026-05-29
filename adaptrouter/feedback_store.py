import sqlite3
import json
import os
from datetime import datetime
from adaptrouter.config import DB_PATH


class FeedbackStore:

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    query_id        TEXT PRIMARY KEY,
                    query_text      TEXT NOT NULL,
                    embedding_json  TEXT,
                    routed_to       TEXT NOT NULL,
                    label           TEXT NOT NULL,
                    confidence      REAL NOT NULL,
                    latency_s       REAL,
                    timestamp       TEXT NOT NULL,
                    domain_tag      TEXT DEFAULT 'general',
                    session_id      TEXT
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id        TEXT NOT NULL,
                    was_helpful     INTEGER,
                    user_rating     INTEGER,
                    implicit_type   TEXT,
                    implicit_conf   REAL,
                    feedback_time   TEXT NOT NULL,
                    used_for_train  INTEGER DEFAULT 0,
                    FOREIGN KEY (query_id)
                        REFERENCES routing_decisions(query_id)
                );

                CREATE TABLE IF NOT EXISTS retraining_history (
                    retrain_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    retrain_time    TEXT NOT NULL,
                    old_accuracy    REAL,
                    new_accuracy    REAL,
                    n_new_examples  INTEGER,
                    status          TEXT,
                    notes           TEXT
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def store_decision(self, result: dict):
        embedding_json = None

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO routing_decisions
                (query_id, query_text, embedding_json, routed_to,
                 label, confidence, latency_s, timestamp, domain_tag, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                result["query_id"],
                result["query"],
                embedding_json,
                result["model_used"],
                result["label"],
                result["confidence"],
                result.get("latency_s", 0),
                datetime.now().isoformat(),
                result.get("domain", "general"),
                result.get("session_id"),
            ))
            conn.commit()
        finally:
            conn.close()

    def store_feedback(self, query_id: str, was_helpful: bool = None,
                       user_rating: int = None, implicit_type: str = None,
                       implicit_conf: float = None):

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO feedback
                (query_id, was_helpful, user_rating, implicit_type,
                 implicit_conf, feedback_time)
                VALUES (?,?,?,?,?,?)
            """, (
                query_id,
                1 if was_helpful is True else (0 if was_helpful is False else None),
                user_rating,
                implicit_type,
                implicit_conf,
                datetime.now().isoformat(),
            ))
            conn.commit()
        finally:
            conn.close()

    def get_labelled_feedback(self, unused_only: bool = True) -> list:
        used_filter = "AND f.used_for_train = 0" if unused_only else ""

        conn = self._get_conn()
        try:
            rows = conn.execute(f"""
                SELECT
                    d.query_id,
                    d.query_text,
                    d.embedding_json,
                    d.routed_to,
                    d.label AS predicted_label,
                    d.confidence,
                    f.was_helpful,
                    f.implicit_type,
                    f.implicit_conf,
                    f.feedback_id
                FROM routing_decisions d
                JOIN feedback f ON d.query_id = f.query_id
                WHERE f.was_helpful IS NOT NULL
                {used_filter}
                ORDER BY f.feedback_time ASC
            """).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            row_dict = dict(row)

            if row_dict["was_helpful"] == 1:
                true_label = row_dict["predicted_label"]
            else:
                true_label = "complex" if row_dict["routed_to"] == \
                             "llama-3.1-8b-instant" else "simple"

            row_dict["true_label"] = true_label
            row_dict["true_label_int"] = 0 if true_label == "simple" else 1

            results.append(row_dict)

        return results

    def count_new_labelled(self) -> int:
        conn = self._get_conn()
        try:
            result = conn.execute("""
                SELECT COUNT(DISTINCT d.query_id)
                FROM routing_decisions d
                JOIN feedback f ON d.query_id = f.query_id
                WHERE f.was_helpful IS NOT NULL
                AND f.used_for_train = 0
            """).fetchone()
            return result[0]
        finally:
            conn.close()

    def mark_feedback_as_used(self, feedback_ids: list):
        if not feedback_ids:
            return

        placeholders = ",".join("?" * len(feedback_ids))

        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE feedback SET used_for_train=1 WHERE feedback_id IN ({placeholders})",
                feedback_ids
            )
            conn.commit()
        finally:
            conn.close()

    def log_retrain_event(self, old_acc: float, new_acc: float,
                          n_examples: int, status: str, notes: str = ""):

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO retraining_history
                (retrain_time, old_accuracy, new_accuracy, n_new_examples, status, notes)
                VALUES (?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(),
                old_acc, new_acc, n_examples, status, notes
            ))
            conn.commit()
        finally:
            conn.close()

    def hours_since_last_retrain(self) -> float:
        conn = self._get_conn()
        try:
            result = conn.execute("""
                SELECT retrain_time FROM retraining_history
                ORDER BY retrain_time DESC LIMIT 1
            """).fetchone()
        finally:
            conn.close()

        if not result:
            return 999.0

        last_time = datetime.fromisoformat(result[0])
        delta = datetime.now() - last_time
        return delta.total_seconds() / 3600

    # ✅ FIXED METHOD
    def get_stats(self) -> dict:
        conn = self._get_conn()
        try:
            total_decisions = conn.execute(
                "SELECT COUNT(*) FROM routing_decisions"
            ).fetchone()[0]

            total_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback"
            ).fetchone()[0]

            implicit_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE implicit_type IS NOT NULL"
            ).fetchone()[0]

            positive_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE was_helpful = 1"
            ).fetchone()[0]

            retraining_events = conn.execute(
                "SELECT COUNT(*) FROM retraining_history"
            ).fetchone()[0]

        finally:
            conn.close()

        # ✅ Safe calculations
        positive_rate = (
            round(positive_feedback / total_feedback, 3)
            if total_feedback > 0 else 0
        )

        feedback_rate_pct = (
            round(total_feedback / total_decisions * 100, 1)
            if total_decisions > 0 else 0.0
        )

        feedback_ratio = total_feedback / max(total_decisions, 1) * 100

        feedback_health = (
            "healthy" if feedback_ratio > 5
            else "low" if feedback_ratio > 2
            else "critical"
        )

        return {
            "total_decisions": total_decisions,
            "total_feedback": total_feedback,
            "implicit_feedback": implicit_feedback,
            "positive_rate": positive_rate,
            "feedback_rate_pct": feedback_rate_pct,
            "feedback_health": feedback_health,
            "retraining_events": retraining_events,
        }

    def close(self):
        pass