"""
History Service — SQLite persistence for multimodal observations.

Stores full structured observation history:
  - id, timestamp, source
  - item_name, detection_confidence
  - temperature_c, humidity_percent, gas_value, sensor_status
  - freshness_status, estimated_remaining_freshness
  - llm_analysis (full JSON)
  - raw_evidence_json (full JSON)
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class HistoryService:
    """
    SQLite persistence layer for fruit/food freshness observations.
    """

    def __init__(self, db_path: str = "database/freshness.db", export_path: str = "exports/analysis.csv"):
        self.db_path = db_path
        self.export_path = export_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        export_dir = os.path.dirname(self.export_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

    def _init_db(self):
        """Create the observations table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT UNIQUE,
                timestamp TEXT,
                source TEXT,
                item_name TEXT,
                detection_confidence REAL,
                temperature_c REAL,
                humidity_percent REAL,
                gas_value REAL,
                sensor_status TEXT,
                freshness_status TEXT,
                estimated_remaining_freshness TEXT,
                llm_analysis TEXT,
                raw_evidence_json TEXT
            )
            """)
            conn.commit()
            logger.info("Database initialized at %s", self.db_path)
        finally:
            conn.close()

    def save_observation(self, analysis_result, evidence_packet: Dict[str, Any]) -> int:
        """
        Save a complete analysis and evidence packet to SQLite.
        Returns the inserted row ID.
        """
        obs_id = evidence_packet.get("observation_id", "")
        now = evidence_packet.get("timestamp", datetime.now().isoformat())
        source = evidence_packet.get("visual_evidence", {}).get("detection_source", "unknown")

        item_name = getattr(analysis_result, "item_name", evidence_packet.get("visual_evidence", {}).get("item_name", "unknown"))
        conf = getattr(analysis_result, "detection_confidence", 0.0)
        freshness = getattr(analysis_result, "freshness_status", "UNKNOWN")

        # Extract remaining freshness string
        est_rf = getattr(analysis_result, "estimated_remaining_freshness", None)
        if est_rf is not None:
            if hasattr(est_rf, "range_description"):
                rf_str = est_rf.range_description
            elif isinstance(est_rf, dict):
                rf_str = est_rf.get("range_description", str(est_rf))
            else:
                rf_str = str(est_rf)
        else:
            rf_str = "N/A"

        sensors = evidence_packet.get("sensor_evidence", {})
        temp = sensors.get("temperature_c")
        hum = sensors.get("humidity_percent")
        gas = sensors.get("gas_value")
        sensor_status = sensors.get("sensor_status", "unavailable")

        # Serialized JSON blobs
        if hasattr(analysis_result, "model_dump_json"):
            analysis_json = analysis_result.model_dump_json()
        elif hasattr(analysis_result, "dict"):
            analysis_json = json.dumps(analysis_result.dict())
        else:
            analysis_json = json.dumps(analysis_result)

        evidence_json = json.dumps(evidence_packet)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO observations (
                observation_id, timestamp, source, item_name, detection_confidence,
                temperature_c, humidity_percent, gas_value, sensor_status,
                freshness_status, estimated_remaining_freshness,
                llm_analysis, raw_evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                obs_id, now, source, item_name, conf,
                temp, hum, gas, sensor_status,
                freshness, rf_str,
                analysis_json, evidence_json
            ))
            conn.commit()
            row_id = cursor.lastrowid
            logger.info("Observation saved: id=%d item=%s status=%s", row_id, item_name, freshness)
            return row_id
        finally:
            conn.close()

    def get_recent_history(self, item_name: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent observations, optionally filtered by item_name."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            if item_name:
                cursor.execute("""
                SELECT timestamp, item_name, detection_confidence,
                       temperature_c, humidity_percent, gas_value,
                       freshness_status, estimated_remaining_freshness
                FROM observations
                WHERE item_name = ?
                ORDER BY id DESC
                LIMIT ?
                """, (item_name, limit))
            else:
                cursor.execute("""
                SELECT timestamp, item_name, detection_confidence,
                       temperature_c, humidity_percent, gas_value,
                       freshness_status, estimated_remaining_freshness
                FROM observations
                ORDER BY id DESC
                LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_history_for_item(self, item_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Convenience alias for get_recent_history for a specific item."""
        return self.get_recent_history(item_name=item_name, limit=limit)

    def get_recent_observations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Convenience alias for getting recent observations across all items."""
        return self.get_recent_history(item_name=None, limit=limit)

    def export_csv(self, export_path: Optional[str] = None):
        """Export all observations to a clean CSV file."""
        target_path = export_path or self.export_path
        export_dir = os.path.dirname(target_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("""
            SELECT id, observation_id, timestamp, source, item_name,
                   detection_confidence, temperature_c, humidity_percent,
                   gas_value, sensor_status, freshness_status,
                   estimated_remaining_freshness
            FROM observations
            ORDER BY id ASC
            """, conn)

            df.to_csv(target_path, index=False)
            logger.info("Exported %d observations to CSV: %s", len(df), target_path)
            return len(df)
        finally:
            conn.close()

    def export_to_csv(self, export_path: Optional[str] = None) -> str:
        """Export observations to CSV and return the file path."""
        target_path = export_path or self.export_path
        self.export_csv(target_path)
        return target_path
