"""Unit tests for SQLite History Service."""

import os
import tempfile
from services.history_service import HistoryService
from services.llm_service import FreshnessAnalysis, RemainingFreshness


def test_save_and_retrieve_history():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        service = HistoryService(db_path=temp_db)

        analysis = FreshnessAnalysis(
            item_name="tomato",
            detection_confidence=0.89,
            freshness_status="FRESH",
            estimated_remaining_freshness=RemainingFreshness(
                value="4-7",
                unit="days",
                range_description="4-7 days",
                confidence="medium",
                basis=["Room temp storage"],
            ),
            visual_observations=["Firm red skin"],
            environmental_context="22C",
            reasoning_summary="Fresh tomato",
            uncertainty_factors=["Surface only"],
            recommendations=["Store at room temp"],
        )

        evidence = {
            "observation_id": "test-uuid-1234",
            "timestamp": "2026-10-02T12:00:00",
            "visual_evidence": {"item_name": "tomato", "confidence": 0.89, "detection_source": "yolo_world"},
            "sensor_evidence": {"temperature_c": 22.0, "humidity_percent": 60.0, "sensor_status": "valid"},
        }

        row_id = service.save_observation(analysis, evidence)
        assert row_id is not None and row_id > 0

        # Retrieve history
        history = service.get_recent_history(item_name="tomato", limit=5)
        assert len(history) == 1
        assert history[0]["item_name"] == "tomato"
        assert history[0]["freshness_status"] == "FRESH"
        assert history[0]["temperature_c"] == 22.0

        # Export CSV test
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as csv_file:
            temp_csv = csv_file.name

        count = service.export_csv(temp_csv)
        assert count == 1
        assert os.path.exists(temp_csv)
        os.remove(temp_csv)

    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
