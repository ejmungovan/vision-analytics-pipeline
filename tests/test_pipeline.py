"""
test_pipeline.py — Integration tests for the full VisionPipeline.

All tests use StubDetector — no GPU, no model files, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.detector import StubDetector
from core.pipeline import VisionPipeline, PipelineConfig
from core.tracker import KalmanBoxTracker
from core.zone_analyzer import Zone
from schemas.events import BoundingBox, Detection, ZoneEvent


# ── Fixtures ───────────────────────────────────────────────────────────────────

ZONES = [
    Zone(name="zone_a", polygon=[(0, 0), (400, 0), (400, 600), (0, 600)]),
    Zone(name="zone_b", polygon=[(400, 0), (800, 0), (800, 600), (400, 600)]),
]

BLANK_FRAME = np.zeros((600, 800, 3), dtype=np.uint8)


def make_pipeline(zones=None, **config_kwargs) -> VisionPipeline:
    KalmanBoxTracker._id_counter = 0
    return VisionPipeline(
        detector=StubDetector(),
        zones=zones or ZONES,
        config=PipelineConfig(headless=True, anonymize=False, **config_kwargs),
    )


# ── Detector ──────────────────────────────────────────────────────────────────

class TestStubDetector:
    def test_returns_default_detections(self):
        d = StubDetector()
        result = d.detect(BLANK_FRAME)
        assert len(result) == 2
        assert all(isinstance(r, Detection) for r in result)

    def test_confidence_filter(self):
        d = StubDetector()
        result = d.detect(BLANK_FRAME, confidence_threshold=0.95)
        assert all(r.confidence >= 0.95 for r in result)

    def test_class_filter(self):
        d = StubDetector()
        result = d.detect(BLANK_FRAME, target_classes={99})  # no class 99 in stub
        assert result == []

    def test_custom_detections(self):
        custom = [Detection(
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=100),
            confidence=0.75, class_id=0, class_name="person"
        )]
        d = StubDetector(fixed_detections=custom)
        assert len(d.detect(BLANK_FRAME)) == 1

    def test_model_name_is_stub(self):
        assert StubDetector().model_name == "stub"


# ── VisionPipeline ────────────────────────────────────────────────────────────

class TestVisionPipeline:
    def test_process_frame_returns_analytics(self):
        pipeline = make_pipeline()
        pipeline.start()
        analytics = pipeline.process_frame(BLANK_FRAME)
        assert analytics.frame_id == 1
        assert analytics.total_detections == 2
        assert analytics.processing_ms is not None

    def test_frame_id_increments(self):
        pipeline = make_pipeline()
        pipeline.start()
        for i in range(5):
            a = pipeline.process_frame(BLANK_FRAME)
        assert a.frame_id == 5

    def test_zone_occupancy_keys_present(self):
        pipeline = make_pipeline()
        pipeline.start()
        analytics = pipeline.process_frame(BLANK_FRAME)
        assert "zone_a" in analytics.zone_occupancy
        assert "zone_b" in analytics.zone_occupancy

    def test_stop_returns_summary(self):
        pipeline = make_pipeline()
        pipeline.start()
        for _ in range(10):
            pipeline.process_frame(BLANK_FRAME)
        summary = pipeline.stop()
        assert summary.total_frames == 10
        assert summary.total_unique_tracks >= 0

    def test_summary_avg_processing_ms_positive(self):
        pipeline = make_pipeline()
        pipeline.start()
        for _ in range(5):
            pipeline.process_frame(BLANK_FRAME)
        summary = pipeline.stop()
        assert summary.avg_processing_ms >= 0.0

    def test_zone_events_contain_enter(self):
        """StubDetector places persons in zone_a — we should see ENTER events."""
        pipeline = make_pipeline(min_tracker_hits=1)
        pipeline.start()
        all_events = []
        for fid in range(1, 6):
            a = pipeline.process_frame(BLANK_FRAME)
            all_events.extend(a.zone_events)
        enters = [e for e in all_events if e.event == ZoneEvent.ENTER]
        assert len(enters) >= 1

    def test_no_crash_on_empty_frame(self):
        """Pipeline should handle frames with no detections gracefully."""
        pipeline = VisionPipeline(
            detector=StubDetector(fixed_detections=[]),
            zones=ZONES,
        )
        pipeline.start()
        analytics = pipeline.process_frame(BLANK_FRAME)
        assert analytics.total_detections == 0
        assert analytics.tracked_objects == []
