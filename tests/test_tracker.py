"""
test_tracker.py — Tests for SORT tracker and Kalman filter.

All tests use synthetic detections — no GPU, no model files required.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.tracker import KalmanBoxTracker, SORTTracker
from schemas.events import BoundingBox, Detection


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_detection(x1, y1, x2, y2, conf=0.9, cls_id=0, cls_name="person") -> Detection:
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
    )


def make_tracker(**kwargs) -> SORTTracker:
    defaults = {"max_age": 3, "min_hits": 2, "iou_threshold": 0.3}
    defaults.update(kwargs)
    return SORTTracker(**defaults)


# ── BoundingBox math ───────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_area(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert bb.area == pytest.approx(20000.0)

    def test_center(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert bb.center == pytest.approx((50.0, 100.0))

    def test_iou_identical_boxes(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bb.iou(bb) == pytest.approx(1.0)

    def test_iou_non_overlapping(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=150, y2=150)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        # intersection = 50*50 = 2500; union = 10000+10000-2500 = 17500
        assert a.iou(b) == pytest.approx(2500 / 17500, rel=1e-4)

    def test_width_height(self):
        bb = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        assert bb.width == pytest.approx(100.0)
        assert bb.height == pytest.approx(200.0)


# ── KalmanBoxTracker ───────────────────────────────────────────────────────────

class TestKalmanBoxTracker:
    def test_initialises_with_detection(self):
        det = make_detection(100, 150, 200, 400)
        trk = KalmanBoxTracker(det)
        assert trk.hits == 1
        assert trk.time_since_update == 0

    def test_predict_advances_state(self):
        det = make_detection(100, 150, 200, 400)
        trk = KalmanBoxTracker(det)
        bbox_before = trk.get_bbox()
        trk.predict()
        # After predict with zero velocity, bbox should be close to initial
        bbox_after = trk.get_bbox()
        assert abs(bbox_after.x1 - bbox_before.x1) < 5.0

    def test_update_resets_time_since_update(self):
        det = make_detection(100, 150, 200, 400)
        trk = KalmanBoxTracker(det)
        trk.predict()
        assert trk.time_since_update == 1
        trk.update(det)
        assert trk.time_since_update == 0

    def test_update_increments_hits(self):
        det = make_detection(100, 150, 200, 400)
        trk = KalmanBoxTracker(det)
        trk.update(det)
        assert trk.hits == 2


# ── SORTTracker ────────────────────────────────────────────────────────────────

class TestSORTTracker:
    def setup_method(self):
        """Reset ID counter between tests."""
        KalmanBoxTracker._id_counter = 0

    def test_empty_detections_returns_empty(self):
        tracker = make_tracker()
        result = tracker.update([], frame_id=1)
        assert result == []

    def test_single_detection_creates_track(self):
        tracker = make_tracker(min_hits=1)
        det = make_detection(100, 150, 200, 400)
        result = tracker.update([det], frame_id=1)
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_track_id_is_stable_across_frames(self):
        tracker = make_tracker(min_hits=1)
        det = make_detection(100, 150, 200, 400)
        r1 = tracker.update([det], frame_id=1)
        r2 = tracker.update([det], frame_id=2)
        assert len(r1) == 1 and len(r2) == 1
        assert r1[0].track_id == r2[0].track_id

    def test_stale_track_deleted_after_max_age(self):
        tracker = SORTTracker(max_age=2, min_hits=1, iou_threshold=0.3)
        det = make_detection(100, 150, 200, 400)
        tracker.update([det], frame_id=1)
        tracker.update([], frame_id=2)
        tracker.update([], frame_id=3)
        result = tracker.update([], frame_id=4)
        assert result == []
        assert tracker.active_track_count == 0

    def test_two_separate_tracks(self):
        tracker = make_tracker(min_hits=1)
        d1 = make_detection(50,  100, 150, 300)
        d2 = make_detection(500, 100, 600, 300)
        result = tracker.update([d1, d2], frame_id=1)
        assert len(result) == 2
        ids = {obj.track_id for obj in result}
        assert len(ids) == 2

    def test_reset_clears_all_tracks(self):
        tracker = make_tracker(min_hits=1)
        det = make_detection(100, 150, 200, 400)
        tracker.update([det], frame_id=1)
        tracker.reset()
        assert tracker.active_track_count == 0
        assert tracker.frame_count == 0

    def test_confirmed_flag_after_min_hits(self):
        tracker = SORTTracker(max_age=5, min_hits=3, iou_threshold=0.3)
        det = make_detection(100, 150, 200, 400)
        for i in range(1, 4):
            result = tracker.update([det], frame_id=i)
        confirmed = [obj for obj in result if obj.is_confirmed]
        assert len(confirmed) >= 1
