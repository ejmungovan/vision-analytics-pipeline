"""
test_zone_analyzer.py — Tests for the polygon zone analytics engine.
"""

from __future__ import annotations

import pytest

from core.zone_analyzer import Zone, ZoneAnalyzer, point_in_polygon
from schemas.events import BoundingBox, TrackedObject, ZoneEvent


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_track(track_id: int, cx: float, cy: float, frame_id: int = 1) -> TrackedObject:
    """Create a tracked object centred at (cx, cy)."""
    return TrackedObject(
        track_id=track_id,
        bbox=BoundingBox(x1=cx - 25, y1=cy - 50, x2=cx + 25, y2=cy + 50),
        confidence=0.9,
        class_name="person",
        frame_id=frame_id,
        is_confirmed=True,
    )


SQUARE_ZONE = Zone(name="square", polygon=[(0, 0), (200, 0), (200, 200), (0, 200)])


# ── point_in_polygon ───────────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_center_inside_square(self):
        assert point_in_polygon((100, 100), [(0,0),(200,0),(200,200),(0,200)]) is True

    def test_outside_square(self):
        assert point_in_polygon((300, 300), [(0,0),(200,0),(200,200),(0,200)]) is False

    def test_on_right_edge(self):
        # Edge behaviour is implementation-defined; just assert no crash
        result = point_in_polygon((200, 100), [(0,0),(200,0),(200,200),(0,200)])
        assert isinstance(result, bool)

    def test_triangle_inside(self):
        tri = [(0, 0), (100, 0), (50, 100)]
        assert point_in_polygon((50, 30), tri) is True

    def test_triangle_outside(self):
        tri = [(0, 0), (100, 0), (50, 100)]
        assert point_in_polygon((90, 90), tri) is False


# ── Zone ──────────────────────────────────────────────────────────────────────

class TestZone:
    def test_contains_inner_point(self):
        assert SQUARE_ZONE.contains((100, 100)) is True

    def test_does_not_contain_outer_point(self):
        assert SQUARE_ZONE.contains((300, 300)) is False


# ── ZoneAnalyzer ──────────────────────────────────────────────────────────────

class TestZoneAnalyzer:
    def setup_method(self):
        self.zone = Zone(name="lobby", polygon=[(0, 0), (400, 0), (400, 400), (0, 400)])
        self.analyzer = ZoneAnalyzer(zones=[self.zone], dwell_threshold_frames=5)

    def test_enter_event_on_first_appearance(self):
        track = make_track(1, cx=200, cy=200, frame_id=1)
        events = self.analyzer.update([track], frame_id=1, timestamp_ms=33.0)
        enters = [e for e in events if e.event == ZoneEvent.ENTER]
        assert len(enters) == 1
        assert enters[0].track_id == 1
        assert enters[0].zone_name == "lobby"

    def test_no_duplicate_enter_on_stay(self):
        track = make_track(1, cx=200, cy=200)
        self.analyzer.update([track], frame_id=1, timestamp_ms=33.0)
        events = self.analyzer.update([track], frame_id=2, timestamp_ms=66.0)
        enters = [e for e in events if e.event == ZoneEvent.ENTER]
        assert len(enters) == 0

    def test_exit_event_on_departure(self):
        inside = make_track(1, cx=200, cy=200)
        outside = make_track(1, cx=600, cy=600)
        self.analyzer.update([inside], frame_id=1, timestamp_ms=33.0)
        events = self.analyzer.update([outside], frame_id=2, timestamp_ms=66.0)
        exits = [e for e in events if e.event == ZoneEvent.EXIT]
        assert len(exits) == 1
        assert exits[0].track_id == 1

    def test_exit_event_on_track_disappear(self):
        inside = make_track(1, cx=200, cy=200)
        self.analyzer.update([inside], frame_id=1, timestamp_ms=33.0)
        events = self.analyzer.update([], frame_id=2, timestamp_ms=66.0)
        exits = [e for e in events if e.event == ZoneEvent.EXIT]
        assert any(e.track_id == 1 for e in exits)

    def test_dwell_event_at_threshold(self):
        track = make_track(1, cx=200, cy=200)
        all_events = []
        for fid in range(1, 12):
            evts = self.analyzer.update([track], frame_id=fid, timestamp_ms=float(fid * 33))
            all_events.extend(evts)
        dwells = [e for e in all_events if e.event == ZoneEvent.DWELL]
        assert len(dwells) >= 1

    def test_occupancy_count(self):
        t1 = make_track(1, cx=100, cy=100)
        t2 = make_track(2, cx=200, cy=200)
        self.analyzer.update([t1, t2], frame_id=1, timestamp_ms=33.0)
        occupancy = self.analyzer.current_occupancy()
        assert occupancy["lobby"] == 2

    def test_reset_clears_state(self):
        track = make_track(1, cx=200, cy=200)
        self.analyzer.update([track], frame_id=1, timestamp_ms=33.0)
        self.analyzer.reset()
        assert self.analyzer.current_occupancy()["lobby"] == 0

    def test_summary_stats_after_entries(self):
        inside = make_track(1, cx=200, cy=200)
        outside = make_track(1, cx=600, cy=600)
        self.analyzer.update([inside], frame_id=1, timestamp_ms=33.0)
        self.analyzer.update([outside], frame_id=2, timestamp_ms=66.0)
        stats = self.analyzer.summary_stats()
        assert stats["total_zone_entries"]["lobby"] == 1
