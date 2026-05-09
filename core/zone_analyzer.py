"""
zone_analyzer.py — Polygon zone analytics engine.

Defines named regions of interest in pixel space, tracks which objects
are inside each zone, and emits enter/exit/dwell events.

Usage
-----
    zones = [
        Zone(name="entrance", polygon=[(0,0),(200,0),(200,400),(0,400)]),
        Zone(name="checkout", polygon=[(500,200),(800,200),(800,600),(500,600)]),
    ]
    analyzer = ZoneAnalyzer(zones)
    events = analyzer.update(tracked_objects, frame_id=42, timestamp_ms=1400.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from schemas.events import ZoneEvent, ZoneEventRecord, TrackedObject

logger = logging.getLogger(__name__)

Point = tuple[float, float]
Polygon = list[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Ray-casting algorithm for point-in-polygon test.
    Pure NumPy — no shapely dependency.
    """
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[-1]
    for i in range(n):
        cx, cy = polygon[i]
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-10) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


@dataclass
class Zone:
    """A named polygon region of interest."""
    name: str
    polygon: Polygon

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def contains_bbox_center(self, bbox_center: Point) -> bool:
        return self.contains(bbox_center)


@dataclass
class TrackZoneState:
    """Tracks which zones a specific track_id is currently inside."""
    track_id: int
    zones_inside: set[str] = field(default_factory=set)
    zone_entry_frame: dict[str, int] = field(default_factory=dict)


class ZoneAnalyzer:
    """
    Emits structured zone events as tracks move through named regions.

    Events emitted:
        ENTER  — track center crosses into a zone
        EXIT   — track center leaves a zone
        DWELL  — track has been inside a zone for >= dwell_threshold frames
                 (emitted once per threshold crossing)
    """

    def __init__(
        self,
        zones: list[Zone],
        dwell_threshold_frames: int = 30,
    ) -> None:
        self.zones = {z.name: z for z in zones}
        self.dwell_threshold = dwell_threshold_frames
        self._track_states: dict[int, TrackZoneState] = {}
        self._zone_entry_counts: dict[str, int] = {z: 0 for z in self.zones}
        self._dwell_totals: dict[str, list[int]] = {z: [] for z in self.zones}

    def update(
        self,
        tracked_objects: list[TrackedObject],
        frame_id: int,
        timestamp_ms: float,
    ) -> list[ZoneEventRecord]:
        """Process one frame of tracked objects. Returns all events this frame."""
        events: list[ZoneEventRecord] = []
        active_ids = {obj.track_id for obj in tracked_objects}

        # Handle objects that have disappeared (implicit EXIT)
        for track_id, state in list(self._track_states.items()):
            if track_id not in active_ids:
                for zone_name in list(state.zones_inside):
                    dwell = frame_id - state.zone_entry_frame.get(zone_name, frame_id)
                    events.append(ZoneEventRecord(
                        track_id=track_id,
                        zone_name=zone_name,
                        event=ZoneEvent.EXIT,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        dwell_frames=dwell,
                    ))
                    if dwell > 0:
                        self._dwell_totals[zone_name].append(dwell)
                del self._track_states[track_id]

        for obj in tracked_objects:
            tid = obj.track_id
            center = obj.bbox.center

            if tid not in self._track_states:
                self._track_states[tid] = TrackZoneState(track_id=tid)

            state = self._track_states[tid]

            for zone_name, zone in self.zones.items():
                currently_inside = zone.contains_bbox_center(center)
                was_inside = zone_name in state.zones_inside

                if currently_inside and not was_inside:
                    # ENTER
                    state.zones_inside.add(zone_name)
                    state.zone_entry_frame[zone_name] = frame_id
                    self._zone_entry_counts[zone_name] += 1
                    events.append(ZoneEventRecord(
                        track_id=tid,
                        zone_name=zone_name,
                        event=ZoneEvent.ENTER,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                    ))

                elif not currently_inside and was_inside:
                    # EXIT
                    dwell = frame_id - state.zone_entry_frame.get(zone_name, frame_id)
                    state.zones_inside.discard(zone_name)
                    if dwell > 0:
                        self._dwell_totals[zone_name].append(dwell)
                    events.append(ZoneEventRecord(
                        track_id=tid,
                        zone_name=zone_name,
                        event=ZoneEvent.EXIT,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        dwell_frames=dwell,
                    ))

                elif currently_inside and was_inside:
                    # Check DWELL threshold
                    frames_inside = frame_id - state.zone_entry_frame.get(zone_name, frame_id)
                    if (
                        frames_inside > 0
                        and frames_inside % self.dwell_threshold == 0
                    ):
                        events.append(ZoneEventRecord(
                            track_id=tid,
                            zone_name=zone_name,
                            event=ZoneEvent.DWELL,
                            frame_id=frame_id,
                            timestamp_ms=timestamp_ms,
                            dwell_frames=frames_inside,
                        ))

        return events

    def current_occupancy(self) -> dict[str, int]:
        """Current number of confirmed tracks inside each zone."""
        occupancy = {name: 0 for name in self.zones}
        for state in self._track_states.values():
            for zone_name in state.zones_inside:
                occupancy[zone_name] += 1
        return occupancy

    def summary_stats(self) -> dict:
        """Aggregate stats for the full session so far."""
        avg_dwell = {
            z: (sum(v) / len(v) if v else 0.0)
            for z, v in self._dwell_totals.items()
        }
        return {
            "total_zone_entries": dict(self._zone_entry_counts),
            "avg_dwell_frames": avg_dwell,
        }

    def reset(self) -> None:
        self._track_states.clear()
        self._zone_entry_counts = {z: 0 for z in self.zones}
        self._dwell_totals = {z: [] for z in self.zones}
