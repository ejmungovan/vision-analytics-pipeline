"""
events.py — Pydantic schemas for all pipeline outputs.

Every detection, track, and zone event is typed here.
Downstream consumers (APIs, dashboards, databases) rely on this contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ZoneEvent(str, Enum):
    ENTER = "enter"
    EXIT = "exit"
    DWELL = "dwell"


class BoundingBox(BaseModel):
    """Pixel-space bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detection(BaseModel):
    """Single-frame object detection (pre-tracking)."""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str


class TrackedObject(BaseModel):
    """A detection that has been assigned a persistent track ID."""
    track_id: int
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    frame_id: int
    age: int = Field(default=1, description="Frames this track has been active")
    is_confirmed: bool = Field(
        default=False,
        description="True after track survives min_hits threshold",
    )


class ZoneEventRecord(BaseModel):
    """An event triggered when a tracked object interacts with a named zone."""
    track_id: int
    zone_name: str
    event: ZoneEvent
    frame_id: int
    timestamp_ms: float = Field(description="Milliseconds from stream start")
    dwell_frames: Optional[int] = Field(
        default=None,
        description="Frames spent in zone (populated on EXIT/DWELL events)",
    )


class FrameAnalytics(BaseModel):
    """Per-frame summary — the main output unit of the pipeline."""
    frame_id: int
    timestamp_ms: float
    tracked_objects: list[TrackedObject]
    zone_events: list[ZoneEventRecord]
    zone_occupancy: dict[str, int] = Field(
        default_factory=dict,
        description="Current count of tracks inside each zone",
    )
    total_detections: int
    processing_ms: Optional[float] = None


class StreamSummary(BaseModel):
    """Aggregate statistics for a completed video or stream segment."""
    total_frames: int
    total_unique_tracks: int
    avg_occupancy_by_zone: dict[str, float]
    peak_occupancy_by_zone: dict[str, int]
    total_zone_entries: dict[str, int]
    avg_dwell_frames_by_zone: dict[str, float]
    avg_processing_ms: float
