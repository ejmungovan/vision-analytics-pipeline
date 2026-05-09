"""
pipeline.py — Top-level orchestrator for the vision analytics pipeline.

Ties together: Detector → Tracker → ZoneAnalyzer → Anonymizer → Output

Supports:
  - Video file input
  - Frame-by-frame API (for RTSP/camera streams)
  - Headless mode (no display window) for production deployment
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.anonymizer import Anonymizer, AnonymizeMode
from core.detector import BaseDetector, StubDetector
from core.tracker import SORTTracker
from core.zone_analyzer import Zone, ZoneAnalyzer
from schemas.events import FrameAnalytics, StreamSummary, TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All tunable pipeline parameters in one place."""
    confidence_threshold: float = 0.4
    target_class_ids: set[int] | None = None      # None = all classes
    max_tracker_age: int = 3
    min_tracker_hits: int = 2
    tracker_iou_threshold: float = 0.3
    dwell_threshold_frames: int = 30
    anonymize: bool = True
    anonymize_mode: AnonymizeMode = AnonymizeMode.BLUR
    headless: bool = True                          # suppress display window


class VisionPipeline:
    """
    End-to-end vision analytics pipeline.

    Process a video file
    --------------------
        pipeline = VisionPipeline(detector, zones, config)
        summary = pipeline.run_video("input.mp4", output_path="output.mp4")

    Process frames from a live stream
    ----------------------------------
        pipeline = VisionPipeline(detector, zones, config)
        pipeline.start()
        for frame in camera_stream():
            analytics = pipeline.process_frame(frame)
            # analytics.zone_events, analytics.tracked_objects, etc.
        summary = pipeline.stop()
    """

    def __init__(
        self,
        detector: BaseDetector,
        zones: list[Zone] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or PipelineConfig()
        self.zones = zones or []

        self._tracker = SORTTracker(
            max_age=self.config.max_tracker_age,
            min_hits=self.config.min_tracker_hits,
            iou_threshold=self.config.tracker_iou_threshold,
        )
        self._zone_analyzer = ZoneAnalyzer(
            zones=self.zones,
            dwell_threshold_frames=self.config.dwell_threshold_frames,
        )
        self._anonymizer = Anonymizer(mode=self.config.anonymize_mode)

        self._frame_id = 0
        self._analytics_log: list[FrameAnalytics] = []
        self._running = False
        self._unique_track_ids: set[int] = set()

    # ── Public: frame-by-frame API ─────────────────────────────────────

    def start(self) -> None:
        """Initialise pipeline state for a new stream session."""
        self._frame_id = 0
        self._analytics_log.clear()
        self._unique_track_ids.clear()
        self._tracker.reset()
        self._zone_analyzer.reset()
        self._running = True
        logger.info("VisionPipeline started.")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float | None = None,
    ) -> FrameAnalytics:
        """
        Process a single frame and return FrameAnalytics.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (standard OpenCV format).
        timestamp_ms : float, optional
            Milliseconds from stream start. Auto-derived from frame_id if None.
        """
        t_start = time.perf_counter()
        self._frame_id += 1
        fid = self._frame_id
        ts = timestamp_ms if timestamp_ms is not None else float(fid * 33.33)  # ~30fps

        # ── Detect ────────────────────────────────────────────────────
        detections = self.detector.detect(
            frame,
            confidence_threshold=self.config.confidence_threshold,
            target_classes=self.config.target_class_ids,
        )

        # ── Track ─────────────────────────────────────────────────────
        tracked = self._tracker.update(detections, frame_id=fid)
        for obj in tracked:
            self._unique_track_ids.add(obj.track_id)

        # ── Zone analytics ────────────────────────────────────────────
        zone_events = self._zone_analyzer.update(tracked, frame_id=fid, timestamp_ms=ts)
        occupancy = self._zone_analyzer.current_occupancy()

        # ── Build analytics record ────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        analytics = FrameAnalytics(
            frame_id=fid,
            timestamp_ms=ts,
            tracked_objects=tracked,
            zone_events=zone_events,
            zone_occupancy=occupancy,
            total_detections=len(detections),
            processing_ms=round(elapsed_ms, 2),
        )
        self._analytics_log.append(analytics)
        return analytics

    def stop(self) -> StreamSummary:
        """Finalise session and return aggregate statistics."""
        self._running = False
        logger.info("VisionPipeline stopped after %d frames.", self._frame_id)
        return self._build_summary()

    # ── Public: video file processing ─────────────────────────────────

    def run_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> StreamSummary:
        """
        Process a video file end-to-end.

        Writes an annotated output video if *output_path* is provided.
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        self.start()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                analytics = self.process_frame(frame, timestamp_ms=ts)

                if writer:
                    annotated = self._draw_frame(frame, analytics)
                    if self.config.anonymize:
                        annotated = self._anonymizer.apply(annotated, analytics.tracked_objects)
                    writer.write(annotated)
        finally:
            cap.release()
            if writer:
                writer.release()

        return self.stop()

    # ── Private helpers ────────────────────────────────────────────────

    def _build_summary(self) -> StreamSummary:
        if not self._analytics_log:
            return StreamSummary(
                total_frames=0,
                total_unique_tracks=0,
                avg_occupancy_by_zone={},
                peak_occupancy_by_zone={},
                total_zone_entries={},
                avg_dwell_frames_by_zone={},
                avg_processing_ms=0.0,
            )

        zone_names = list(self._zone_analyzer.zones.keys())
        occupancy_sums: dict[str, int] = {z: 0 for z in zone_names}
        peak_occ: dict[str, int] = {z: 0 for z in zone_names}

        for fa in self._analytics_log:
            for z, count in fa.zone_occupancy.items():
                occupancy_sums[z] = occupancy_sums.get(z, 0) + count
                peak_occ[z] = max(peak_occ.get(z, 0), count)

        n = len(self._analytics_log)
        avg_occ = {z: occupancy_sums[z] / n for z in zone_names}
        zone_stats = self._zone_analyzer.summary_stats()
        avg_proc = sum(
            fa.processing_ms for fa in self._analytics_log if fa.processing_ms
        ) / n

        return StreamSummary(
            total_frames=self._frame_id,
            total_unique_tracks=len(self._unique_track_ids),
            avg_occupancy_by_zone=avg_occ,
            peak_occupancy_by_zone=peak_occ,
            total_zone_entries=zone_stats["total_zone_entries"],
            avg_dwell_frames_by_zone=zone_stats["avg_dwell_frames"],
            avg_processing_ms=round(avg_proc, 2),
        )

    def _draw_frame(
        self,
        frame: np.ndarray,
        analytics: FrameAnalytics,
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and zone overlays onto *frame*."""
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw zones
        for zone in self.zones:
            pts = np.array([[int(x), int(y)] for x, y in zone.polygon], dtype=np.int32)
            cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            if pts.size > 0:
                cx, cy = pts.mean(axis=0).astype(int)
                occ = analytics.zone_occupancy.get(zone.name, 0)
                cv2.putText(
                    output, f"{zone.name} ({occ})",
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA,
                )

        # Draw tracked objects
        for obj in analytics.tracked_objects:
            x1, y1 = int(obj.bbox.x1), int(obj.bbox.y1)
            x2, y2 = int(obj.bbox.x2), int(obj.bbox.y2)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
            cv2.putText(
                output, label, (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA,
            )

        # HUD
        cv2.putText(
            output,
            f"Frame {analytics.frame_id} | Tracks: {len(analytics.tracked_objects)} "
            f"| {analytics.processing_ms:.1f}ms",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return output
