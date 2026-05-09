"""
tracker.py — SORT (Simple Online and Realtime Tracking) implementation.

Built from scratch to demonstrate algorithmic depth:
  - Kalman filter for motion prediction (constant velocity model)
  - Hungarian algorithm for detection-track assignment
  - Track lifecycle management (tentative → confirmed → lost → deleted)

Reference: Bewley et al., "Simple Online and Realtime Tracking" (ICIP 2016)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

from schemas.events import BoundingBox, Detection, TrackedObject


# ── Kalman Filter ──────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Kalman filter tracker for a single bounding box.

    State vector: [x_center, y_center, area, aspect_ratio, dx, dy, d_area]
    Measurement:  [x_center, y_center, area, aspect_ratio]

    Constant velocity motion model. The filter predicts box position between
    detections and corrects when a new detection is associated.
    """

    _id_counter = 0

    def __init__(self, detection: Detection) -> None:
        KalmanBoxTracker._id_counter += 1
        self.track_id = KalmanBoxTracker._id_counter
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

        # State: [cx, cy, s, r, dcx, dcy, ds]  (s=area, r=aspect ratio)
        self.x = self._bbox_to_state(detection.bbox)

        # Kalman matrices
        self.F = np.eye(7)          # State transition
        self.F[0, 4] = 1            # cx += dcx
        self.F[1, 5] = 1            # cy += dcy
        self.F[2, 6] = 1            # s  += ds

        self.H = np.zeros((4, 7))   # Measurement matrix
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7) * 10.0   # Initial covariance
        self.P[4:, 4:] *= 1000.0    # High velocity uncertainty

        self.Q = np.eye(7)          # Process noise
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4)          # Measurement noise
        self.R[2:, 2:] *= 10.0

    # ── Kalman predict ──────────────────────────────────────────────────

    def predict(self) -> BoundingBox:
        """Advance state by one step. Returns predicted bbox."""
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1

        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._state_to_bbox(self.x)

    def update(self, detection: Detection) -> None:
        """Correct state with a new detection (measurement update)."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = detection.confidence
        self.class_name = detection.class_name

        z = self._bbox_to_state(detection.bbox)[:4]
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_bbox(self) -> BoundingBox:
        return self._state_to_bbox(self.x)

    # ── Coordinate transforms ───────────────────────────────────────────

    @staticmethod
    def _bbox_to_state(bbox: BoundingBox) -> np.ndarray:
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        s = bbox.area
        r = bbox.width / (bbox.height + 1e-6)
        return np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> BoundingBox:
        s = max(state[2], 1.0)
        r = max(state[3], 0.1)
        w = np.sqrt(s * r)
        h = s / w
        x1 = state[0] - w / 2
        y1 = state[1] - h / 2
        return BoundingBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)


# ── SORT Tracker ───────────────────────────────────────────────────────────────

class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracker.

    Maintains a set of KalmanBoxTrackers. On each frame:
      1. Predict all existing tracks forward.
      2. Compute IoU cost matrix between predictions and new detections.
      3. Solve assignment with the Hungarian algorithm.
      4. Update matched tracks; create new tracks for unmatched detections.
      5. Delete stale tracks that have not been updated.

    Parameters
    ----------
    max_age : int
        Frames a track survives without a detection match before deletion.
    min_hits : int
        Minimum detection hits before a track is marked "confirmed".
    iou_threshold : float
        Minimum IoU for a detection-track pair to be considered a match.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0  # Reset IDs per tracker instance

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> list[TrackedObject]:
        """
        Process one frame of detections and return active tracked objects.

        Only confirmed tracks (hits >= min_hits) are returned unless
        it's the very first frame.
        """
        self.frame_count += 1

        # ── Step 1: Predict all tracks ──────────────────────────────────
        predicted_bboxes = [t.predict() for t in self.trackers]

        # ── Step 2: Build IoU cost matrix ───────────────────────────────
        track_ids, det_ids, matched, unmatched_dets, unmatched_trks = (
            self._associate(predicted_bboxes, detections)
        )

        # ── Step 3: Update matched tracks ───────────────────────────────
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx])

        # ── Step 4: Create new tracks for unmatched detections ───────────
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d_idx]))

        # ── Step 5: Cull stale tracks; collect output ───────────────────
        active: list[TrackedObject] = []
        survivors: list[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue  # delete
            survivors.append(trk)
            is_confirmed = trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            if trk.time_since_update < 1 and is_confirmed:
                active.append(TrackedObject(
                    track_id=trk.track_id,
                    bbox=trk.get_bbox(),
                    confidence=trk.confidence,
                    class_name=trk.class_name,
                    frame_id=frame_id,
                    age=trk.age,
                    is_confirmed=is_confirmed,
                ))

        self.trackers = survivors
        return active

    def reset(self) -> None:
        """Clear all tracks and reset frame counter."""
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0

    # ── Private: Hungarian assignment ──────────────────────────────────

    def _associate(
        self,
        predicted: list[BoundingBox],
        detections: list[Detection],
    ) -> tuple[list[int], list[int], list[tuple[int, int]], list[int], list[int]]:
        """Return (track_ids, det_ids, matched, unmatched_dets, unmatched_trks)."""
        if not predicted or not detections:
            return (
                list(range(len(predicted))),
                list(range(len(detections))),
                [],
                list(range(len(detections))),
                list(range(len(predicted))),
            )

        iou_matrix = np.zeros((len(predicted), len(detections)), dtype=float)
        for t_i, pred_box in enumerate(predicted):
            for d_i, det in enumerate(detections):
                iou_matrix[t_i, d_i] = pred_box.iou(det.bbox)

        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: list[tuple[int, int]] = []
        unmatched_trks = list(range(len(predicted)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_trks.remove(r)
                unmatched_dets.remove(c)

        return (
            list(range(len(predicted))),
            list(range(len(detections))),
            matched,
            unmatched_dets,
            unmatched_trks,
        )

    @property
    def active_track_count(self) -> int:
        return len(self.trackers)
