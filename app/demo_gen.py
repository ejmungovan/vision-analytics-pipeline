"""
demo_gen.py — Synthetic scene generator for the Streamlit demo.

Creates deterministic, reproducible "persons" moving through named zones
so the full pipeline can be demonstrated without any real video file.

Three actors with distinct trajectories:
  - Actor A: walks entrance → main_area → exits right  (long dwell)
  - Actor B: quick pass through entrance → exit
  - Actor C: lingers in main_area (triggers DWELL event)

Each actor's position is a smooth function of frame_id, so the demo
plays out the same way every time and exercises every analytics feature.
"""

from __future__ import annotations

import math
import numpy as np

from schemas.events import BoundingBox, Detection

# Demo canvas size
DEMO_W = 800
DEMO_H = 480

# Zone geometry (matches app defaults)
ZONE_POLYGONS = {
    "entrance":  [(0, 0),   (250, 0),   (250, DEMO_H), (0, DEMO_H)],
    "main_area": [(250, 0), (600, 0),   (600, DEMO_H), (250, DEMO_H)],
    "exit":      [(600, 0), (DEMO_W, 0),(DEMO_W, DEMO_H),(600, DEMO_H)],
}

TOTAL_DEMO_FRAMES = 240   # 8 seconds at 30fps
PERSON_W, PERSON_H = 50, 110   # bounding box size


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _smooth(t: float) -> float:
    """Smoothstep easing for natural movement."""
    return t * t * (3 - 2 * t)


def _actor_a_position(frame: int) -> tuple[float, float] | None:
    """Actor A: entrance → main → exit. Present frames 0–220."""
    if frame > 220:
        return None
    t = frame / 220
    if t < 0.2:
        cx = _lerp(30,  200, _smooth(t / 0.2))
        cy = _lerp(150, 200, _smooth(t / 0.2))
    elif t < 0.8:
        cx = _lerp(200, 580, _smooth((t - 0.2) / 0.6))
        cy = _lerp(200, 250, _smooth((t - 0.2) / 0.6))
    else:
        cx = _lerp(580, 790, _smooth((t - 0.8) / 0.2))
        cy = _lerp(250, 260, _smooth((t - 0.8) / 0.2))
    return cx, cy


def _actor_b_position(frame: int) -> tuple[float, float] | None:
    """Actor B: quick pass, entrance → exit. Present frames 30–130."""
    if frame < 30 or frame > 130:
        return None
    t = (frame - 30) / 100
    cx = _lerp(60, 760, _smooth(t))
    cy = 380.0
    return cx, cy


def _actor_c_position(frame: int) -> tuple[float, float] | None:
    """Actor C: dwells in main_area. Present frames 60–210."""
    if frame < 60 or frame > 210:
        return None
    t = (frame - 60) / 150
    if t < 0.15:
        cx = _lerp(40, 380, _smooth(t / 0.15))
    elif t < 0.85:
        # Loitering in main_area with slight drift
        cx = 380 + 40 * math.sin(t * math.pi * 3)
    else:
        cx = _lerp(380, 760, _smooth((t - 0.85) / 0.15))
    cy = 200 + 30 * math.sin(t * math.pi * 2)
    return cx, cy


_ACTORS = [
    (_actor_a_position, 0.90, 0),   # (pos_fn, confidence, class_id)
    (_actor_b_position, 0.85, 0),
    (_actor_c_position, 0.88, 0),
]


def generate_frame(frame_id: int) -> np.ndarray:
    """
    Return an 800×480 BGR frame with coloured actor rectangles drawn.
    Background has a subtle grid to look like a floor plan.
    """
    frame = np.full((DEMO_H, DEMO_W, 3), (28, 30, 36), dtype=np.uint8)

    # Draw subtle grid
    grid_color = (40, 42, 50)
    for x in range(0, DEMO_W, 40):
        frame[:, x] = grid_color
    for y in range(0, DEMO_H, 40):
        frame[y, :] = grid_color

    # Draw zone boundaries
    import cv2
    zone_colors = {
        "entrance":  (60, 120, 200),
        "main_area": (60, 180, 100),
        "exit":      (200, 100, 60),
    }
    for name, poly in ZONE_POLYGONS.items():
        pts = np.array(poly, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], zone_colors[name])
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=zone_colors[name], thickness=1)
        # Label
        cx_z = int(sum(p[0] for p in poly) / len(poly))
        cy_z = 20
        cv2.putText(frame, name, (cx_z - 30, cy_z),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    tuple(int(c * 1.5) for c in zone_colors[name]), 1, cv2.LINE_AA)

    # Draw actors
    actor_colors = [(100, 220, 255), (255, 180, 80), (180, 255, 120)]
    for i, (pos_fn, _, _) in enumerate(_ACTORS):
        pos = pos_fn(frame_id)
        if pos is None:
            continue
        cx, cy = int(pos[0]), int(pos[1])
        x1 = max(0, cx - PERSON_W // 2)
        y1 = max(0, cy - PERSON_H // 2)
        x2 = min(DEMO_W, cx + PERSON_W // 2)
        y2 = min(DEMO_H, cy + PERSON_H // 2)
        color = actor_colors[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        # Silhouette head
        cv2.circle(frame, (cx, y1 - 14), 14, color, -1)

    return frame


def generate_detections(frame_id: int) -> list[Detection]:
    """Return synthetic detections for this frame (no model needed)."""
    detections = []
    for pos_fn, conf, cls_id in _ACTORS:
        pos = pos_fn(frame_id)
        if pos is None:
            continue
        cx, cy = pos[0], pos[1]
        detections.append(Detection(
            bbox=BoundingBox(
                x1=cx - PERSON_W / 2,
                y1=cy - PERSON_H / 2,
                x2=cx + PERSON_W / 2,
                y2=cy + PERSON_H / 2,
            ),
            confidence=conf,
            class_id=cls_id,
            class_name="person",
        ))
    return detections


class DemoDetector:
    """
    Drop-in detector that returns pre-computed synthetic detections.
    Conforms to the BaseDetector interface without subclassing
    (avoids importing the ABC in the demo layer).
    """

    name = "demo"
    model_name = "demo"
    _frame = 0

    def detect(self, frame: np.ndarray, **_) -> list[Detection]:
        dets = generate_detections(self._frame)
        self._frame += 1
        return dets

    def reset(self) -> None:
        self._frame = 0
