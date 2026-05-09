"""
anonymizer.py — Privacy-preserving detection anonymization.

Blurs bounding box regions before frames are stored or transmitted.
This is a first-class design concern — not an afterthought.

GDPR / CCPA relevance: any system that processes video of people in public
or semi-public spaces must handle personal data appropriately. This module
ensures no identifiable face or body imagery is retained in logs or exports.

Modes
-----
BLUR     — Gaussian blur over the bounding box region (default)
PIXELATE — Block pixelation (lower compute, useful for edge devices)
BLACKOUT — Replace region with solid color (maximum privacy)
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from schemas.events import BoundingBox, TrackedObject


class AnonymizeMode(str, Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACKOUT = "blackout"


class Anonymizer:
    """
    Applies privacy-preserving masking over tracked object bounding boxes.

    Usage
    -----
        anon = Anonymizer(mode=AnonymizeMode.BLUR, blur_strength=31)
        anonymized_frame = anon.apply(frame, tracked_objects)
    """

    def __init__(
        self,
        mode: AnonymizeMode = AnonymizeMode.BLUR,
        blur_strength: int = 31,
        pixel_block_size: int = 12,
        blackout_color: tuple[int, int, int] = (0, 0, 0),
        target_classes: set[str] | None = None,
    ) -> None:
        self.mode = mode
        self.blur_strength = blur_strength | 1  # must be odd
        self.pixel_block_size = pixel_block_size
        self.blackout_color = blackout_color
        # If None, anonymize all classes. Otherwise filter.
        self.target_classes = target_classes or {"person"}

    def apply(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> np.ndarray:
        """Return a copy of *frame* with all target detections anonymized."""
        output = frame.copy()
        h, w = output.shape[:2]

        for obj in tracked_objects:
            if obj.class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = self._clamp_bbox(obj.bbox, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            region = output[y1:y2, x1:x2]

            if self.mode == AnonymizeMode.BLUR:
                output[y1:y2, x1:x2] = cv2.GaussianBlur(
                    region, (self.blur_strength, self.blur_strength), 0
                )
            elif self.mode == AnonymizeMode.PIXELATE:
                output[y1:y2, x1:x2] = self._pixelate(region)
            elif self.mode == AnonymizeMode.BLACKOUT:
                output[y1:y2, x1:x2] = self.blackout_color

        return output

    # ------------------------------------------------------------------

    def _pixelate(self, region: np.ndarray) -> np.ndarray:
        h, w = region.shape[:2]
        bh = max(1, h // self.pixel_block_size)
        bw = max(1, w // self.pixel_block_size)
        small = cv2.resize(region, (bw, bh), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _clamp_bbox(bbox: BoundingBox, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        return x1, y1, x2, y2
