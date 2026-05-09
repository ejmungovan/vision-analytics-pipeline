"""
detector.py — Object detection wrapper with YOLOv8 backend.

Design:
- BaseDetector ABC enforces a clean interface for any future backend
  (YOLOv8, ONNX Runtime, TFLite, custom model).
- YOLOv8Detector wraps ultralytics with lazy model loading.
- StubDetector enables full pipeline testing without GPU or model files.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from schemas.events import BoundingBox, Detection

logger = logging.getLogger(__name__)

# COCO class names (80 classes)
COCO_CLASSES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 14: "stop sign", 15: "parking meter", 16: "bench",
    17: "bird", 18: "cat", 19: "dog", 20: "horse", 21: "sheep", 22: "cow",
    24: "elephant", 25: "bear", 26: "zebra", 27: "giraffe", 28: "backpack",
    63: "laptop", 64: "mouse", 67: "cell phone", 73: "book",
}

# Default classes to track (persons + vehicles)
DEFAULT_TRACK_CLASSES = {0, 1, 2, 3, 5, 6, 7}


class BaseDetector(ABC):
    """Abstract base for all detection backends."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.4,
        target_classes: set[int] | None = None,
    ) -> list[Detection]:
        """Run inference on *frame*, return filtered detections."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class YOLOv8Detector(BaseDetector):
    """
    YOLOv8 detector via the ultralytics library.

    Lazy-loads the model on first call. Supports ONNX-exported weights
    transparently (pass model_path ending in .onnx).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._model = None

    @property
    def model_name(self) -> str:
        return Path(self._model_path).stem

    def _load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import]
            self._model = YOLO(self._model_path)
            logger.info("YOLOv8Detector loaded: %s on %s", self._model_path, self._device)
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from exc

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.4,
        target_classes: set[int] | None = None,
    ) -> list[Detection]:
        if self._model is None:
            self._load()

        results = self._model.predict(
            source=frame,
            conf=confidence_threshold,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if target_classes and class_id not in target_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=COCO_CLASSES.get(class_id, f"class_{class_id}"),
                ))

        return detections


class StubDetector(BaseDetector):
    """
    Deterministic stub detector for testing and CI.

    Returns a configurable fixed set of detections on every call —
    no model files, no GPU, no network required.
    """

    def __init__(self, fixed_detections: list[Detection] | None = None) -> None:
        self._detections = fixed_detections if fixed_detections is not None else [
            Detection(
                bbox=BoundingBox(x1=100, y1=150, x2=200, y2=400),
                confidence=0.92,
                class_id=0,
                class_name="person",
            ),
            Detection(
                bbox=BoundingBox(x1=320, y1=160, x2=420, y2=390),
                confidence=0.87,
                class_id=0,
                class_name="person",
            ),
        ]

    @property
    def model_name(self) -> str:
        return "stub"

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.4,
        target_classes: set[int] | None = None,
    ) -> list[Detection]:
        return [
            d for d in self._detections
            if d.confidence >= confidence_threshold
            and (target_classes is None or d.class_id in target_classes)
        ]
