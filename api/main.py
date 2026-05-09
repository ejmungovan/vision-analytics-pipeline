"""
main.py — FastAPI REST interface for the Vision Analytics Pipeline.

Endpoints:
  POST /analyze/frame   — analyze a single base64-encoded frame
  POST /analyze/video   — submit a video file path for processing
  GET  /health          — liveness + model info
  GET  /zones           — list configured zones
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from core.detector import StubDetector
from core.pipeline import VisionPipeline, PipelineConfig
from core.zone_analyzer import Zone
from schemas.events import FrameAnalytics, StreamSummary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Default zones (configurable via YAML in production) ───────────────────────
DEFAULT_ZONES = [
    Zone(name="entrance",  polygon=[(0, 0), (320, 0), (320, 720), (0, 720)]),
    Zone(name="main_area", polygon=[(320, 0), (960, 0), (960, 720), (320, 720)]),
    Zone(name="exit",      polygon=[(960, 0), (1280, 0), (1280, 720), (960, 720)]),
]

_pipeline: VisionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    detector = StubDetector()          # swap for YOLOv8Detector in production
    config = PipelineConfig(headless=True, anonymize=True)
    _pipeline = VisionPipeline(detector=detector, zones=DEFAULT_ZONES, config=config)
    logger.info("VisionPipeline initialised. Zones: %s", [z.name for z in DEFAULT_ZONES])
    yield


app = FastAPI(
    title="Vision Analytics Pipeline",
    description=(
        "Real-time multi-object tracking + zone analytics. "
        "Privacy-first: all detections anonymised before storage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    frame_b64: str = Field(..., description="Base64-encoded BGR JPEG frame")
    timestamp_ms: float = Field(default=0.0)


class HealthResponse(BaseModel):
    status: str
    version: str
    detector: str
    zones: list[str]


class ZonesResponse(BaseModel):
    zones: list[dict]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return HealthResponse(
        status="ok",
        version=app.version,
        detector=_pipeline.detector.model_name,
        zones=[z.name for z in DEFAULT_ZONES],
    )


@app.get("/zones", response_model=ZonesResponse, tags=["Config"])
async def list_zones() -> ZonesResponse:
    return ZonesResponse(zones=[
        {"name": z.name, "polygon": z.polygon} for z in DEFAULT_ZONES
    ])


@app.post("/analyze/frame", response_model=FrameAnalytics, tags=["Inference"])
async def analyze_frame(request: FrameRequest) -> FrameAnalytics:
    """
    Analyze a single video frame.

    Submit a base64-encoded BGR JPEG. Returns full FrameAnalytics
    including tracked objects, zone events, and occupancy counts.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    try:
        img_bytes = base64.b64decode(request.frame_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid frame data: {exc}") from exc

    if not _pipeline._running:
        _pipeline.start()

    return _pipeline.process_frame(frame, timestamp_ms=request.timestamp_ms)


@app.post("/analyze/video", response_model=StreamSummary, tags=["Inference"])
async def analyze_video(file: UploadFile = File(...)) -> StreamSummary:
    """
    Process an uploaded video file end-to-end.

    Returns aggregate StreamSummary with occupancy stats, dwell times,
    and zone entry counts across the full video.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    tmp_path = Path(f"/tmp/{file.filename}")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        _pipeline.start()
        summary = _pipeline.run_video(input_path=tmp_path)
        return summary
    except Exception as exc:
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
