# Changelog

## [1.0.0] — 2025-05-09

### feat: initial production scaffold

- `core/detector.py` — BaseDetector ABC, YOLOv8Detector (lazy-load), StubDetector
- `core/tracker.py` — Full SORT implementation from scratch (Kalman + Hungarian)
- `core/zone_analyzer.py` — Polygon zones with ENTER/EXIT/DWELL event emission
- `core/anonymizer.py` — Privacy-first frame masking (blur / pixelate / blackout)
- `core/pipeline.py` — VisionPipeline orchestrator (video file + frame-by-frame API)
- `schemas/events.py` — Typed Pydantic schemas for all pipeline outputs
- `api/main.py` — FastAPI: POST /analyze/frame, POST /analyze/video, GET /health
- `export/onnx_export.py` — YOLOv8 → ONNX export utility for edge deployment

### fix: StubDetector empty-list falsy bug
- `fixed_detections or [...]` replaced with `fixed_detections if not None else [...]`
- Prevented empty detection lists from falling through to default stub detections

### test: 44-test suite (zero external dependencies)
- `test_tracker.py` — BoundingBox IoU math, Kalman filter, SORT assignment
- `test_zone_analyzer.py` — Ray-casting, zone enter/exit/dwell events
- `test_pipeline.py` — Full pipeline integration with StubDetector

### ci: GitHub Actions pipeline
- Matrix: Python 3.11 + 3.12
- Ruff lint, pytest coverage, Codecov upload

## [1.1.0] — 2025-05-09

### feat: Streamlit monitoring dashboard

- `app/streamlit_app.py` — interactive demo dashboard
  - Live video frame with bounding box and zone overlays
  - Zone occupancy line chart (rolling 120-frame window)
  - Real-time ENTER/EXIT/DWELL event log (colour-coded)
  - Session summary statistics panel
  - Upload mode for real .mp4 files
  - Sidebar: confidence threshold, tracker params, anonymize toggle, playback FPS
  - Custom dark-theme CSS (professional monitoring aesthetic)

- `app/demo_gen.py` — deterministic synthetic scene generator
  - 3 actors with scripted trajectories through entrance/main_area/exit zones
  - Exercises all event types: ENTER, EXIT, DWELL
  - 240 frames, smoothstep-eased movement, zero dependencies beyond NumPy/OpenCV
  - 473 total synthetic detections; 10 enter, 10 exit, 17 dwell events verified

### chore: add streamlit + pandas to requirements
