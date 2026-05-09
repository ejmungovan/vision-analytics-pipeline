# 📹 Vision Analytics Pipeline

[![CI](https://github.com/ejmungovan/vision-analytics-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ejmungovan/vision-analytics-pipeline/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time **multi-object tracking + zone analytics** system. Drop a camera feed in, get structured data about who is where, for how long, and when they move — without storing a single identifiable image.

Built for any physical space that needs to understand how people or vehicles move through it: retail stores, airports, factories, hospitals, smart buildings.

---

## Problem Statement

Camera feeds generate enormous volumes of video that operations teams cannot monitor manually. The value isn't in the raw pixels — it's in the structured signal: *how many people entered zone A*, *average dwell time at the checkout*, *when did occupancy peak*. This pipeline extracts that signal in real time, with privacy built into the architecture from the ground up.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        VisionPipeline                            │
│                                                                  │
│  Video / Stream                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐    ┌────────────────────────────────────────┐  │
│  │  BaseDetector│    │  YOLOv8Detector  (production)          │  │
│  │  (abstract)  │───▶│  StubDetector    (testing / offline)   │  │
│  └──────────────┘    └──────────────┬─────────────────────────┘  │
│                                     │  list[Detection]           │
│                                     ▼                            │
│                      ┌──────────────────────────────────────┐    │
│                      │          SORTTracker                  │    │
│                      │  Kalman filter (constant velocity)    │    │
│                      │  Hungarian assignment (scipy)         │    │
│                      │  Track lifecycle: tentative→confirmed │    │
│                      └──────────────┬───────────────────────┘    │
│                                     │  list[TrackedObject]       │
│                                     ▼                            │
│                      ┌──────────────────────────────────────┐    │
│                      │         ZoneAnalyzer                  │    │
│                      │  Polygon containment (ray-casting)    │    │
│                      │  ENTER / EXIT / DWELL events          │    │
│                      │  Occupancy counts per zone            │    │
│                      └──────────────┬───────────────────────┘    │
│                                     │  FrameAnalytics            │
│                                     ▼                            │
│                      ┌──────────────────────────────────────┐    │
│                      │         Anonymizer                    │    │
│                      │  Gaussian blur / pixelate / blackout  │    │
│                      │  Applied before any frame is stored   │    │
│                      └──────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
  REST API (FastAPI)         JSON event log          StreamSummary
  POST /analyze/frame    →   FrameAnalytics          aggregate stats
  POST /analyze/video    →   StreamSummary
```

---

## Technical Stack

| Component | Technology | Notes |
|---|---|---|
| Detection | YOLOv8 (ultralytics) | Swappable via BaseDetector ABC |
| Tracking | SORT (built from scratch) | Kalman filter + Hungarian algorithm |
| Zone logic | Ray-casting algorithm | Pure NumPy, no shapely |
| Privacy | OpenCV Gaussian blur | Applied before any frame storage |
| API | FastAPI + Pydantic v2 | Fully typed request/response |
| Edge export | ONNX Runtime | CPU / Jetson / Raspberry Pi |
| Testing | pytest (44 tests) | Fully offline, no GPU required |
| CI | GitHub Actions | Python 3.11 + 3.12 matrix |

---

## Project Structure

```
vision-analytics-pipeline/
├── core/
│   ├── detector.py       # BaseDetector ABC, YOLOv8Detector, StubDetector
│   ├── tracker.py        # SORT: KalmanBoxTracker + SORTTracker
│   ├── zone_analyzer.py  # Polygon zones, ENTER/EXIT/DWELL events
│   ├── anonymizer.py     # Privacy-preserving frame masking
│   └── pipeline.py       # Top-level orchestrator
├── schemas/
│   └── events.py         # Pydantic schemas: Detection, TrackedObject,
│                         #   ZoneEventRecord, FrameAnalytics, StreamSummary
├── api/
│   └── main.py           # FastAPI: /analyze/frame, /analyze/video, /health
├── export/
│   └── onnx_export.py    # YOLOv8 → ONNX for edge deployment
├── tests/
│   ├── test_tracker.py      # Kalman filter, IoU, SORT assignment
│   ├── test_zone_analyzer.py # Point-in-polygon, zone events
│   └── test_pipeline.py     # Full pipeline integration tests
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/ejmungovan/vision-analytics-pipeline.git
cd vision-analytics-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Optional: enable live YOLOv8 inference
pip install ultralytics
```

---

## Demo — Zero Setup Required

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` — hit **▶ Run Demo** and the full pipeline runs immediately with synthetic actors moving through three zones. No video file, no GPU, no API keys.

**What the demo shows:**
- 3 synthetic actors with distinct trajectories (long-dwell, quick-pass, loiter)
- Real-time ENTER / EXIT / DWELL event log
- Occupancy chart over time per zone
- Per-session summary statistics
- Anonymization toggle (Gaussian blur over all detections)
- Adjustable tracker settings and playback speed

---

## Quickstart

### Process a video file

```python
from core.detector import YOLOv8Detector
from core.pipeline import VisionPipeline, PipelineConfig
from core.zone_analyzer import Zone

zones = [
    Zone(name="entrance", polygon=[(0, 0), (320, 0), (320, 720), (0, 720)]),
    Zone(name="main_floor", polygon=[(320, 0), (1280, 0), (1280, 720), (320, 720)]),
]

pipeline = VisionPipeline(
    detector=YOLOv8Detector("yolov8n.pt"),
    zones=zones,
    config=PipelineConfig(anonymize=True),
)

summary = pipeline.run_video("store_footage.mp4", output_path="annotated.mp4")
print(f"Unique tracks: {summary.total_unique_tracks}")
print(f"Entrance entries: {summary.total_zone_entries['entrance']}")
print(f"Avg dwell (main floor): {summary.avg_dwell_frames_by_zone['main_floor']:.1f} frames")
```

### REST API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Analyze a single frame
curl -X POST http://localhost:8000/analyze/frame \
  -H "Content-Type: application/json" \
  -d '{"frame_b64": "<base64-encoded JPEG>", "timestamp_ms": 1000.0}'

# Upload and analyze a full video
curl -X POST http://localhost:8000/analyze/video \
  -F "file=@store_footage.mp4"
```

### Export to ONNX (edge deployment)

```bash
python -m export.onnx_export --model yolov8n.pt --imgsz 640
# → yolov8n.onnx  (runs on CPU, Jetson, Raspberry Pi)
```

---

## Running Tests

```bash
# Full suite with coverage
pytest tests/ --cov=core --cov=api --cov=schemas --cov-report=term-missing

# Fast: just unit tests
pytest tests/test_tracker.py tests/test_zone_analyzer.py -v

# Lint
ruff check .
```

**No GPU, no model files, no network access required** — all tests use the built-in StubDetector.

---

## Privacy Design

Anonymization is a first-class concern, not a plugin:

1. The `Anonymizer` runs **before** any frame is written to disk or transmitted.
2. Three modes: `BLUR` (Gaussian), `PIXELATE`, `BLACKOUT`.
3. Target class filtering — only persons are anonymized by default; vehicles are not.
4. The event log stores **only structured metadata** (track IDs, timestamps, zone names) — zero image data.

This makes the system compliant-by-default for GDPR/CCPA contexts where video of individuals requires consent or anonymization.

---

## Design Decisions

**Why implement SORT from scratch instead of importing a library?**
To demonstrate understanding of the underlying algorithm — Kalman filter state estimation, Hungarian assignment, and track lifecycle management. The implementation is ~150 lines and fully tested.

**Why pure NumPy ray-casting instead of shapely?**
Fewer dependencies, faster on small polygon counts, and directly testable with synthetic points. For complex GIS-style polygons, shapely would be the right swap.

**Why `BaseDetector` ABC?**
Any detection backend (ONNX Runtime, TFLite, cloud API) can be swapped in by implementing two methods. Tests never touch real model weights.

---

## Roadmap

- [ ] ByteTrack re-ID module (reduce ID switches in crowded scenes)
- [ ] Heatmap generation (dwell density overlay)
- [ ] Configurable zones via YAML/JSON (no code changes)
- [ ] Async frame processing (concurrent detection + tracking)
- [ ] Docker + docker-compose deployment
- [ ] Streamlit dashboard for live monitoring
- [ ] Benchmark suite (FPS, MOTA, ID switch rate)

---

## License

MIT — see [LICENSE](LICENSE)
