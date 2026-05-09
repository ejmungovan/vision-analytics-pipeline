"""
streamlit_app.py — Vision Analytics Pipeline — Interactive Demo Dashboard

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.demo_gen import (
    DEMO_W, DEMO_H, TOTAL_DEMO_FRAMES,
    ZONE_POLYGONS, DemoDetector,
    generate_frame, generate_detections,
)
from core.pipeline import VisionPipeline, PipelineConfig
from core.tracker import KalmanBoxTracker, SORTTracker
from core.zone_analyzer import Zone, ZoneAnalyzer
from schemas.events import FrameAnalytics, ZoneEvent


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Analytics Pipeline",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark dashboard theme */
  .stApp { background-color: #0e1117; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #1c1f26;
    border: 1px solid #2d3142;
    border-radius: 10px;
    padding: 16px 20px;
  }
  [data-testid="stMetric"] label { color: #8b92a5 !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #e8eaf0 !important; font-size: 2rem !important; font-weight: 700; }
  [data-testid="stMetricDelta"] { color: #4caf8a !important; }

  /* Event log rows */
  .event-enter { color: #4caf8a; font-weight: 600; }
  .event-exit  { color: #e07b54; font-weight: 600; }
  .event-dwell { color: #f0c040; font-weight: 600; }
  .event-row   { font-family: 'Courier New', monospace; font-size: 0.82rem; padding: 3px 0; border-bottom: 1px solid #1c1f26; }

  /* Section headers */
  .section-header {
    color: #8b92a5;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 4px;
  }

  /* Video frame border */
  [data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #2d3142;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] { background-color: #13161e; border-right: 1px solid #2d3142; }

  /* Progress bar accent */
  [data-testid="stProgressBar"] > div > div { background-color: #4c8af0; }

  /* Hide Streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📹 Vision Analytics")
    st.markdown("<div class='section-header'>Mode</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Input source",
        ["🎬 Demo (synthetic)", "📂 Upload video"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("<div class='section-header'>Tracker Settings</div>", unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence threshold", 0.1, 0.95, 0.4, 0.05)
    max_age = st.slider("Track max age (frames)", 1, 10, 3)
    min_hits = st.slider("Min hits to confirm track", 1, 5, 2)
    dwell_thresh = st.slider("Dwell threshold (frames)", 5, 60, 20)

    st.divider()
    st.markdown("<div class='section-header'>Privacy</div>", unsafe_allow_html=True)
    anonymize = st.toggle("Anonymize detections", value=False,
                           help="Applies Gaussian blur over all tracked persons")

    st.divider()
    st.markdown("<div class='section-header'>Playback</div>", unsafe_allow_html=True)
    playback_fps = st.slider("Demo speed (fps)", 5, 60, 24)


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "running": False,
        "frame_id": 0,
        "analytics_log": [],
        "event_log": deque(maxlen=50),
        "occupancy_history": {z: [] for z in ZONE_POLYGONS},
        "unique_tracks": set(),
        "total_entries": {z: 0 for z in ZONE_POLYGONS},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _reset_state() -> None:
    keys = ["running", "frame_id", "analytics_log", "event_log",
            "occupancy_history", "unique_tracks", "total_entries"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    KalmanBoxTracker._id_counter = 0
    _init_state()

_init_state()


# ── Build pipeline ─────────────────────────────────────────────────────────────

@st.cache_resource
def _get_zones() -> list[Zone]:
    return [Zone(name=name, polygon=poly) for name, poly in ZONE_POLYGONS.items()]


def _make_tracker() -> SORTTracker:
    KalmanBoxTracker._id_counter = 0
    return SORTTracker(max_age=max_age, min_hits=min_hits, iou_threshold=0.3)


def _make_zone_analyzer() -> ZoneAnalyzer:
    return ZoneAnalyzer(zones=_get_zones(), dwell_threshold_frames=dwell_thresh)


# ── Draw annotated frame ───────────────────────────────────────────────────────

def _draw_analytics(
    frame: np.ndarray,
    analytics: FrameAnalytics,
    zones: list[Zone],
    do_anonymize: bool,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Zone overlays
    for zone in zones:
        pts = np.array([[int(x), int(y)] for x, y in zone.polygon], dtype=np.int32)
        occ = analytics.zone_occupancy.get(zone.name, 0)
        color = (100, 220, 100) if occ > 0 else (80, 90, 110)
        cv2.polylines(out, [pts], True, color, 2)
        if pts.size:
            cx_z = int(pts[:, 0].mean())
            cv2.putText(out, f"{zone.name} [{occ}]", (cx_z - 45, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Tracked objects
    for obj in analytics.tracked_objects:
        x1, y1 = max(0, int(obj.bbox.x1)), max(0, int(obj.bbox.y1))
        x2, y2 = min(w, int(obj.bbox.x2)), min(h, int(obj.bbox.y2))

        if do_anonymize:
            region = out[y1:y2, x1:x2]
            if region.size:
                out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (31, 31), 0)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 200, 255), 2)
            label = f"#{obj.track_id}"
            cv2.putText(out, label, (x1 + 4, max(y1 + 16, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1, cv2.LINE_AA)

    # HUD
    hud = (f"Frame {analytics.frame_id:04d}  |  "
           f"Tracks: {len(analytics.tracked_objects)}  |  "
           f"{analytics.processing_ms:.1f} ms")
    cv2.putText(out, hud, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 145, 160), 1, cv2.LINE_AA)
    return out


# ── Main layout ────────────────────────────────────────────────────────────────

st.markdown("# 📹 Vision Analytics Pipeline")
st.markdown(
    "<span style='color:#8b92a5;font-size:0.85rem'>"
    "Real-time multi-object tracking · Zone analytics · Privacy-first"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

# Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
metric_tracks    = m1.empty()
metric_occupancy = m2.empty()
metric_entries   = m3.empty()
metric_fps       = m4.empty()
metric_frames    = m5.empty()

def _update_metrics(
    unique: int,
    occ: dict[str, int],
    entries: dict[str, int],
    fps: float,
    frame: int,
) -> None:
    total_occ = sum(occ.values())
    total_entries = sum(entries.values())
    metric_tracks.metric("Unique Tracks", unique)
    metric_occupancy.metric("Current Occupancy", total_occ)
    metric_entries.metric("Total Zone Entries", total_entries)
    metric_fps.metric("Processing FPS", f"{fps:.0f}")
    metric_frames.metric("Frames Processed", frame)

_update_metrics(0, {}, {}, 0.0, 0)

st.divider()

# Main columns: video left, analytics right
col_video, col_right = st.columns([3, 2], gap="large")

with col_video:
    st.markdown("<div class='section-header'>Live Feed</div>", unsafe_allow_html=True)
    video_placeholder = st.empty()
    progress_bar = st.empty()

    # Placeholder frame
    blank = np.full((DEMO_H, DEMO_W, 3), (20, 22, 28), dtype=np.uint8)
    cv2.putText(blank, "Press  ▶  Run Demo  to start",
                (DEMO_W // 2 - 160, DEMO_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 85, 100), 1, cv2.LINE_AA)
    video_placeholder.image(blank, channels="BGR", use_container_width=True)

with col_right:
    tab_zones, tab_events, tab_summary = st.tabs(["Zone Occupancy", "Event Log", "Summary"])

    with tab_zones:
        st.markdown("<div class='section-header'>Occupancy over time</div>",
                    unsafe_allow_html=True)
        occ_chart = st.empty()

    with tab_events:
        st.markdown("<div class='section-header'>Recent events</div>",
                    unsafe_allow_html=True)
        event_container = st.empty()

    with tab_summary:
        st.markdown("<div class='section-header'>Session statistics</div>",
                    unsafe_allow_html=True)
        summary_container = st.empty()


def _render_event_log(log: deque) -> None:
    if not log:
        event_container.markdown("*Waiting for events...*")
        return
    rows = []
    for ev in reversed(list(log)[-20:]):
        cls = f"event-{ev['type']}"
        icon = {"enter": "→", "exit": "←", "dwell": "⏱"}.get(ev["type"], "·")
        rows.append(
            f"<div class='event-row'>"
            f"<span class='{cls}'>{icon} {ev['type'].upper()}</span>"
            f" &nbsp; <span style='color:#c0c8d8'>#{ev['track_id']}</span>"
            f" &nbsp; <span style='color:#8b92a5'>{ev['zone']}</span>"
            f" &nbsp; <span style='color:#555'>f{ev['frame']:04d}</span>"
            f"</div>"
        )
    event_container.markdown("\n".join(rows), unsafe_allow_html=True)


def _render_occ_chart(history: dict[str, list[int]]) -> None:
    import pandas as pd
    max_len = max((len(v) for v in history.values()), default=0)
    if max_len == 0:
        return
    data = {k: v[-120:] for k, v in history.items()}
    df = pd.DataFrame(data)
    occ_chart.line_chart(df, height=200)


def _render_summary(
    unique: int,
    total_entries: dict[str, int],
    occ_history: dict[str, list[int]],
    n_frames: int,
) -> None:
    if n_frames == 0:
        summary_container.info("Run the demo to see summary stats.")
        return
    avg_occ = {
        z: (sum(v) / len(v) if v else 0.0)
        for z, v in occ_history.items()
    }
    lines = [
        f"**Frames processed:** {n_frames}",
        f"**Unique tracks:** {unique}",
        "",
        "**Zone entries:**",
    ]
    for z, n in total_entries.items():
        lines.append(f"- `{z}`: {n} entries")
    lines += ["", "**Avg occupancy:**"]
    for z, a in avg_occ.items():
        lines.append(f"- `{z}`: {a:.2f}")
    summary_container.markdown("\n".join(lines))


# ── Controls ───────────────────────────────────────────────────────────────────

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])

run_btn   = ctrl_col1.button("▶  Run Demo",  type="primary",  use_container_width=True)
reset_btn = ctrl_col2.button("↺  Reset",     type="secondary", use_container_width=True)

if reset_btn:
    _reset_state()
    st.rerun()

# ── Demo execution loop ────────────────────────────────────────────────────────

if run_btn and not st.session_state.running:
    st.session_state.running = True
    _reset_state()

    tracker = _make_tracker()
    zone_analyzer = _make_zone_analyzer()
    zones = _get_zones()
    fps_times: deque = deque(maxlen=30)

    for fid in range(TOTAL_DEMO_FRAMES):
        t0 = time.perf_counter()

        # Generate synthetic frame + detections
        raw_frame = generate_frame(fid)
        detections = generate_detections(fid)

        # Filter by confidence
        detections = [d for d in detections if d.confidence >= conf_threshold]

        # Track
        tracked = tracker.update(detections, frame_id=fid)
        for obj in tracked:
            st.session_state.unique_tracks.add(obj.track_id)

        # Zone analytics
        ts = float(fid * 33.33)
        zone_events = zone_analyzer.update(tracked, frame_id=fid, timestamp_ms=ts)
        occupancy = zone_analyzer.current_occupancy()

        # Update occupancy history
        for z in ZONE_POLYGONS:
            st.session_state.occupancy_history[z].append(occupancy.get(z, 0))

        # Record zone events
        for ev in zone_events:
            st.session_state.event_log.append({
                "type": ev.event.value,
                "track_id": ev.track_id,
                "zone": ev.zone_name,
                "frame": fid,
            })
            if ev.event == ZoneEvent.ENTER:
                st.session_state.total_entries[ev.zone_name] = (
                    st.session_state.total_entries.get(ev.zone_name, 0) + 1
                )

        # FPS tracking
        elapsed = time.perf_counter() - t0
        fps_times.append(elapsed)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

        # Build FrameAnalytics (for draw fn)
        from schemas.events import FrameAnalytics as FA, TrackedObject
        analytics = FA(
            frame_id=fid,
            timestamp_ms=ts,
            tracked_objects=tracked,
            zone_events=zone_events,
            zone_occupancy=occupancy,
            total_detections=len(detections),
            processing_ms=round(elapsed * 1000, 2),
        )

        # Draw + display
        annotated = _draw_analytics(raw_frame, analytics, zones, anonymize)
        video_placeholder.image(annotated, channels="BGR", use_container_width=True)

        # Update progress
        progress_bar.progress(
            (fid + 1) / TOTAL_DEMO_FRAMES,
            text=f"Frame {fid + 1} / {TOTAL_DEMO_FRAMES}"
        )

        # Update metrics
        _update_metrics(
            len(st.session_state.unique_tracks),
            occupancy,
            st.session_state.total_entries,
            fps,
            fid + 1,
        )

        # Update charts & logs
        _render_event_log(st.session_state.event_log)
        _render_occ_chart(st.session_state.occupancy_history)
        _render_summary(
            len(st.session_state.unique_tracks),
            st.session_state.total_entries,
            st.session_state.occupancy_history,
            fid + 1,
        )

        # Speed control
        target_interval = 1.0 / playback_fps
        sleep_time = max(0.0, target_interval - elapsed)
        time.sleep(sleep_time)

    st.session_state.running = False
    progress_bar.progress(1.0, text="✅ Demo complete")
    st.success(f"Session complete — {len(st.session_state.unique_tracks)} unique tracks detected.")


# ── Upload mode ────────────────────────────────────────────────────────────────

elif "📂 Upload" in mode:
    st.info(
        "Upload a `.mp4` video and the pipeline will process it frame-by-frame. "
        "Configure zones and tracker settings in the sidebar.",
        icon="📂",
    )
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded:
        tmp = Path(f"/tmp/{uploaded.name}")
        tmp.write_bytes(uploaded.read())

        from core.detector import StubDetector
        detector = StubDetector()  # swap for YOLOv8Detector("yolov8n.pt") with real model
        pipeline = VisionPipeline(
            detector=detector,
            zones=_get_zones(),
            config=PipelineConfig(
                confidence_threshold=conf_threshold,
                max_tracker_age=max_age,
                min_tracker_hits=min_hits,
                dwell_threshold_frames=dwell_thresh,
                anonymize=anonymize,
                headless=True,
            ),
        )
        with st.spinner("Processing video..."):
            pipeline.start()
            cap = cv2.VideoCapture(str(tmp))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fa = pipeline.process_frame(frame)
                for ev in fa.zone_events:
                    st.session_state.event_log.append({
                        "type": ev.event.value,
                        "track_id": ev.track_id,
                        "zone": ev.zone_name,
                        "frame": fa.frame_id,
                    })
                for z in ZONE_POLYGONS:
                    st.session_state.occupancy_history[z].append(
                        fa.zone_occupancy.get(z, 0)
                    )
                for obj in fa.tracked_objects:
                    st.session_state.unique_tracks.add(obj.track_id)
            cap.release()
            summary = pipeline.stop()
        tmp.unlink(missing_ok=True)

        st.success(f"Done — {summary.total_frames} frames, {summary.total_unique_tracks} unique tracks")
        _render_event_log(st.session_state.event_log)
        _render_occ_chart(st.session_state.occupancy_history)
        _render_summary(
            summary.total_unique_tracks,
            summary.total_zone_entries,
            st.session_state.occupancy_history,
            summary.total_frames,
        )
