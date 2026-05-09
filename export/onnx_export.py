"""
onnx_export.py — Export YOLOv8 model to ONNX for edge deployment.

Exported models run on:
  - CPU (any platform)
  - NVIDIA Jetson (TensorRT via ONNX)
  - Raspberry Pi 5 (ONNX Runtime)
  - Apple Silicon (CoreML via ONNX)

Usage
-----
    python -m export.onnx_export --model yolov8n.pt --output models/yolov8n.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str | None = None,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    """
    Export a YOLOv8 model to ONNX format.

    Parameters
    ----------
    model_path   : Path to .pt weights file
    output_path  : Output .onnx path (defaults to same dir as model_path)
    imgsz        : Input image size (square)
    opset        : ONNX opset version
    simplify     : Apply onnx-simplifier for inference speed
    dynamic      : Enable dynamic batch size axis

    Returns
    -------
    Path to the exported .onnx file
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("ultralytics not installed: pip install ultralytics") from exc

    model = YOLO(model_path)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )

    exported_path = Path(str(exported))
    logger.info("ONNX export complete: %s (%.1f MB)", exported_path, exported_path.stat().st_size / 1e6)
    return exported_path


def validate_onnx(onnx_path: str) -> bool:
    """Quick structural validation of an exported ONNX model."""
    try:
        import onnx  # type: ignore[import]
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("ONNX model validation passed: %s", onnx_path)
        return True
    except ImportError:
        logger.warning("onnx package not installed — skipping validation")
        return True
    except Exception as exc:
        logger.error("ONNX validation failed: %s", exc)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    out = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
    )
    validate_onnx(str(out))
    print(f"Exported to: {out}")
