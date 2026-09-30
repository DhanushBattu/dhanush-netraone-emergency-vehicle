#!/usr/bin/env python3
"""Run emergency-vehicle inference on a still image or a short traffic video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {
    "indian_ambulance": (30, 220, 30),
    "indian_police_vehicle": (255, 140, 0),
    "indian_fire_tender": (40, 40, 235),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect Indian emergency vehicles and render priority alerts.")
    parser.add_argument("--source", required=True, help="Path or URL to an image/video")
    parser.add_argument("--weights", default="weights/best.pt", help="best.pt or best.onnx")
    parser.add_argument("--conf", type=float, default=0.45, help="Minimum confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default=None, help="0 for GPU, cpu for CPU; default is Ultralytics auto")
    parser.add_argument("--output", type=Path, default=None, help="Output media path; inferred if omitted")
    parser.add_argument("--show", action="store_true", help="Preview frames (desktop only)")
    return parser.parse_args()


def destination(source: str, is_image: bool) -> Path:
    path = Path(source)
    suffix = path.suffix if is_image else ".mp4"
    return Path("test_samples") / f"detected_{path.stem}{suffix}"


def annotate(frame, result) -> tuple[object, list[str]]:
    names = result.names
    detected: list[str] = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            class_name = str(names[int(box.cls[0])])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = COLORS.get(class_name, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            detected.append(class_name)
    if detected:
        unique = ", ".join(sorted(set(detected)))
        text = f"PRIORITY RESCUE VEHICLE DETECTED: {unique.upper()}"
        banner_height = max(48, int(frame.shape[0] * 0.09))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (0, 150, 0), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.putText(frame, text, (12, int(banner_height * 0.66)), cv2.FONT_HERSHEY_SIMPLEX,
                    min(0.72, frame.shape[1] / 1400), (255, 255, 255), 2, cv2.LINE_AA)
    return frame, detected


def run_image(model: YOLO, args: argparse.Namespace, output: Path) -> int:
    frame = cv2.imread(args.source)
    if frame is None:
        raise ValueError(f"Cannot read image: {args.source}")
    result = model(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
    rendered, detected = annotate(frame, result)
    if not cv2.imwrite(str(output), rendered):
        raise OSError(f"Could not write {output}")
    print(f"Saved {output}. " + (f"Priority alert: {', '.join(sorted(set(detected)))}" if detected else "No emergency vehicle above threshold."))
    if args.show:
        cv2.imshow("Netra-One", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_video(model: YOLO, args: argparse.Namespace, output: Path) -> int:
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot create video: {output}")
    frames, alert_frames = 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            rendered, detected = annotate(frame, result)
            writer.write(rendered)
            frames += 1
            alert_frames += bool(detected)
            if args.show:
                cv2.imshow("Netra-One", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Saved {output}. Alerts on {alert_frames}/{frames} frames.")
    return 0


def main() -> int:
    args = parse_args()
    if not Path(args.weights).exists() and not args.weights.startswith(("http://", "https://")):
        raise FileNotFoundError(f"Weights not found: {args.weights}. Train first or download best.pt into weights/.")
    is_image = Path(args.source).suffix.lower() in IMAGE_SUFFIXES
    output = args.output or destination(args.source, is_image)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    return run_image(model, args, output) if is_image else run_video(model, args, output)


if __name__ == "__main__":
    raise SystemExit(main())
