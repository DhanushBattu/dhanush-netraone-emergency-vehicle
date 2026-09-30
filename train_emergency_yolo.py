#!/usr/bin/env python3
"""Train, evaluate, and export a YOLOv8n emergency-vehicle detector."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n and export best.pt + best.onnx.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint or model YAML")
    parser.add_argument("--epochs", type=int, default=35, choices=range(25, 51))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 selects automatic batch size")
    parser.add_argument("--device", default="auto", help="0 for Colab T4, cpu for CPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path("runs/emergency"))
    parser.add_argument("--name", default="yolov8n")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    return parser.parse_args()


def metric_value(metrics: object, suffix: str) -> float | None:
    results = getattr(metrics, "results_dict", {})
    for key, value in results.items():
        if key.endswith(suffix):
            return float(value)
    return None


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing {args.data}. Run pipeline/auto_label.py and review labels first.")
    device = 0 if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        workers=args.workers, project=str(args.project), name=args.name, patience=args.patience,
        seed=args.seed, pretrained=True, plots=True, exist_ok=True,
    )
    run_dir = args.project / args.name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training completed but no checkpoint was found at {best_pt}")
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(data=str(args.data), imgsz=args.imgsz, device=device, plots=True)
    exported = Path(best_model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=True))
    destination_pt = args.weights_dir / "best.pt"
    destination_onnx = args.weights_dir / "best.onnx"
    shutil.copy2(best_pt, destination_pt)
    shutil.copy2(exported, destination_onnx)
    class_names = [str(best_model.names[index]) for index in range(len(best_model.names))]
    per_class_ap50 = getattr(metrics.box, "ap50", metrics.box.maps)
    report = {
        "model": "YOLOv8n", "epochs_requested": args.epochs, "imgsz": args.imgsz,
        "precision": metric_value(metrics, "metrics/precision(B)"),
        "recall": metric_value(metrics, "metrics/recall(B)"),
        "map50": metric_value(metrics, "metrics/mAP50(B)"),
        "map50_95": metric_value(metrics, "metrics/mAP50-95(B)"),
        "per_class_map50": {name: float(value) for name, value in zip(class_names, per_class_ap50.tolist())},
        "per_class_map50_95": {name: float(value) for name, value in zip(class_names, metrics.box.maps.tolist())},
        "checkpoint": str(destination_pt), "onnx": str(destination_onnx), "run_dir": str(run_dir),
    }
    report_path = args.weights_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Confusion matrix and PR curves: {run_dir}; deployable weights: {args.weights_dir}")


if __name__ == "__main__":
    main()
