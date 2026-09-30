#!/usr/bin/env python3
"""Bootstrap YOLO labels with Grounding DINO and create a stratified dataset.

The detector produces *provisional* labels. Inspect the `--review-dir` images
before training, especially when signs, toy vehicles, or vehicle interiors are
present. A `--min-score` around 0.35 is deliberately conservative.
"""
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

CLASS_NAMES = ["indian_ambulance", "indian_police_vehicle", "indian_fire_tender"]
PROMPTS = [
    "Indian ambulance emergency vehicle",
    "Indian police patrol vehicle",
    "Indian fire tender fire truck",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box: tuple[float, float, float, float]  # xyxy pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding-DINO label bootstrapper and YOLO dataset splitter.")
    parser.add_argument("--input", type=Path, default=Path("dataset/clean"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--review-dir", type=Path, default=Path("dataset/review"))
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="0 labels every candidate image")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated images/labels")
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return requested


def source_images(input_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = input_dir / class_name
        if not class_dir.exists():
            continue
        found.extend((class_id, path) for path in sorted(class_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS)
    return found


def detect(
    image: Image.Image, processor: AutoProcessor, model: AutoModelForZeroShotObjectDetection,
    device: str, class_id: int, min_score: float, text_threshold: float, nms_iou: float,
) -> list[Detection]:
    # Each source folder represents one requested class. A focused prompt avoids
    # accidental cross-class labels from a multi-class caption.
    inputs = processor(images=image, text=[PROMPTS[class_id]], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=min_score, text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]
    raw: list[Detection] = []
    for box, score in zip(result["boxes"].detach().cpu(), result["scores"].detach().cpu()):
        x1, y1, x2, y2 = map(float, box.tolist())
        if x2 > x1 and y2 > y1:
            raw.append(Detection(class_id, float(score), (x1, y1, x2, y2)))
    kept: list[Detection] = []
    for class_id in range(len(CLASS_NAMES)):
        same = [item for item in raw if item.class_id == class_id]
        if not same:
            continue
        xywh = [[d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]] for d in same]
        indices = cv2.dnn.NMSBoxes(xywh, [d.score for d in same], min_score, nms_iou)
        for index in np.array(indices).reshape(-1).tolist() if len(indices) else []:
            kept.append(same[index])
    return kept


def yolo_lines(detections: list[Detection], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for item in detections:
        x1, y1, x2, y2 = item.box
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        x_center, y_center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
        box_width, box_height = ((x2 - x1) / width, (y2 - y1) / height)
        values = (x_center, y_center, box_width, box_height)
        if all(0.0 <= value <= 1.0 for value in values):
            lines.append(f"{item.class_id} " + " ".join(f"{value:.6f}" for value in values))
    return lines


def save_review(image: Image.Image, detections: list[Detection], destination: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for item in detections:
        draw.rectangle(item.box, outline=(0, 255, 0), width=3)
        draw.text((item.box[0], max(0, item.box[1] - 15)), f"{CLASS_NAMES[item.class_id]} {item.score:.2f}", fill=(0, 255, 0))
    canvas.save(destination, "JPEG", quality=92)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    device = select_device(args.device)
    candidates = source_images(args.input)
    if not candidates:
        raise FileNotFoundError(f"No images found in {args.input}. Run scrape_and_clean.py first.")
    if args.max_images:
        candidates = candidates[:args.max_images]
    generated_roots = [args.output / "images", args.output / "labels"]
    if not args.overwrite and any(any(root.rglob("*.*")) for root in generated_roots if root.exists()):
        raise FileExistsError(f"Generated files already exist under {args.output}; pass --overwrite to rebuild them.")
    if args.overwrite:
        for root in generated_roots:
            if root.exists():
                shutil.rmtree(root)
    print(f"Loading {args.model} on {device}; labeling {len(candidates)} images.")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    args.review_dir.mkdir(parents=True, exist_ok=True)
    labelled: dict[int, list[tuple[Path, list[str]]]] = {index: [] for index in range(len(CLASS_NAMES))}
    for source_class, path in tqdm(candidates, desc="Auto-labeling"):
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            detections = detect(image, processor, model, device, source_class, args.min_score, args.text_threshold, args.nms_iou)
            lines = yolo_lines(detections, image.width, image.height)
            if not lines:
                continue  # Do not turn an uncertain candidate into a negative example.
            # Stratify by the scrape category, while labels retain every detected object class.
            labelled[source_class].append((path, lines))
            save_review(image, detections, args.review_dir / f"{path.stem}.jpg")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Skipping {path}: {exc}")
    randomizer = random.Random(args.seed)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for class_id, records in labelled.items():
        randomizer.shuffle(records)
        val_count = max(1, round(len(records) * args.val_ratio)) if len(records) > 1 else 0
        for index, (source, lines) in enumerate(records):
            split = "val" if index < val_count else "train"
            target_stem = f"{CLASS_NAMES[class_id]}_{source.stem}"
            target_image = args.output / "images" / split / f"{target_stem}.jpg"
            target_label = args.output / "labels" / split / f"{target_stem}.txt"
            with Image.open(source) as opened:
                opened.convert("RGB").save(target_image, "JPEG", quality=95)
            target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[split] += 1
    print(f"Dataset ready: {counts}. Review overlays in {args.review_dir} before training.")


if __name__ == "__main__":
    main()
