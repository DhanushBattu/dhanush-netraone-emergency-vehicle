#!/usr/bin/env python3
"""Fail fast if a generated YOLO dataset contains invalid labels."""
from __future__ import annotations

import argparse
from pathlib import Path

CLASS_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO labels are paired and normalized.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    labels_seen = 0
    for split in ("train", "val"):
        images = args.dataset / "images" / split
        labels = args.dataset / "labels" / split
        for image in images.glob("*"):
            label = labels / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"Missing label for {image}")
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                pieces = line.split()
                if len(pieces) != 5:
                    errors.append(f"{label}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(pieces[0])
                    values = [float(value) for value in pieces[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_number}: fields must be numeric")
                    continue
                if not 0 <= class_id < CLASS_COUNT:
                    errors.append(f"{label}:{line_number}: invalid class {class_id}")
                x, y, width, height = values
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    errors.append(f"{label}:{line_number}: values outside YOLO bounds")
                labels_seen += 1
    if errors:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(errors[:50]))
    print(f"Dataset valid: {labels_seen} object labels checked.")


if __name__ == "__main__":
    main()
