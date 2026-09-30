#!/usr/bin/env python3
"""Build an input ZIP with source code and the cleaned candidates for Colab."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = [
    ".gitignore", "README.md", "data.yaml", "requirements.txt", "train_emergency_yolo.py",
    "detect_emergency.py", "preview_server.py", "colab_train.ipynb", "make_colab_bundle.py", "pipeline/__init__.py",
    "pipeline/scrape_and_clean.py", "pipeline/auto_label.py", "pipeline/validate_dataset.py",
    "dataset/clean/manifest.csv", "weights/.gitkeep", "test_samples/.gitkeep",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package this project and clean images for Colab T4 training.")
    parser.add_argument("--output", type=Path, default=Path("netraone_colab_input.zip"))
    parser.add_argument("--include-images", action="store_true", help="Include dataset/clean images and manifest")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in SOURCE_FILES if (ROOT / name).is_file()]
    if args.include_images:
        included = set(files)
        files.extend(path for path in sorted((ROOT / "dataset" / "clean").rglob("*")) if path.is_file() and path not in included)
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required project files: " + ", ".join(missing))
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, Path("netraone") / path.relative_to(ROOT))
    print(f"Created {output} ({output.stat().st_size / (1024**2):.1f} MiB)")
    if not args.include_images:
        print("This ZIP contains source code only. Pass --include-images to package the local cleaned collection.")


if __name__ == "__main__":
    main()
