#!/usr/bin/env python3
"""Collect and clean public Indian emergency-vehicle image candidates.

This is intentionally a *candidate* scraper: public image search results carry
their own licences and terms. Review the produced manifest before redistributing
the dataset or training a commercial model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import cv2
import imagehash
import numpy as np
import requests
try:  # Package was renamed from duckduckgo-search to ddgs.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - compatibility with older Colab images
    from duckduckgo_search import DDGS
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False

CLASSES = {
    "indian_ambulance": [
        "Indian 108 ambulance Force Traveller", "India Maruti Omni ambulance",
        "Indian ambulance emergency vehicle traffic",
        "Indian ambulance highway ambulance vehicle", "Indian private hospital ambulance van",
        "India ambulance rear view street", "Indian ambulance side view vehicle",
    ],
    "indian_police_vehicle": [
        "Indian police PCR Bolero vehicle", "India police Scorpio patrol car",
        "Indian police Ertiga beacon vehicle",
        "Indian police PCR van road", "Indian police Bolero gypsy patrol",
        "Indian police vehicle red blue beacon", "Indian police response vehicle traffic",
    ],
    "indian_fire_tender": [
        "Indian fire tender fire brigade truck", "India fire rescue truck",
        "Indian fire engine emergency vehicle",
        "Indian fire tender road fire brigade", "Indian municipal fire truck",
        "Indian fire rescue tender vehicle", "Indian fire engine street",
        "Indian fire brigade appliance truck", "Delhi fire tender vehicle",
        "Maharashtra fire tender truck", "Indian fire services water tender",
        "Indian fire brigade rescue vehicle", "Indian fire emergency truck road",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape, validate, and deduplicate image candidates.")
    p.add_argument("--output", type=Path, default=Path("dataset/clean"))
    p.add_argument("--target-per-class", type=int, default=150, help="Recommended: 120-200")
    p.add_argument("--max-results-per-query", type=int, default=90)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--min-size", type=int, default=300)
    p.add_argument("--phash-distance", type=int, default=6,
                   help="Reject perceptual hashes within this Hamming distance.")
    p.add_argument("--timeout", type=float, default=15)
    p.add_argument("--delay", type=float, default=0.2, help="Delay between search queries.")
    p.add_argument("--progress", action="store_true", help="Show a per-download progress bar.")
    p.add_argument("--class", dest="classes", action="append", choices=CLASSES,
                   help="May be supplied repeatedly; default is all classes.")
    return p.parse_args()


def search_urls(query: str, limit: int) -> list[dict[str, str]]:
    """Return URL/source-page pairs, accepting both current DDGS APIs."""
    records: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.images(query, max_results=limit, safesearch="moderate"):
            url = item.get("image")
            if url:
                records.append({"url": url, "source": item.get("url", ""), "query": query})
    return records


def download_candidate(record: dict[str, str], timeout: float) -> tuple[dict[str, str], bytes | None]:
    try:
        response = requests.get(record["url"], timeout=timeout,
                                headers={"User-Agent": "Mozilla/5.0 Netra-One research scraper"})
        response.raise_for_status()
        if len(response.content) > 20_000_000:
            return record, None
        return record, response.content
    except requests.RequestException:
        return record, None


def validate_and_normalize(payload: bytes, min_size: int) -> tuple[Image.Image, imagehash.ImageHash] | None:
    """Decode via Pillow and OpenCV, then return a normalized RGB image and pHash."""
    try:
        with Image.open(io.BytesIO(payload)) as im:
            im.verify()
        with Image.open(io.BytesIO(payload)) as im:
            rgb = im.convert("RGB")
            if min(rgb.size) < min_size:
                return None
            # OpenCV decode provides a second independent corruption/readability check.
            encoded = np.frombuffer(payload, dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[0] < min_size or decoded.shape[1] < min_size:
                return None
            return rgb.copy(), imagehash.phash(rgb)
    except (UnidentifiedImageError, OSError, ValueError, cv2.error):
        return None


def is_near_duplicate(candidate: imagehash.ImageHash, known: list[imagehash.ImageHash], threshold: int) -> bool:
    return any(candidate - previous <= threshold for previous in known)


def reconcile_manifest(records: list[dict[str, str]], output: Path) -> list[dict[str, str]]:
    """Keep one provenance row per active image and retain known source metadata."""
    active = {
        (class_name, path.name): path
        for class_name in CLASSES
        for path in (output / class_name).glob("*.jpg")
    }
    by_image: dict[tuple[str, str], dict[str, str]] = {}
    for row in records:
        key = (row.get("class", ""), Path(row.get("file", "")).name)
        if key in active:
            current = by_image.get(key, {})
            by_image[key] = {field: row.get(field, "") or current.get(field, "") for field in
                             ("class", "file", "image_url", "source_page", "query", "phash")}
    result: list[dict[str, str]] = []
    for key, path in sorted(active.items()):
        row = by_image.get(key, {})
        row["class"] = key[0]
        row["file"] = path.as_posix()
        if not row.get("phash"):
            try:
                with Image.open(path) as image:
                    row["phash"] = str(imagehash.phash(image.convert("RGB")))
            except (UnidentifiedImageError, OSError):
                row["phash"] = ""
        for field in ("image_url", "source_page", "query"):
            row.setdefault(field, "")
        result.append(row)
    return result


def collect_class(class_name: str, args: argparse.Namespace, manifest: list[dict[str, str]]) -> int:
    destination = args.output / class_name
    destination.mkdir(parents=True, exist_ok=True)
    known_hashes: list[imagehash.ImageHash] = []
    manifested = {(row.get("class", ""), Path(row.get("file", "")).name) for row in manifest}
    existing_files = sorted(destination.glob("*.jpg"))
    for existing in existing_files:
        try:
            with Image.open(existing) as image:
                fingerprint = imagehash.phash(image.convert("RGB"))
            known_hashes.append(fingerprint)
            if (class_name, existing.name) not in manifested:
                manifest.append({
                    "class": class_name, "file": existing.as_posix(), "image_url": "",
                    "source_page": "", "query": "", "phash": str(fingerprint),
                })
        except (UnidentifiedImageError, OSError):
            logging.warning("Ignoring unreadable pre-existing file: %s", existing)
    accepted = len(existing_files)
    if accepted >= args.target_per_class:
        logging.info("%s already has %d files; manifest metadata reconciled.", class_name, accepted)
        return accepted

    candidates: list[dict[str, str]] = []
    for query in CLASSES[class_name]:
        try:
            candidates.extend(search_urls(query, args.max_results_per_query))
        except Exception as exc:  # Search backends change often; continue with the other queries.
            logging.warning("Search failed for %r: %s", query, exc)
        time.sleep(args.delay)
    # Preserve order while removing repeat URLs.
    seen_urls: set[str] = set()
    candidates = [r for r in candidates if not (r["url"] in seen_urls or seen_urls.add(r["url"]))]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_candidate, record, args.timeout) for record in candidates]
        for future in tqdm(as_completed(futures), total=len(futures), desc=class_name, disable=not args.progress):
            if accepted >= args.target_per_class:
                continue
            record, payload = future.result()
            if payload is None:
                continue
            checked = validate_and_normalize(payload, args.min_size)
            if checked is None:
                continue
            image, fingerprint = checked
            if is_near_duplicate(fingerprint, known_hashes, args.phash_distance):
                continue
            digest = hashlib.sha256(payload).hexdigest()[:16]
            image_path = destination / f"{class_name}_{accepted:04d}_{digest}.jpg"
            image.save(image_path, "JPEG", quality=95, optimize=True)
            known_hashes.append(fingerprint)
            manifest.append({
                "class": class_name, "file": str(image_path.as_posix()), "image_url": record["url"],
                "source_page": record["source"], "query": record["query"], "phash": str(fingerprint),
            })
            manifested.add((class_name, image_path.name))
            accepted += 1
    return accepted


def main() -> None:
    args = parse_args()
    if not 120 <= args.target_per_class <= 200:
        logging.warning("Target %d is outside the brief's recommended 120-200 range.", args.target_per_class)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    manifest: list[dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            manifest.extend(csv.DictReader(handle))
    totals = {name: collect_class(name, args, manifest) for name in (args.classes or list(CLASSES))}
    manifest = reconcile_manifest(manifest, args.output)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "file", "image_url", "source_page", "query", "phash"])
        writer.writeheader()
        writer.writerows(manifest)
    logging.info("Completed: %s. Attribution/review manifest: %s", totals, manifest_path)
    if any(value < args.target_per_class for value in totals.values()):
        logging.warning("Some classes fell short. Re-run with more queries/results; do not pad with duplicates.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
