# Netra-One: Indian Emergency Vehicle Detection

An end-to-end, reproducible YOLOv8n pipeline for detecting three India-specific emergency-responder vehicle categories in traffic imagery:

| ID | YOLO class | Typical target |
|---:|---|---|
| 0 | `indian_ambulance` | 108, Force Traveller, Maruti Omni ambulances |
| 1 | `indian_police_vehicle` | PCR vans, Bolero, Scorpio, Ertiga patrol vehicles |
| 2 | `indian_fire_tender` | Red fire brigade rescue trucks / fire tenders |

The repository does **not** claim unmeasured metrics or include placeholder checkpoints. The source code and cleaned image collection are ready; run training on a Colab T4 to produce the real weights and evaluation figures.

## What the pipeline does

```text
DuckDuckGo/Bing public-image search
       │
       ├─ Pillow + OpenCV readability checks; reject < 300×300
       ├─ perceptual-hash near-duplicate rejection
       ▼
dataset/clean/<class>/ + attribution manifest.csv
       │
       ├─ Grounding DINO zero-shot box proposals
       ├─ confidence threshold + classwise NMS
       ├─ visual review overlays (human quality gate)
       ▼
80/20 stratified YOLO dataset → YOLOv8n training → PT + ONNX
       │
       ▼
image/video inference with a green priority-rescue banner
```

## Quick start — Google Colab T4

1. Build an upload bundle on your computer. It contains the source files and cleaned image set (about 111 MiB):

   ```bash
   py -3.14 make_colab_bundle.py --include-images
   ```

2. Open `colab_train.ipynb` in Google Colab, select **Runtime → Change runtime type → T4 GPU**, and run the cells in order. When prompted, upload `netraone_colab_input.zip`. The notebook labels the images, shows review examples, validates labels, trains YOLOv8n for 35 epochs, evaluates, exports both weights, creates one inference image per class, and downloads `netraone_submission.zip`. To include a traffic-video example, set `run_video_demo = True` in the final notebook cell and upload a short clip when prompted.

3. To collect a fresh set instead, run the scraper in Colab or locally. It writes `dataset/clean/manifest.csv` with source-page URLs where available.

4. For a manual command-line run, bootstrap labels and inspect review overlays. Re-run with a higher `--min-score` if false boxes appear; correct or remove bad labels before training. Auto-labels are proposals, not ground truth.

   ```bash
   !python pipeline/auto_label.py --device cuda --min-score 0.35
   !python pipeline/validate_dataset.py
   ```

5. Train 35 epochs (within the requested 25–50 range), validate, create plots, and export both formats:

   ```bash
   !python train_emergency_yolo.py --device 0 --epochs 35 --batch -1 --imgsz 640
   ```

6. The training run generates these deliverables:

   ```text
   weights/best.pt
   weights/best.onnx
   weights/metrics.json
   runs/emergency/yolov8n/confusion_matrix.png
   runs/emergency/yolov8n/results.png
   ```

### Local laptop

Create a virtual environment, install the CUDA-matched PyTorch build if using NVIDIA GPU, then install `requirements.txt`. Use `--device 0` for GPU, or `--device cpu --batch 4 --workers 0` for CPU. CPU label bootstrapping is slow; start with `--max-images 30` only as a pipeline smoke test, then run the full set on Colab.

## Dataset contract

After `auto_label.py`, the generated layout exactly matches [`data.yaml`](data.yaml):

```text
dataset/
├── images/train, images/val
├── labels/train, labels/val       # one YOLO .txt per image
└── review/                        # proposed boxes for human inspection
```

Each YOLO label is checked before write and uses normalized coordinates:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Values are clamped to image bounds, invalid boxes are dropped, and splits are reproducible (`--seed 42`) and stratified by scraper category. Images with no confident proposal are excluded rather than mislabeled as negatives. This keeps the first model honest but makes the review pass essential: inspect `dataset/review/` and correct or remove bad labels before the final training run. If you need to regenerate, pass `--overwrite`; it replaces generated files under `dataset/images/` and `dataset/labels/`.

The scraper reconciles the manifest to active image files when resumed. The original interrupted collection lost some source URLs; 174 of the current 450 rows have blank URL fields and should not be treated as attribution records. Fresh collection runs retain image and source-page URLs. The Colab training bundle includes the images for private training, while the final submission ZIP includes the manifest but excludes the scraped images.

## Evaluation record

`train_emergency_yolo.py` runs a final validation on `best.pt`, then saves machine-readable values in `weights/metrics.json`:

| Required measure | Where to find it after the run |
|---|---|
| Overall mAP@0.5 | `weights/metrics.json` → `map50` |
| Precision / recall | `weights/metrics.json` → `precision`, `recall` |
| Per-class mAP@0.5 | `weights/metrics.json` → `per_class_map50` |
| Confusion matrix | `runs/emergency/yolov8n/confusion_matrix.png` |
| PR and validation plots | `runs/emergency/yolov8n/` |

Do not report a target such as mAP@0.5 ≥ 0.65 as achieved until it appears in this run report. If performance is weak, first improve label review and diversity (angles, day/night, occlusion, background) rather than simply increasing epochs.

## Traffic inference

```bash
python detect_emergency.py \
  --source traffic_sample.mp4 \
  --weights weights/best.pt \
  --conf 0.45
```

For an image:

```bash
python detect_emergency.py --source traffic_sample.jpg --weights weights/best.onnx --conf 0.45
```

The command writes `test_samples/detected_<source-name>.*`. Each detection has a class-coloured box and confidence label; any positive frame gets this green overlay:

```text
PRIORITY RESCUE VEHICLE DETECTED: INDIAN_AMBULANCE
```

Use `--output test_samples/scene_01.mp4` to set the path and `--show` for an interactive local preview. The output video is re-encoded without original audio, which is normal for this OpenCV-oriented challenge demonstrator.

## Local visual preview

After collecting images, run this local dashboard to inspect the three classes and the current Grounding DINO proposal overlay:

```bash
python preview_server.py
```

Open `http://127.0.0.1:8765` in your browser. Once training creates `weights/best.pt`, use `detect_emergency.py` for final YOLO inference on traffic footage.

## Repository map

```text
.
├── data.yaml
├── pipeline/
│   ├── scrape_and_clean.py     # search, decode validation, pHash filtering
│   └── auto_label.py           # Grounding DINO proposals, review images, YOLO split
├── train_emergency_yolo.py     # train, validate, metrics, PT/ONNX export
├── detect_emergency.py         # image/video alert-overlay CLI
├── preview_server.py            # local visual collection and auto-label dashboard
├── colab_train.ipynb             # end-to-end T4 labeling, training, evaluation, export
├── make_colab_bundle.py          # creates the Colab upload ZIP
├── weights/                    # add release copies of best.pt and best.onnx
└── test_samples/               # add 3–5 rendered output examples for submission
```

## Responsible data use

The scraper retrieves publicly indexed URLs, not a rights grant. Use the generated manifest to check source terms, copyright, and consent before dataset redistribution. Avoid treating this proof-of-concept detector as an autonomous dispatch or enforcement decision system; validate it against local traffic conditions, rare vehicle variants, weather, and false-positive safety cases.
