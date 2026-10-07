# AgriSentinel — potato spoilage detection pipeline

Two-stage hybrid pipeline, structured after Mishra et al. (Scientific Reports,
2026) but adapted from whole-leaf classification to per-tuber detection:

- **Stage 1 (localizer):** a lightweight single-class YOLO model finds every
  potato in a storage-bin photo. It only answers "is this a potato" — it
  doesn't need to know its condition, so it can be small and fast.
- **Stage 2 (classifier):** each cropped tuber is graded into one of the
  dataset's five condition classes by a ResNet+SE ("combined") backbone,
  swappable for AlexNet / ResNet / EfficientNet to reproduce the base
  paper's architecture comparison table.
- **Aggregation:** per-tuber predictions are combined into a percent-defective
  figure for that inspection — the number your spoilage-tracking logic
  actually needs.

## Setup

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY="your_key_here"   # never hardcode this in a script
```

**Security note:** rotate the Roboflow key you pasted earlier in chat —
treat any key that's been shared in plaintext as compromised.

## Run order

1. `python data_prep.py`
   Downloads the dataset (version 3, workspace `workspace-tedkk`,
   project `potato-detection-phjcg`), then builds two derived sets from the
   same source annotations:
   - `data/localizer/` — YOLO format, every box collapsed to one class
     (`potato`), for Stage 1.
   - `data/classifier/<split>/<class>/*.jpg` — every box cropped out with
     its *original* class label, for Stage 2.

2. `python train_localizer.py --epochs 50`
   Trains the Stage 1 detector (YOLOv8n by default — swap with `--model`).

3. `python train_classifier.py --model seresnet50 --epochs 30`
   Trains the Stage 2 classifier. Re-run with `--model resnet50`,
   `--model alexnet`, or `--model efficientnet_b0` (all hyperparameters held
   fixed) to build the same style of comparison table as the base paper's
   Table 4 — just add a `seresnet50` row for the combined model.

4. `python inference_pipeline.py --image path/to/bin_photo.jpg`
   Runs the full pipeline on one photo: detects every tuber, classifies
   each crop, and prints per-class counts plus percent defective.

## Notes

- Class names are read from the dataset's `data.yaml` at download time and
  saved to `data/class_names.json` — don't hardcode class order, since it
  can change between Roboflow dataset versions.
- `train_localizer.py` relies on Ultralytics' built-in `model.val()` for
  mAP@0.5 / mAP@0.5:0.95 — no separate evaluation script needed for Stage 1.
- `train_classifier.py` writes `results_<model>.json` per run (epoch history,
  test precision/recall/F1, confusion matrix) — feed several of these into
  your comparative-results table and ablation study.
- Before training anything, spot-check a sample of crops per class in
  `data/classifier/` — the dataset's own description notes it's still being
  labeled, and Damaged / Defected / Diseased-fungal likely overlap.
