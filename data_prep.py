"""
Stage 0: Dataset Acquisition & Preparation for Potato Spoilage Pipeline
Processes the local COCO dataset from 'potato dataset/potato detection.v3i.coco'
into two derived datasets:
  1. data/localizer/ — Single-class YOLO dataset ('potato') for Stage 1 tuber detector.
  2. data/classifier/ — Cropped tuber images organized by condition for Stage 2 classifier:
     ['Potato', 'Damaged potato', 'Defected potato', 'Diseased-fungal potato', 'Sprouted potato']
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import yaml

DATA_DIR = Path("data")
LOCALIZER_DIR = DATA_DIR / "localizer"
CLASSIFIER_DIR = DATA_DIR / "classifier"

DEFAULT_COCO_DIR = Path("potato dataset") / "potato detection.v3i.coco"


def find_source_dataset() -> Path:
    """Locates the source potato COCO dataset on disk."""
    candidates = [
        DEFAULT_COCO_DIR,
        Path("potato dataset"),
        DATA_DIR / "raw",
    ]
    for c in candidates:
        if (c / "train" / "_annotations.coco.json").exists():
            return c
        for sub in c.glob("*.coco"):
            if (sub / "train" / "_annotations.coco.json").exists():
                return sub
    raise FileNotFoundError(
        "Could not find '_annotations.coco.json' under 'potato dataset' or 'data/raw'."
    )


def extract_classes_and_annotations(coco_split_dir: Path):
    """Loads COCO json and returns images dict, annotations dict grouped by image, and category mapping."""
    json_path = coco_split_dir / "_annotations.coco.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Categories: filter out any empty supercategory container (id 0)
    cat_map = {}
    for c in data.get("categories", []):
        if c["name"].lower() != "potatoes":
            cat_map[c["id"]] = c["name"]

    # Image lookup: image_id -> image info
    images = {img["id"]: img for img in data.get("images", [])}

    # Group annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        if cat_id in cat_map:
            anns_by_img[ann["image_id"]].append(ann)

    return images, anns_by_img, cat_map


def build_localizer_and_classifier(
    src_root: Path,
    limit_crops_per_class: int = None,
    skip_localizer: bool = False
):
    """
    Parses COCO dataset and produces both:
      - data/localizer/ (YOLO single-class format)
      - data/classifier/ (Cropped image folders by condition)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover all unique class names across train split
    _, _, cat_map = extract_classes_and_annotations(src_root / "train")
    class_names = sorted(list(set(cat_map.values())))
    # Ensure standard healthy class is first
    if "Potato" in class_names:
        class_names.remove("Potato")
        class_names = ["Potato"] + class_names

    with open(DATA_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Condition Classes ({len(class_names)}): {class_names}")

    # 2. Process each split (train, valid, test)
    classifier_counts = {split: Counter() for split in ("train", "valid", "test")}

    for split in ("train", "valid", "test"):
        split_src = src_root / split
        if not (split_src / "_annotations.coco.json").exists():
            print(f"Split {split} not found, skipping...")
            continue

        print(f"\nProcessing {split} split...")
        images, anns_by_img, split_cat_map = extract_classes_and_annotations(split_src)

        # Output paths
        loc_img_dir = LOCALIZER_DIR / split / "images"
        loc_lbl_dir = LOCALIZER_DIR / split / "labels"
        if not skip_localizer:
            loc_img_dir.mkdir(parents=True, exist_ok=True)
            loc_lbl_dir.mkdir(parents=True, exist_ok=True)

        for c_name in class_names:
            (CLASSIFIER_DIR / split / c_name).mkdir(parents=True, exist_ok=True)

        processed_images = 0
        total_crops = 0

        for img_id, img_info in images.items():
            fname = img_info["file_name"]
            img_path = split_src / fname
            if not img_path.exists():
                continue

            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue

            img = None
            w = img_info.get("width")
            h = img_info.get("height")

            # Localizer YOLO format lines
            yolo_lines = []

            for i, ann in enumerate(anns):
                cat_name = split_cat_map.get(ann["category_id"])
                if not cat_name:
                    continue

                bbox = ann["bbox"]  # [x, y, width, height]
                bx, by, bw, bh = bbox

                # Ensure image dimensions
                if w is None or h is None or img is None:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        break
                    h, w = img.shape[:2]

                # Convert to normalized YOLO format: class_id center_x center_y width height
                cx = (bx + bw / 2.0) / w
                cy = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                # Crop for classifier if limit not reached
                if limit_crops_per_class is None or classifier_counts[split][cat_name] < limit_crops_per_class:
                    if img is None:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue

                    x1 = max(0, int(bx))
                    y1 = max(0, int(by))
                    x2 = min(w, int(bx + bw))
                    y2 = min(h, int(by + bh))

                    if x2 > x1 + 5 and y2 > y1 + 5:
                        crop = img[y1:y2, x1:x2]
                        out_crop_path = (
                            CLASSIFIER_DIR / split / cat_name
                            / f"{Path(fname).stem}_crop_{i}.jpg"
                        )
                        cv2.imwrite(str(out_crop_path), crop)
                        classifier_counts[split][cat_name] += 1
                        total_crops += 1

            # Save YOLO annotation & copy image for localizer
            if not skip_localizer and yolo_lines:
                lbl_out = loc_lbl_dir / f"{Path(fname).stem}.txt"
                lbl_out.write_text("\n".join(yolo_lines), encoding="utf-8")
                # Copy image to localizer
                dst_img = loc_img_dir / fname
                if not dst_img.exists():
                    shutil.copy(img_path, dst_img)

            processed_images += 1
            if processed_images % 500 == 0:
                print(f"  Processed {processed_images}/{len(images)} images ({total_crops} crops created)...")

        print(f"  Finished {split}: {processed_images} images, {total_crops} tuber crops extracted.")

    # 3. Write data.yaml for YOLO training
    if not skip_localizer:
        yaml_content = {
            "path": str(LOCALIZER_DIR.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": {0: "potato"},
        }
        with open(LOCALIZER_DIR / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)
        print(f"\nWritten YOLO localizer config to {LOCALIZER_DIR / 'data.yaml'}")

    # 4. Print final composition summary
    print("\n" + "=" * 60)
    print("CLASSIFIER CROPPED DATASET COMPOSITION:")
    print("=" * 60)
    for split in ("train", "valid", "test"):
        print(f"[{split.upper()} SPLIT]")
        for c_name in class_names:
            cnt = classifier_counts[split].get(c_name, 0)
            print(f"  - {c_name:25s}: {cnt:5d} crops")
    print("=" * 60)
    print("Done! See data/localizer and data/classifier.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Potato COCO dataset for AgriSentinel")
    parser.add_argument("--src", type=str, default=None, help="Path to potato COCO root folder")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Limit crops per class (e.g. 500 for fast testing)")
    parser.add_argument("--skip-localizer", action="store_true",
                        help="Skip building YOLO localizer images/labels to save time/disk")
    args = parser.parse_args()

    source_path = Path(args.src) if args.src else find_source_dataset()
    print(f"Source COCO dataset located at: {source_path}")

    build_localizer_and_classifier(
        src_root=source_path,
        limit_crops_per_class=args.limit_per_class,
        skip_localizer=args.skip_localizer
    )
