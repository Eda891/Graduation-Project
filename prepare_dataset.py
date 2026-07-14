"""
prepare_dataset.py — Desktop/Jcode/scripts/
Run this ONCE after collecting your LabelImg-labeled images.

What it does:
  1. Scans dataset/images/ and dataset/labels/ for your files
  2. Splits them 80/20 into train and val subfolders
  3. Verifies every image has a matching label file
  4. Prints a summary

LabelImg saves .txt files in YOLO format already — no conversion needed.

Usage:
  python scripts/prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent   # Desktop/Jcode/
IMG_DIR    = BASE_DIR / "dataset" / "images"
LABEL_DIR  = BASE_DIR / "dataset" / "labels"

TRAIN_SPLIT = 0.80
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare():
    # Gather all images at the top level of dataset/images/
    all_images = [
        f for f in IMG_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    ]

    if not all_images:
        print(f"[ERROR] No images found in {IMG_DIR}")
        print("  → Put your images directly in Desktop/Jcode/dataset/images/")
        print("  → Put your LabelImg .txt files in Desktop/Jcode/dataset/labels/")
        return

    # Check label coverage
    missing = []
    for img in all_images:
        label = LABEL_DIR / (img.stem + ".txt")
        if not label.exists():
            missing.append(img.name)

    if missing:
        print(f"[WARNING] {len(missing)} images have no label file:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            return

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_SPLIT)
    train_imgs = all_images[:split_idx]
    val_imgs   = all_images[split_idx:]

    # Create subfolders
    for split in ["train", "val"]:
        (IMG_DIR / split).mkdir(exist_ok=True)
        (LABEL_DIR / split).mkdir(exist_ok=True)

    # Copy files
    def copy_split(imgs, split_name):
        copied_imgs   = 0
        copied_labels = 0
        for img in imgs:
            shutil.copy2(img, IMG_DIR / split_name / img.name)
            copied_imgs += 1
            label = LABEL_DIR / (img.stem + ".txt")
            if label.exists():
                shutil.copy2(label, LABEL_DIR / split_name / label.name)
                copied_labels += 1
        return copied_imgs, copied_labels

    ti, tl = copy_split(train_imgs, "train")
    vi, vl = copy_split(val_imgs, "val")

    print("\n✅ Dataset prepared successfully!")
    print(f"   Train: {ti} images, {tl} labels")
    print(f"   Val:   {vi} images, {vl} labels")
    print(f"\nNext step: open data.yaml and confirm the path is correct,")
    print(f"then run:  python scripts/train.py")


if __name__ == "__main__":
    prepare()
