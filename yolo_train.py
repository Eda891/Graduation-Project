import os
import yaml
import glob
import shutil
import random
from pathlib import Path
from zipfile import ZipFile
from ultralytics import YOLO
import torch

GRAD_DIR = r'C:\Users\22000268\Desktop\Grad - Copy 2 newer'

DATA_ZIP     = os.path.join(GRAD_DIR, 'livingroom (1).zip')
DATA_DIR     = os.path.join(GRAD_DIR, 'custom_data')
CLASSES_TXT  = os.path.join(DATA_DIR, 'classes.txt')
DATA_YAML    = os.path.join(GRAD_DIR, 'data.yaml')
RUNS_DIR     = os.path.join(GRAD_DIR, 'runs', 'detect')
MODEL_DIR    = os.path.join(GRAD_DIR, 'my_model')
MODEL_PT     = os.path.join(MODEL_DIR, 'my_model.pt')
MODEL_ZIP    = os.path.join(GRAD_DIR, 'my_model.zip')
PREDICT_DIR  = os.path.join(RUNS_DIR, 'predict')
TRAIN_IMG    = os.path.join(DATA_DIR, 'train', 'images')
VAL_IMG      = os.path.join(DATA_DIR, 'validation', 'images')


def check_gpu():
    print("=" * 70)
    print("YOLO TRAINING  —  RTX 4090 / Desktop\\Grad")
    print("=" * 70)
    print(f"\n   All output -> {GRAD_DIR}")
    print("\n=== GPU Status ===")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected — training will be slow!")


def prepare_data():
    print("\n=== Step 1: Preparing Data ===")

    if not os.path.exists(DATA_DIR):
        if not os.path.exists(DATA_ZIP):
            raise FileNotFoundError(
                f"Dataset zip not found: {DATA_ZIP}\n"
                f"Place 'livingroom (1).zip' in {GRAD_DIR}"
            )

        print(f"Extracting {DATA_ZIP} ...")

        with ZipFile(DATA_ZIP, 'r') as z:
            z.extractall(DATA_DIR)

        print("Dataset extracted")

    else:
        print(f"Dataset already at {DATA_DIR}")


def train_val_split(train_pct=0.9):
    print("\n=== Step 2: Splitting Dataset ===")

    images_dir = Path(DATA_DIR) / 'images'
    labels_dir = Path(DATA_DIR) / 'labels'

    train_img_dir = Path(TRAIN_IMG)
    train_lbl_dir = Path(DATA_DIR) / 'train' / 'labels'
    val_img_dir   = Path(VAL_IMG)
    val_lbl_dir   = Path(DATA_DIR) / 'validation' / 'labels'

    if train_img_dir.exists() and any(train_img_dir.iterdir()):
        print("Train/val split already exists, skipping")
        return

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    random.seed(42)
    random.shuffle(image_files)

    split_idx    = int(len(image_files) * train_pct)
    train_images = image_files[:split_idx]
    val_images   = image_files[split_idx:]

    print(f"Total: {len(image_files)} | Train: {len(train_images)} | Val: {len(val_images)}")

    for img in train_images:
        lbl = labels_dir / (img.stem + '.txt')

        shutil.copy(img, train_img_dir / img.name)

        if lbl.exists():
            shutil.copy(lbl, train_lbl_dir / lbl.name)

    for img in val_images:
        lbl = labels_dir / (img.stem + '.txt')

        shutil.copy(img, val_img_dir / img.name)

        if lbl.exists():
            shutil.copy(lbl, val_lbl_dir / lbl.name)

    print("Split complete")


def create_data_yaml():
    print("\n=== Step 3: Creating data.yaml ===")

    with open(CLASSES_TXT, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    data = {
        'path': DATA_DIR,
        'train': 'train/images',
        'val':   'validation/images',
        'nc':    len(classes),
        'names': classes,
    }

    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created: {DATA_YAML}")
    print(f"   Classes ({len(classes)}): {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")


def train_model():
    print("\n=== Step 4: Training YOLO Model ===")
    print("   Model  : yolo11s.pt")
    print("   Epochs : 60  |  Batch: 16  |  ImgSz: 640")
    print(f"   Output : {RUNS_DIR}")

    model = YOLO('yolo11s.pt')

    model.train(
        data=DATA_YAML,
        epochs=60,
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        project=RUNS_DIR,
        name='train',
        cache='disk',
        plots=True,
        verbose=True,
        amp=True,
        patience=20,
        exist_ok=True,
    )

    print("Training complete!")


def find_latest_run():
    pattern  = os.path.join(RUNS_DIR, 'train*')
    run_dirs = glob.glob(pattern)

    if not run_dirs:
        raise FileNotFoundError(
            f"No training runs found in {RUNS_DIR}\n"
            "Make sure training completed successfully."
        )

    run_dirs.sort(key=os.path.getmtime)

    latest  = run_dirs[-1]
    best_pt = os.path.join(latest, 'weights', 'best.pt')

    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"best.pt not found in {latest}")

    print(f"   Latest run : {latest}")

    return best_pt, latest


def run_predictions():
    print("\n=== Step 5: Making Predictions ===")

    best_pt, _ = find_latest_run()

    print(f"   Model  : {best_pt}")
    print(f"   Images : {VAL_IMG}")
    print(f"   Output : {PREDICT_DIR}")

    model = YOLO(best_pt)

    model.predict(
        source=VAL_IMG,
        save=True,
        project=RUNS_DIR,
        name='predict',
        conf=0.25,
        iou=0.45,
        exist_ok=True,
    )

    pred_images = glob.glob(os.path.join(PREDICT_DIR, '*.jpg'))

    print(f"Saved {len(pred_images)} prediction images -> {PREDICT_DIR}")

    for i, p in enumerate(pred_images[:5], 1):
        print(f"   {i}. {os.path.basename(p)}")


def save_model():
    print("\n=== Step 6: Saving Model ===")

    best_pt, run_dir = find_latest_run()

    os.makedirs(MODEL_DIR, exist_ok=True)

    shutil.copy(best_pt, MODEL_PT)

    print(f"Copied weights -> {MODEL_PT}")

    train_copy = os.path.join(MODEL_DIR, 'train')

    if os.path.exists(train_copy):
        shutil.rmtree(train_copy)

    shutil.copytree(run_dir, train_copy)

    with ZipFile(MODEL_ZIP, 'w') as zipf:
        zipf.write(MODEL_PT, arcname='my_model.pt')

        for root, dirs, files in os.walk(train_copy):
            for file in files:
                file_path = os.path.join(root, file)
                arcname   = os.path.relpath(file_path, MODEL_DIR)

                zipf.write(file_path, arcname=arcname)

    print(f"Zipped model -> {MODEL_ZIP}")

    print("\n" + "=" * 70)
    print("ALL DONE — everything saved to Desktop\\Grad")
    print("=" * 70)
    print(f"   Weights     : {MODEL_PT}")
    print(f"   Zip         : {MODEL_ZIP}")
    print(f"   Predictions : {PREDICT_DIR}")
    print(f"   Train plots : {run_dir}\\results.png")
    print("=" * 70)


if __name__ == '__main__':
    check_gpu()
    prepare_data()
    train_val_split(train_pct=0.9)
    create_data_yaml()
    train_model()
    run_predictions()
    save_model()
