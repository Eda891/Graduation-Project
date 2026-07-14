from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

# ─── AYARLAR ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
MODEL_SIZE   = "yolo11m"
DATA_YAML    = str(BASE_DIR / "dataset" / "data.yaml")
PROJECT_NAME = str(BASE_DIR / "outputs" / "runs" / "detect")
RUN_NAME     = "exp1"

EPOCHS        = 200
IMGSZ         = 640
BATCH         = -1             # auto (GPU VRAM'e göre)
WORKERS       = 0              # Windows'ta multiprocessing hatası için 0 olmalı

LR0           = 0.00005
LRF           = 0.1
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3.0

OPTIMIZER    = "AdamW"
PATIENCE     = 20
SAVE_PERIOD  = 25

PRETRAINED   = True
FREEZE       = 0
AMP          = True
CACHE        = True
PROFILE      = False

# ─── AUGMENTATION ─────────────────────────────────────────────
AUGMENT_CFG = dict(
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    degrees      = 5.0,
    translate    = 0.1,
    scale        = 0.5,
    shear        = 2.0,
    perspective  = 0.0,
    flipud       = 0.0,
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.1,
    copy_paste   = 0.1,
    erasing      = 0.4,
    crop_fraction= 1.0,
)

# ─── LOSS WEIGHTS ─────────────────────────────────────────────
LOSS_CFG = dict(
    box = 7.5,
    cls = 0.5,
    dfl = 1.5,
)

# ─── WINDOWS MULTIPROCESSING GUARD ────────────────────────────
# Bu satır Windows'ta zorunludur. workers > 0 ile çalışırken
# DataLoader yeni process spawn etmeye çalışır ve bu guard olmadan crash olur.
if __name__ == "__main__":

    # ─── SETUP ────────────────────────────────────────────────
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_info = yaml.safe_load(f)
    print(f"\nSınıf Sayısı: {data_info.get('nc')}")
    print(f"Sınıflar: {data_info.get('names')}")

    # ─── MODEL ────────────────────────────────────────────────
    # Option A: Fine-tune from YOUR existing best.pt (recommended)
    weights = str(BASE_DIR / "models" / "best.pt")

    # Option B: Start fresh from COCO pretrained yolo11m (uncomment to use)
    # weights = f"{MODEL_SIZE}.pt" if PRETRAINED else f"{MODEL_SIZE}.yaml"

    model = YOLO(weights)
    print(f"\nModel: {MODEL_SIZE} | Params: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

    # ─── EĞİTİM ───────────────────────────────────────────────
    results = model.train(
        data         = DATA_YAML,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH,
        workers      = WORKERS,

        lr0          = LR0,
        lrf          = LRF,
        momentum     = MOMENTUM,
        weight_decay = WEIGHT_DECAY,
        warmup_epochs= WARMUP_EPOCHS,

        optimizer    = OPTIMIZER,
        patience     = PATIENCE,
        save_period  = SAVE_PERIOD,

        freeze       = FREEZE if FREEZE > 0 else None,
        amp          = AMP,
        cache        = CACHE,
        profile      = PROFILE,

        project      = PROJECT_NAME,
        name         = RUN_NAME,
        exist_ok     = True,
        verbose      = True,
        plots        = True,

        **AUGMENT_CFG,
        **LOSS_CFG,
    )

    # ─── SONUÇLAR ─────────────────────────────────────────────
    best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
    print(f"\nEğitim tamamlandı.")
    print(f"En iyi model: {best_model_path}")
    print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

    # ─── VALİDASYON ───────────────────────────────────────────
    best_model = YOLO(best_model_path)
    val_results = best_model.val(
        data      = DATA_YAML,
        imgsz     = IMGSZ,
        batch     = 16,        # val() does not support batch=-1
        conf      = 0.001,
        iou       = 0.6,
        plots     = True,
        save_json = True,
        project   = PROJECT_NAME,   # add this
        name      = "val1",   
    )

    print(f"\nValidation mAP50:    {val_results.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_results.box.map:.4f}")

