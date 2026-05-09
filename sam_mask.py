import os
import zipfile
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NUM_WORKERS = 20
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
USE_FP16 = True
YOLO_BATCH_SIZE = 6

import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = cv2.imread(str(path))

        if image is None:
            return None, str(path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_rgb, str(path)


def collate_skip_none(batch):
    return [(img, path) for img, path in batch if img is not None]


def _find_yolo_model():
    zip_candidates = glob.glob("*.zip") + glob.glob("**/*.zip")

    for zip_path in zip_candidates:
        with zipfile.ZipFile(zip_path, "r") as zf:
            pt_files = [f for f in zf.namelist() if f.endswith(".pt")]

            if pt_files:
                extract_dir = Path(zip_path).stem

                zf.extractall(extract_dir)

                extracted_pts = [str(Path(extract_dir) / f) for f in pt_files]

                best = next(
                    (p for p in extracted_pts if "best" in Path(p).name),
                    None
                )

                return best or extracted_pts[0]

    for candidate in ["my_model.pt", "my_model/my_model.pt"]:
        if os.path.exists(candidate):
            return candidate

    run_weights = glob.glob("runs/detect/train*/weights/best.pt")

    if run_weights:
        return sorted(run_weights)[-1]

    raise FileNotFoundError("No YOLO model found!")


def setup_models(yolo_model_path=None):
    print("=" * 70)
    print("YOLO + SAM - WINDOWS FIXED VERSION")
    print("=" * 70)

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    cudnn.benchmark = True

    print("\nLoading YOLO...")

    if yolo_model_path is None:
        yolo_model_path = _find_yolo_model()

    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device)

    print("YOLO loaded")

    print(f"\nLoading SAM ({SAM_MODEL_TYPE.upper()})...")

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"{SAM_CHECKPOINT} not found!")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

        raise FileNotFoundError(SAM_CHECKPOINT)

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)

    print("FP16: Using autocast")
    print("torch.compile: Disabled")

    predictor = SamPredictor(sam)

    print("SAM loaded")

    torch.cuda.empty_cache()

    return yolo_model, predictor, device


def process_images(
    yolo_model,
    sam_predictor,
    device,
    input_dir=r"C:\Users\22000268\Desktop\Grad - Copy 2 newer\custom_data\images",
    output_dir=r"C:\Users\22000268\Desktop\Grad - Copy 2 newer\runs\sam_output",
    conf_threshold=0.25
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        print("No images found!")
        return

    print(f"Processing {len(image_paths)} images...")

    dataset = ImageDataset(image_paths)

    dataloader = DataLoader(
        dataset,
        batch_size=YOLO_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_skip_none,
        prefetch_factor=4,
        persistent_workers=True,
    )

    save_executor = ThreadPoolExecutor(max_workers=8)

    for batch in dataloader:
        if not batch:
            continue

        batch_images = [item[0] for item in batch]
        batch_paths = [item[1] for item in batch]

        yolo_results = yolo_model(
            batch_images,
            conf=conf_threshold,
            verbose=False
        )

        for image_rgb, img_path, result in zip(batch_images, batch_paths, yolo_results):
            img_path = Path(img_path)

            annotated = image_rgb.copy()

            if len(result.boxes) == 0:
                print(f"[{img_path.name}] No detections")

                _save_image(
                    save_executor,
                    annotated,
                    output_dir,
                    img_path.name
                )

                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            sam_predictor.set_image(image_rgb)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_FP16):
                    masks, _, _ = sam_predictor.predict(
                        box=np.array([x1, y1, x2, y2], dtype=np.float32),
                        multimask_output=False,
                    )

                mask = masks[0]

                color = get_class_color(cls_id)

                annotated[mask] = (
                    annotated[mask] * 0.6 +
                    np.array(color) * 0.4
                ).astype(np.uint8)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                class_name = yolo_model.names[cls_id]

                label = f"{class_name} {conf:.2f}"

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - 8),
                    (x1 + tw + 4, y1),
                    color,
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            _save_image(
                save_executor,
                annotated,
                output_dir,
                img_path.name
            )

            print(f"[{img_path.name}] {len(boxes)} objects")

    save_executor.shutdown(wait=True)

    print(f"\nDone! Results saved to: {output_dir}")


def _save_image(executor, image_rgb, output_dir, filename):
    save_path = os.path.join(output_dir, filename)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    executor.submit(cv2.imwrite, save_path, bgr)


def get_class_color(class_id):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255)
    ]

    return colors[class_id % len(colors)]


if __name__ == "__main__":
    try:
        yolo_model, sam_predictor, device = setup_models()

        input_dir = r"C:\Users\22000268\Desktop\Grad - Copy 2 newer\custom_data\images"

        output_dir = r"C:\Users\22000268\Desktop\Grad - Copy 2 newer\runs\sam_output"

        process_images(
            yolo_model,
            sam_predictor,
            device,
            input_dir=input_dir,
            output_dir=output_dir
        )

    except Exception as e:
        print(f"\nError: {e}")

        import traceback

        traceback.print_exc()
