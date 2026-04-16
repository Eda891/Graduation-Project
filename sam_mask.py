import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO


def run_sam():
    print("Running SAM...")

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Safety check
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"{sam_checkpoint} not found. Download it manually.")

    # Load SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    torch.cuda.empty_cache()  # 🔥 important for small GPU

    import glob
    weight_files = glob.glob("runs/detect/train*/weights/best.pt")
    if not weight_files:
        raise FileNotFoundError("No YOLO weights found. Train the model first.")
    latest_weights = sorted(weight_files)[-1]
    print(f"Loading YOLO from: {latest_weights}")
    yolo_model = YOLO(latest_weights)

    input_dir = "custom_data/images"
    output_dir = "runs/sam_output"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO detection
        with torch.no_grad():
            results = yolo_model(image_rgb)

        predictor.set_image(image_rgb)

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                with torch.no_grad():
                    masks, _, _ = predictor.predict(
                        box=np.array([x1, y1, x2, y2]),
                        multimask_output=False
                    )

                mask = masks[0]

                # Better visualization
                image_rgb[mask] = (
                    image_rgb[mask] * 0.5 +
                    np.array([255, 0, 0]) * 0.5
                )

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        print(f"Processed {img_name}")