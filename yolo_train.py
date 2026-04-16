import os
import zipfile
import yaml
from ultralytics import YOLO
import glob
from ultralytics import YOLO

def unzip_dataset():
    zip_path = "livingroom (1).zip"
    extract_path = "custom_data"

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    print("YOLO dataset extracted to custom_data")


def create_data_yaml(classes_path, output_yaml):
    with open(classes_path, "r") as f:
        classes = [c.strip() for c in f.readlines()]

    data = {
        "path": os.path.abspath("custom_data"),
        "train": "images",
        "val": "images",   # using same folder
        "nc": len(classes),
        "names": classes
    }

    with open(output_yaml, "w") as f:
        yaml.dump(data, f)

    print("Created data.yaml")
    return output_yaml


def train_yolo(data_yaml, epochs=5, imgsz=640):
    model = YOLO("yolo11s.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=4,      
        workers=0     
    )

    return model


def predict_yolo():
    weight_files = glob.glob("runs/detect/train*/weights/best.pt")
    if not weight_files:
        raise FileNotFoundError("No YOLO weights found. Train the model first.")

    model_path = sorted(weight_files)[-1]  # pick the latest
    print(f"Loading YOLO model from: {model_path}")

    model = YOLO(model_path)
    model.predict(
        source="custom_data/images",
        save=True
    )

    print("Prediction completed.")