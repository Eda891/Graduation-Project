import os
from yolo_train import unzip_dataset, create_data_yaml, train_yolo, predict_yolo
from sam_mask import run_sam
from diffusion_train import unzip_diffusion, train_diffusion, run_diffusion_inference

def main():

    print("==== STEP 1: YOLO =====")
    unzip_dataset()
    create_data_yaml("custom_data/classes.txt", "data.yaml")
    train_yolo("data.yaml")
    predict_yolo()

    print("==== STEP 2: SAM =====")
    run_sam()

    print("==== STEP 3: DIFFUSION TRAINING =====")
    unzip_diffusion()
    # Stage 1: fast learning at 128px
    train_diffusion(epochs=30, img_size=128)
    # Stage 2: refine at 256px
    train_diffusion(epochs=20, img_size=256, resume=True)

    print("==== STEP 4: INFERENCE =====")
    os.makedirs("results", exist_ok=True)
    before_folder = "beforeandafter_examples/before images"
    for img_file in sorted(os.listdir(before_folder)):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            run_diffusion_inference(
                input_image_path=os.path.join(before_folder, img_file),
                output_path=f"results/{img_file}",
                strength=0.5
            )

if __name__ == "__main__":
    main()