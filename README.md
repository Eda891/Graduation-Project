# 🏠 AI-Based Minimalist Interior Redesign System

This project is a hybrid AI system that redesigns interior room images into a **minimalist style** using computer vision and diffusion-based deep learning.

⚠️ **Important:**
This project is currently **under development** and **NOT a finished version**.
Several components (frontend, optimization, advanced inpainting, etc.) are still incomplete.

---

## 🚀 Project Overview

The system takes a room image and generates a **minimalist redesign** by:

* Detecting objects in the room (YOLO11)
* Segmenting objects precisely (SAM)
* Removing clutter (inpainting)
* Applying a **custom-trained diffusion model** for redesign

---

## 🧠 Technologies Used

* **Python**
* **YOLO** → Object Detection
* **SAM (Segment Anything Model)** → Image Segmentation
* **Stable Diffusion (custom training)** → Redesign
* **PyTorch + Diffusers** → Model training/inference
* **OpenCV** → Image processing

---

## ⚙️ System Pipeline

```
Input Image
   ↓
YOLO11 → Object detection
   ↓
SAM → Precise segmentation masks
   ↓
Custom Diffusion Model → Minimalist redesign
   ↓
Output Image
```

---

## 📁 Project Structure

```
project/
│
├── main.py
├── diffusion/
│
├── yolo/
├── sam/
│
├── beforeandafter_examples/   # Extracted dataset
├── beforeandafter.zip         # Dataset archive
│
├── latent_proj.pt
├── unet_attn.pt
│
└──
```

---

## 📊 Required Data

### 1️⃣ Diffusion Training Dataset (REQUIRED)

You must provide paired images in the following format:

```
beforeandafter_examples/
   before images/
       room1_before.jpg
       room2_before.jpg

   after images/
       room1_after.jpg
       room2_after.jpg
```

### ⚠️ Naming is VERY IMPORTANT:

* `room1_before.jpg` → `room1_after.jpg`
* Must match exactly

### Dataset Notes:

* Same room, same angle
* Only style/layout should change
* Recommended: 100+ image pairs

---

### 2️⃣ YOLO Dataset (OPTIONAL)

Only required if you want to train YOLO yourself:

```
image.jpg
image.txt  # YOLO format
```

Otherwise:
👉 Pretrained YOLO is enough

---

## 🧪 Diffusion Model (Your Implementation)

This project uses a **custom-trained Stable Diffusion pipeline** with:

### ✔ Latent conditioning (before → after)

* Uses encoded "before" image as conditioning input
* Not standard img2img — more advanced

### ✔ Fine-tuned UNet attention layers

* Only attention layers are trained
* Efficient and GPU-friendly

### ✔ Custom latent projection network

* Maps image latents → conditioning embeddings

### ✔ SNR-weighted loss

* Improves training stability

---

## ▶️ How to Run

### 1️⃣ Extract dataset

```bash
python diffusion_train.py
```

(or call `unzip_diffusion()` manually)

---

### 2️⃣ Train diffusion model

```bash
python diffusion_train.py
```

This will generate:

* `latent_proj.pt`
* `unet_attn.pt`
* best checkpoints

---

## ⚠️ Limitations (Current Version)

* Dataset is small (~78–100 images)
* No frontend yet
* Inpainting is basic
* Model sometimes alters room structure
* Requires strong GPU (recommended RTX series)

---

## 🔮 Future Improvements

* Improve dataset size and diversity
* Add ControlNet for better structure preservation
* Add web interface (Flask / React)
* Improve inference speed
* Add furniture recommendation system

---

## 📌 Notes

* GPU is strongly recommended (RTX 30/40 series ideal)
* Training uses **mixed precision (fp16)**
* Large models may require high VRAM (8GB+)

---

## 📷 Output Example

(Currently shows side-by-side comparison)

```
[Before Image | Generated Minimalist Image]
```

---

## 🤝 Contribution

This is an academic project under development.
Contributions and suggestions are welcome.

---

## 📄 License

This project is for educational purposes only.
