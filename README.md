# 🏠 AI Interior Redesign Studio (YOLO + SAM + SDXL Inpainting)

An interactive, terminal-based tool that redesigns interior room photos. It detects
furniture with a **custom-trained YOLO11 model**, segments it precisely with **SAM**,
and edits it with a **pretrained SDXL inpainting pipeline** — driven entirely from a
rich-text menu in your terminal.

⚠️ This is the part I worked on the project it is not the whole project.

---

## ✍️ What I actually built

The part of this project that's my own trained model is the **YOLO11 object
detector** (`scripts/prepare_dataset.py` + `scripts/train.py`). Everything
downstream of detection — SAM segmentation and SDXL inpainting — uses other
people's pretrained weights as-is, called through prompt/mask engineering I wrote,
but not trained by me.

**To reproduce or extend the detector, you need this kind of dataset:**

- Photos of room interiors (any room type you want to detect furniture in)
- Each photo labeled in **YOLO format** using a tool like **LabelImg** — one `.txt`
  per image, with bounding boxes for each object class you care about (sofa, chair,
  table, carpet, curtain, lamp, pillow, etc. — see `Config.APPLIANCE_KEYWORDS` in
  `inpaint_pipeline.py` for the full class list this project expects)
- Images and their matching label files placed in `dataset/images/` and
  `dataset/labels/` respectively, with matching filenames (`room1.jpg` ↔
  `room1.txt`)
- Ideally a few hundred+ labeled images per class for a decent detector; more is
  better

**Given that dataset, here's what the code does with it:**

1. `scripts/prepare_dataset.py` checks every image has a matching label file, then
   splits everything 80/20 into `train/` and `val/` subfolders automatically.
2. `scripts/train.py` fine-tunes a YOLO11m model on that split (either from your own
   existing `best.pt` or from COCO pretrained weights), with augmentation, AdamW,
   and early stopping already configured, and reports mAP50 / mAP50-95 when done.
3. The resulting `best.pt` is what `inpaint_pipeline.py` loads to detect furniture in
   new room photos, which SAM then turns into precise masks for SDXL to edit.

So: **you supply the labeled room-photo dataset, the code trains the detector and
plugs it straight into the existing SAM + SDXL editing pipeline.**

---

## 🚀 What it does

You give it one room photo. It first runs an automatic "initial redesign" pass
(declutter, and optionally restyle), then drops you into an interactive loop where
you can keep editing:

- **Modify a single object** — recolor it, change its material, or restyle it, while
  keeping (or intentionally changing) its shape
- **Global change** — redesign the whole room while locking structural elements
  (walls, windows, doors, ceiling) so the room's architecture doesn't shift
- Every iteration is saved as its own image, with a mask preview and a session
  report so you can see exactly what changed and when

---

## 🧠 Technologies used

| Stage | Tool | Trained by you? |
|---|---|---|
| Object detection | **YOLO11** (Ultralytics) | ✅ Yes — `train.py` fine-tunes this on your own labeled photos |
| Segmentation | **SAM** (`vit_b`, Meta's Segment Anything) | ❌ No — pretrained checkpoint, used as-is |
| Image editing | **SDXL Inpainting 1.0** + **SDXL Refiner 1.0** (via 🤗 Diffusers) | ❌ No — pretrained, off-the-shelf, used as-is |

Other libraries: PyTorch, OpenCV, `rich` (terminal UI), `segment-anything`, `transformers`.

---

## ⚙️ System pipeline

```
Room photo
   ↓
YOLO11 (best.pt, your fine-tuned weights) → detects furniture/decor
   ↓
SAM (vit_b, pretrained) → precise per-object segmentation masks
   ↓
SDXL Inpainting + Refiner (pretrained, prompt-driven) → recolor / restyle / remove / redesign
   ↓
Saved iteration image + mask preview + session report
```

---
## ▶️ How to run

### 1️⃣ Prepare your dataset

```bash
python prepare_dataset.py
```

Checks that every image in `dataset/images/` has a matching LabelImg YOLO-format
label in `dataset/labels/`, then splits everything 80/20 into `train/` and `val/`
subfolders under each.

### 2️⃣ Train (or fine-tune) the YOLO detector

```bash
python yolo_train.py
```

Fine-tunes `yolo11m` from your existing `models/best.pt` (or from COCO pretrained
weights, see the commented option in the script) on `dataset/data.yaml`, then
validates and reports mAP50 / mAP50-95. 

### 3️⃣ Run an interactive redesign session

```bash
python inpaint_pipeline.py
```

You'll be asked for an image path, then for an initial style
(`low` / `balanced` / `creative`):

- **low** — declutter only (removes wall art, extra pillows, etc.); furniture color
  and shape are untouched
- **balanced** / **creative** — declutter *and* restyle the room, with structural
  elements locked

After that, you get a menu each iteration:
[1] Modify an object   — recolor / re-material / reshape one detected item
[2] Global change      — redesign the whole room (furniture-only or full mask)
[3] Exit

Every change is saved along with a mask preview and an
entry in the session report.

---

## 📌 Notes

- GPU strongly recommended (RTX 30/40 series ideal); the pipeline runs on CPU as a
  fallback but is very slow
- SDXL runs in fp16 on GPU with model CPU offload and VAE tiling to reduce VRAM use
- YOLO training defaults to automatic batch sizing (`batch=-1`) and mixed precision

---


## 📄 License

This project is for educational purposes only.
