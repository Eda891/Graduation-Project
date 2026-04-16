# diffusion_train.py
import os
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["PYTORCH_ATTENTION_IMPL"] = "math"

import zipfile
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from accelerate import Accelerator


class BeforeAfterDataset(Dataset):
    def __init__(self, before_dir, after_dir, transform=None):
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.transform = transform

        before_files = sorted([f for f in os.listdir(before_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        after_files_set = set(os.listdir(after_dir))
        self.pairs = []

        for before_file in before_files:
            base = before_file.replace("_before", "")
            after_file = base.replace(".jpg","_after.jpg").replace(".png","_after.png").replace(".jpeg","_after.jpeg")
            if after_file in after_files_set:
                self.pairs.append((before_file, after_file))

        print(f"Found {len(self.pairs)} valid pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        before_file, after_file = self.pairs[idx]
        before = Image.open(os.path.join(self.before_dir, before_file)).convert("RGB")
        after  = Image.open(os.path.join(self.after_dir,  after_file)).convert("RGB")

        if self.transform:
            before = self.transform(before)
            after  = self.transform(after)

        return before, after


def unzip_diffusion():
    zip_file_path = "beforeandafter.zip"
    extract_dir   = "beforeandafter_examples"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    print("Diffusion dataset extracted.")


def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas    = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas / sqrt_one_minus) ** 2
    return snr


def train_diffusion(epochs=50, batch_size=1, lr=1e-4, img_size=128, resume=False):
    print("Training diffusion model...")
    before_dir = "beforeandafter_examples/before images"
    after_dir  = "beforeandafter_examples/after images"

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset    = BeforeAfterDataset(before_dir, after_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    accelerator = Accelerator(mixed_precision="no")
    device      = accelerator.device

    # Use Realistic Vision for better photorealistic results
    MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )
    pipeline.unet.set_attn_processor(AttnProcessor())

    unet = pipeline.unet
    vae  = pipeline.vae

    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.to(device, dtype=torch.float32)

    # Freeze UNet then unfreeze only cross-attention k/v
    unet.train()
    for p in unet.parameters():
        p.requires_grad_(False)

    attn_count = 0
    for name, module in unet.named_modules():
        if "attn2" in name and hasattr(module, "to_k"):
            for pname, p in module.named_parameters():
                if "to_k" in pname or "to_v" in pname:
                    p.requires_grad_(True)
                    attn_count += 1

    print(f"Unfrozen {attn_count} cross-attention k/v parameters")
    unet.to(device, dtype=torch.float32)

    latent_proj = torch.nn.Sequential(
        torch.nn.Linear(4, 512),
        torch.nn.LayerNorm(512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 768),
        torch.nn.LayerNorm(768),
    ).to(device)

    # Resume from checkpoint if requested
    if resume:
        if os.path.exists("latent_proj_best.pt"):
            latent_proj.load_state_dict(torch.load("latent_proj_best.pt"))
            print("Resumed latent_proj from checkpoint")
        if os.path.exists("unet_attn_best.pt"):
            unet.load_state_dict(torch.load("unet_attn_best.pt"), strict=False)
            print("Resumed UNet attention from checkpoint")

    optimizer = torch.optim.AdamW([
        {"params": [p for p in unet.parameters() if p.requires_grad], "lr": lr * 0.1},
        {"params": latent_proj.parameters(), "lr": lr}
    ], weight_decay=1e-4)

    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    unet, vae, latent_proj, optimizer, dataloader = accelerator.prepare(
        unet, vae, latent_proj, optimizer, dataloader
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler"
    )

    best_loss = float("inf")

    for epoch in range(epochs):
        unet.train()
        latent_proj.train()
        total_loss = 0.0

        for before, after in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            before = before.to(device, dtype=torch.float32)
            after  = after.to(device,  dtype=torch.float32)

            with torch.no_grad():
                after_latents  = vae.encode(after ).latent_dist.sample() * vae.config.scaling_factor
                before_latents = vae.encode(before).latent_dist.sample() * vae.config.scaling_factor

            noise     = torch.randn_like(after_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (after_latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(after_latents, noise, timesteps)

            B, C, H, W = before_latents.shape
            cond_seq = before_latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
            encoder_hidden_states = latent_proj(cond_seq)

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            snr = compute_snr(noise_scheduler, timesteps)
            loss_weight = snr / (snr + 1.0)
            loss = (F.mse_loss(noise_pred, noise, reduction="none")
                    .mean(dim=[1,2,3]) * loss_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad] +
                list(latent_proj.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler_lr.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {scheduler_lr.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(latent_proj.state_dict(), "latent_proj_best.pt")
            torch.save(
                {k: v for k, v in unet.state_dict().items() if "attn2" in k},
                "unet_attn_best.pt"
            )
            print(f"  ✓ Saved best model (loss={best_loss:.4f})")

    torch.save(latent_proj.state_dict(), "latent_proj.pt")
    torch.save(
        {k: v for k, v in unet.state_dict().items() if "attn2" in k},
        "unet_attn.pt"
    )
    print("Diffusion training complete.")


# def run_diffusion_inference(input_image_path, output_path="output.png", strength=0.5):
#     print(f"Running inference on {input_image_path}...")
#     from diffusers.models.attention_processor import AttnProcessor
#     from PIL import Image, ImageEnhance
#     import torch
#     import os

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

#     pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float32
#     )
#     pipe.unet.set_attn_processor(AttnProcessor())
#     pipe = pipe.to(device)

#     # Load fine-tuned attention weights if available
#     unet_ckpt = "unet_attn_best.pt" if os.path.exists("unet_attn_best.pt") else "unet_attn.pt"
#     if os.path.exists(unet_ckpt):
#         print(f"Loading fine-tuned UNet attention from {unet_ckpt}")
#         attn_state = torch.load(unet_ckpt, map_location=device)
#         pipe.unet.load_state_dict(attn_state, strict=False)

#     prompt = (
#         "same living room after professional renovation, "
#         "same layout same furniture positions same window same walls, "
#         "modern interior design, clean bright space, "
#         "RAW photo, 8k uhd, sharp focus, "
#         "photorealistic, high quality, professional photography"
#     )
#     negative_prompt = (
#         "different room, different angle, different furniture layout, "
#         "(deformed, distorted:1.3), blurry, "
#         "low quality, watermark, text, people, "
#         "cluttered, messy, dark, dirty"
#     )

#     before = Image.open(input_image_path).convert("RGB")
#     before_resized = before.resize((512, 512), Image.LANCZOS)

#     result = pipe(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         image=before_resized,
#         strength=strength,
#         guidance_scale=10.0,
#         num_inference_steps=30,
#     ).images[0]

#     # Sharpen and boost contrast
#     result = ImageEnhance.Sharpness(result).enhance(1.4)
#     result = ImageEnhance.Contrast(result).enhance(1.1)

#     comparison = Image.new("RGB", (1024, 512))
#     comparison.paste(before_resized, (0, 0))
#     comparison.paste(result, (512, 0))
#     comparison.save(output_path, quality=95)
#     print(f"Saved to {output_path}")
#     return result