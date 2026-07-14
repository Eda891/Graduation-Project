import os
import sys
import time
import torch
import cv2
import logging
import warnings
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter




from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import transformers
from diffusers.utils import logging as diffusers_logging

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
diffusers_logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
log = logging.getLogger("rich")

if not torch.cuda.is_available():
    log.warning("GPU not found — running on CPU. Inference will be VERY slow.")
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
    log.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─── CONFIGURATION ────────────────────────────────────────────
class Config:
    BASE_DIR   = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    YOLO_MODEL = "best.pt"

    SAM_CHECKPOINT = str(MODELS_DIR / "sam_vit_b_01ec64.pth")
    SAM_MODEL_TYPE = "vit_b"

    SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
    SDXL_VAE_FP16_FIX  = "madebyollin/sdxl-vae-fp16-fix"

    OUTPUT_FORMAT = "WEBP"
    WEBP_QUALITY  = 98

    SYSTEM_PROMPT = (
        "ultra photorealistic minimalist interior design, professional architectural photography, "
        "decluttered and pristine modern aesthetic, rich and highly detailed textures. "
        "razor-sharp focus, distinct object separation, crisp boundaries between objects and surfaces. "
        "perfectly matching furniture placement, high-end materials, 8k uhd."
    )

    SYSTEM_NEGATIVE_PROMPT = (
        "pastel, crayon, painting, illustration, cartoon, washed out colors, "
        "blending colors, messy boundaries, blurry, low quality, worst quality, "
        "deformed, grainy noise, colorful spots, jpeg artifacts, cluttered, "
        "distorted walls, chaotic textures, unrealistic shadows, mutated proportions"
    )

    ANTI_REDESIGN_PREFIX = (
        "exact same furniture exact same shape exact same proportions exact same style "
        "exact same design exact same silhouette, only the color or material is changed, "
        "photo-realistic color edit not AI redesign, preserve all geometry, "
    )

    SHAPE_NEGATIVE = (
        "rounded shape, bubble shape, oval shape, egg shape, futuristic shape, "
        "curved edges, black trim, black outline, black border, dark outline, "
        "new silhouette, new furniture design, redesigned furniture, modern redesign, "
        "futuristic furniture, different furniture, replaced furniture, new furniture, "
        "different sofa, different chair, different shape sofa, different shape chair, "
        "reshaped, morphed, distorted proportions, warped geometry, "
        "different number of cushions, different arm style, different leg style, "
        "different back style, different base, unrealistic furniture shape, "
    )

    NEON_NEGATIVE = (
        "glowing, neon, luminescent, fluorescent, backlit, oversaturated, "
        "flat solid color, painted floor, unrealistic color, artificial lighting effect, "
        "glowing rug, glowing carpet, light emission, bloom, "
    )

    GLOBAL_STRUCTURAL_NEGATIVE = (
        "new windows, new doors, added windows, extra windows, different windows, "
        "new doorways, new openings in walls, different room shape, "
        "new ceiling features, added skylights, recessed lighting, new light fixtures, "
        "new built-in furniture, new shelving units built into wall, new built-ins, "
        "different room layout, different floor plan, extra rooms, new walls, "
        "changed room proportions, new architectural elements, new room structure, "
        "different architecture, new wainscoting, new paneling, different baseboard, "
    )

    WALL_ART_NEGATIVE = (
        "picture, painting, photograph, artwork, print, poster, canvas, illustration, "
        "frame, picture frame, framed art, wall art, gallery wall, art gallery, "
        "mirror, round mirror, oval mirror, circular mirror, flower mirror, sunburst mirror, "
        "decorative mirror, wall mirror, convex mirror, mirror cluster, mirror grouping, "
        "wall decoration, wall hanging, ornament, plaque, tapestry, "
        "any object on wall, anything mounted on wall, anything hanging on wall, "
        "wall sconce, shelf on wall, floating shelf, bracket, hook, "
    )

    STRUCTURAL_KEYWORDS = ["window", "glass", "wall", "ceiling", "floor", "outside_environment"]

    APPLIANCE_KEYWORDS = [
        "tv", "plant", "sofa", "carpet", "pillow", "tv unit", "cabinet",
        "lamp", "chair", "middle table", "vase", "curtain", "candle",
        "books", "shelf", "picture", "table", "book shelf", "door",
        "side table", "radiator", "mirror", "blanket"
    ]

    FLAT_SURFACE_EXCLUSIONS = [
        "table", "middle table", "side table", "coffee table", "desk",
        "chair", "sofa", "lamp", "vase", "plant", "candle", "books",
        "cabinet", "tv", "tv unit", "shelf", "book shelf", "radiator",
        "floor", "hardwood", "wood floor", "tile",
    ]

    BRIGHT_COLORS = [
        "white", "pure white", "pearl white", "off white", "warm white", "cool white",
        "antique white", "ivory", "cream", "linen", "snow", "chalk", "bright white",
        "eggshell", "cotton", "milk white", "porcelain",
    ]

    TARGET_ALIASES: dict[str, list[str]] = {
        "sofa":    ["sofa", "couch", "loveseat", "sectional"],
        "chair":   ["chair", "armchair", "accent chair"],
        "carpet":  ["carpet", "rug", "mat", "floor mat"],
        "rug":     ["rug", "carpet", "mat"],
        "curtain": ["curtain", "drape", "blind"],
        "table":   ["table", "middle table", "side table", "coffee table", "desk"],
        "lamp":    ["lamp", "light", "floor lamp"],
        "shelf":   ["shelf", "book shelf", "shelving"],
        "pillow":  ["pillow", "cushion"],
        "blanket": ["blanket", "throw"],
        "picture": ["picture", "frame", "artwork", "painting", "photograph"],
        "mirror":  ["mirror", "round mirror"],
    }

    MODES = {
        # low strength 0.40→0.58 (was too weak vs balanced), guidance 9→8.5 and more
        # steps for all three → finer detail = more realistic, less "soft" redesigns.
        "low":     {"strength": 0.58, "guidance": 8.5,  "steps": 65, "refiner_start": 0.78},
        "balanced":{"strength": 0.70, "guidance": 8.0,  "steps": 72, "refiner_start": 0.75},
        "creative":{"strength": 0.90, "guidance": 7.5,  "steps": 80, "refiner_start": 0.70},

        # ── Friend's redesign settings (INITIAL auto-redesign only) ──────────
        # strength/guidance/steps come straight from the friend's bedroom code.
        # Lower steps + different strength → fixes the "balanced redesign not
        # clear/HD" issue (gentler denoise keeps more of the source photo's detail).
        # refiner_start is ours (the friend's spec gave only strength/guidance/steps);
        # it mirrors the existing redesign convention — gentler mode hands off to the
        # refiner later (more polish). Initial redesign routing:
        #   menu 'balanced' → redesign_moderate, menu 'creative' → redesign_creative.
        # redesign_low is defined per spec but intentionally NOT wired: menu 'low'
        # stays declutter-only (no redesign) per the locked decision.
        "redesign_low":      {"strength": 0.60, "guidance": 8.8, "steps": 36, "refiner_start": 0.78},
        "redesign_moderate": {"strength": 0.75, "guidance": 8.5, "steps": 45, "refiner_start": 0.75},
        "redesign_creative": {"strength": 0.85, "guidance": 7.8, "steps": 50, "refiner_start": 0.70},

        # strength 0.33→0.50 + guidance 12→14: 0.33 was too weak to shift hue
        # (e.g. green sofa → teal barely moved). Recolor has no color hint, so
        # strength/guidance are the only levers that drive the new color.
        # strength 0.50→0.42: lower strength preserves the furniture SHAPE/STYLE
        # (high strength let SDXL redesign the chairs). The color hint (now enabled)
        # carries the color, so strength can stay low to keep the original geometry.
        "recolor":          {"strength": 0.42, "guidance": 13.0, "steps": 72, "refiner_start": 0.82},
        "style":            {"strength": 0.42, "guidance": 11.0, "steps": 65, "refiner_start": 0.80},
        # flat surfaces (rugs/curtains) have no "style" to preserve — keep strong.
        "flat_recolor":     {"strength": 0.50, "guidance": 13.0, "steps": 70, "refiner_start": 0.82},

        # ── FIX 1a: guidance 12→16, strength 0.72→0.75, steps 75→80 ─────────
        # guidance=12 was too weak for an extreme luminance jump (beige→black).
        # Higher guidance forces strict text compliance: "jet black edge to edge".
        "dark_shift":       {"strength": 0.40, "guidance": 14.0, "steps": 65, "refiner_start": 0.82},
        "flat_dark_shift":  {"strength": 0.75, "guidance": 16.0, "steps": 80, "refiner_start": 0.78},

        "bright_shift":     {"strength": 0.55, "guidance": 15.0, "steps": 65, "refiner_start": 0.82},
        "flat_bright_shift":{"strength": 0.70, "guidance": 14.0, "steps": 70, "refiner_start": 0.80},

        "erase":            {"strength": 0.99, "guidance": 7.5,  "steps": 70, "refiner_start": 0.75},

        # strength 0.82 keeps the pre-filled hardwood base so the rug is removed
        # (full 1.0 noise destroys the prefill and the rug returns). refiner_start
        # 0.70→0.64 + steps 90→100: the refiner now runs longer over the filled area
        # to smooth the rug-edge seam and blend the new floor tone into the existing.
        "erase_large":      {"strength": 0.82, "guidance": 9.0,  "steps": 100, "refiner_start": 0.64},

        "erase_rug":        {"strength": 0.88, "guidance": 12.0, "steps": 80, "refiner_start": 0.72},
        "erase_wall":       {"strength": 0.99, "guidance": 6.5,  "steps": 70, "refiner_start": 0.75},
        # Curtains kept coming back because erase_large's 0.82 strength (tuned to PRESERVE
        # the rug pre-fill) is too gentle to destroy a fabric panel — SDXL just redrew the
        # curtain. erase_curtain uses near-full strength to wipe the fabric, with low
        # guidance so it fills with plain wall + window instead of inventing decor.
        "erase_curtain":    {"strength": 0.97, "guidance": 6.5,  "steps": 80, "refiner_start": 0.70},
        "pillow_recolor":   {"strength": 0.52, "guidance": 18.0, "steps": 65, "refiner_start": 0.80},

        # strength 0.38→0.50, guidance 11→13: 0.38 was too gentle to convert a DARK sofa
        # (e.g. dark green) to a vivid colour — green+purple just muddied to gray/black.
        # 0.50 lets SDXL actually reach the new hue; 13 keeps it strong without neon.
        "vivid_recolor":    {"strength": 0.50, "guidance": 13.0, "steps": 72, "refiner_start": 0.83},

        # guidance 15→11 for the same realism reason. The color hint (now enabled)
        # carries the hue uniformly across all masked objects, so guidance no longer
        # needs to be cranked up to force saturation — lower guidance = more realistic.
        "vivid_warm":       {"strength": 0.40, "guidance": 11.0, "steps": 72, "refiner_start": 0.83},
    }

    LARGE_OBJECTS = [
        "sofa", "couch", "carpet", "rug", "mat", "curtain", "drape",
        "chair", "armchair", "table", "middle table", "coffee table",
        "shelf", "book shelf", "cabinet", "tv unit",
    ]

    FLAT_SURFACE_KEYWORDS = ["carpet", "rug", "mat", "curtain", "drape", "blind"]
    SMALL_OBJECTS = ["pillow", "cushion", "vase", "candle", "books", "blanket", "throw"]
    WALL_ART_OBJECTS = ["picture", "frame", "artwork", "painting", "photograph", "mirror"]

    # ── FIX B: Split into near-black (needs hint) vs vivid-dark (no hint, strength-driven) ──
    # near-black: extreme luminance drop → hint + dark_shift
    # vivid-dark: saturated mid-tone → vivid_recolor (no hint, guidance drives saturation)
    NEAR_BLACK_COLORS = [
        "black", "jet black", "charcoal", "charcoal gray", "dark gray", "dark grey",
        "anthracite", "gunmetal", "espresso", "dark brown",
    ]
    VIVID_DARK_COLORS = [
        "navy", "navy blue", "midnight blue", "dark blue", "deep blue",
        "dark green", "forest green", "hunter green", "dark teal",
        "dark red", "deep red", "burgundy", "maroon",
        "deep purple", "dark purple", "purple", "plum", "indigo",
        "slate", "cobalt",
    ]
    # Keep DARK_COLORS for backward compat (union of both)
    DARK_COLORS = NEAR_BLACK_COLORS + VIVID_DARK_COLORS

    # ── FIX 1b: warm/mid-tone vivid colors that need strength boost ──────────────
    # These are NOT in DARK_COLORS or BRIGHT_COLORS so they fall to recolor mode.
    # Adding them here routes them to vivid_warm mode instead.
    VIVID_WARM_COLORS = [
        "coral", "salmon", "terracotta", "terra cotta", "burnt orange",
        "orange", "pink", "hot pink", "blush", "blush pink", "dusty pink",
        "dusty rose", "rose", "red", "crimson", "scarlet",
        "sky blue", "light blue", "powder blue", "royal blue", "cobalt blue",
        "electric blue", "periwinkle",
        "mustard", "gold", "amber", "ochre", "honey",
        "olive", "sage", "sage green", "army green", "mint", "mint green",
        "lime", "lime green", "emerald",
        "lavender", "lilac", "mauve", "violet", "orchid",
        "turquoise", "aqua", "teal blue",
        "caramel", "cognac", "rust", "copper", "bronze",
        "forest green",
    ]

    MATERIAL_KEYWORDS = [
        "velvet", "leather", "linen", "wool", "silk", "suede", "chenille",
        "boucle", "bouclé", "tweed", "cotton", "polyester", "microfiber",
        "faux leather", "faux fur", "wood", "marble", "rattan", "wicker",
    ]

    CONF_THRESHOLD        = 0.15
    PILLOW_CONF_THRESHOLD = 0.08
    TABLE_CONF_THRESHOLD  = 0.05

    # ── FIX 2a: MASK_EXPANSION 25→12, MASK_BLUR 10→4 ────────────────────────
    # Wide expansion + high blur created a ~35px soft halo around the SAM mask.
    # SDXL inpainted into that halo → color bled onto surrounding rug/floor.
    # Tight values keep the inpaint strictly inside the object silhouette.
    MASK_EXPANSION        = 8
    FLAT_MASK_EXPANSION   = 6
    MASK_BLUR             = 3
    FLAT_MASK_BLUR        = 3

    OUTPUT_ROOT           = BASE_DIR / "outputs" / "sessions"

    MAX_REMOVAL_MASK_COVERAGE = 0.15
    CLAMPED_ERASE_EXPANSION   = 10

    SOFT_HINT_OPACITY      = 0.25
    # ── FIX 1b: separate opacity for extreme flat dark shifts ─────────────────
    # 25% was too faint to pre-condition SDXL toward black on a beige surface.
    # 65% gives a strong dark visual cue while still preserving some texture.
    FLAT_DARK_HINT_OPACITY = 0.65

    KNOWN_COLORS = sorted([
        "dark blue", "navy blue", "forest green", "dark green", "light blue",
        "sky blue", "off white", "dark red", "dark gray", "light gray",
        "dusty pink", "dusty rose", "dusty blue", "pale pink", "pale blue",
        "pale green", "pale yellow", "warm white", "cool white", "pure white",
        "jet black", "midnight blue", "midnight black", "deep red", "deep blue",
        "deep green", "deep purple", "royal blue", "royal purple",
        "cobalt blue", "prussian blue", "electric blue", "powder blue",
        "army green", "sage green", "hunter green", "mint green", "lime green",
        "burnt orange", "terra cotta", "terracotta", "burnt sienna",
        "hot pink", "blush pink", "rose gold", "antique white",
        "gunmetal gray", "charcoal gray", "slate gray", "ash gray",
        "dark brown", "dark teal",
        "red", "orange", "yellow", "green", "lime", "cyan", "teal", "blue",
        "navy", "purple", "violet", "magenta", "pink", "white", "black",
        "gray", "grey", "brown", "beige", "cream", "gold", "silver",
        "turquoise", "coral", "salmon", "burgundy", "maroon", "charcoal",
        "ivory", "rust", "mustard", "olive", "lavender", "lilac",
        "crimson", "scarlet", "indigo", "cobalt", "emerald", "mint",
        "rose", "sand", "chocolate", "tan", "khaki", "copper", "bronze",
        "steel", "midnight", "forest", "sage", "slate", "anthracite",
        "blush", "plum", "mauve", "taupe", "champagne", "caramel",
        "espresso", "cognac", "amber", "ochre", "ecru",
    ], key=len, reverse=True)

    ALL_COLOR_WORDS = [
        "red", "orange", "yellow", "green", "lime", "cyan", "teal", "blue",
        "navy", "purple", "violet", "magenta", "pink", "white", "black",
        "gray", "grey", "brown", "beige", "cream", "gold", "silver",
        "turquoise", "coral", "salmon", "burgundy", "maroon", "charcoal",
        "ivory", "rust", "mustard", "olive", "lavender", "crimson",
        "scarlet", "indigo", "cobalt", "emerald", "rose", "sand",
        "tan", "copper", "bronze", "plum", "mauve", "taupe", "amber",
    ]

    COLOR_RGB_MAP: dict[str, tuple] = {
        "jet black": (10, 10, 10),   "black": (15, 15, 15),
        "charcoal": (54, 54, 54),    "dark gray": (60, 60, 60),   "dark grey": (60, 60, 60),
        "anthracite": (50, 55, 60),  "gunmetal": (44, 53, 57),
        "navy blue": (0, 0, 128),    "navy": (0, 0, 128),         "midnight blue": (25, 25, 112),
        "dark blue": (0, 0, 139),    "deep blue": (0, 0, 180),
        "dark green": (0, 100, 0),   "forest green": (34, 139, 34), "hunter green": (53, 94, 59),
        "dark teal": (0, 80, 80),    "dark brown": (101, 67, 33),  "espresso": (70, 40, 20),
        "dark red": (139, 0, 0),     "deep red": (139, 0, 0),     "burgundy": (128, 0, 32),
        "deep purple": (48, 25, 52), "dark purple": (48, 25, 52),
        "pure white": (255, 255, 255), "bright white": (255, 255, 255),
        "white": (252, 252, 252),    "off white": (245, 242, 235),
        "ivory": (255, 255, 240),    "cream": (255, 253, 208),    "linen": (250, 240, 230),
        "pearl white": (240, 240, 245), "warm white": (255, 248, 220),
        "antique white": (250, 235, 215), "eggshell": (240, 234, 214),
        "gray": (150, 150, 150),     "grey": (150, 150, 150),     "silver": (192, 192, 192),
        "beige": (245, 245, 220),    "taupe": (72, 60, 50),       "tan": (210, 180, 140),
        "brown": (139, 90, 43),      "chocolate": (123, 63, 0),
        "red": (200, 30, 30),        "orange": (230, 120, 30),    "yellow": (230, 210, 30),
        "green": (50, 160, 50),      "teal": (0, 128, 128),       "cyan": (0, 200, 200),
        "blue": (30, 100, 210),      "purple": (128, 0, 128),     "violet": (143, 0, 255),
        "pink": (255, 105, 180),     "magenta": (255, 0, 255),    "rose": (255, 0, 127),
        "coral": (255, 127, 80),     "salmon": (250, 128, 114),
        "gold": (212, 175, 55),      "copper": (184, 115, 51),    "bronze": (140, 120, 83),
        "lavender": (230, 230, 250), "lilac": (200, 162, 200),
        "olive": (128, 128, 0),      "sage": (143, 151, 121),     "mint": (152, 255, 152),
        "turquoise": (64, 224, 208), "cobalt": (0, 71, 171),      "indigo": (75, 0, 130),
        "crimson": (220, 20, 60),    "scarlet": (255, 36, 0),     "maroon": (128, 0, 0),
        "rust": (183, 65, 14),       "mustard": (255, 219, 88),   "amber": (255, 191, 0),
        "plum": (142, 69, 133),      "mauve": (224, 176, 255),
    }

    @classmethod
    def color_name_to_rgb(cls, color_name: str) -> tuple | None:
        cn = color_name.lower().strip()
        for key in sorted(cls.COLOR_RGB_MAP.keys(), key=len, reverse=True):
            if key in cn or cn in key:
                return cls.COLOR_RGB_MAP[key]
        return None

    STYLE_HINTS = [
        "Scandinavian", "Industrial", "Bohemian", "Mid-Century Modern",
        "Minimalist", "Art Deco", "Japandi", "Coastal", "Rustic", "Contemporary",
    ]

    MATERIAL_DESCRIPTORS: dict[str, str] = {
        "velvet":       "rich velvet upholstery with soft sheen, crushed pile texture, light-catching highlights, deep lustrous velvet fabric",
        "leather":      "genuine leather upholstery with natural grain texture, subtle sheen, realistic hide surface, stitched seams",
        "suede":        "suede upholstery with matte nap texture, soft brushed surface, directional pile, no sheen",
        "linen":        "linen upholstery with natural woven texture, subtle slub weave, matte fabric surface",
        "wool":         "wool upholstery with soft textured surface, natural fiber weave, matte finish, slight nap",
        "silk":         "silk upholstery with smooth lustrous sheen, light reflective surface, fine tight weave",
        "boucle":       "boucle upholstery with looped curly texture, nubby woven surface, tactile raised loops",
        "bouclé":       "bouclé upholstery with looped curly texture, nubby woven surface, tactile raised loops",
        "chenille":     "chenille upholstery with soft tufted pile, velvety texture, plush velvety surface",
        "faux leather": "faux leather upholstery with smooth matte surface, natural grain embossed pattern",
        "faux fur":     "faux fur covering with dense fluffy pile, soft texture, plush deep surface",
        "rattan":       "rattan weave with natural wicker texture, open lattice pattern, organic material",
        "wicker":       "wicker weave with natural open lattice texture, organic earthy material",
    }


def clean_gpu():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ─── ALIAS RESOLUTION ─────────────────────────────────────────
def resolve_target_classes(user_target: str) -> list[str]:
    user_target = user_target.lower().strip()
    if user_target in Config.TARGET_ALIASES:
        aliases = Config.TARGET_ALIASES[user_target]
        log.info(f"[dim]Alias expansion: '{user_target}' → {aliases}[/dim]")
        return aliases
    for key, values in Config.TARGET_ALIASES.items():
        if user_target in values:
            log.info(f"[dim]Alias expansion via value match: '{user_target}' → {Config.TARGET_ALIASES[key]}[/dim]")
            return Config.TARGET_ALIASES[key]
    return [user_target]


def is_flat_surface(target: str) -> bool:
    return any(kw in target.lower() for kw in Config.FLAT_SURFACE_KEYWORDS)


def _color_match(user_input: str, color_list: list[str]) -> bool:
    """One-direction substring matching: list_entry ⊆ user_input only.
    Prevents 'gray' matching 'dark gray', 'green' matching 'dark green'.
    Still allows 'navy' matching 'navy blue' (navy ⊆ navy blue ✓).
    """
    cl = user_input.lower().strip()
    for entry in color_list:
        if entry == cl:    return True   # exact match
        if entry in cl:    return True   # list entry is part of user input (navy→navy blue)
        # DO NOT match cl in entry: prevents 'green'→'dark green' false positive
    return False


def is_dark_color(color: str | None) -> bool:
    if color is None:
        return False
    return _color_match(color, Config.DARK_COLORS)


def is_near_black(color: str | None) -> bool:
    """True only for near-black colors (charcoal, espresso, jet black…) → use dark_shift + hint."""
    if color is None:
        return False
    return _color_match(color, Config.NEAR_BLACK_COLORS)


def is_vivid_dark(color: str | None) -> bool:
    """True for saturated deep colors (navy, purple, teal…) → use vivid_recolor, no hint."""
    if color is None:
        return False
    return _color_match(color, Config.VIVID_DARK_COLORS)


def is_vivid_warm(color: str | None) -> bool:
    """True for warm/mid-tone vivid colors (coral, salmon, orange, pink, sky blue…).
    These are NOT dark or bright but still need strength=0.50 to shift from gray furniture.
    → use vivid_warm mode (no hint, guidance=15 drives warm saturation).
    """
    if color is None:
        return False
    # Don't double-classify with dark or bright
    if is_dark_color(color) or is_bright_color(color):
        return False
    return _color_match(color, Config.VIVID_WARM_COLORS)


def is_bright_color(color: str | None) -> bool:
    if color is None:
        return False
    return _color_match(color, Config.BRIGHT_COLORS)


def extract_material(instruction: str) -> str | None:
    instr_lower = instruction.lower()
    for mat in Config.MATERIAL_KEYWORDS:
        if mat in instr_lower:
            return mat
    return None


def is_small_object(target: str) -> bool:
    return any(s in target.lower() for s in Config.SMALL_OBJECTS)


def is_wall_art(target: str) -> bool:
    return any(w in target.lower() for w in Config.WALL_ART_OBJECTS)


def box_overlap_frac(inner, outer) -> float:
    """Fraction of the `inner` box's area that lies inside `outer`.
    Used to veto a weak mis-detection (e.g. a side table weakly seen as a 'chair')
    when a higher-confidence detection of another class covers the same spot.
    """
    ix1 = max(float(inner[0]), float(outer[0])); iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2])); iy2 = min(float(inner[3]), float(outer[3]))
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area  = max(1e-6, (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1])))
    return inter / area


# ─── PROMPT HELPERS ───────────────────────────────────────────
def extract_target_color(instruction: str) -> str | None:
    instruction_lower = instruction.lower()
    for color in Config.KNOWN_COLORS:
        if color in instruction_lower:
            return color
    return None


def _safe_blocked_colors(target_color: str) -> list[str]:
    tc_lower = target_color.lower()
    return [
        c for c in Config.ALL_COLOR_WORDS
        if c not in tc_lower and tc_lower not in c
    ]


def build_recolor_prompts(
    target: str,
    instruction: str,
    previous_color: str | None = None,
    flat_surface: bool = False,
    dark_shift: bool = False,
    bright_shift: bool = False,
    vivid_recolor: bool = False,   # ── FIX F: vivid saturated colors (navy, purple) ──
    vivid_warm: bool = False,          # ── FIX 1f: warm/mid vivid colors (coral, orange, pink…) ──
):
    target_color = extract_target_color(instruction)
    material     = extract_material(instruction)
    shape_neg    = Config.SHAPE_NEGATIVE if not flat_surface else ""

    # ── Pillow / cushion early-return ─────────────────────────────────────
    if target_color and ("pillow" in target.lower() or "cushion" in target.lower()):
        blocked = _safe_blocked_colors(target_color)
        content_prompt = (
            f"{Config.ANTI_REDESIGN_PREFIX}"
            f"{target_color} {target}, decorative {target_color} throw pillow, "
            f"solid {target_color} fabric pillow with natural textile woven texture, "
            f"realistic soft pillow shape with gentle folds and natural volume, "
            f"same pillow same position same size on sofa, only color changed to {target_color}, "
            f"matte {target_color} fabric surface with visible weave grain, "
            f"photorealistic, sharp focus, crisp edges, "
            f"surrounding sofa fabric and all other objects completely unchanged."
        )
        style_prompt = (
            f"{target_color} throw pillow on sofa, decorative {target_color} cushion, "
            f"realistic fabric texture with soft folds and natural shading, "
            f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
        )
        negative_prompt = (
            f"{', '.join(blocked[:15])}, blob, amorphous shape, flat shape, painted shape, "
            f"no texture, no fabric detail, painted look, glowing, neon, solid paint glob, "
            + Config.SHAPE_NEGATIVE
            + Config.SYSTEM_NEGATIVE_PROMPT
        )
        return content_prompt, style_prompt, negative_prompt

    # ── FIX F: vivid_recolor branch — uniform saturation, no hint, no bleed ──────
    # Called when mode=vivid_recolor (navy, purple, teal, cobalt, etc.).
    # Separate from dark_shift to avoid: patchy highlights, hint bleed, shape drift.
    if target_color and vivid_recolor:
        blocked = _safe_blocked_colors(target_color)
        blocked_str = ", ".join(blocked[:20])
        content_prompt = (
            f"{Config.ANTI_REDESIGN_PREFIX}"
            f"{target_color} {target}, uniformly {target_color} upholstery across every surface, "
            f"same {target} same shape same size same silhouette, "
            f"identical proportions identical legs identical arms identical cushions, "
            f"only the fabric color changed to {target_color}, "
            f"consistent {target_color} tone on seat back armrests and all cushions, "
            f"no gray undertone no original color showing through no blotchy patches, "
            f"rich {target_color} fabric with natural woven texture, "
            f"photorealistic sharp focus crisp edges, "
            f"exact same position, surrounding area completely unchanged."
        )
        style_prompt = (
            f"uniformly {target_color} {target}, rich {target_color} upholstered furniture, "
            f"consistent {target_color} fabric tone, natural material texture, "
            f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
        )
        negative_prompt = (
            f"{blocked_str}, "
            "gray undertone, original color, patchy, blotchy, uneven color, "
            "two-tone, highlights showing through, "
            "flat painted look, solid color blob, no texture, "
            + Config.SHAPE_NEGATIVE
            + Config.SYSTEM_NEGATIVE_PROMPT
        )
        return content_prompt, style_prompt, negative_prompt

    # ── FIX 1g: vivid_warm prompt — warm hue commit, no hint, uniform coverage ──
    if target_color and vivid_warm:
        blocked = _safe_blocked_colors(target_color)
        blocked_str = ", ".join(blocked[:20])
        content_prompt = (
            f"{Config.ANTI_REDESIGN_PREFIX}"
            f"{target_color} {target}, uniformly {target_color} upholstery across every surface, "
            f"same {target} same shape same size same silhouette, "
            f"identical proportions identical legs identical arms identical cushions, "
            f"only the fabric color changed to {target_color}, "
            f"consistent warm {target_color} tone on seat back armrests and all cushions, "
            f"no gray undertone no original color showing through no blotchy patches, "
            f"rich {target_color} fabric with natural woven textile texture, "
            f"photorealistic sharp focus crisp edges, "
            f"exact same position, surrounding area completely unchanged."
        )
        style_prompt = (
            f"uniformly {target_color} {target}, warm {target_color} upholstered furniture, "
            f"consistent {target_color} fabric tone, natural material texture, "
            f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
        )
        negative_prompt = (
            f"{blocked_str}, "
            "gray undertone, original color, patchy, blotchy, uneven color, "
            "two-tone, highlights showing through, "
            "flat painted look, solid color blob, no texture, "
            + Config.SHAPE_NEGATIVE
            + Config.SYSTEM_NEGATIVE_PROMPT
        )
        return content_prompt, style_prompt, negative_prompt

    if target_color:

        # ── DARK SHIFT ────────────────────────────────────────────────────
        if dark_shift:
            prev_block = f"{previous_color}, " if previous_color else ""
            light_block = (
                "white, beige, cream, ivory, tan, sand, off white, light gray, silver, "
                "champagne, ecru, linen, warm white, light beige, khaki, wheat"
            )
            blocked_other = [
                c for c in Config.ALL_COLOR_WORDS
                if c not in target_color.lower()
                and target_color.lower() not in c
                and c not in light_block
            ]
            flat_tgt = is_flat_surface(target)

            if flat_tgt:
                # ── FIX 1c: Explicit edge-to-edge color forcing ───────────────
                # Old prompt: "solid woven area rug" — SDXL still left beige edges
                # because it anchored to the existing texture pattern.
                # New prompt: hammers "no beige, no original color, edge to edge"
                # at multiple token positions so the model has no excuse to keep
                # any of the original color at the borders or center.
                content_prompt = (
                    f"{Config.ANTI_REDESIGN_PREFIX}"
                    f"deep {target_color} area rug, entire rug surface {target_color} from edge to edge, "
                    f"same rug same position same weave structure only color changed to {target_color}, "
                    f"deep rich {target_color} woven textile with preserved pile texture, "
                    f"no beige no cream no light patches no original color visible anywhere, "
                    f"uniform {target_color} tone across whole rug — all four corners {target_color}, "
                    f"entire border {target_color}, center {target_color}, "
                    f"matte {target_color} fabric no glow no neon no reflections, "
                    f"photorealistic natural fiber rug sharp focus, "
                    f"exact same size and position, surrounding floor completely unchanged."
                )
                style_prompt = (
                    f"deep {target_color} area rug, dark {target_color} woven floor rug, "
                    f"realistic textile pattern preserved, natural fiber carpet, matte fabric, "
                    f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
                )
                negative_prompt = (
                    f"{prev_block}{light_block}, "
                    f"light colored, wrong color, color drift, "
                    + Config.NEON_NEGATIVE
                    + Config.SYSTEM_NEGATIVE_PROMPT
                )
            else:
                content_prompt = (
                    f"{Config.ANTI_REDESIGN_PREFIX}"
                    f"{target_color} {target}, same {target} same shape same size, "
                    f"same fabric texture same material grain, only color changed to {target_color}, "
                    f"identical silhouette, identical proportions, identical legs, identical arms, "
                    f"{target_color} upholstery on the exact same {target} frame, "
                    f"deep {target_color} fabric with natural texture, solid {target_color} {target}, "
                    f"photorealistic natural fabric, sharp focus, crisp edges, "
                    f"exact same position, surrounding area completely unchanged."
                )
                style_prompt = (
                    f"deep {target_color} {target}, same {target} recolored {target_color}, "
                    f"natural fabric texture preserved, "
                    f"photorealistic interior photography, 8k uhd, sharp focus, studio quality."
                )
                negative_prompt = (
                    f"{prev_block}{light_block}, "
                    f"light colored, wrong color, color drift, "
                    f"{', '.join(blocked_other[:12])}, "
                    + shape_neg
                    + Config.SYSTEM_NEGATIVE_PROMPT
                )

        # ── BRIGHT SHIFT ON FURNITURE ─────────────────────────────────────
        elif bright_shift and not flat_surface:
            prev_block = f"{previous_color}, " if previous_color else ""
            gray_block = (
                "gray, grey, dark gray, light gray, ash gray, silver, charcoal, "
                "beige, tan, taupe, sand, khaki, linen, ecru, warm white, "
                "off white, dirty white, yellowish, grayish, tinted, colored"
            )
            blocked = [
                c for c in Config.ALL_COLOR_WORDS
                if c not in target_color.lower()
                and target_color.lower() not in c
                and c not in gray_block
            ]
            content_prompt = (
                f"{Config.ANTI_REDESIGN_PREFIX}"
                f"pure {target_color} {target}, bright {target_color} upholstery, "
                f"clean {target_color} fabric {target} with natural material texture, "
                f"solid {target_color} {target}, same {target} same shape same size, "
                f"only color changed to {target_color}, identical silhouette, "
                f"identical proportions, identical legs, identical arms, "
                f"{target_color} {target} with natural fabric grain, no gray, no beige, "
                f"photorealistic, sharp focus, crisp edges, "
                f"exact same size and position, surrounding area completely unchanged."
            )
            style_prompt = (
                f"pure {target_color} {target}, bright {target_color} furniture, "
                f"natural fabric texture, photorealistic interior photography, "
                f"8k uhd, natural daylight, sharp focus."
            )
            negative_prompt = (
                f"{prev_block}{gray_block}, "
                f"wrong color, color drift, uneven color, patchy, discolored, faded, "
                f"{', '.join(blocked[:12])}, "
                + shape_neg
                + Config.SYSTEM_NEGATIVE_PROMPT
            )

        # ── FLAT SURFACE RECOLOR ──────────────────────────────────────────
        elif flat_surface:
            prev_block = f"{previous_color}, " if previous_color else ""
            bright = is_bright_color(target_color)

            if bright:
                gray_block = (
                    "gray, grey, dark gray, light gray, ash gray, silver, charcoal, "
                    "beige, tan, taupe, sand, khaki, cream, linen, ecru, warm white, "
                    "off white, dirty white, yellowish white, grayish white, "
                    "brown, dark brown, navy, blue, colored, tinted"
                )
                blocked = [
                    c for c in Config.ALL_COLOR_WORDS
                    if c not in target_color.lower()
                    and target_color.lower() not in c
                    and c not in gray_block
                ]
                content_prompt = (
                    f"{Config.ANTI_REDESIGN_PREFIX}"
                    f"bright {target_color} {target}, "
                    f"same rug pattern same geometric weave design, "
                    f"identical carpet texture structure only color changed to {target_color}, "
                    f"pure {target_color} woven floor covering with preserved textile pattern, "
                    f"clean {target_color} rug with natural fiber pile texture and original weave, "
                    f"all four corners {target_color}, entire border {target_color}, "
                    f"edge to edge {target_color}, no gray tint, no neon, "
                    f"matte natural fiber surface, photorealistic, sharp focus, "
                    f"exact same size and position, surrounding area completely unchanged."
                )
                style_prompt = (
                    f"pure {target_color} {target}, bright {target_color} woven floor rug, "
                    f"natural fiber textile with preserved weave pattern, "
                    f"photorealistic architectural photography, 8k uhd, natural daylight, sharp focus."
                )
                negative_prompt = (
                    f"{prev_block}{gray_block}, "
                    f"wrong color, color drift, uneven color, patchy, discolored, faded, "
                    f"original color at edges, beige border, {', '.join(blocked[:10])}, "
                    + Config.NEON_NEGATIVE
                    + Config.SYSTEM_NEGATIVE_PROMPT
                )
            else:
                blocked = _safe_blocked_colors(target_color)
                content_prompt = (
                    f"{Config.ANTI_REDESIGN_PREFIX}"
                    f"{target_color} {target}, "
                    f"same rug pattern same geometric design same border design, "
                    f"identical carpet weave texture only color changed to {target_color}, "
                    f"realistic {target_color} woven area rug with preserved original pattern, "
                    f"natural fiber pile texture and original weave structure intact, "
                    f"uniform {target_color} tone, matte fabric surface, no glow, no neon, "
                    f"all four corners {target_color}, entire border {target_color}, "
                    f"consistent {target_color} color across entire rug, "
                    f"no original color at edges, photorealistic natural fabric, sharp focus, "
                    f"exact same size and position, surrounding floor completely unchanged."
                )
                style_prompt = (
                    f"solid {target_color} {target}, {target_color} woven textile with pattern, "
                    f"natural fiber rug with preserved design, matte fabric surface, "
                    f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
                )
                negative_prompt = (
                    f"{prev_block}"
                    f"{', '.join(blocked[:20])}, wrong color, color drift, uneven color, "
                    f"patchy, multicolor, original color at edges, beige border, cream border, "
                    + Config.NEON_NEGATIVE
                    + Config.SYSTEM_NEGATIVE_PROMPT
                )

        # ── STANDARD FURNITURE RECOLOR ────────────────────────────────────
        else:
            prev_block = f"{previous_color}, " if previous_color else ""
            blocked = _safe_blocked_colors(target_color)
            content_prompt = (
                f"{Config.ANTI_REDESIGN_PREFIX}"
                f"{target_color} {target}, same {target} same shape same size, "
                f"same fabric texture same material surface, "
                f"only the color changed to {target_color}, identical silhouette, "
                f"identical proportions, {target_color} colored {target}, "
                f"{target_color} {target} finish with natural material texture, "
                f"same design recolored {target_color}, "
                f"photorealistic natural texture, sharp focus, crisp edges, "
                f"exact same size and position, surrounding area completely unchanged."
            )
            style_prompt = (
                f"{target_color} {target}, {target_color} color same {target}, "
                f"natural material texture preserved, "
                f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
            )
            negative_prompt = (
                f"{prev_block}"
                f"{', '.join(blocked[:20])}, wrong color, color drift, "
                f"flat painted look, no texture, solid color blob, "
                + shape_neg
                + Config.SYSTEM_NEGATIVE_PROMPT
            )

    # ── MATERIAL CHANGE ───────────────────────────────────────────────────
    elif material:
        log.info(f"[dim]Material keyword detected: '{material}' — building material-specific prompt.[/dim]")
        mat_desc = Config.MATERIAL_DESCRIPTORS.get(
            material, f"{material} texture, realistic {material} surface"
        )
        content_prompt = (
            f"{Config.ANTI_REDESIGN_PREFIX}"
            f"a {target} upholstered in {material}, {mat_desc}, "
            f"same {target} same shape same size, only the upholstery material changed to {material}, "
            f"identical silhouette, identical proportions, identical legs and frame, "
            f"close-up visible {material} texture and surface detail, "
            f"photorealistic, sharp focus, crisp edges, high detail, "
            f"exact same position, surrounding area completely unchanged."
        )
        style_prompt = (
            f"{material} {target}, realistic {mat_desc}, "
            f"photorealistic interior photography, 8k uhd, studio lighting, sharp focus."
        )
        negative_prompt = (
            f"wrong material, wrong texture, smooth plain fabric, "
            f"no texture, flat surface, painted look, "
            + shape_neg
            + Config.SYSTEM_NEGATIVE_PROMPT
        )

    # ── GENERIC STYLE EDIT ────────────────────────────────────────────────
    else:
        log.info("[dim]No color or material found — treating as generic style/instruction edit.[/dim]")
        content_prompt = (
            f"{Config.ANTI_REDESIGN_PREFIX}"
            f"a {target}, {instruction}, "
            f"same {target} same shape same size, only the material or style changed, "
            f"identical silhouette, identical proportions, "
            f"photorealistic, sharp focus, crisp edges, "
            f"exact same size and position, surrounding area completely unchanged."
        )
        style_prompt = (
            f"{target} with {instruction}, same {target} restyled, "
            f"photorealistic interior photography, 8k uhd, natural lighting, sharp focus."
        )
        negative_prompt = (
            "wrong material, wrong texture, wrong style, color shift, "
            + shape_neg
            + Config.SYSTEM_NEGATIVE_PROMPT
        )

    return content_prompt, style_prompt, negative_prompt


# ─── SESSION REPORT ───────────────────────────────────────────
class SessionReport:
    def __init__(self):
        self._entries: list[dict] = []

    def add(self, **kwargs):
        self._entries.append(kwargs)

    def print_report(self):
        if not self._entries:
            console.print("[dim]No changes recorded.[/dim]")
            return

        console.rule("[bold cyan]SESSION REPORT[/bold cyan]")

        table = Table(
            title="Changes Applied",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            show_lines=True,
        )
        table.add_column("#",         style="dim", width=4,  justify="right")
        table.add_column("Iteration", width=9,  justify="center")
        table.add_column("Target",    width=14)
        table.add_column("Change",    width=40)
        table.add_column("Mode",      width=15)
        table.add_column("File",      width=30, overflow="fold")

        for i, e in enumerate(self._entries, 1):
            iteration   = str(e.get("iteration", "—"))
            target      = e.get("target") or "—"
            action      = e.get("action", "")
            from_color  = e.get("from_color")
            to_color    = e.get("to_color")
            to_material = e.get("to_material")
            instruction = e.get("instruction", "")
            mode        = e.get("mode", "—")
            passes      = e.get("passes", 1)
            saved_path  = e.get("saved_path")

            if action == "initial_redesign":
                change_desc = f"Initial redesign — {instruction[:50]}"
            elif action == "global":
                change_desc = f"Global redesign — {instruction[:50]}"
            elif to_color:
                if from_color:
                    change_desc = f"Color: [bold]{from_color}[/bold] → [bold green]{to_color}[/bold green]"
                else:
                    change_desc = f"Color → [bold green]{to_color}[/bold green]"
                if passes > 1:
                    change_desc += f" ({passes}-pass)"
            elif to_material:
                change_desc = f"Material → [bold cyan]{to_material}[/bold cyan]"
            else:
                change_desc = instruction[:55]

            fname = saved_path.name if saved_path else "—"
            table.add_row(str(i), iteration, target, change_desc, mode, fname)

        console.print(table)
        console.print("\n[bold yellow]Summary[/bold yellow]")

        for e in self._entries:
            a = e.get("action", "")
            if e.get("to_color"):
                fc = f"from [dim]{e['from_color']}[/dim] " if e.get("from_color") else ""
                p  = f" ({e['passes']}-pass)" if e.get("passes", 1) > 1 else ""
                console.print(f"  • [cyan]{e['target']}[/cyan] color {fc}→ [bold green]{e['to_color']}[/bold green]{p}")
            elif e.get("to_material"):
                console.print(f"  • [cyan]{e['target']}[/cyan] material → [bold cyan]{e['to_material']}[/bold cyan]")
            elif a == "style":
                console.print(f"  • [cyan]{e['target']}[/cyan] style: {e['instruction'][:60]}")
            elif a == "global":
                console.print(f"  • Global redesign: {e['instruction'][:60]}")
            elif a == "remove":
                console.print(f"  • [red]Removed[/red] [cyan]{e['target']}[/cyan]")

        console.print(f"\n[bold]Total iterations:[/bold] {len(self._entries)}")
        last = self._entries[-1].get("saved_path") if self._entries else None
        if last:
            console.print(f"[bold]Final output:[/bold] [cyan]{last}[/cyan]")
        console.rule()


# ─── SDXL ENGINE ──────────────────────────────────────────────
class SDXLFixedEngine:
    def __init__(self):
        from diffusers import (StableDiffusionXLInpaintPipeline,
                               StableDiffusionXLImg2ImgPipeline,
                               AutoencoderKL)

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        log.info("[bold cyan]Loading VAE with FP16 Fix...[/bold cyan]")
        vae_fix = AutoencoderKL.from_pretrained(Config.SDXL_VAE_FP16_FIX, torch_dtype=dtype)

        log.info("Loading SDXL Base Inpaint...")
        self.base = StableDiffusionXLInpaintPipeline.from_pretrained(
            Config.SDXL_INPAINT_MODEL, vae=vae_fix, torch_dtype=dtype,
            variant="fp16" if DEVICE == "cuda" else None, use_safetensors=True
        )
        if DEVICE == "cuda":
            self.base.enable_model_cpu_offload()
            self.base.enable_vae_tiling()
        else:
            self.base.to("cpu")

        log.info("Loading SDXL Refiner...")
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            Config.SDXL_REFINER_MODEL, vae=vae_fix,
            text_encoder_2=self.base.text_encoder_2,
            torch_dtype=dtype,
            variant="fp16" if DEVICE == "cuda" else None,
            use_safetensors=True
        )
        if DEVICE == "cuda":
            self.refiner.enable_model_cpu_offload()

    def run(self, image, mask, mode_settings, content_prompt, style_prompt,
            negative_prompt=None, passes: int = 1, color_hint: tuple | None = None,
            mode_name: str = ""):
        """
        mode_name: pass the mode key string (e.g. 'flat_dark_shift') so the engine
        can choose the correct hint opacity. Defaults to "" (uses SOFT_HINT_OPACITY).
        """
        if negative_prompt is None:
            negative_prompt = Config.SYSTEM_NEGATIVE_PROMPT

        s = mode_settings
        w, h = image.size
        # Round UP to the next multiple of 64 instead of DOWN. Rounding down forced a
        # bigger downscale (e.g. 700→640) that softened detail and compounded over
        # iterations. Rounding up (700→704) is a tiny upscale that keeps detail. This
        # is a clean LANCZOS resize (NOT edge-padding, which caused artifacts before).
        target_w = ((w + 63) // 64) * 64
        target_h = ((h + 63) // 64) * 64

        current_image = image

        if color_hint is not None:
            hint_arr   = np.array(current_image.convert("RGB")).copy().astype(np.float32)
            mask_arr   = np.array(mask.convert("L"))
            solid      = mask_arr > 127
            hint_color = np.array(color_hint, dtype=np.float32)

            # ── FIX 1e: flat_dark_shift uses 65% opacity; everything else 25% ─
            # Reason: beige→black is an extreme luminance jump. 25% hint barely
            # nudges the starting latent; SDXL snaps back to beige on its prior.
            # 65% pre-fills the mask with a clearly dark value so the model only
            # needs to refine "dark" rather than invent it from a bright base.
            if mode_name == "flat_dark_shift":
                _hint_opacity = Config.FLAT_DARK_HINT_OPACITY
            elif mode_name == "flat_recolor":
                # 0.55: a carpet is a big flat area with strong existing pattern;
                # it needs a strong pre-tint so the new color covers the WHOLE rug
                # and is clearly visible (0.40 left it looking mostly unchanged).
                _hint_opacity = 0.55
            elif mode_name == "pillow_recolor":
                # 0.50: ensures EVERY pillow in the mask takes the color, not just one.
                _hint_opacity = 0.50
            elif mode_name in ("vivid_recolor", "vivid_warm"):
                # 0.52: vivid/saturated colours (purple, navy, teal…) on a DARK sofa need a
                # stronger pre-tint or they mix with the dark original into gray/black.
                _hint_opacity = 0.52
            elif mode_name == "recolor":
                # 0.40: balanced for ordinary colours — carries the hue while keeping fabric
                # texture (0.55 looked flat/fake, so 0.40 stays for the realistic look).
                _hint_opacity = 0.40
            elif mode_name == "dark_shift":
                # 0.45: black/charcoal on furniture came out only gray at 0.25 — the dark
                # pre-tint was too weak. 0.45 pushes it to a true dark without going fully
                # flat like the carpet path (0.65).
                _hint_opacity = 0.45
            else:
                _hint_opacity = Config.SOFT_HINT_OPACITY
            hint_arr[solid] = (
                hint_arr[solid] * (1.0 - _hint_opacity) +
                hint_color      *        _hint_opacity
            )
            current_image = Image.fromarray(hint_arr.astype(np.uint8))
            log.info(f"[dim]Soft hint: RGB{color_hint} at {_hint_opacity*100:.0f}% "
                     f"over {solid.sum()} px.[/dim]")

        for pass_num in range(1, passes + 1):
            if passes > 1:
                log.info(f"[bold cyan]Pass {pass_num}/{passes}...[/bold cyan]")

            img_r  = current_image.resize((target_w, target_h), Image.LANCZOS)
            mask_r = mask.resize((target_w, target_h), Image.LANCZOS)

            seed = int(time.time()) + int(np.random.randint(0, 99999))
            generator = torch.Generator(DEVICE).manual_seed(seed)
            log.info(f"SDXL seed (pass {pass_num}): {seed}")

            latents = self.base(
                prompt=content_prompt,
                prompt_2=style_prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                image=img_r,
                mask_image=mask_r,
                strength=s["strength"],
                guidance_scale=s["guidance"],
                num_inference_steps=s["steps"],
                denoising_end=s["refiner_start"],
                output_type="latent",
                num_images_per_prompt=1,
                generator=generator,
            ).images

            refined = self.refiner(
                prompt=content_prompt,
                prompt_2=style_prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                image=latents,
                num_inference_steps=s["steps"],
                denoising_start=s["refiner_start"],
                num_images_per_prompt=1,
                generator=generator,
            ).images[0]

            # ── Hard-boundary composite ──────────────────────────────────
            #   mask < 50  → alpha = 0.0  (fully original — zero bleed outside silhouette)
            #   mask 50–200 → linear ramp (small feather for smooth edge)
            #   mask > 200  → alpha = 1.0 (fully refined)
            if (w, h) != (target_w, target_h):
                refined_full = refined.resize((w, h), Image.LANCZOS)
                mask_full    = mask_r.resize((w, h), Image.LANCZOS).convert("L")
                base_arr     = np.array(current_image.convert("RGB")).astype(float)
                refined_arr  = np.array(refined_full.convert("RGB")).astype(float)
                alpha        = np.array(mask_full).astype(float)           # 0–255
                alpha        = np.clip((alpha - 50.0) / 150.0, 0.0, 1.0)  # hard cutoff
                alpha        = alpha[:, :, np.newaxis]
                blended      = (refined_arr * alpha + base_arr * (1 - alpha)).astype(np.uint8)
                current_image = Image.fromarray(blended)
            else:
                current_image = refined

        return current_image


# ─── INTERACTIVE PIPELINE ─────────────────────────────────────
class InteractivePipeline:
    def __init__(self):
        console.print(Panel.fit(
            "[bold cyan]Interior AI Studio — Local VSCode Edition[/bold cyan]\n"
            "[white]YOLO11s · SAM · SDXL[/white]",
            border_style="blue"
        ))
        Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        from segment_anything import sam_model_registry, SamPredictor
        log.info(f"Loading SAM from {Config.SAM_CHECKPOINT}")
        sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
        self.sam_predictor = SamPredictor(sam)

        from ultralytics import YOLO
        yolo_path = str(Config.MODELS_DIR / Config.YOLO_MODEL)
        log.info(f"Loading YOLO from {yolo_path}")
        self.yolo = YOLO(yolo_path)
        if DEVICE == "cuda":
            self.yolo.to("cuda")

        self.sdxl   = SDXLFixedEngine()
        self.report = SessionReport()

    def save_iteration(self, image: Image.Image, iteration: int) -> Path:
        target_dir = Config.OUTPUT_ROOT / "session"
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = target_dir / f"iter_{iteration:02d}_{int(time.time())}.webp"
        image.save(filename, "WEBP", quality=Config.WEBP_QUALITY)
        return filename

    def preview_mask(self, image_pil: Image.Image, mask: Image.Image,
                     target: str, iteration: int) -> Path:
        img_np   = np.array(image_pil.convert("RGB"))
        mask_np  = np.array(mask.convert("L"))
        overlay  = img_np.copy()
        red_area = mask_np > 30
        overlay[red_area, 0] = np.clip(overlay[red_area, 0].astype(int) + 130, 0, 255)
        overlay[red_area, 1] = (overlay[red_area, 1] * 0.35).astype(np.uint8)
        overlay[red_area, 2] = (overlay[red_area, 2] * 0.35).astype(np.uint8)

        preview_dir  = Config.OUTPUT_ROOT / "mask_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"mask_{iteration:02d}_{target.replace(' ', '_')}.png"
        Image.fromarray(overlay).save(preview_path)

        h, w     = img_np.shape[:2]
        coverage = red_area.sum() / (h * w) * 100
        if coverage < 2.0:
            log.warning(f"[yellow]Mask coverage {coverage:.1f}% — SAM may have missed '{target}'.[/yellow]")
        elif coverage > 45.0:
            log.warning(f"[yellow]Mask coverage {coverage:.1f}% — may be covering too much.[/yellow]")
        else:
            log.info(f"Mask coverage: {coverage:.1f}% ✔")

        log.info(f"[dim]Mask preview: {preview_path}[/dim]")
        return preview_path

    def get_detected_objects(self, rgb_img: np.ndarray) -> list:
        results = self.yolo(rgb_img, conf=Config.CONF_THRESHOLD, verbose=False)[0]
        return list(set(self.yolo.names[int(b.cls[0])].lower() for b in results.boxes))

    def generate_mask(self, rgb_img: np.ndarray,
                      target_classes: list = None,
                      avoid_structural: bool = True,
                      flat_surface: bool = False,
                      conf_override: float = None,
                      expansion_override: int = None,
                      exclude_classes: list = None,
                      exclude_erode: int = 0,
                      constrain_to_boxes: bool = False,
                      low_conf_aux: float = None):
        # exclude_classes: object classes to carve OUT of the target mask (e.g. carve
        #   pillows out of a sofa recolor, or the window out of a curtain removal).
        # exclude_erode: >0 shrinks the carve so a small lip of the target still covers
        #   the excluded object's edge (kills seams); <0 grows it for extra protection.
        # The raw exclusion is also stored on self._last_user_exclusion so callers that
        # do their own post-dilation (removal flow) can re-protect after they dilate.
        conf = conf_override if conf_override is not None else Config.CONF_THRESHOLD
        results = self.yolo(rgb_img, conf=conf, verbose=False)[0]
        detected_appliances = set()

        if DEVICE == "cuda":
            self.sam_predictor.model.to("cuda")
        self.sam_predictor.set_image(rgb_img)

        # all detections (name, conf, box) — used for the cross-class veto below so a
        # target match that's really a different, higher-confidence object is rejected.
        all_dets = [(self.yolo.names[int(b.cls[0])].lower(),
                     float(b.conf[0]),
                     b.xyxy[0].cpu().numpy()) for b in results.boxes]

        # DIAGNOSTIC: dump exactly what YOLO saw + where, so we can tell whether a bad
        # mask is a mis-detection vs SAM bleed instead of guessing. Shows class:conf(x1,y1).
        if target_classes and all_dets:
            log.info("[dim]YOLO detections: " + ", ".join(
                f"{n}:{c:.2f}@({int(b[0])},{int(b[1])})" for n, c, b in all_dets) + "[/dim]")

        # Optional LOW-CONFIDENCE auxiliary pass. Catches weak detections of NON-target
        # objects (e.g. a small side table only faintly seen) so they can still (a) VETO a
        # target mis-match sitting on top of them and (b) be carved out of the mask. Target
        # matching still uses the main `conf` ONLY — the aux pass never adds a target, so it
        # cannot cause a false seating recolor; it only protects neighbours.
        aux_dets = []
        if low_conf_aux is not None and low_conf_aux < conf:
            aux_results = self.yolo(rgb_img, conf=low_conf_aux, verbose=False)[0]
            aux_dets = [(self.yolo.names[int(b.cls[0])].lower(),
                         float(b.conf[0]),
                         b.xyxy[0].cpu().numpy()) for b in aux_results.boxes]
            if aux_dets:
                log.info("[dim]Low-conf aux (veto/carve only): " + ", ".join(
                    f"{n}:{c:.2f}" for n, c, _ in aux_dets) + "[/dim]")

        combined_mask  = np.zeros(rgb_img.shape[:2], dtype=bool)
        exclusion_mask = np.zeros(rgb_img.shape[:2], dtype=bool)
        user_exclusion_mask = np.zeros(rgb_img.shape[:2], dtype=bool)
        self._last_user_exclusion = None
        matched_classes = []

        for box in results.boxes:
            name = self.yolo.names[int(box.cls[0])].lower()

            if target_classes:
                if flat_surface and any(ex in name for ex in Config.FLAT_SURFACE_EXCLUSIONS):
                    m_ex, _, _ = self.sam_predictor.predict(
                        box=box.xyxy[0].cpu().numpy(), multimask_output=False)
                    exclusion_mask |= m_ex[0]

                if exclude_classes and any(ex in name for ex in exclude_classes):
                    m_ux, _, _ = self.sam_predictor.predict(
                        box=box.xyxy[0].cpu().numpy(), multimask_output=False)
                    user_exclusion_mask |= m_ux[0]

                if any(t in name or name in t for t in target_classes):
                    bxy = box.xyxy[0].cpu().numpy()
                    conf_here = float(box.conf[0])
                    # Cross-class veto: if a DIFFERENT, non-target class with >= confidence
                    # covers most of this box, this match is probably a mis-detection of
                    # that object (e.g. a side table weakly seen as a 'chair'). Skip it so
                    # only real furniture of the target type is masked — nothing else.
                    if constrain_to_boxes:
                        vetoed = False
                        for on, oc, ob in (all_dets + aux_dets):
                            if on == name:
                                continue
                            if any(t in on or on in t for t in target_classes):
                                continue                      # another real target → keep
                            if box_overlap_frac(bxy, ob) <= 0.60:
                                continue
                            # A table/desk that strongly overlaps a 'chair'/'sofa' match is
                            # almost certainly the real object (seating is never a table), so
                            # let it veto even at LOWER confidence — this is the recurring
                            # "side table recolored as a chair" bug. Other classes still need
                            # >= confidence to veto (so a real seat isn't wrongly dropped).
                            is_hard_nonseat = any(k in on for k in ("table", "desk"))
                            if oc >= conf_here or is_hard_nonseat:
                                vetoed = True
                                log.info(f"[dim]Veto match '{name}' ({conf_here:.2f}) — "
                                         f"covered by '{on}' ({oc:.2f}).[/dim]")
                                break
                        if vetoed:
                            continue
                    matched_classes.append(name)
                    if constrain_to_boxes:
                        # ── Recolor masking (friend's color technique) ───────────────
                        # SAM with the box + a centre point, multimask, then take the
                        # highest-SCORE (most precise) mask and close small holes. This
                        # hugs the furniture accurately — no box-rectangle blob over the
                        # table/rug/floor, and no neighbour bleed. A generous box-clip is
                        # kept only as a safety net (won't trim the piece).
                        cx = float((bxy[0] + bxy[2]) / 2.0)
                        cy = float((bxy[1] + bxy[3]) / 2.0)
                        ms, sc, _ = self.sam_predictor.predict(
                            point_coords=np.array([[cx, cy]]),
                            point_labels=np.array([1]),
                            box=bxy,
                            multimask_output=True,
                        )
                        obj_mask = ms[int(np.argmax(sc))].astype(np.uint8)
                        obj_mask = cv2.morphologyEx(
                            obj_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                        # Fill enclosed interior holes so a patch SAM left inside the sofa
                        # (e.g. a shadow/seam it skipped) still gets recolored. Only fills
                        # holes SURROUNDED by the mask — never grows outward onto neighbours.
                        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(obj_mask, cnts, -1, 1, cv2.FILLED)
                        obj_mask = obj_mask.astype(bool)
                        H, W = obj_mask.shape
                        pad  = 10
                        x1 = max(0, int(bxy[0]) - pad); y1 = max(0, int(bxy[1]) - pad)
                        x2 = min(W, int(bxy[2]) + pad); y2 = min(H, int(bxy[3]) + pad)
                        box_rect = np.zeros_like(obj_mask)
                        box_rect[y1:y2, x1:x2] = True
                        obj_mask = obj_mask & box_rect
                    else:
                        # Removal path — unchanged single-box mask (the user is happy with it).
                        m, _, _ = self.sam_predictor.predict(box=bxy, multimask_output=False)
                        obj_mask = m[0]
                    combined_mask |= obj_mask
                continue

            for app in Config.APPLIANCE_KEYWORDS:
                if app in name:
                    detected_appliances.add(app)

            if avoid_structural and any(kw in name for kw in Config.STRUCTURAL_KEYWORDS):
                continue

            m, _, _ = self.sam_predictor.predict(
                box=box.xyxy[0].cpu().numpy(), multimask_output=False)
            combined_mask |= m[0]

        # Carve aux (low-conf) exclusion detections too, so a faintly-seen side table is
        # protected from a seating recolor even when it was below the main threshold.
        if target_classes and exclude_classes and aux_dets:
            for on, oc, ob in aux_dets:
                if any(ex in on for ex in exclude_classes):
                    m_ax, _, _ = self.sam_predictor.predict(box=ob, multimask_output=False)
                    user_exclusion_mask |= m_ax[0]

        if target_classes:
            if matched_classes:
                log.info(f"[dim]Matched YOLO classes: {list(set(matched_classes))}[/dim]")
            else:
                log.warning(
                    f"[yellow]No boxes matching {target_classes} — trying fallback...[/yellow]"
                )
                if results.boxes:
                    TABLE_KWS = ["table", "middle table", "coffee table", "desk"]
                    is_table_target = any(any(tk in tc for tk in TABLE_KWS) for tc in target_classes)
                    h_f, w_f = rgb_img.shape[:2]

                    if is_table_target:
                        cx_img, cy_img = w_f / 2, h_f * 0.55
                        best = min(
                            results.boxes,
                            key=lambda b: (
                                ((b.xyxy[0][0] + b.xyxy[0][2]) / 2 - cx_img) ** 2 +
                                ((b.xyxy[0][1] + b.xyxy[0][3]) / 2 - cy_img) ** 2
                            )
                        )
                        best_name = self.yolo.names[int(best.cls[0])].lower()
                        log.warning(f"[yellow]Table fallback: center-closest box '{best_name}'[/yellow]")
                    else:
                        best = max(
                            results.boxes,
                            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
                        )
                        best_name = self.yolo.names[int(best.cls[0])].lower()
                        log.warning(f"[yellow]Using largest box: '{best_name}'[/yellow]")

                    m, _, _ = self.sam_predictor.predict(
                        box=best.xyxy[0].cpu().numpy(), multimask_output=False)
                    combined_mask |= m[0]

                if not combined_mask.any():
                    log.warning("[yellow]Fallback also empty — returning None.[/yellow]")
                    if DEVICE == "cuda":
                        self.sam_predictor.model.to("cpu")
                    clean_gpu()
                    return None, detected_appliances

        # Fill enclosed holes in the flat-surface (rug) mask BEFORE carving furniture.
        # SAM under-segments a busy patterned rug — it masks the dominant field but drops
        # contrasting motifs (medallions/borders) as separate regions, leaving holes that
        # survive as gray remnants after removal ("carpet not removed cleanly"). Filling
        # holes SURROUNDED by rug pulls those motifs in; it never grows outward, and the
        # furniture sitting on the rug is re-protected below, so this can't delete furniture.
        if flat_surface and combined_mask.any():
            cm = combined_mask.astype(np.uint8)
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cm, cnts, -1, 1, cv2.FILLED)
            filled = int(cm.sum()) - int(combined_mask.sum())
            if filled > 0:
                log.info(f"[dim]Flat-surface hole-fill: +{filled} px (rug motifs SAM dropped).[/dim]")
            combined_mask = cm.astype(bool)

        if flat_surface and exclusion_mask.any():
            before = combined_mask.sum()
            combined_mask = combined_mask & ~exclusion_mask
            after  = combined_mask.sum()
            log.info(f"[dim]Exclusion mask removed {before - after} px from flat-surface mask.[/dim]")

        if expansion_override is not None:
            expansion = expansion_override
            blur_r    = Config.FLAT_MASK_BLUR
        else:
            expansion = Config.FLAT_MASK_EXPANSION if flat_surface else Config.MASK_EXPANSION
            blur_r    = Config.FLAT_MASK_BLUR      if flat_surface else Config.MASK_BLUR

        combined_mask_uint8 = combined_mask.astype(np.uint8)
        if expansion > 0:
            kernel = np.ones((expansion, expansion), np.uint8)
            combined_mask_uint8 = cv2.dilate(combined_mask_uint8, kernel)

        # Re-protect furniture AFTER dilation: the exclusion above runs before the
        # dilation, so the dilation can re-grow the carpet mask back onto furniture
        # legs/bases sitting on the rug → removal deletes parts of the furniture.
        # Carving the (slightly dilated) furniture out here keeps the mask off them.
        # 12px margin: paired with the wider carpet erase_expansion (6px) below, this keeps
        # the NET furniture clearance at ~6px (12 carve − 6 regrow) — same protection as
        # before — while letting the rug mask reach 4px further at its free edges so it
        # stops leaving rug slivers behind. (At 8px the table got clipped earlier; 12px
        # keeps it safe.)
        if flat_surface and exclusion_mask.any():
            excl_protect = cv2.dilate(exclusion_mask.astype(np.uint8),
                                      np.ones((12, 12), np.uint8))
            combined_mask_uint8[excl_protect > 0] = 0

        # Carve user-requested exclusions (e.g. pillows out of a sofa recolor). A small
        # erosion (exclude_erode>0) leaves a thin lip of the target covering the object's
        # edge so there's no seam; <0 grows the protected region for extra safety.
        if exclude_classes is not None and user_exclusion_mask.any():
            self._last_user_exclusion = user_exclusion_mask.copy()
            carve = user_exclusion_mask.astype(np.uint8)
            if exclude_erode > 0:
                carve = cv2.erode(carve, np.ones((exclude_erode, exclude_erode), np.uint8))
            elif exclude_erode < 0:
                carve = cv2.dilate(carve, np.ones((-exclude_erode, -exclude_erode), np.uint8))
            combined_mask_uint8[carve > 0] = 0
            log.info(f"[dim]Excluded {exclude_classes} from mask "
                     f"(erode={exclude_erode}px, {int(user_exclusion_mask.sum())} px protected).[/dim]")

        pil_mask = Image.fromarray(combined_mask_uint8 * 255).convert("L")
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur_r))

        if DEVICE == "cuda":
            self.sam_predictor.model.to("cpu")
        clean_gpu()
        return pil_mask, detected_appliances

    def _sample_surrounding_color(self, mask: Image.Image, rgb_img: np.ndarray,
                                   expand_px: int = 90,
                                   prefer_dark: bool = False) -> tuple | None:
        mask_arr = np.array(mask.convert("L"))
        kern     = np.ones((expand_px, expand_px), np.uint8)
        expanded = cv2.dilate(mask_arr, kern)
        surround = (expanded > 127) & (mask_arr <= 30)
        pixels   = rgb_img[surround]
        if len(pixels) < 200:
            return None
        # prefer_dark: a big rug fills most of the floor, so the surrounding ring is a
        # mix of dark hardwood + light rug. Averaging gives a rug-ish tone and SDXL just
        # regenerates the rug ("carpet remove did not work"). Taking the darker pixels
        # (bottom 35% by brightness) locks onto the hardwood so the fill is real floor.
        if prefer_dark:
            brightness = pixels.mean(axis=1)
            cutoff     = np.percentile(brightness, 35)
            dark_px    = pixels[brightness <= cutoff]
            if len(dark_px) >= 50:
                return tuple(dark_px.mean(axis=0).astype(int))
        return tuple(pixels.mean(axis=0).astype(int))

    def _keep_large_components(self, mask: Image.Image, min_area_frac: float = 0.004) -> Image.Image:
        """Drop small stray blobs from a mask, keeping only the sizeable regions (the real
        furniture). A component is kept if its area >= min_area_frac of the image, OR it is
        the largest component (so we never delete everything). Soft feathered edges kept.
        """
        mask_arr = np.array(mask.convert("L"))
        binary   = (mask_arr > 127).astype(np.uint8)
        if binary.sum() == 0:
            return mask

        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 2:                      # background + a single component → nothing to drop
            return mask

        total    = binary.shape[0] * binary.shape[1]
        min_area = max(int(total * min_area_frac), 1)
        largest  = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])

        keep    = np.zeros_like(binary)
        dropped = 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area or i == largest:
                keep[labels == i] = 1
            else:
                dropped += area

        if dropped == 0:
            return mask
        keep_dil = cv2.dilate(keep, np.ones((7, 7), np.uint8))   # recover feather
        out = np.where(keep_dil > 0, mask_arr, 0).astype(np.uint8)
        log.info(f"[dim]Cluster cleanup: dropped {dropped} px of stray blobs.[/dim]")
        return Image.fromarray(out).convert("L")

    def direct_recolor(self, image: Image.Image, mask: Image.Image,
                       color_name: str) -> Image.Image | None:
        """Deterministic pixel recolor (no SDXL). Referenced by the apply step; only used
        if _use_direct_recolor is set (currently never), kept so the call never errors."""
        target_rgb = Config.color_name_to_rgb(color_name) if color_name else None
        if target_rgb is None:
            return None
        img_np  = np.array(image.convert("RGB")).astype(np.float32)
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
        mask_np = np.clip(mask_np * 1.6 - 0.25, 0.0, 1.0)
        mask_np = cv2.GaussianBlur(mask_np, (0, 0), 0.6)
        mask_np = np.clip(mask_np, 0.0, 1.0)
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        cn   = (color_name or "").lower()
        is_light = any(w in cn for w in ["white", "cream", "linen", "ivory", "beige", "pearl"])
        if is_light:
            shading, alpha_strength = 0.72 + gray * 0.45, 0.96
        else:
            shading, alpha_strength = 0.55 + gray * 0.65, 0.94
        target = np.empty_like(img_np)
        for c in range(3):
            target[:, :, c] = np.clip(target_rgb[c] * shading, 0, 255)
        alpha  = (mask_np * alpha_strength)[:, :, None]
        result = np.clip(img_np * (1.0 - alpha) + target * alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _ask_style_prompt(self) -> tuple[str, str]:
        style_hints = ", ".join(Config.STYLE_HINTS)
        mode = Prompt.ask(
            "Select processing level",
            choices=["low", "balanced", "creative"],
            default="balanced"
        )
        user_prompt = Prompt.ask(
            f"Any specific styles or instructions?\n"
            f"  [dim](e.g. {style_hints})[/dim]\n"
            f"  Instruction"
        )
        return mode, user_prompt

    def _detect_surface_under_mask(self, mask: Image.Image, rgb_img: np.ndarray) -> str:
        mask_np = np.array(mask.convert("L"))
        ys, xs  = np.where(mask_np > 127)
        if not len(ys):
            return "floor"
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        sample_y = min(cy + 80, rgb_img.shape[0] - 1)
        surrounding = rgb_img[sample_y, cx]
        brightness  = int(surrounding.mean())
        log.info(f"[dim]Surface probe at ({cx},{sample_y}): RGB={surrounding}, brightness={brightness}[/dim]")
        if brightness > 160:
            return "rug"
        else:
            return "floor"

    def _wall_art_bbox_mask(self, rgb_img: np.ndarray, pad: int = 12) -> Image.Image | None:
        """Solid bounding-box mask covering the WHOLE of every wall-art / frame detection
        (border + content), so removal leaves clean wall instead of an empty frame shell.
        SAM masks the picture inside the frame but misses the frame border — this fixes the
        'gray empty frames left on the wall' problem.
        """
        results = self.yolo(rgb_img, conf=Config.CONF_THRESHOLD, verbose=False)[0]
        h, w = rgb_img.shape[:2]
        m = np.zeros((h, w), np.uint8)
        art = ["picture", "frame", "artwork", "painting", "photograph", "mirror", "poster"]
        count = 0
        for box in results.boxes:
            name = self.yolo.names[int(box.cls[0])].lower()
            if any(c in name for c in art):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
                m[y1:y2, x1:x2] = 255
                count += 1
        if count == 0:
            return None
        log.info(f"[dim]Force-masked {count} wall-art frame bbox(es) for clean removal.[/dim]")
        return Image.fromarray(m).convert("L")

    def start_session(self, initial_image_path: Path):
        raw_bgr = cv2.imread(str(initial_image_path))
        if raw_bgr is None:
            log.error(f"Image not found: {initial_image_path}")
            return

        current_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        current_pil = Image.fromarray(current_rgb)

        iteration = 1
        console.rule("[bold green]Starting Initial Auto-Redesign Phase[/bold green]")

        mode, user_prompt = self._ask_style_prompt()

        # Clutter = wall art + surface decor + extra soft furnishings. NOT the main
        # furniture (sofa/chair/table/carpet) — those keep their colour. Masking the
        # clutter and removing it is how we ACTUALLY declutter (prompt-only didn't work).
        declutter_classes = [
            "picture", "frame", "artwork", "painting", "photograph", "mirror",
            "pillow", "cushion", "vase", "candle", "books", "book",
        ]
        clutter_negative = (
            "cluttered, busy, crowded, many throw pillows, many framed pictures, gallery "
            "wall, rows of picture frames, knick-knacks, clutter on shelves, too many "
            "objects, "
        )

        if mode == "low":
            # ── LOW = declutter ONLY (no recolor / no restyle) ────────────────
            # Mask just the clutter and remove it; the furniture is NOT masked, so its
            # colour and shape stay exactly the same.
            with console.status("[bold blue]Masking clutter to remove..."):
                mask, _ = self.generate_mask(
                    current_rgb, target_classes=declutter_classes, avoid_structural=True)
                # Force-mask the FULL frame bboxes so we don't leave empty frame shells.
                frame_bbox = self._wall_art_bbox_mask(current_rgb, pad=12)
                if frame_bbox is not None:
                    base = (np.array(mask.convert("L")) if mask is not None
                            else np.zeros(current_rgb.shape[:2], np.uint8))
                    mask = Image.fromarray(np.maximum(
                        base, np.array(frame_bbox))).convert("L")
            if mask is None:
                log.warning("[yellow]No clutter detected — leaving the room unchanged.[/yellow]")
                saved_path = self.save_iteration(current_pil, iteration)
            else:
                content_prompt = (
                    "clean empty wall and clear tidy surfaces where the clutter was removed, "
                    "completely bare smooth painted wall with no picture frames, no empty "
                    "frames, no frame outlines, no wall art and no gallery wall, empty "
                    "shelves, no extra throw pillows, seamless wall and surface matching the "
                    "surrounding area, no ghost outline. Keep all furniture exactly the same "
                    "colour, shape and position. photorealistic, sharp, 8k uhd."
                )
                style_prompt = (
                    "minimalist decluttered interior, clean empty smooth walls and bare "
                    "surfaces, photorealistic interior photography, 8k uhd, natural lighting."
                )
                negative_prompt = (
                    "picture, frame, framed picture, empty frame, blank frame, frame outline, "
                    "frame border, gray frame, painting, wall art, gallery wall, poster, "
                    "mirror, vase, candle, books, knick-knacks, clutter, extra pillows, "
                    "new furniture, recolored furniture, restyled furniture, different colours, "
                    + Config.SYSTEM_NEGATIVE_PROMPT
                )
                with console.status("[bold magenta]Decluttering (low — no colour change)..."):
                    current_pil = self.sdxl.run(
                        current_pil, mask, Config.MODES["erase_wall"],
                        content_prompt, style_prompt,
                        negative_prompt=negative_prompt, mode_name="erase_wall",
                    )
                    saved_path = self.save_iteration(current_pil, iteration)
        else:
            # ── BALANCED / CREATIVE = declutter + redesign ────────────────────
            with console.status("[bold blue]Analyzing & masking objects..."):
                mask, appliances = self.generate_mask(current_rgb, avoid_structural=True)
                # also fold the clutter + full frame bboxes into the mask so the redesign
                # actually removes them (and doesn't leave empty frame shells on the wall)
                clutter_mask, _ = self.generate_mask(
                    current_rgb, target_classes=declutter_classes, avoid_structural=True)
                base = np.array(mask.convert("L"))
                if clutter_mask is not None:
                    base = np.maximum(base, np.array(clutter_mask.convert("L")))
                frame_bbox = self._wall_art_bbox_mask(current_rgb, pad=12)
                if frame_bbox is not None:
                    base = np.maximum(base, np.array(frame_bbox))
                # WALL-PATCH FIX: feather the mask so the rectangular frame regions blend
                # into the wall instead of regenerating as hard-edged colored blocks.
                base = cv2.GaussianBlur(base, (0, 0), 9)
                mask = Image.fromarray(base).convert("L")
            app_str = f"Must seamlessly integrate modern {', '.join(appliances)}. " if appliances else ""
            declutter_str = (
                "minimalist and decluttered: remove most throw pillows, remove the wall "
                "picture frames and gallery wall, clear and bare surfaces, generous negative "
                "space. "
            )
            redesign_negative = clutter_negative + Config.SYSTEM_NEGATIVE_PROMPT
            # Route the initial redesign through the friend's redesign settings (lower
            # steps + different strength → clearer/HD result). menu 'low' never reaches
            # here (it's the declutter-only branch above); 'balanced'→redesign_moderate,
            # 'creative'→redesign_creative.
            redesign_mode = {"balanced": "redesign_moderate",
                             "creative": "redesign_creative"}.get(mode, mode)
            log.info(f"[dim]Initial redesign: menu '{mode}' → {redesign_mode} "
                     f"{Config.MODES[redesign_mode]}[/dim]")
            with console.status(f"[bold magenta]Generating iteration {iteration}..."):
                current_pil = self.sdxl.run(
                    current_pil, mask, Config.MODES[redesign_mode],
                    f"{app_str}{declutter_str}Custom specifics: {user_prompt}",
                    Config.SYSTEM_PROMPT,
                    negative_prompt=redesign_negative,
                    mode_name=redesign_mode,
                )
                saved_path = self.save_iteration(current_pil, iteration)

        log.info(f"✔ [bold green]Iteration {iteration} saved:[/bold green] {saved_path}")
        self.report.add(
            iteration=iteration, action="initial_redesign", target=None,
            from_color=None, to_color=None, to_material=None,
            instruction=user_prompt or "auto redesign",
            mode=mode, passes=1, saved_path=saved_path,
        )

        object_color_history: dict[str, str] = {}

        while True:
            console.rule(f"[bold yellow]Interactive Session — Iteration {iteration + 1}[/bold yellow]")
            console.print("[1] Modify an object")
            console.print("[2] Global change")
            console.print("[3] Exit")
            action = Prompt.ask("What would you like to do?",
                                choices=["1", "2", "3"], default="3")

            if action == "3":
                self.report.print_report()
                console.print(Panel(
                    f"Session finished!\nFinal image: [cyan]{saved_path}[/cyan]",
                    border_style="green"
                ))
                break

            current_rgb = np.array(current_pil)

            mask            = None
            content_prompt  = ""
            style_prompt    = Config.SYSTEM_PROMPT
            negative_prompt = Config.SYSTEM_NEGATIVE_PROMPT
            mode            = "balanced"
            passes          = 1
            target          = None
            new_instruction = ""
            target_color    = None
            target_material = None
            _is_remove      = False
            _prefill_image  = None
            _use_direct_recolor = False   # solid-furniture color → deterministic tint, no SDXL

            # ── ACTION 1: MODIFY ─────────────────────────────────────────
            if action == "1":
                console.print("\n[1] Remove")
                console.print("[2] Change color")
                console.print("[3] Change material")
                sub_action = Prompt.ask("What modification?",
                                        choices=["1", "2", "3"], default="2")

                # ── SUB 1: REMOVE ─────────────────────────────────────────
                if sub_action == "1":
                    _is_remove = True
                    detected_items = self.get_detected_objects(current_rgb)
                    if not detected_items:
                        console.print("[bold red]No objects detected.[/bold red]")
                        continue

                    console.print(f"\n[bold cyan]Detected:[/bold cyan] [yellow]{', '.join(sorted(detected_items))}[/yellow]")
                    target = Prompt.ask("Which object to remove?").lower().strip()
                    target_classes  = resolve_target_classes(target)
                    is_large        = any(kw in target for kw in Config.LARGE_OBJECTS)
                    is_flat_removal = is_flat_surface(target)
                    # Treat shelves/bookshelves as wall items → bare-wall erase (strength
                    # 0.99) instead of the general erase, so they fill clean wall properly
                    # instead of leaving a blank panel.
                    is_shelf        = any(kw in target for kw in ["shelf", "book shelf", "bookshelf", "shelving"])
                    is_wall_item    = is_wall_art(target) or is_shelf
                    is_table        = any(kw in target for kw in ["table", "middle table", "coffee table", "desk"])
                    is_curtain      = any(kw in target for kw in ["curtain", "drape", "blind"])
                    # Small decor (books/vase/candle/pillow/blanket) sits ON furniture that
                    # often sits ON the rug — the old 20px dilation reached down onto the
                    # carpet and the erase damaged its pattern. These need only a tight margin.
                    is_small_removal = is_small_object(target) and not (is_large or is_flat_removal
                                       or is_wall_item or is_table or is_curtain)

                    conf_remove = Config.TABLE_CONF_THRESHOLD if is_table else None

                    with console.status(f"[bold blue]Masking '{target}' for removal..."):
                        mask, _ = self.generate_mask(
                            current_rgb,
                            target_classes=target_classes,
                            avoid_structural=False,
                            flat_surface=is_flat_removal,
                            conf_override=conf_remove,
                        )

                    if mask is None:
                        console.print(f"[bold red]Could not detect '{target}' — skipping.[/bold red]")
                        continue

                    if is_table:
                        mask_np_ext = np.array(mask.convert("L"))
                        leg_shift   = 60
                        leg_pad     = np.zeros_like(mask_np_ext)
                        leg_pad[leg_shift:, :] = mask_np_ext[:-leg_shift, :]
                        mask_np_ext = np.maximum(mask_np_ext, leg_pad)
                        mask = Image.fromarray(mask_np_ext).convert("L")
                        log.info(f"[dim]Table leg padding +{leg_shift}px applied.[/dim]")

                    mask_np = np.array(mask.convert("L"))

                    h_img, w_img  = mask_np.shape[:2]
                    raw_coverage  = (mask_np > 127).sum() / (h_img * w_img)
                    if raw_coverage > Config.MAX_REMOVAL_MASK_COVERAGE:
                        log.warning(
                            f"[yellow]Mask {raw_coverage*100:.1f}% > threshold — "
                            f"clamping dilation to {Config.CLAMPED_ERASE_EXPANSION}px.[/yellow]"
                        )
                        erase_expansion = Config.CLAMPED_ERASE_EXPANSION
                    elif is_curtain:
                        # Curtain: no furniture legs to protect, so it needs full
                        # coverage to actually be removed (the tight carpet value left
                        # the curtain behind). This is the old working behavior.
                        erase_expansion = 10
                        log.info(f"[dim]Curtain removal: erase_expansion 10px.[/dim]")
                    elif is_flat_removal:
                        # Carpet/rug: 6px so the mask reaches the rug's edges and stops
                        # leaving slivers. The furniture is still protected because the
                        # furniture-carve above was widened to 12px (net ~6px clearance).
                        erase_expansion = 6
                        log.info(f"[dim]Flat removal: erase_expansion 6px (furniture kept clear via 12px carve).[/dim]")
                    elif is_small_removal:
                        # Tight margin so a small object on the table doesn't bleed the erase
                        # down onto the rug/floor and wipe out its pattern.
                        erase_expansion = 10
                        log.info(f"[dim]Small-object removal: erase_expansion 10px (no carpet bleed).[/dim]")
                    else:
                        erase_expansion = 40 if is_large else 20

                    kernel  = np.ones((erase_expansion, erase_expansion), np.uint8)
                    mask_np = cv2.dilate(mask_np, kernel)
                    _, mask_np = cv2.threshold(mask_np, 30, 255, cv2.THRESH_BINARY)
                    mask = Image.fromarray(mask_np).convert("L")
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=6))

                    self.preview_mask(current_pil, mask, f"REMOVE_{target}", iteration + 1)

                    if is_table:
                        surface = self._detect_surface_under_mask(mask, current_rgb)
                        bg_color = self._sample_surrounding_color(mask, current_rgb, expand_px=90)

                        if bg_color is not None:
                            pf_arr  = np.array(current_pil.convert("RGB")).copy().astype(np.float32)
                            msk_arr = np.array(mask.convert("L"))
                            solid   = msk_arr > 127
                            pf_arr[solid] = (
                                pf_arr[solid] * 0.35 +
                                np.array(bg_color, dtype=np.float32) * 0.65
                            )
                            _prefill_image = Image.fromarray(pf_arr.astype(np.uint8))
                            log.info(f"[dim]Surface pre-fill: RGB{bg_color} at 65% over {solid.sum()} px.[/dim]")

                        if surface == "rug":
                            fill_surface = (
                                "seamless rug continuation matching the surrounding carpet, "
                                "same rug pattern texture and color as the area around it, "
                                "continuous floor covering with no gap, clean rug surface, "
                                "completely empty, absolutely nothing on the rug"
                            )
                            mode = "erase_rug"
                        else:
                            fill_surface = "empty hardwood floor, bare natural wood floor boards, clean open floor area, absolutely nothing on the floor"
                            mode = "erase_large"
                        passes = 1
                        # Block the table AND everything that was sitting on it (vase, flowers,
                        # books, bowl, tray, decor) so removal leaves clean empty space and
                        # doesn't regenerate those as new floating objects.
                        no_obj_neg = (
                            f"{target}, coffee table, wooden table, rustic table, square table, "
                            f"rectangular table, round table, oval table, center table, "
                            f"any table, table top, table legs, table surface, any furniture here, "
                            f"vase, flowers, flower vase, plant, potted plant, books, book, bowl, "
                            f"tray, candle, cup, mug, decor, ornament, stack of books, "
                            f"any object on the floor, any item placed here, "
                            f"replacement object, new object placed here, "
                        )
                        log.info(
                            f"[bold yellow]Table erase — surface={surface}, mode={mode}, "
                            f"pre-fill={'yes' if _prefill_image else 'no'}.[/bold yellow]"
                        )

                    # ── FIX 3c: pre-fill for ALL large non-table removals ─────
                    # Carpet removal: without this, SDXL starts from bright beige
                    # pixels and outputs a pale/white ghost patch.
                    # We sample the surrounding floor color (dark hardwood ~RGB 60,40,30)
                    # and pre-fill the masked region at 75% opacity so SDXL starts
                    # from a visually correct dark-floor base.
                    elif is_large and not is_wall_item:
                        # prefer_dark for carpets/rugs: lock the pre-fill onto the dark
                        # hardwood, not the light rug, so the rug is actually replaced by
                        # floor instead of regenerated.
                        bg_color = self._sample_surrounding_color(
                            mask, current_rgb, expand_px=120,
                            prefer_dark=is_flat_removal and not is_curtain,
                        )
                        if bg_color is not None:
                            pf_arr  = np.array(current_pil.convert("RGB")).copy().astype(np.float32)
                            msk_arr = np.array(mask.convert("L"))
                            solid   = msk_arr > 127
                            pf_arr[solid] = (
                                pf_arr[solid] * 0.15 +
                                np.array(bg_color, dtype=np.float32) * 0.85
                            )
                            _prefill_image = Image.fromarray(pf_arr.astype(np.uint8))
                            log.info(f"[dim]Large-obj pre-fill: RGB{bg_color} at 85% over {solid.sum()} px.[/dim]")

                    # ── Wall art / shelf removal ──────────────────────────────
                    if is_wall_item:
                        fill_surface = (
                            "completely bare flat painted interior wall, "
                            "smooth empty wall surface, clean plaster wall, no objects"
                        )
                        no_obj_neg = Config.WALL_ART_NEGATIVE
                        if is_shelf:
                            # also block the shelf + everything stored on it so it doesn't
                            # come back as a shelf, books, or a blank panel.
                            no_obj_neg += (
                                "shelf, bookshelf, book shelf, shelving, floating shelf, "
                                "wall shelf, white panel, blank panel, cabinet, books, book, "
                                "stack of books, vase, plant, potted plant, decor, ornament, "
                            )

                    elif not is_table:
                        if any(kw in target for kw in ["sofa", "couch", "chair", "armchair"]):
                            fill_surface = "hardwood floor and plain wall, matching floor boards, empty room area"
                            no_obj_neg   = f"{target}, sofa, couch, chair, armchair, any seating furniture, "
                        elif any(kw in target for kw in ["carpet", "rug", "mat"]):
                            fill_surface = "hardwood floor, natural wood floor boards, clean bare floor"
                            # On bare floor SDXL "completes" the flush coffee table by adding
                            # legs. Block invented legs / furniture parts so the table is left
                            # exactly as it is and only the floor is filled.
                            no_obj_neg   = (
                                f"{target}, rug, carpet, mat, any floor covering, "
                                "table leg, table legs, coffee table leg, furniture leg, "
                                "new leg, added leg, raised table, table on legs, table base, "
                                "new furniture, extra furniture part, chair leg, "
                            )
                        elif any(kw in target for kw in ["curtain", "drape", "blind"]):
                            # The curtain kept getting REDRAWN (white sheer -> tan drape)
                            # instead of removed. Push hard for an uncovered window + bare
                            # wall, and block every fabric-panel synonym so SDXL can't just
                            # paint another curtain. Also block lamps/sconces (earlier they
                            # appeared when the wall was cleared).
                            fill_surface = (
                                "bare uncovered window with clear glass and plain frame, "
                                "naked window with no window treatment, "
                                "plain smooth painted wall on both sides of the window, "
                                "empty wall, no fabric of any kind"
                            )
                            no_obj_neg   = (
                                f"{target}, curtain, curtains, drape, drapes, drapery, "
                                "sheer curtain, valance, window treatment, window covering, "
                                "blind, blinds, shade, fabric panel, hanging fabric, "
                                "cloth, textile, drapery rod with fabric, "
                                "lamp, wall lamp, table lamp, floor lamp, sconce, wall sconce, "
                                "light fixture, light, lantern, candle, "
                                "picture, frame, artwork, mirror, shelf, plant, "
                                "wall decoration, wall hanging, any new object, any furniture, "
                            )
                        elif any(kw in target for kw in ["lamp", "vase", "candle", "plant"]):
                            fill_surface = "clean empty surface, bare table top or floor, no object"
                            no_obj_neg   = f"{target}, lamp, vase, candle, plant, any small object, "
                        else:
                            fill_surface = "clean interior surface, empty floor and wall"
                            no_obj_neg   = f"{target}, {', '.join(target_classes)}, any furniture, "

                    # ── FIX 3b: passes=2 for erase_large ─────────────────────
                    if not is_table:
                        if is_wall_item:
                            mode   = "erase_wall"
                            passes = 1
                        elif is_curtain:
                            # High-strength curtain wipe (erase_large was too gentle and
                            # just restyled the curtain). 2 passes: destroy, then blend.
                            mode   = "erase_curtain"
                            passes = 2
                        elif is_large:
                            mode   = "erase_large"
                            passes = 2  # 2-pass: first pass covers the object, second blends seams
                        else:
                            mode   = "erase"
                            passes = 1

                    content_prompt = (
                        f"intentionally empty {fill_surface} where {target} was removed, "
                        f"deliberately bare surface with no object placed here, "
                        f"seamless {fill_surface} continuation matching surrounding area, "
                        f"no ghost outline, no remnant shadow, no object, nothing here, "
                        f"photorealistic surface, sharp focus, 8k uhd."
                    )
                    style_prompt = (
                        f"empty {fill_surface}, clean interior, bare seamless surface, "
                        f"photorealistic architectural photography, 8k uhd."
                    )
                    negative_prompt = (
                        no_obj_neg
                        + "any replacement object, any new object, anything placed here, "
                        + "ghost, remnant, outline, shadow of object, floating element, "
                        + "artifact, smearing, blurry fill, visible seam, "
                        + Config.SYSTEM_NEGATIVE_PROMPT
                    )

                    new_instruction = f"remove {target}"
                    target_color    = None
                    target_material = None
                    log.info(
                        f"[bold yellow]Remove '{target}' — mode={mode}, passes={passes}, "
                        f"large={is_large}, wall={is_wall_item}.[/bold yellow]"
                    )

                # ── SUB 2: CHANGE COLOR  /  SUB 3: CHANGE MATERIAL ───────
                else:
                    detected_items = self.get_detected_objects(current_rgb)
                    if not detected_items:
                        console.print("[bold red]No objects detected. Try option 2 for global redesign.[/bold red]")
                        continue

                    console.print(f"\n[bold cyan]Detected:[/bold cyan] [yellow]{', '.join(sorted(detected_items))}[/yellow]")
                    target = Prompt.ask("Which object to target?").lower().strip()

                    target_classes = resolve_target_classes(target)
                    # Recoloring a sofa should recolor ALL the seating (armchairs/accent
                    # chairs) so the set matches — the user treats every seat as "the sofas"
                    # and flags it when one is left a different colour. (Recolor only; the
                    # removal path keeps its own narrower aliases.)
                    if any(s in target for s in ["sofa", "couch", "loveseat", "sectional"]):
                        for extra in ["chair", "armchair", "accent chair"]:
                            if extra not in target_classes:
                                target_classes.append(extra)
                    if len(target_classes) > 1:
                        console.print(f"[dim]Will also match: {', '.join(target_classes[1:])}[/dim]")

                    if sub_action == "2":
                        console.print(f"\n[dim]Examples: 'dark green'  /  'navy blue'  /  'charcoal gray'  /  'pink'[/dim]")
                        new_instruction = Prompt.ask(f"What color for '{target}'?")
                    else:
                        console.print(f"\n[dim]Examples: 'velvet'  /  'leather'  /  'boucle'  /  'linen'  /  'marble'[/dim]")
                        new_instruction = Prompt.ask(f"What material for '{target}'?")

                    target_color    = extract_target_color(new_instruction)
                    target_material = extract_material(new_instruction)
                    flat            = is_flat_surface(target)
                    dark            = is_dark_color(target_color)
                    bright          = is_bright_color(target_color)
                    small           = is_small_object(target)
                    vivid           = is_vivid_dark(target_color)   # ── FIX D ──
                    vivid_w         = is_vivid_warm(target_color)   # ── FIX 1i ──
                    # Rugs/carpets/mats only (NOT curtains/blinds) → separate direct-tint path.
                    is_rug          = any(k in target for k in ("carpet", "rug", "mat"))

                    if small and target_color:
                        mode   = "pillow_recolor"
                        passes = 1
                        log.info(f"[bold yellow]Small object '{target}' → pillow_recolor 1-pass.[/bold yellow]")
                    elif is_rug and target_color:
                        # ── Separate flat-surface recolor (rugs/carpets/mats) ─────────
                        # SDXL flattened the rug's weave and left a hard boundary LINE.
                        # Use the deterministic pixel-tint instead: it multiplies the
                        # target colour by the rug's own luminance (weave/pattern kept)
                        # and feathers the mask edge via alpha blend (no seam/line).
                        # Applies to ALL rug colours incl. black/white — coverage is
                        # guaranteed (unlike SDXL flat_dark/bright_shift, which left
                        # original-colour edges). Curtains/blinds are NOT rugs → SDXL.
                        mode   = "flat_recolor"   # SDXL fallback if colour has no RGB map
                        passes = 1
                        _use_direct_recolor = True
                        log.info(f"[bold yellow]Rug recolor '{target}' → direct pixel-tint "
                                 f"(weave preserved, no SDXL boundary line).[/bold yellow]")
                    elif dark and flat:
                        mode   = "flat_dark_shift"
                        passes = 2
                        log.info(f"[bold yellow]Dark+flat → flat_dark_shift 2-pass.[/bold yellow]")
                    elif is_vivid_dark(target_color) and not flat:
                        mode   = "vivid_recolor"
                        passes = 1
                        log.info(f"[bold yellow]Vivid dark '{target_color}' → vivid_recolor 1-pass.[/bold yellow]")
                    elif is_vivid_warm(target_color) and not flat:
                        # ── FIX 1d: warm/mid vivid colors (coral, salmon, orange, pink…) ──
                        # recolor mode (0.33/12) is too weak for gray→coral hue+lum shift.
                        # vivid_warm: strength=0.50 commits fully; guidance=15 drives warm sat.
                        # No hint — text prompt alone is sufficient; hint risks warm-tone bleed.
                        mode   = "vivid_warm"
                        passes = 1
                        log.info(f"[bold yellow]Vivid warm '{target_color}' → vivid_warm 1-pass.[/bold yellow]")
                    elif dark:
                        mode   = "dark_shift"
                        passes = 1
                        log.info(f"[bold yellow]Dark furniture → dark_shift 1-pass.[/bold yellow]")
                    elif flat and bright:
                        mode   = "flat_bright_shift"
                        passes = 2
                        log.info(f"[bold yellow]Bright+flat → flat_bright_shift 2-pass.[/bold yellow]")
                    elif bright:
                        mode   = "bright_shift"
                        passes = 1
                        log.info(f"[bold yellow]Bright furniture → bright_shift 1-pass.[/bold yellow]")
                    elif flat:
                        mode   = "flat_recolor" if target_color else "style"
                        passes = 1
                        log.info(f"[bold yellow]Flat surface → {mode} 1-pass.[/bold yellow]")
                    elif target_material:
                        mode   = "style"
                        passes = 1
                        log.info(f"[bold yellow]Material edit '{target_material}' → style 1-pass.[/bold yellow]")
                    else:
                        mode   = "recolor" if target_color else "style"
                        passes = 1
                        log.info(f"[dim]Auto-selected: {mode} 1-pass.[/dim]")

                    conf_ov = Config.PILLOW_CONF_THRESHOLD if small else None
                    # seating conf stays at the default 0.15 (0.10 falsely grabbed the side table).
                    _is_solid_furniture = (not flat) and (not small)
                    # Mask expansion: hug the SAM/YOLO furniture outline — do NOT puff past
                    # the furniture lines. 2px for solid furniture recolor (was 8) so the
                    # mask follows the segmentation and stops bridging onto the side table.
                    if small:
                        exp_ov = 5
                    elif _is_solid_furniture and target_color:
                        exp_ov = 2
                    else:
                        exp_ov = None

                    # Carve pillows out always; carve tables out too (the middle/side table
                    # was bleeding into the sofa mask) — but NOT when the target IS a table,
                    # or we'd carve the table out of its own mask. Tables sit beside the sofa,
                    # so carving them can't gap the sofa.
                    recolor_excl = None
                    if _is_solid_furniture and target_color:
                        recolor_excl = ["pillow", "cushion"]
                        if not any(t in target for t in ["table", "desk"]):
                            recolor_excl += ["table", "side table", "coffee table",
                                             "middle table", "desk"]

                    with console.status(f"[bold blue]Masking '{target}'..."):
                        mask, _ = self.generate_mask(
                            current_rgb,
                            target_classes=target_classes,
                            avoid_structural=False,
                            flat_surface=flat,
                            conf_override=conf_ov,
                            expansion_override=exp_ov,
                            # Keep each object's mask inside its own detected box so the
                            # colour can't bleed onto a neighbour (e.g. the side table).
                            constrain_to_boxes=_is_solid_furniture and bool(target_color),
                            exclude_classes=recolor_excl,
                            exclude_erode=3,
                            # Low-conf pass so a faintly-detected side table can veto a
                            # 'chair' mis-match AND be carved out — fixes the side table
                            # getting recolored with the sofa. Veto/carve only, never a target.
                            low_conf_aux=(Config.TABLE_CONF_THRESHOLD
                                          if (_is_solid_furniture and bool(target_color)) else None),
                        )

                    # Just drop tiny stray blobs; no leg protection (recolor the mask as-is).
                    if mask is not None and _is_solid_furniture and target_color:
                        mask = self._keep_large_components(mask)

                    if mask is None:
                        console.print(f"[bold red]Could not create mask for '{target}'.[/bold red]")
                        use_manual = Prompt.ask(
                            "Enter manual bounding box x1,y1,x2,y2 (or 'skip')",
                            default="skip"
                        )
                        if use_manual.lower() == "skip":
                            continue
                        try:
                            x1, y1, x2, y2 = [int(v.strip()) for v in use_manual.split(",")]
                            h_img, w_img = current_rgb.shape[:2]
                            manual_mask  = np.zeros((h_img, w_img), dtype=np.uint8)
                            manual_mask[y1:y2, x1:x2] = 255
                            kernel = np.ones((Config.MASK_EXPANSION, Config.MASK_EXPANSION), np.uint8)
                            manual_mask = cv2.dilate(manual_mask, kernel)
                            mask = Image.fromarray(manual_mask).convert("L")
                            mask = mask.filter(ImageFilter.GaussianBlur(radius=Config.MASK_BLUR))
                        except Exception as e:
                            console.print(f"[bold red]Invalid coordinates: {e}[/bold red]")
                            continue

                    self.preview_mask(current_pil, mask, target, iteration + 1)

                    previous_color = object_color_history.get(target)
                    if previous_color:
                        log.info(f"Previous color for '{target}': {previous_color}")

                    content_prompt, style_prompt, negative_prompt = build_recolor_prompts(
                        target, new_instruction, previous_color,
                        flat_surface=(mode in ("flat_recolor", "flat_bright_shift", "flat_dark_shift") and flat),
                        dark_shift=(mode in ("dark_shift", "flat_dark_shift")),
                        bright_shift=(mode in ("bright_shift", "flat_bright_shift")),
                        vivid_recolor=(mode == "vivid_recolor"),
                        vivid_warm=(mode == "vivid_warm"),        # ── FIX 1e ──
                    )

                    log.info(f"[dim]Color    : {target_color or 'none'}[/dim]")
                    log.info(f"[dim]Material : {target_material or 'none'}[/dim]")
                    log.info(f"[dim]Mode     : {mode} | Passes: {passes}[/dim]")

                    if target_color:
                        object_color_history[target] = target_color

            # ── ACTION 2: GLOBAL REDESIGN ────────────────────────────────
            elif action == "2":
                mode, new_instruction = self._ask_style_prompt()

                if mode == "creative":
                    mask            = Image.new("L", current_pil.size, 255)
                    negative_prompt = Config.SYSTEM_NEGATIVE_PROMPT
                    log.info("[bold yellow]Creative global redesign — full mask, structures may change.[/bold yellow]")
                else:
                    with console.status("[bold blue]Masking furniture for redesign..."):
                        mask, appliances = self.generate_mask(current_rgb, avoid_structural=True)
                    if mask is None:
                        log.warning("[yellow]No furniture detected — falling back to full mask.[/yellow]")
                        mask = Image.new("L", current_pil.size, 255)
                    negative_prompt = Config.GLOBAL_STRUCTURAL_NEGATIVE + Config.SYSTEM_NEGATIVE_PROMPT
                    log.info(f"[bold yellow]{mode.title()} global redesign — furniture mask only, structure locked.[/bold yellow]")

                content_prompt = new_instruction
                style_prompt   = Config.SYSTEM_PROMPT

            # ── GUARD ────────────────────────────────────────────────────
            if mask is None:
                console.print("[bold red]No mask was created — skipping.[/bold red]")
                continue

            log.info(f"[dim]Prompt  : {content_prompt[:120]}[/dim]")
            log.info(f"[dim]Style   : {style_prompt[:80]}[/dim]")
            log.info(f"[dim]Negative: {negative_prompt[:80]}...[/dim]")

            # ── FIX 1d: hint decision — flat_dark_shift bypasses skip ────
            # Previously flat surfaces were always excluded from hints.
            # For extreme shifts (beige→black), that left SDXL with no dark
            # visual cue → it output near-beige. Now flat_dark_shift gets a
            # 65% dark hint (via FLAT_DARK_HINT_OPACITY) to force the jump.
            color_hint_rgb = None
            if target_color and action == "1":
                # The hint pre-tints EVERY masked pixel with the target color, so the
                # whole object converts uniformly. This is ESSENTIAL for flat surfaces
                # (carpet) and small objects (pillows) — without it the color barely
                # shows (beige carpet, only one pillow recolored). So we now enable the
                # hint for ALL recolor modes, including flat_recolor and pillow_recolor.
                use_hint  = mode in ("dark_shift", "flat_dark_shift",
                                     "bright_shift", "flat_bright_shift",
                                     "vivid_recolor", "vivid_warm",
                                     "recolor", "flat_recolor", "pillow_recolor")
                if use_hint:
                    color_hint_rgb = Config.color_name_to_rgb(target_color)
                    if color_hint_rgb:
                        log.info(f"[dim]Color hint enabled for {mode} "
                                 f"(covers the WHOLE masked area uniformly).[/dim]")
                    else:
                        log.info(f"[dim]No RGB mapping for '{target_color}' — color via prompt only.[/dim]")
                else:
                    log.info(f"[dim]Standard recolor — no hint, strength/guidance drives color.[/dim]")

            with console.status("[bold magenta]Applying changes..."):
                direct_done = False
                if _use_direct_recolor:
                    recolored = self.direct_recolor(current_pil, mask, target_color)
                    if recolored is not None:
                        current_pil = recolored
                        direct_done = True
                    else:
                        log.info("[dim]No RGB mapping — falling back to SDXL recolor.[/dim]")
                if not direct_done:
                    run_image = _prefill_image if _prefill_image is not None else current_pil
                    current_pil = self.sdxl.run(
                        run_image, mask, Config.MODES[mode],
                        content_prompt, style_prompt,
                        negative_prompt=negative_prompt,
                        passes=passes,
                        color_hint=color_hint_rgb,
                        mode_name=mode,      # ← pass mode so run() picks correct hint opacity
                    )
                iteration += 1
                saved_path = self.save_iteration(current_pil, iteration)

            log.info(f"✔ [bold green]Iteration {iteration} saved:[/bold green] {saved_path}")

            if action == "2":
                report_action = "global"
            elif _is_remove:
                report_action = "remove"
            elif target_color:
                report_action = "recolor"
            elif target_material:
                report_action = "material"
            else:
                report_action = "style"

            self.report.add(
                iteration=iteration,
                action=report_action,
                target=target,
                from_color=object_color_history.get(target) if action == "1" else None,
                to_color=target_color,
                to_material=target_material,
                instruction=new_instruction,
                mode="direct_recolor" if direct_done else mode,
                passes=passes,
                saved_path=saved_path,
            )


if __name__ == "__main__":
    img_input = Prompt.ask("[bold white]Enter image path (or drag & drop)[/bold white]")
    path = Path(img_input.strip().strip('"'))

    if path.exists():
        app = InteractivePipeline()
        app.start_session(path)
    else:
        console.print(f"[bold red]File not found:[/bold red] {path}")
