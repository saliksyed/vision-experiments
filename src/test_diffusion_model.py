# test_generate_font_pngs.py
#
# Given:
#   --ttf_path     path to a TTF/OTF font
#   --diff_ckpt    checkpoint from SDFFontDiffusionLit training
#
# This script will:
#   1) Enumerate renderable unicode codepoints in the font (via cmap)
#   2) For each glyph:
#        - render its SDF as ctx_img (K=1 context: "this glyph alone")
#        - compute z_font using the frozen pretrained z_font model inside diffusion ckpt
#        - run DDPM sampling to generate an SDF for the SAME glyph_id (reconstruction-ish)
#        - convert the generated SDF to a raster PNG (ink = sdf < 0)
#   3) Save PNGs to an output directory
#
# Notes / assumptions:
# - Your diffusion checkpoint was trained with SDFFontDiffusionLit (SDF pixel diffusion).
# - The checkpoint contains the frozen z_font encoder module and the gid embedding table.
# - Glyph IDs are the "gid_map" codepoint->id mapping used during training.
#   This script needs that same gid_map. Easiest: export it at training time.
#
# Recommended: during training, save gid_map as JSON once:
#   json.dump(dm_embed.gid_map, open("gid_map.json","w"))
#
# Requirements:
#   pip install torch pytorch-lightning numpy fonttools freetype-py scipy pillow
#
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from fontTools.ttLib import TTFont
import freetype
from scipy.ndimage import distance_transform_edt

# Import your Lightning modules
from embedding import EmbeddingModel
from diffusion_model import DiffusionModel
from constants import MODEL_CHECKPOINT_DIR

# -----------------------------
# Rendering helpers (match training)
# -----------------------------

def font_codepoints(ttf_path: str) -> List[int]:
    tt = TTFont(ttf_path, recalcBBoxes=False, recalcTimestamp=False)
    cmap = tt.getBestCmap() or {}
    cps = sorted(int(cp) for cp in cmap.keys())
    tt.close()
    return cps


def render_char_mask(face: freetype.Face, ch: str, *, canvas: int, ppem: int, thresh: int) -> Optional[np.ndarray]:
    try:
        face.set_char_size(ppem * 64)
        face.load_char(ch, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    except Exception:
        return None

    bmp = face.glyph.bitmap
    if bmp.width == 0 or bmp.rows == 0:
        return None

    buf = np.asarray(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width)

    if buf.shape[0] > canvas or buf.shape[1] > canvas:
        return None

    out = np.zeros((canvas, canvas), dtype=np.uint8)
    y0 = (canvas - buf.shape[0]) // 2
    x0 = (canvas - buf.shape[1]) // 2
    out[y0 : y0 + buf.shape[0], x0 : x0 + buf.shape[1]] = buf

    mask = (out > thresh).astype(np.uint8)
    if int(mask.sum()) < 10:
        return None
    return mask


def mask_to_sdf(mask01: np.ndarray) -> np.ndarray:
    inside = mask01.astype(bool)
    dist_in = distance_transform_edt(inside).astype(np.float32)
    dist_out = distance_transform_edt(~inside).astype(np.float32)
    return dist_out - dist_in  # positive outside, negative inside


def sdf_to_png(
    sdf: np.ndarray,
    *,
    invert: bool = False,
    mode: str = "binary",
    clamp: float = 32.0,
) -> Image.Image:
    """
    sdf: float32 [H,W], negative inside strokes.
    mode:
      - "binary": ink = sdf < 0 (pure black/white)
      - "soft": map sdf to grayscale with tanh-ish clamping
    """
    if mode == "binary":
        ink = (sdf < 0).astype(np.uint8) * 255
        img = ink if not invert else (255 - ink)
        return Image.fromarray(img, mode="L")
    else:
        x = np.clip(sdf / clamp, -1.0, 1.0)
        # map [-1,1] to [0,255] so inside (neg) is darker
        g = ((x + 1.0) * 0.5 * 255.0).astype(np.uint8)
        img = g if not invert else (255 - g)
        return Image.fromarray(img, mode="L")


# -----------------------------
# DDPM sampling (uses module buffers)
# -----------------------------

@torch.no_grad()
def ddpm_sample(
    *,
    model: SDFFontDiffusionLit,
    cond: torch.Tensor,            # [1, cond_dim]
    shape: Tuple[int, int, int, int],  # [B=1,1,H,W]
    steps: Optional[int] = None,   # if set, use fewer steps by skipping (simple stride)
) -> torch.Tensor:
    """
    Generate x0 ~ p(x0 | cond) via DDPM sampling, returning x0 tensor [1,1,H,W].
    This uses the trained UNet to predict epsilon.
    """
    device = cond.device
    T = model.T
    x = torch.randn(shape, device=device)

    if steps is None or steps >= T:
        ts = list(range(T - 1, -1, -1))
    else:
        # simple stride schedule (not DDIM, just skipping timesteps)
        stride = max(1, T // steps)
        ts = list(range(T - 1, -1, -stride))
        if ts[-1] != 0:
            ts.append(0)

    for t_int in ts:
        t = torch.tensor([t_int], device=device, dtype=torch.long)

        # Predict epsilon
        # Build time embedding inside model the same way as training step does
        cond_t = torch.cat([cond[:, : model.z_dim + model.gid_emb.embedding_dim], model.t_emb(t)], dim=-1)
        eps = model.unet(x, cond_t)  # [1,1,H,W]

        # Coeffs
        beta_t = model.betas[t_int]
        alpha_t = model.alphas[t_int]
        alpha_bar_t = model.alphas_cumprod[t_int]

        # x0 estimate
        x0_hat = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        if t_int == 0:
            x = x0_hat
            break

        # posterior mean (DDPM)
        alpha_bar_prev = model.alphas_cumprod[t_int - 1]
        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef1 * x0_hat + coef2 * x

        # posterior variance
        var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        noise = torch.randn_like(x)
        x = mean + torch.sqrt(var) * noise

    return x


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ttf_path", type=str, required=True)
    p.add_argument("--gid_map_json", type=str, required=True, help="JSON file with codepoint->gid mapping used in training.")
    p.add_argument("--out_dir", type=str, default="out_pngs")
    p.add_argument("--max_glyphs", type=int, default=200)
    p.add_argument("--canvas", type=int, default=256)
    p.add_argument("--ppem", type=int, default=256)
    p.add_argument("--thresh", type=int, default=16)

    # sampling speed/quality
    p.add_argument("--steps", type=int, default=1000, help="Sampling steps (<=T). Use 50-200 for quick tests.")
    p.add_argument("--png_mode", type=str, default="binary", choices=["binary", "soft"])
    p.add_argument("--save_reference", action="store_true", help="Also save the input glyph raster/SDF visualization.")
    return p.parse_args()


def safe_name(cp: int) -> str:
    # readable filename fragment
    ch = chr(cp)
    if ch.isalnum():
        return f"U{cp:04X}_{ch}"
    return f"U{cp:04X}"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_model_ckpt = MODEL_CHECKPOINT_DIR + "/embedding_model.ckpt"
    diffusion_model_ckpt = MODEL_CHECKPOINT_DIR + "/diffusion_model.ckpt"

    gid_map: Dict[int, int] = {int(k): int(v) for k, v in json.load(open(args.gid_map_json, "r")).items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load diffusion model checkpoint
    diff_model: DiffusionModel = DiffusionModel.load_from_checkpoint(
        diffusion_model_ckpt,
        zfont_module_cls=EmbeddingModel,
        zfont_ckpt_path=embedding_model_ckpt
    ).to(device)
    diff_model.eval()

    # Load font
    face = freetype.Face(args.ttf_path)
    cps = font_codepoints(args.ttf_path)
    cps = [cp for cp in cps if cp in gid_map]
    cps = cps[: args.max_glyphs]

    if not cps:
        raise SystemExit("No codepoints in this font match your gid_map (did you use the same gid_map as training?).")

    print(f"Found {len(cps)} glyphs to generate (capped). Saving to: {out_dir}")

    for i, cp in enumerate(cps):
        ch = chr(cp)
        mask = render_char_mask(face, ch, canvas=args.canvas, ppem=args.ppem, thresh=args.thresh)
        if mask is None:
            continue

        sdf = mask_to_sdf(mask)  # [H,W]
        ctx_img = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        ctx_img = ctx_img.to(device)

        # Build ctx batch for encoder: [B=1,K=1,1,H,W]
        ctx_batch = ctx_img.unsqueeze(1)  # [1,1,1,H,W]
        ctx_mask = torch.ones((1, 1), device=device, dtype=torch.float32)

        # Compute z_font from this single glyph as context
        with torch.no_grad():
            z_font = diff_model.compute_z_font(ctx_batch, ctx_mask)  # [1,z_dim]

        # Condition uses (z_font, gid_emb(gid), t_emb(t)) in training.
        gid = torch.tensor([gid_map[cp]], device=device, dtype=torch.long)
        cond_base = torch.cat([z_font, diff_model.gid_emb(gid)], dim=-1)  # [1, z_dim+gid_dim]

        # Sample SDF for this glyph
        x0 = ddpm_sample(
            model=diff_model,
            cond=cond_base,
            shape=(1, 1, args.canvas, args.canvas),
            steps=args.steps if args.steps > 0 else None,
        )

        sdf_gen = x0.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Save generated PNG
        img = sdf_to_png(sdf_gen, mode=args.png_mode)
        fname = f"{i:04d}_{safe_name(cp)}_gen.png"
        img.save(out_dir / fname)

        # Optionally save reference input visualization
        if args.save_reference:
            ref = sdf_to_png(sdf, mode=args.png_mode)
            ref.save(out_dir / f"{i:04d}_{safe_name(cp)}_ref.png")

    print("Done.")


if __name__ == "__main__":
    main()
