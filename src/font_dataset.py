# font_masked_datamodule.py
#
# PyTorch Lightning DataModule + Dataset that:
# - Takes a list of .ttf/.otf paths
# - For each font, samples:
#     ctx_img, ctx_mask  (Kc glyph SDFs, Kc in [10,32])
#     tgt_img, tgt_mask  (Kt held-out glyph SDFs)
#     tgt_gid            (glyph IDs for targets)
# - Returns padded tensors suitable for the MaskedZFontLit training loop.
#
# Requirements:
#   pip install pytorch-lightning torch numpy fonttools freetype-py scipy
#
# Notes:
# - "gid" here defaults to a contiguous ID mapping built from all seen codepoints
#   across the provided fonts (capped by max_vocab). This keeps embedding tables small.
# - Fonts with too few renderable glyphs are skipped automatically.
#
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from fontTools.ttLib import TTFont
import freetype
from scipy.ndimage import distance_transform_edt


# -----------------------------
# SDF rendering helpers
# -----------------------------

def _mask_to_sdf(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: uint8/bool {0,1} where 1 = ink/inside
    returns: float32 SDF where negative=inside, positive=outside (in pixels)
    """
    inside = mask01.astype(bool)
    dist_in = distance_transform_edt(inside).astype(np.float32)
    dist_out = distance_transform_edt(~inside).astype(np.float32)
    return dist_out - dist_in


def _render_char_mask(
    face: freetype.Face,
    ch: str,
    *,
    canvas: int,
    ppem: int,
    thresh: int = 16,
) -> Optional[np.ndarray]:
    """
    Render a unicode character using FreeType to a centered square canvas.
    Returns uint8 mask {0,1} of shape [canvas, canvas], or None if not renderable.
    """
    try:
        face.set_char_size(ppem * 64)  # 26.6 fixed point
        face.load_char(ch, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    except Exception:
        return None

    bmp = face.glyph.bitmap
    if bmp.width == 0 or bmp.rows == 0:
        return None

    buf = np.asarray(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width)

    if buf.shape[0] > canvas or buf.shape[1] > canvas:
        # For now: skip too-large glyphs (or you can resize)
        return None

    out = np.zeros((canvas, canvas), dtype=np.uint8)
    y0 = (canvas - buf.shape[0]) // 2
    x0 = (canvas - buf.shape[1]) // 2
    out[y0 : y0 + buf.shape[0], x0 : x0 + buf.shape[1]] = buf

    mask = (out > thresh).astype(np.uint8)
    if int(mask.sum()) < 10:
        return None
    return mask


def _font_codepoints(ttf_path: str) -> List[int]:
    """
    Get unicode codepoints in the font using getBestCmap.
    """
    tt = TTFont(ttf_path, recalcBBoxes=False, recalcTimestamp=False)
    cmap = tt.getBestCmap() or {}
    cps = sorted(int(cp) for cp in cmap.keys())
    tt.close()
    return cps


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class FontSampleConfig:
    canvas: int = 256
    ppem: int = 256
    thresh: int = 16
    samples_per_font: int = 16

    # Context/target sampling
    ctx_k_min: int = 10
    ctx_k_max: int = 32
    tgt_k_min: int = 4
    tgt_k_max: int = 16

    # Fonts with < min_glyphs renderable are skipped
    min_glyphs: int = 14

    # How many codepoints to try per font when building its usable glyph list
    max_codepoints_per_font: int = 512

    # LRU-ish cache size for glyph SDFs per dataset instance
    glyph_cache_max: int = 8192


class FontMaskedDataset(Dataset):
    """
    Each item corresponds to one FONT (not one glyph).
    It returns variable-size context/target sets (collate_fn pads).

    Output item dict (unpadded):
      {
        "ctx_img": Tensor [Kc,1,H,W]
        "tgt_img": Tensor [Kt,1,H,W]
        "tgt_gid": Tensor [Kt] (long)
      }
    """

    def __init__(
        self,
        font_paths: Sequence[str | Path],
        *,
        gid_map: Dict[int, int],
        cfg: FontSampleConfig = FontSampleConfig(),
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.font_paths = [str(Path(p).expanduser().resolve()) for p in font_paths]
        self.gid_map = gid_map  # codepoint -> contiguous id

        # Build per-font usable codepoints by actually trying to render them
        self.font_glyphs: List[Tuple[str, List[int]]] = []  # (path, usable_codepoints)
        for fp in self.font_paths:
            print(f"Processing font: {fp}")
            cps = _font_codepoints(fp)
            if not cps:
                continue
            cps = cps[: self.cfg.max_codepoints_per_font]

            face = freetype.Face(fp)
            usable: List[int] = []
            for cp in cps:
                gid = self.gid_map.get(cp)
                if gid is None:
                    continue
                m = _render_char_mask(
                    face,
                    chr(cp),
                    canvas=self.cfg.canvas,
                    ppem=self.cfg.ppem,
                    thresh=self.cfg.thresh,
                )
                if m is not None:
                    usable.append(cp)

            if len(usable) >= self.cfg.min_glyphs:
                self.font_glyphs.append((fp, usable))

        # Tiny cache: (font_path, codepoint) -> Tensor[1,H,W]
        self._glyph_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._glyph_cache_keys: List[Tuple[str, int]] = []

    def __len__(self) -> int:
        return len(self.font_glyphs) * self.cfg.samples_per_font

    def _cache_put(self, k: Tuple[str, int], v: torch.Tensor) -> None:
        if k in self._glyph_cache:
            return
        self._glyph_cache[k] = v
        self._glyph_cache_keys.append(k)
        if len(self._glyph_cache_keys) > self.cfg.glyph_cache_max:
            old = self._glyph_cache_keys.pop(0)
            self._glyph_cache.pop(old, None)

    def _get_glyph_sdf(self, font_path: str, cp: int) -> Optional[torch.Tensor]:
        """
        Returns Tensor [1,H,W] float32 (SDF), or None if not renderable.
        """
        k = (font_path, cp)
        if k in self._glyph_cache:
            return self._glyph_cache[k]

        face = freetype.Face(font_path)
        mask = _render_char_mask(
            face,
            chr(cp),
            canvas=self.cfg.canvas,
            ppem=self.cfg.ppem,
            thresh=self.cfg.thresh,
        )
        if mask is None:
            return None

        sdf = _mask_to_sdf(mask)  # [H,W] float32
        t = torch.from_numpy(sdf).unsqueeze(0)  # [1,H,W]
        self._cache_put(k, t)
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        font_path, cps = self.font_glyphs[idx // self.cfg.samples_per_font]
        # Sample Kc and Kt from available glyphs
        n = len(cps)

        # Choose Kc first
        Kc = self.rng.randint(self.cfg.ctx_k_min, self.cfg.ctx_k_max)
        Kc = min(Kc, n - 1)  # leave room for targets
        if Kc < 1:
            Kc = 1

        # Choose Kt from remaining
        remaining = n - Kc
        Kt = self.rng.randint(self.cfg.tgt_k_min, self.cfg.tgt_k_max)
        Kt = min(Kt, max(1, remaining))

        # Sample without overlap
        chosen = self.rng.sample(cps, Kc + Kt)
        ctx_cps = chosen[:Kc]
        tgt_cps = chosen[Kc:]

        # Render SDFs (retry a few times if something fails unexpectedly)
        ctx_imgs: List[torch.Tensor] = []
        for cp in ctx_cps:
            t = self._get_glyph_sdf(font_path, cp)
            if t is not None:
                ctx_imgs.append(t)
        tgt_imgs: List[torch.Tensor] = []
        tgt_gids: List[int] = []
        for cp in tgt_cps:
            t = self._get_glyph_sdf(font_path, cp)
            gid = self.gid_map.get(cp)
            if t is not None and gid is not None:
                tgt_imgs.append(t)
                tgt_gids.append(int(gid))

        # If we ended up too sparse due to render failures, resample deterministically
        # (Keep it simple: fall back to first usable glyphs)
        if len(ctx_imgs) < min(self.cfg.ctx_k_min, 1 + len(cps) // 2) or len(tgt_imgs) < 1:
            fallback = cps[: max(self.cfg.min_glyphs, 8)]
            # context
            ctx_imgs = []
            for cp in fallback[: min(len(fallback) - 1, self.cfg.ctx_k_min)]:
                t = self._get_glyph_sdf(font_path, cp)
                if t is not None:
                    ctx_imgs.append(t)
            # targets
            tgt_imgs = []
            tgt_gids = []
            for cp in fallback[min(len(fallback) - 1, self.cfg.ctx_k_min) : min(len(fallback), self.cfg.ctx_k_min + self.cfg.tgt_k_min)]:
                t = self._get_glyph_sdf(font_path, cp)
                gid = self.gid_map.get(cp)
                if t is not None and gid is not None:
                    tgt_imgs.append(t)
                    tgt_gids.append(int(gid))

        ctx = torch.stack(ctx_imgs, dim=0) if ctx_imgs else torch.zeros((0, 1, self.cfg.canvas, self.cfg.canvas))
        tgt = torch.stack(tgt_imgs, dim=0) if tgt_imgs else torch.zeros((0, 1, self.cfg.canvas, self.cfg.canvas))
        gid = torch.tensor(tgt_gids, dtype=torch.long)

        return {"ctx_img": ctx, "tgt_img": tgt, "tgt_gid": gid}


# -----------------------------
# Collate: pad to batch max
# -----------------------------

def _pad_set(batch_tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of [K,1,H,W] tensors into [B,Kmax,1,H,W] and returns mask [B,Kmax].
    """
    B = len(batch_tensors)
    Kmax = max((t.shape[0] for t in batch_tensors), default=0)
    if Kmax == 0:
        # Degenerate
        return torch.zeros((B, 0, 1, 1, 1)), torch.zeros((B, 0))

    _, C, H, W = batch_tensors[0].shape
    out = torch.full((B, Kmax, C, H, W), pad_value, dtype=batch_tensors[0].dtype)
    mask = torch.zeros((B, Kmax), dtype=torch.float32)

    for i, t in enumerate(batch_tensors):
        k = t.shape[0]
        if k == 0:
            continue
        out[i, :k] = t
        mask[i, :k] = 1.0

    return out, mask


def font_masked_collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    ctx_list = [it["ctx_img"] for it in items]
    tgt_list = [it["tgt_img"] for it in items]
    gid_list = [it["tgt_gid"] for it in items]

    ctx_img, ctx_mask = _pad_set(ctx_list, pad_value=0.0)
    tgt_img, tgt_mask = _pad_set(tgt_list, pad_value=0.0)

    # pad gids to Ktmax
    B = len(items)
    Ktmax = tgt_img.shape[1]
    tgt_gid = torch.zeros((B, Ktmax), dtype=torch.long)
    for i, g in enumerate(gid_list):
        k = g.numel()
        if k > 0:
            tgt_gid[i, :k] = g

    return {
        "ctx_img": ctx_img,
        "ctx_mask": ctx_mask,
        "tgt_img": tgt_img,
        "tgt_mask": tgt_mask,
        "tgt_gid": tgt_gid,
    }


# -----------------------------
# DataModule
# -----------------------------

class FontDataModule(pl.LightningDataModule):
    """
    Builds a contiguous glyph-id mapping (codepoint -> gid) from the provided font paths,
    then provides train/val dataloaders yielding the required batch dict.
    """

    def __init__(
        self,
        font_paths: Sequence[str | Path],
        *,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.05,
        seed: int = 0,
        cfg: FontSampleConfig = FontSampleConfig(),
        # Vocab control:
        max_vocab: int = 2048,
        # If True, prioritize common ASCII + Latin-1 first, then fill with others seen.
        prioritize_basic_latin: bool = True,
    ):
        super().__init__()
        self.font_paths = [str(Path(p).expanduser().resolve()) for p in font_paths]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.cfg = cfg
        self.max_vocab = max_vocab
        self.prioritize_basic_latin = prioritize_basic_latin

        self.gid_map: Dict[int, int] = {}
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None

    def _build_gid_map(self) -> Dict[int, int]:
        # Gather codepoints seen across fonts
        seen: set[int] = set()
        for fp in self.font_paths:
            try:
                cps = _font_codepoints(fp)
            except Exception:
                continue
            for cp in cps:
                seen.add(int(cp))

        # Optional: seed with ASCII + Latin-1 + some punctuation
        prioritized: List[int] = []
        if self.prioritize_basic_latin:
            prioritized.extend(list(range(32, 127)))      # basic printable ASCII
            prioritized.extend(list(range(160, 256)))     # Latin-1 supplement
        # Keep only those actually seen
        prioritized = [cp for cp in prioritized if cp in seen]

        rest = sorted(seen.difference(prioritized))
        vocab = (prioritized + rest)[: self.max_vocab]

        return {cp: i for i, cp in enumerate(vocab)}

    def setup(self, stage: Optional[str] = None):
        if not self.gid_map:
            self.gid_map = self._build_gid_map()

        full = FontMaskedDataset(
            self.font_paths,
            gid_map=self.gid_map,
            cfg=self.cfg,
            seed=self.seed,
        )

        if len(full) == 0:
            raise RuntimeError("No usable fonts found (too few renderable glyphs after filtering).")

        # split
        val_n = max(1, int(len(full) * self.val_split))
        train_n = len(full) - val_n
        self._train, self._val = random_split(
            full,
            [train_n, val_n],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=font_masked_collate,
        )

    def val_dataloader(self):
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=font_masked_collate,
        )

    @property
    def glyph_vocab_size(self) -> int:
        return max(self.gid_map.values(), default=-1) + 1


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: list TTF files
    paths = list(Path("/path/to/fonts").rglob("*.ttf"))[:2000]

    dm = FontMaskedDataModule(paths, batch_size=4, num_workers=2, max_vocab=2048)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    for k, v in batch.items():
        print(k, tuple(v.shape), v.dtype)
    print("glyph_vocab_size:", dm.glyph_vocab_size)
