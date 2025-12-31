# font_diffusion_datamodule.py
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

from fontTools.ttLib import TTFont
import freetype
from scipy.ndimage import distance_transform_edt


def _mask_to_sdf(mask01: np.ndarray) -> np.ndarray:
    inside = mask01.astype(bool)
    dist_in = distance_transform_edt(inside).astype(np.float32)
    dist_out = distance_transform_edt(~inside).astype(np.float32)
    return dist_out - dist_in


def _render_char_mask(face: freetype.Face, ch: str, *, canvas: int, ppem: int, thresh: int) -> Optional[np.ndarray]:
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


def _font_codepoints(ttf_path: str) -> List[int]:
    tt = TTFont(ttf_path, recalcBBoxes=False, recalcTimestamp=False)
    cmap = tt.getBestCmap() or {}
    cps = sorted(int(cp) for cp in cmap.keys())
    tt.close()
    return cps


@dataclass
class DiffusionFontCfg:
    canvas: int = 256
    ppem: int = 256
    thresh: int = 16

    ctx_k_min: int = 10
    ctx_k_max: int = 32
    samples_per_font: int = 8

    # need at least ctx+1 glyph
    min_glyphs: int = 20
    max_codepoints_per_font: int = 512


class FontDiffusionDataset(Dataset):
    """
    Each item corresponds to one (font, target glyph) sample.
    Returns:
      ctx_img: [Kc,1,H,W]
      ctx_mask: [Kc]
      tgt_img: [1,H,W]
      tgt_gid: scalar long
    """

    def __init__(
        self,
        font_paths: Sequence[str | Path],
        *,
        gid_map: Dict[int, int],  # codepoint -> gid
        cfg: DiffusionFontCfg = DiffusionFontCfg(),
        seed: int = 0,
    ):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.gid_map = gid_map
        self.font_paths = [str(Path(p).expanduser().resolve()) for p in font_paths]

        # Build usable glyph list per font (by renderability)
        self.font_glyphs: List[Tuple[str, List[int]]] = []
        for fp in self.font_paths:
            try:
                cps = _font_codepoints(fp)[: cfg.max_codepoints_per_font]
            except Exception:
                continue
            if not cps:
                continue
            face = freetype.Face(fp)
            usable: List[int] = []
            for cp in cps:
                if cp not in gid_map:
                    continue
                m = _render_char_mask(face, chr(cp), canvas=cfg.canvas, ppem=cfg.ppem, thresh=cfg.thresh)
                if m is not None:
                    usable.append(cp)
            if len(usable) >= cfg.min_glyphs:
                self.font_glyphs.append((fp, usable))

        if not self.font_glyphs:
            raise RuntimeError("No usable fonts after filtering; lower min_glyphs or check font paths.")

        # Create an index of (font_idx, cp) targets
        self.targets: List[Tuple[int, int]] = []
        for fi, (_, cps) in enumerate(self.font_glyphs):
            for cp in cps:
                self.targets.append((fi, cp))

    def __len__(self) -> int:
        return len(self.targets) * self.cfg.samples_per_font

    def _get_sdf(self, font_path: str, cp: int) -> Optional[torch.Tensor]:
        face = freetype.Face(font_path)
        mask = _render_char_mask(face, chr(cp), canvas=self.cfg.canvas, ppem=self.cfg.ppem, thresh=self.cfg.thresh)
        if mask is None:
            return None
        sdf = _mask_to_sdf(mask)
        return torch.from_numpy(sdf).float().unsqueeze(0)  # [1,H,W]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi, tgt_cp = self.targets[idx // self.cfg.samples_per_font]
        font_path, cps = self.font_glyphs[fi]

        # sample context excluding target
        avail = [cp for cp in cps if cp != tgt_cp]
        Kc = self.rng.randint(self.cfg.ctx_k_min, self.cfg.ctx_k_max)
        Kc = min(Kc, len(avail))
        if Kc < 1:
            Kc = 1

        ctx_cps = self.rng.sample(avail, Kc)

        ctx_imgs: List[torch.Tensor] = []
        for cp in ctx_cps:
            t = self._get_sdf(font_path, cp)
            if t is not None:
                ctx_imgs.append(t)

        tgt_img = self._get_sdf(font_path, tgt_cp)
        if tgt_img is None or len(ctx_imgs) == 0:
            # very rare; resample a neighboring example deterministically
            # (simple fallback: return first available)
            tgt_cp = cps[0]
            tgt_img = self._get_sdf(font_path, tgt_cp)
            ctx_imgs = [self._get_sdf(font_path, cp) for cp in cps[1 : 1 + min(len(cps) - 1, self.cfg.ctx_k_min)]]
            ctx_imgs = [x for x in ctx_imgs if x is not None]

        ctx = torch.stack(ctx_imgs, dim=0)  # [Kc,1,H,W]
        ctx_mask = torch.ones((ctx.shape[0],), dtype=torch.float32)
        tgt_gid = torch.tensor(self.gid_map[int(tgt_cp)], dtype=torch.long)

        return {"ctx_img": ctx, "ctx_mask": ctx_mask, "tgt_img": tgt_img, "tgt_gid": tgt_gid}


def diffusion_collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # pad ctx to Kmax
    B = len(items)
    H, W = items[0]["tgt_img"].shape[-2:]
    Kmax = max(it["ctx_img"].shape[0] for it in items)

    ctx_img = torch.zeros((B, Kmax, 1, H, W), dtype=torch.float32)
    ctx_mask = torch.zeros((B, Kmax), dtype=torch.float32)

    tgt_img = torch.stack([it["tgt_img"] for it in items], dim=0)  # [B,1,H,W]
    tgt_gid = torch.stack([it["tgt_gid"] for it in items], dim=0)  # [B]

    for b, it in enumerate(items):
        k = it["ctx_img"].shape[0]
        ctx_img[b, :k] = it["ctx_img"]
        ctx_mask[b, :k] = 1.0

    return {"ctx_img": ctx_img, "ctx_mask": ctx_mask, "tgt_img": tgt_img, "tgt_gid": tgt_gid}


class DiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        font_paths: Sequence[str | Path],
        *,
        gid_map: Dict[int, int],
        cfg: DiffusionFontCfg = DiffusionFontCfg(),
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.01,
        seed: int = 0,
    ):
        super().__init__()
        self.font_paths = list(font_paths)
        self.gid_map = gid_map
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self._train = None
        self._val = None

    def setup(self, stage: Optional[str] = None):
        full = FontDiffusionDataset(self.font_paths, gid_map=self.gid_map, cfg=self.cfg, seed=self.seed)
        val_n = max(1, int(len(full) * self.val_split))
        train_n = len(full) - val_n
        self._train, self._val = random_split(
            full,
            [train_n, val_n],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=diffusion_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=diffusion_collate,
        )
