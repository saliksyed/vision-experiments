from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# -------------------------
# Model components
# -------------------------

class GlyphEncoder(nn.Module):
    """Encode glyph image (SDF or mask) -> embedding e_g."""
    def __init__(self, in_ch: int = 1, base: int = 64, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 4, stride=2, padding=1),      # /2
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),  # /4
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),  # /8
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(base * 4, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,1,H,W]
        h = self.net(x).flatten(1)
        return self.fc(h)  # [N,emb_dim]


class DeepSetsPosterior(nn.Module):
    """
    Set encoder q(z_font | {e_g}).
    Inputs:
      e: [B,K,D]
      mask: [B,K] (1 present, 0 pad)
    Outputs:
      mu, logvar: [B,Z]
    """
    def __init__(self, emb_dim: int = 256, hidden: int = 512, z_dim: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.to_mu = nn.Linear(hidden, z_dim)
        self.to_logvar = nn.Linear(hidden, z_dim)

    def forward(self, e: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # e: [B,K,D]
        h = self.phi(e)  # [B,K,H]
        m = mask.unsqueeze(-1).to(h.dtype)  # [B,K,1]
        h = h * m
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom  # mean pooling (permutation-invariant)
        pooled = self.rho(pooled)
        return self.to_mu(pooled), self.to_logvar(pooled)


class TargetLatentHead(nn.Module):
    """
    Predict held-out glyph latent y from z_font and glyph_id.
    """
    def __init__(
        self,
        z_dim: int = 64,
        glyph_vocab: int = 1024,
        gid_dim: int = 128,
        latent_dim: int = 256,
        hidden: int = 512,
    ):
        super().__init__()
        self.gid = nn.Embedding(glyph_vocab, gid_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + gid_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_font: torch.Tensor, glyph_id: torch.Tensor) -> torch.Tensor:
        g = self.gid(glyph_id)  # [N,gid_dim]
        x = torch.cat([z_font, g], dim=-1)
        return self.net(x)      # [N,latent_dim]


# -------------------------
# Latent encoder interface (replace with your VAE encoder)
# -------------------------

class FrozenLatentEncoderStub(nn.Module):
    """
    Replace with a trained VAE encoder producing a flat latent vector per glyph.
    The only requirement for masked modeling is a stable target representation y_g.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, latent_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,1,H,W]
        h = self.enc(x).flatten(1)
        return self.fc(h)  # [N,latent_dim]


# -------------------------
# Lightning module (z_font only)
# -------------------------

class FontEmbeddingModel(pl.LightningModule):
    """
    Batch format expected:

      batch = {
        # Context glyphs: sample Kc in [10,32] per font (pad to max Kc in batch)
        "ctx_img":  FloatTensor [B, Kc, 1, H, W]
        "ctx_mask": FloatTensor [B, Kc]            (1=present, 0=pad)

        # Targets (held-out glyphs): any number >=1 (pad to max Kt in batch)
        "tgt_img":  FloatTensor [B, Kt, 1, H, W]
        "tgt_mask": FloatTensor [B, Kt]
        "tgt_gid":  LongTensor  [B, Kt]            glyph IDs (pad values ignored when tgt_mask=0)

        # (Optional) context glyph ids, if you want extra regularizers; not required:
        # "ctx_gid": LongTensor [B, Kc]
      }

    The "10–32 glyphs produce the same z_font" requirement is satisfied because:
    - we compute ONE z_font per font by aggregating the set of glyph embeddings.
    """
    def __init__(
        self,
        glyph_vocab: int = 2048,
        img_ch: int = 1,
        glyph_emb_dim: int = 256,
        z_dim: int = 64,
        latent_dim: int = 256,
        lr: float = 2e-4,
        kl_weight: float = 1e-4,
        recon_weight: float = 1.0,
        # Optional regularizer strength encouraging consistency across random context splits
        split_consistency_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Frozen target latent encoder (your VAE encoder goes here)
        self.latent_enc = FrozenLatentEncoderStub(latent_dim=latent_dim)
        for p in self.latent_enc.parameters():
            p.requires_grad = False
        self.latent_enc.eval()

        self.glyph_enc = GlyphEncoder(in_ch=img_ch, emb_dim=glyph_emb_dim)
        self.posterior = DeepSetsPosterior(emb_dim=glyph_emb_dim, z_dim=z_dim)
        self.head = TargetLatentHead(z_dim=z_dim, glyph_vocab=glyph_vocab, latent_dim=latent_dim)

        self.lr = lr
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.split_consistency_weight = split_consistency_weight

    def configure_optimizers(self):
        params = list(self.glyph_enc.parameters()) + list(self.posterior.parameters()) + list(self.head.parameters())
        return torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.99), weight_decay=1e-4)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # per-sample KL(q||N(0,1))
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)

    def infer_zfont(self, ctx_img: torch.Tensor, ctx_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ctx_img: [B,Kc,1,H,W], ctx_mask: [B,Kc]
        -> z_font [B,Z], mu [B,Z], logvar [B,Z]
        """
        B, Kc, _, H, W = ctx_img.shape
        x = ctx_img.view(B * Kc, 1, H, W)
        e = self.glyph_enc(x).view(B, Kc, -1)  # [B,Kc,D]
        mu, logvar = self.posterior(e, ctx_mask)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        ctx_img = batch["ctx_img"]
        ctx_mask = batch["ctx_mask"]
        tgt_img = batch["tgt_img"]
        tgt_mask = batch["tgt_mask"]
        tgt_gid = batch["tgt_gid"]

        B, Kt, _, H, W = tgt_img.shape

        # 1) Infer ONE z_font per font from the context set (10–32 glyphs)
        z_font, mu, logvar = self.infer_zfont(ctx_img, ctx_mask)  # [B,Z]

        # 2) Encode targets to stable latent y (frozen)
        with torch.no_grad():
            y_all = self.latent_enc.encode(tgt_img.view(B * Kt, 1, H, W)).view(B, Kt, -1)  # [B,Kt,L]

        # 3) Unpad targets
        m = tgt_mask.bool()
        if m.sum() == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train/loss", loss, prog_bar=True)
            return loss

        y = y_all[m]       # [N,L]
        gid = tgt_gid[m]   # [N]
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, Kt)[m]  # [N]
        z_rep = z_font[b_idx]  # [N,Z] (same z for all glyphs of that font)

        # 4) Predict held-out glyph latents from z_font + glyph_id
        y_hat = self.head(z_rep, gid)
        recon = F.mse_loss(y_hat, y)

        # 5) KL regularization on z_font posterior
        kl = self.kl_to_standard_normal(mu, logvar).mean()

        # 6) Optional: split-consistency regularizer
        # Encourage two random sub-splits of the same context set to produce similar mu.
        # This directly enforces "any subset of 10–32 glyphs yields same embedding".
        split_cons = torch.tensor(0.0, device=self.device)
        if self.split_consistency_weight > 0.0:
            # Build two random sub-masks from the existing ctx_mask for each font.
            # This assumes ctx_mask already indicates Kc valid glyphs (<=32).
            cm = ctx_mask.clone().bool()  # [B,Kc]
            if cm.sum(dim=1).min().item() >= 10:
                # Create submasks selecting ~half of available glyphs, at least 10
                sub1 = torch.zeros_like(cm)
                sub2 = torch.zeros_like(cm)
                for b in range(B):
                    idx = torch.where(cm[b])[0]
                    idx = idx[torch.randperm(idx.numel(), device=self.device)]
                    k = idx.numel()
                    k1 = max(10, k // 2)
                    k2 = max(10, k - k1)
                    sub1[b, idx[:k1]] = True
                    sub2[b, idx[-k2:]] = True

                # Infer mu for both sub-splits (use same images, different masks)
                _, mu1, _ = self.infer_zfont(ctx_img, sub1.float())
                _, mu2, _ = self.infer_zfont(ctx_img, sub2.float())
                split_cons = F.mse_loss(mu1, mu2)

        loss = self.recon_weight * recon + self.kl_weight * kl + self.split_consistency_weight * split_cons

        self.log_dict(
            {
                "train/loss": loss,
                "train/recon": recon,
                "train/kl": kl,
                "train/split_cons": split_cons,
                "train/ctx_count": ctx_mask.sum(dim=1).float().mean(),
                "train/tgt_count": tgt_mask.sum(dim=1).float().mean(),
            },
            prog_bar=True,
        )
        return loss
