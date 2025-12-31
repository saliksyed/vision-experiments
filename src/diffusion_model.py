from __future__ import annotations

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ---------- schedule helpers ----------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    # from Nichol & Dhariwal
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.02)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    # a: [T], t: [B]
    out = a.gather(0, t)
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))


# ---------- conditioning + UNet ----------

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, channels)
        self.to_shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W], cond: [B,cond_dim]
        s = self.to_scale(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b


class ResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = FiLM(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SimpleUNet(nn.Module):
    """
    A small UNet that predicts noise epsilon.
    Conditioning is done via FiLM in ResBlocks.
    """
    def __init__(self, in_ch: int = 1, base: int = 64, cond_dim: int = 256):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = nn.Sequential(
            ResBlock(base, cond_dim),
            ResBlock(base, cond_dim),
        )
        self.downsample1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.down2 = nn.Sequential(
            ResBlock(base * 2, cond_dim),
            ResBlock(base * 2, cond_dim),
        )
        self.downsample2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.mid = nn.Sequential(
            ResBlock(base * 4, cond_dim),
            ResBlock(base * 4, cond_dim),
        )

        self.upsample2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2 = nn.Sequential(
            ResBlock(base * 2, cond_dim),
            ResBlock(base * 2, cond_dim),
        )

        self.upsample1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1 = nn.Sequential(
            ResBlock(base, cond_dim),
            ResBlock(base, cond_dim),
        )

        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        d1 = self.down1(x, cond) if isinstance(self.down1, ResBlock) else self._forward_seq(self.down1, x, cond)
        x = self.downsample1(d1)

        d2 = self._forward_seq(self.down2, x, cond)
        x = self.downsample2(d2)

        x = self._forward_seq(self.mid, x, cond)

        x = self.upsample2(x)
        x = x + d2
        x = self._forward_seq(self.up2, x, cond)

        x = self.upsample1(x)
        x = x + d1
        x = self._forward_seq(self.up1, x, cond)

        return self.out(x)

    @staticmethod
    def _forward_seq(seq: nn.Sequential, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for m in seq:
            if isinstance(m, ResBlock):
                x = m(x, cond)
            else:
                x = m(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # sinusoidal
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------- Lightning diffusion module ----------

class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        *,
        # pretrained embedding model (your MaskedZFontLit checkpoint)
        zfont_ckpt_path: str,
        # import class for embedding model
        zfont_module_cls,
        glyph_vocab: int,
        gid_dim: int = 128,
        z_dim: int = 64,
        time_dim: int = 128,
        unet_base: int = 64,
        T: int = 1000,
        lr: float = 2e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["zfont_module_cls"])

        # ---- load pretrained z_font encoder and freeze it ----
        self.zfont_model = zfont_module_cls.load_from_checkpoint(zfont_ckpt_path)
        self.zfont_model.eval()
        for p in self.zfont_model.parameters():
            p.requires_grad = False

        # ---- conditioning ----
        self.gid_emb = nn.Embedding(glyph_vocab, gid_dim)
        self.t_emb = TimeEmbedding(time_dim)
        cond_dim = z_dim + gid_dim + time_dim

        self.unet = SimpleUNet(in_ch=1, base=unet_base, cond_dim=cond_dim)

        # ---- diffusion buffers ----
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.T = T
        self.lr = lr
        self.z_dim = z_dim

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=1e-4)

    @torch.no_grad()
    def compute_z_font(self, ctx_img: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        # ctx_img: [B,K,1,H,W], ctx_mask: [B,K]
        # Uses the pretrained model’s inference path
        z, _, _ = self.zfont_model.infer_zfont(ctx_img, ctx_mask)
        return z  # [B,z_dim]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        ctx_img = batch["ctx_img"]   # [B,K,1,H,W]
        ctx_mask = batch["ctx_mask"] # [B,K]
        x0 = batch["tgt_img"]        # [B,1,H,W]  (SDF)
        gid = batch["tgt_gid"]       # [B]

        B = x0.shape[0]
        device = x0.device

        # 1) compute z_font from context glyphs (frozen)
        with torch.no_grad():
            z_font = self.compute_z_font(ctx_img, ctx_mask)  # [B,z_dim]

        # 2) sample t and noise
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)

        # 3) forward diffuse
        xt = self.q_sample(x0, t, noise)

        # 4) build conditioning vector
        cond = torch.cat(
            [
                z_font,
                self.gid_emb(gid),
                self.t_emb(t),
            ],
            dim=-1,
        )

        # 5) predict noise and compute loss
        noise_hat = self.unet(xt, cond)
        loss = F.mse_loss(noise_hat, noise)

        self.log("train/loss", loss, prog_bar=True)
        return loss
