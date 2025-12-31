from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from embedding_dataset import EmbeddingDataModule, FontSampleConfig
from embedding_model import EmbeddingModel
from diffusion_dataset import DiffusionDataModule, DiffusionFontCfg
from diffusion_model import DiffusionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fonts_dir", type=str, required=True, help="Directory to recursively scan for .ttf/.otf files.")
    p.add_argument("--zfont_ckpt", type=str, required=True, help="Path to pretrained MaskedZFontLit checkpoint.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_fonts", type=int, default=0, help="If >0, limit number of font files used.")
    p.add_argument("--max_vocab", type=int, default=2048, help="Must match the embedding stage to keep gid embedding aligned.")

    # Rasterization / SDF
    p.add_argument("--canvas", type=int, default=256)
    p.add_argument("--ppem", type=int, default=256)
    p.add_argument("--thresh", type=int, default=16)

    # Context sampling for z_font inference DURING diffusion training
    p.add_argument("--ctx_k_min", type=int, default=10)
    p.add_argument("--ctx_k_max", type=int, default=32)
    p.add_argument("--min_glyphs", type=int, default=20)

    # Diffusion hparams
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--unet_base", type=int, default=64)

    # These MUST match how you trained MaskedZFontLit
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--gid_dim", type=int, default=128)
    p.add_argument("--time_dim", type=int, default=128)

    # Trainer
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "mps"])
    p.add_argument("--seed", type=int, default=0)

    # Logging/output
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="sdf_diffusion")
    p.add_argument("--save_top_k", type=int, default=3)
    p.add_argument("--val_split", type=float, default=0.01)

    return p.parse_args()


def find_font_files(fonts_dir: str, max_fonts: int = 0):
    root = Path(fonts_dir).expanduser().resolve()
    paths = list(root.rglob("*.ttf")) + list(root.rglob("*.otf"))
    paths = [str(p) for p in paths]
    paths.sort()
    if max_fonts and max_fonts > 0:
        paths = paths[:max_fonts]
    if not paths:
        raise SystemExit(f"No .ttf/.otf files found under: {root}")
    return paths


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    font_paths = find_font_files(args.fonts_dir, args.max_fonts)

    # ------------------------------------------------------------
    # IMPORTANT: Build the SAME gid_map used during embedding stage
    # ------------------------------------------------------------
    #
    # We reuse FontMaskedDataModule's gid_map builder deterministically, so
    # the glyph_id embedding table in diffusion stays aligned with what you trained.
    #
    dummy_cfg = FontSampleConfig(
        canvas=args.canvas,
        ppem=args.ppem,
        thresh=args.thresh,
        ctx_k_min=args.ctx_k_min,
        ctx_k_max=args.ctx_k_max,
        tgt_k_min=4,
        tgt_k_max=16,
        min_glyphs=max(args.min_glyphs, 14),
    )
    dm_embed = FontMaskedDataModule(
        font_paths,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
        cfg=dummy_cfg,
        max_vocab=args.max_vocab,
        val_split=0.01,
        prioritize_basic_latin=True,
    )
    dm_embed.setup()
    gid_map = dm_embed.gid_map
    glyph_vocab = dm_embed.glyph_vocab_size

    # ------------------------------------------------------------
    # Diffusion DataModule
    # ------------------------------------------------------------
    diff_cfg = DiffusionFontCfg(
        canvas=args.canvas,
        ppem=args.ppem,
        thresh=args.thresh,
        ctx_k_min=args.ctx_k_min,
        ctx_k_max=args.ctx_k_max,
        min_glyphs=args.min_glyphs,
        max_codepoints_per_font=512,
    )
    dm_diff = DiffusionDataModule(
        font_paths,
        gid_map=gid_map,
        cfg=diff_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    dm_diff.setup()

    # ------------------------------------------------------------
    # Diffusion LightningModule (loads & freezes pretrained z_font model)
    # ------------------------------------------------------------
    model = DiffusionModel(
        zfont_ckpt_path=args.zfont_ckpt,
        zfont_module_cls=EmbeddingModel,
        glyph_vocab=glyph_vocab,
        gid_dim=args.gid_dim,
        z_dim=args.z_dim,
        time_dim=args.time_dim,
        unet_base=args.unet_base,
        T=args.T,
        lr=args.lr,
    )

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.name)

    ckpt = ModelCheckpoint(
        monitor="train/loss",   # simplest; add val step later if you want monitor="val/loss"
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        filename="diff-{epoch:02d}-{train_loss:.4f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=25,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=dm_diff)
    trainer.save_checkpoint("diffusion_model.ckpt")
    print("Saved checkpoint to diffusion_model.ckpt")


if __name__ == "__main__":
    main()
