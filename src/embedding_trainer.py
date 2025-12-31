from __future__ import annotations

import argparse
from pathlib import Path
import csv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import torch

from embedding_dataset import EmbeddingDataModule, FontSampleConfig
from embedding_model import FontEmbeddingModel
from constants import GOOGLE_FONTS_METADATA_DIR, MODEL_CHECKPOINT_DIR

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_fonts", type=int, default=0, help="If >0, limit number of font files used.")
    p.add_argument("--max_vocab", type=int, default=2048)

    # Glyph rasterization / SDF
    p.add_argument("--canvas", type=int, default=256)
    p.add_argument("--ppem", type=int, default=256)
    p.add_argument("--thresh", type=int, default=16)

    # Sampling
    p.add_argument("--ctx_k_min", type=int, default=10)
    p.add_argument("--ctx_k_max", type=int, default=32)
    p.add_argument("--tgt_k_min", type=int, default=4)
    p.add_argument("--tgt_k_max", type=int, default=16)
    p.add_argument("--min_glyphs", type=int, default=14)

    # Model hparams
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--glyph_emb_dim", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--kl_weight", type=float, default=1e-4)
    p.add_argument("--split_consistency_weight", type=float, default=0.0)

    # Trainer
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "mps"])
    p.add_argument("--seed", type=int, default=0)

    # Logging / output
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="zfont")
    p.add_argument("--save_top_k", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    font_paths = []

    with open(f"{GOOGLE_FONTS_METADATA_DIR}/curated_fonts.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            font_paths.append(row["font_path"])
    font_paths = font_paths[:15]
    
    cfg = FontSampleConfig(
        canvas=args.canvas,
        ppem=args.ppem,
        thresh=args.thresh,
        ctx_k_min=args.ctx_k_min,
        ctx_k_max=args.ctx_k_max,
        tgt_k_min=args.tgt_k_min,
        tgt_k_max=args.tgt_k_max,
        min_glyphs=args.min_glyphs,
    )

    dm = EmbeddingDataModule(
        font_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        cfg=cfg,
        max_vocab=args.max_vocab,
        val_split=0.05,
        prioritize_basic_latin=True,
    )
    dm.setup()
    glyph_vocab = dm.glyph_vocab_size

    model = FontEmbeddingModel(
        glyph_vocab=glyph_vocab,
        z_dim=args.z_dim,
        glyph_emb_dim=args.glyph_emb_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        kl_weight=args.kl_weight,
        recon_weight=1.0,
        split_consistency_weight=args.split_consistency_weight,
    )

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.name)

    ckpt = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        filename="zfont-{epoch:02d}-{train_loss:.4f}",
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
        accumulate_grad_batches=1,
    )

    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(f"{MODEL_CHECKPOINT_DIR}/embedding_model.ckpt")
    print("Embedding model saved to", f"{MODEL_CHECKPOINT_DIR}/embedding_model.ckpt")


if __name__ == "__main__":
    main()
