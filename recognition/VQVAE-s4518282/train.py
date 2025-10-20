"""
train.py
---------
Main training loop for the VQ-VAE model on HipMRI data.

Handles:
- Model creation and optimizer setup
- Data loading (train/val)
- Training and validation loops
- Loss computation (reconstruction + VQ loss)
- SSIM metric for reconstruction quality
- Checkpoint saving

Author: Youngsu Choi
"""

import os, math, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim

from modules import VQVAE
from dataset import get_dataloader
from config import *


# ============================================================
# **** Initialise Model, Optimizer, and Data
# ============================================================
model = VQVAE(in_c=1, out_c=1, channels=CHANNELS, embed_num=EMBED_NUM, embed_dim=EMBED_DIM, 
              beta=BETA, n_res=N_RES).to(DEVICE)

train_loader = get_dataloader("train", batch_size=BATCH_SIZE, workers=0, earlyStop=EARLY_STOP)
val_loader = get_dataloader("val", batch_size=BATCH_SIZE, workers=0, earlyStop=EARLY_STOP)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)


# ============================================================
# **** Training Loop
# ============================================================
print(f"Training on {DEVICE} for {EPOCHS} epochs...\n")

# Track only total loss per epoch
train_loss_history = []
val_loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for batch_idx, x in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
        x = x.to(DEVICE)
        optimizer.zero_grad()

        # Predict and calculate loss
        pred, vq_loss = model(x)
        recon_loss = F.l1_loss(pred, x)
        total_loss = recon_loss + vq_loss

        # Grad Descent
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f"[Epoch {epoch:02d}] Step {batch_idx+1:04d} | Total Loss: {total_loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    print(f"Epoch {epoch} Train — Avg Total Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_running_ssim = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(DEVICE)
            pred, vq_loss = model(x)
            recon_loss = F.l1_loss(pred, x)
            total_loss = recon_loss + vq_loss
            val_running_loss += total_loss.item()

            # Compute SSIM for each batch (values in [-1, 1])
            ssim_score = ssim(
                torch.clamp(pred, -1, 1),
                torch.clamp(x, -1, 1),
                data_range=2.0,
            )
            val_running_ssim += ssim_score.item()

    avg_val_loss = val_running_loss / len(val_loader)
    avg_val_ssim = val_running_ssim / len(val_loader)
    val_loss_history.append(avg_val_loss)

    print(f"Validation — Avg Total Loss: {avg_val_loss:.4f} | Avg SSIM: {avg_val_ssim:.4f}")

    # Save checkpoint
    if epoch:
        ckpt_path = os.path.join(SAVE_DIR, f"vqvae_epoch{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}\n")
