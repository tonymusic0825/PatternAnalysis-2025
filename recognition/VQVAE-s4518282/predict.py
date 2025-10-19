"""
predict.py
---------
Handles all analysis of VQ-VAE model. 

Handles:
- Plotting loss
- Calculate SSIM scores on test set
- ...

Author: Youngsu Choi
"""

import torch
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from modules import VQVAE

# ============================================================
# Load model checkpoint
# ============================================================
def load_model(checkpoint_path):
    """
    Given a checkpoint path, loads model 

    Args:
        checkpoint_path (str): Path to model checkpoint

    Returns:
        nn.Module: A trained / loaded VQ-VAE model 
    """
    model = VQVAE(in_c=1, out_c=1, channels=CHANNELS, embed_num=EMBED_NUM, embed_dim=EMBED_DIM, 
                beta=BETA, n_res=N_RES).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")

    return model

# ============================================================
# Plot loss curves
# ============================================================
def plot_loss_curves(checkpoint_path, save=False, save_path="loss.png"):
    """
    Given a checkpoint path, loads loss histories (train + val) and plots it.

    Args:
        checkpoint_path (str): Path to model checkpoint
        save (bool): If True, saves the plot to save_path rather than showing it
        save_path (str): Path to save
    """
    # Load data
    data = torch.load(checkpoint_path, weights_only=True)
    train_loss = data["train_loss_history"]
    val_loss = data["val_loss_history"]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("VQVAE Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":

    # Load Model and Test Data
    model = load_model(CHECKPOINT_PATH)
    test_loader = get_dataloader("train", batch_size=BATCH_SIZE, workers=NUM_WORKERS, earlyStop=EARLY_STOP)

    # Plot loss graph
    # plot_loss_curves(CHECKPOINT_PATH) 