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
# Entry Point
# ============================================================
if __name__ == "__main__":

    # Load Model and Test Data
    model = load_model(CHECKPOINT_PATH)
    test_loader = get_dataloader("train", batch_size=BATCH_SIZE, workers=NUM_WORKERS, earlyStop=EARLY_STOP)