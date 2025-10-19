"""
config.py
---------
Configuration file for hyperparameters

Author: Youngsu Choi
"""
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# **** Data
# ============================================================
BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS = 50
EARLY_STOP = False  

# ============================================================
# **** Model
# ============================================================
CHANNELS = (64, 128, 256)
EMBED_NUM = 1024
EMBED_DIM = 64
BETA = 0.25
N_RES = 5

# ============================================================
# **** Optim
# ============================================================
LEARNING_RATE = 3e-4

# ============================================================
# **** Save / Logs
# ============================================================
LOG_INTERVAL = 50
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# For testing only
CHECKPOINT_PARENT = "checkpoints/"
CHECKPOINT_PATH = "checkpoints/vqvae_epoch019.pt"