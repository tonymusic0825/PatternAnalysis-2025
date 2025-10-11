import os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from modules import Generator, Discriminator
from dataset import create_dataloader, CLASS_TO_IDX


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 50

# Optimizer Parameters (StyleGAN2 recommended)
LR = 2e-3
BETA1 = 0.0
BETA2 = 0.99
ADAM_EPS = 1e-8

# Regularization
R1_GAMMA = 10.0  # Weight for the R1 gradient penalty
PL_DECAY = 0.01  # Path Length Regularization decay (path length moving average)
PL_WEIGHT = 2.0  # Weight for the Path Length regularization
R1_FREQ = 16     # Apply R1 regularization every N steps (saves time)
PL_FREQ = 4      # Apply Path Length regularization every N steps

# Models
G = Generator()
D = Discriminator()

# Loss Functions ==================================================================================
# * StyleGAN is typically trained using Logistic Loss

def g_loss_fn(d_fake):
    """Logistic loss function for generator"""
    return F.softplus(-d_fake).mean()

def d_loss_fn(d_real, d_fake, x_real, y, D, gamma):
    """
    This is the logistic loss function for the discriminator
    that also adds R1 regularization
    """
    
    # Loss function 
    loss_total = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

    # Apply r1 regularization ((gamma / 2) * L2Norm(Grad(D(x)))
    r1_penalty = torch.tensor(0.0, device=x_real.device)
    x_real.requires_grad_(True)
    d_out, _ = D(x_real, y)

    # Compute grad w.r.t real image
    grad = torch.autograd.grad(d_out.sum(), x_real, create_graph=True, retain_graph=True, only_inputs=True)[0]

    # L2 Norm
    r1 = grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()

    # Final penalty
    r1_penalty = (gamma * 0.5) * r1
    x_real.requires_grad_(False)

    return loss_total + r1_penalty, r1_penalty.detach()

# ==================================================================================================

# Training Loop ====================================================================================

