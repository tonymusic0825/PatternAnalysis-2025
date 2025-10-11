import os, math, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from utils import plot_images, plot_loss
from tqdm import tqdm

from modules import Generator, Discriminator, PathLengthPenalty
from dataset import create_dataloader, CLASS_TO_IDX, denorm


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
WORKERS = 4

# Optimizer Parameters (StyleGAN2 recommended)
LR = 1e-3
BETA1 = 0.0
BETA2 = 0.99
ADAM_EPS = 1e-8

# Model Parameters
Z_DIM = 256
W_DIM = 256
LAMBDA_GP = 10 # For gradient penalty

# Loss Functions ==================================================================================
# * StyleGAN is typically trained using Logistic Loss

def g_loss_fn(d_fake, batch_idx, w, fake, ppl):
    """Loss function for generator (w/ PPL)"""
    loss = -torch.mean(d_fake)

    if batch_idx % 16 == 0:
        plp = ppl(w, fake)

        if not torch.isnan(plp):
            loss += plp

    return loss

def d_loss_fn(d_real, d_fake, real, fake, disc):
    """
    This is the logistic loss function for the discriminator
    that also adds R1 regularization
    """
    
    # R1 term
    r1_term = 0.001 * torch.mean(d_real ** 2)

    # Gradient Penalty Calc
    gp_term = gradient_penalty(disc, real, fake, device=device)

    # Loss
    loss = -(torch.mean(d_real) - torch.mean(d_fake))
    loss = loss + (LAMBDA_GP * gp_term) + r1_term


    return loss, gp_term, r1_term

def gradient_penalty(critic, real, fake, device=device):
    """
    Calculates the WGAN-GP Gradient Penalty.
    (This function is directly copied from your provided code.)
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

# ==================================================================================================

# Training Loop ====================================================================================
G = Generator()
D = Discriminator()
path_length_penalty = PathLengthPenalty(0.99).to(device)
dataloader, _ = create_dataloader(batch_size=BATCH_SIZE, num_workers=1)
print("DATA LOADER SANITY CHECK")
print(len(dataloader))
print(len(next(iter(dataloader))))
opt_g = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2), eps=ADAM_EPS)
opt_d = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2), eps=ADAM_EPS)
step = 0
g_loss_t = []
d_loss_t = []

def train(test=False):
    pass

if __name__ == "__main__":
    train(True)

