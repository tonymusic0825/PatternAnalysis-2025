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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
WORKERS = 4
N_CLASSES = 2
LOG_RESOLUTION = 8

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

def g_loss_fn(d_fake, batch_idx, w, ppl):
    """Loss function for generator (w/ PPL)"""
    loss = -torch.mean(d_fake)

    if batch_idx % 16 == 0:
        plp = ppl(w, d_fake)

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
    gp_term = gradient_penalty(disc, real, fake, DEVICE=DEVICE)

    # Loss
    loss = -(torch.mean(d_real) - torch.mean(d_fake))
    loss = loss + (LAMBDA_GP * gp_term) + r1_term


    return loss, gp_term, r1_term

def gradient_penalty(disc, real, fake, DEVICE=DEVICE):
    """
    Calculates the WGAN-GP Gradient Penalty.
    (This function is directly copied from your provided code.)
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(DEVICE)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = disc(interpolated_images)
 
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

def get_z_y(batch_size):
    """
    Generates the necessary inputs for the Generator (Z and Y).
    
    NOTE: Generator is assumed to handle its own noise generation internally.
    
    Args:
        batch_size (int): The current batch size.
        
    Returns:
        tuple: (z_noise, y_labels)
    """
    # Latent Noise Z
    z = torch.randn(batch_size, Z_DIM, device=DEVICE)
    
    # Conditional Label Y
    y = torch.randint(0, N_CLASSES, (batch_size,), device=DEVICE)
    
    return z, y

def get_noise(batch_size):
    noise = []
    res = 4

    for i in range(LOG_RESOLUTION):
        n1 = torch.randn(batch_size, 1, res, res, device=DEVICE) if i != 0 else None
        n2 = torch.randn(batch_size, 1, res, res, device=DEVICE)
        noise.append((n1, n2))
        res *= 2

    return noise

def get_w(batch_size, mapping):
    z, y = get_z_y(batch_size)
    w = mapping(z, y)

    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

# Train Step =======================================================================================
def train_fn(
    disc,
    gen,
    path_length_penalty,
    dataloader,
    opt_critic,
    opt_gen,
):
    """
    Performs one full training loop over the dataset.
    """
    loop = tqdm(dataloader, leave=True)

    epoch_losses = {
        'loss_d': 0.0,
        'loss_g': 0.0,
        'n_items': 0 
    }

    for batch_idx, (real, y_labels_real) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        epoch_losses['n_items'] += cur_batch_size

        # 1. Prepare Inputs (Z, Y). Noise is handled internally by the Generator.
        w = get_w(cur_batch_size)
        noise = get_noise(cur_batch_size)
        
        # *** Train Discriminator/Critic 
        with torch.cuda.amp.autocast():
            # Generator now takes (Z, Y) and returns (fake_image, w_single)
            # w_single is the BxW_DIM latent code needed for PPL
            # The generator generates its own noise internally.
            fake = gen(w, noise)
            
            # Detach fake for D training
            d_fake = disc(fake.detach())
            d_real = disc(real)
            
            # Calculate total D loss
            loss_disc, gp, r1_term = d_loss_fn(d_real, d_fake, real, fake.detach(), disc)

        disc.zero_grad()
        loss_disc.backward()
        opt_critic.step()
        epoch_losses['loss_d'] += loss_disc.item() * cur_batch_size

        # *** Train Generator 
        gen_fake = disc(fake)
        
        # Calculate total G loss
        loss_gen = g_loss_fn(
            gen_fake, 
            batch_idx,
            w,
            path_length_penalty
        )
        
        # The Mapping Network is part of 'gen', so we just optimize 'gen'.
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        epoch_losses['loss_g'] += loss_gen.item() * cur_batch_size

        # --- Update Postfix / Logging ---
        loop.set_postfix(
            gp=gp.item(),
            r1=r1_term.item(),
            loss_disc=loss_disc.item(),
            loss_gen=loss_gen.item()
        )
    
    avg_losses = {}
    n_items = epoch_losses['n_items']
    if n_items > 0:
        avg_losses = {
            'loss_d_avg': epoch_losses['loss_d'] / n_items,
            'loss_g_avg': epoch_losses['loss_g'] / n_items,
        }

    return avg_losses


# Training Loop ====================================================================================
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)
dataloader, _ = create_dataloader(batch_size=BATCH_SIZE, num_workers=1)
print("DATA LOADER SANITY CHECK")
print(len(dataloader))
print(len(next(iter(dataloader))))
opt_g = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2), eps=ADAM_EPS)
opt_d = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2), eps=ADAM_EPS)
step = 0
g_loss_t = []
d_loss_t = []

for epoch in range(EPOCHS):
    avg_losses = train_fn(
        D,
        G,
        path_length_penalty,
        dataloader,
        opt_d,
        opt_g,
    )

    d_loss_t.append(avg_losses.get('loss_d_avg', 0))
    g_loss_t.append(avg_losses.get('loss_g_avg', 0))

    plot_images(G, Z_DIM, DEVICE)


