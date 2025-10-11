import os, math, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from utils import plot_images, plot_loss

from modules import Generator, Discriminator
from dataset import create_dataloader, CLASS_TO_IDX, denorm


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 50
WORKERS = 8

# Optimizer Parameters (StyleGAN2 recommended)
LR = 2e-3
BETA1 = 0.0
BETA2 = 0.99
ADAM_EPS = 1e-8

# Regularization
R1_GAMMA = 10.0  # Weight for the R1 gradient penalty
# PL_DECAY = 0.01  # Path Length Regularization decay (path length moving average)
# PL_WEIGHT = 2.0  # Weight for the Path Length regularization
R1_FREQ = 16     # Apply R1 regularization every N steps (saves time)
# PL_FREQ = 4      # Apply Path Length regularization every N steps

# Loss Functions ==================================================================================
# * StyleGAN is typically trained using Logistic Loss

def g_loss_fn(d_fake):
    """Logistic loss function for generator"""
    return F.softplus(-d_fake).mean()

def d_loss_fn(d_real, d_fake, x_real, y, D, gamma, do_r1):
    """
    This is the logistic loss function for the discriminator
    that also adds R1 regularization
    """
    
    # Loss function 
    loss_total = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

    # Apply r1 regularization ((gamma / 2) * L2Norm(Grad(D(x)))
    r1_penalty = torch.tensor(0.0, device=x_real.device)

    if do_r1:
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
G = Generator()
D = Discriminator()
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
    
    for epoch in range(1, EPOCHS + 1):
        for idx, (real_img, real_label) in enumerate(dataloader):
            g_step += 1
            real_img = real_img.to(device)
            real_label = real_label.to(device)

            # *** TRAIN DISCRIMINATOR
            opt_d.zero_grad()

            # Generate fake images
            z = torch.randn(BATCH_SIZE, 256, device=device)

            with torch.no_grad():
                fake_img = G(z, real_label)
            
            # Forward + backward pass
            # We do r1 penality only every R1_FREQ step
            d_real, _ = D(real_img, real_label)
            d_fake, _ = D(fake_img.detach(), real_label)

            do_r1_step = g_step % R1_FREQ == 0
            d_loss, r1_penalty = d_loss_fn(
                d_real=d_real,
                d_fake=d_fake,
                x_real=real_img,
                y=real_label,
                D = D,
                gamma=R1_GAMMA,
                do_r1=do_r1_step
            )
            
            d_loss_t.append(d_loss.item())
            d_loss.backward()
            opt_d.step()

            # *** TRAIN GENERATOR
            opt_g.zero_grad()

            z = torch.randn(BATCH_SIZE, 256, device=device)
            fake_img = G(z, real_label) 

            # Forward pass through D but backward using G
            d_fake_for_g, _ = D(fake_img, real_label)
            g_loss = g_loss_fn(d_fake_for_g)
            g_loss_t.append(g_loss.item())
            g_loss.backward()
            opt_g.step()

            # *** Print and Plot
            if epoch == 1:
                z = torch.randn(BATCH_SIZE, 256, device=device)
                fake_img1 = G(z, 0)
                fake_img2 = G(z, 1)

                plot_images(fake_img1, fake_img2)
                plot_loss(d_loss_t, g_loss_t)

        print(f"EPOCH{epoch}: D_loss = {d_loss_t[-1]}, G_loss = {g_loss_t[-1]}")

        if test and epoch == 5:
            break

if __name__ == "__main__":
    train(True)

