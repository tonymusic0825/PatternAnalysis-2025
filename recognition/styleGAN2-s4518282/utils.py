import matplotlib.pyplot as plt
from dataset import denorm
import numpy as np
import torch

def plot_images(gen, z_dim, device):
    """
    Plots the NC and AD generated image from StyleGAN2

    Expects still normalized [-1, 1] images from generator
    """
    titles = ["NC (Normal Control)", "AD (Alzheimer's Disease)"]
    labels_to_generate = torch.tensor([0, 1], device=device) 
    batch_size = 2 
    gen.eval()

    # Generate images
    with torch.no_grad():
        z = torch.randn(batch_size, z_dim, device=device)
        generated_images = gen(z, labels_to_generate, False)

    gen.train()

    img_nc = denorm(generated_images[0]) # NC image (label 0)
    img_ad = denorm(generated_images[1]) # AD image (label 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    nc_plot = np.squeeze(img_nc)
    axes[0].imshow(nc_plot, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(titles[0])
    axes[0].axis('off') 

    ad_plot = np.squeeze(img_ad)
    axes[1].imshow(ad_plot, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.show()

def plot_loss(d_loss, g_loss):
    """
    Plots the loss function of both Discriminator and Generator

    Expects lists
    """
    steps = np.arange(len(d_loss))
    plt.figure(figsize=(12,6))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="Generator")
    plt.plot(d_loss, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    