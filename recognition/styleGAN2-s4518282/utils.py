import matplotlib.pyplot as plt
from dataset import denorm
import numpy as np

def plot_images(img_nc, img_ad):
    """
    Plots the NC and AD generated image from StyleGAN2

    Expects still normalized [-1, 1] images from generator
    """
    titles = ["NC (Normal Control)", "AD (Alzheimer's Disease)"]
    img_nc_np = denorm(img_nc)
    img_ad_np = denorm(img_ad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    nc_plot = np.squeeze(img_nc_np)
    axes[0].imshow(nc_plot, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off') 

    ad_plot = np.squeeze(img_ad_np)
    axes[1].imshow(ad_plot, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(titles[1], fontsize=14)
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


    