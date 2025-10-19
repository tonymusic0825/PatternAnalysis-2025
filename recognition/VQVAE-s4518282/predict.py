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
# Visualize reconstructions + compute SSIM
# ============================================================
def evaluate_model(model, test_loader, show_img=True, batch_max=2):
    """
    Given a loaded model, evaluates its performance on test data using SSIM scores.
    Also plots the original vs reconstruction images.

    Args:
        model (nn.Module): A fully trained / loaded model
        test_loader (torch.utils.data.DataLoader): A test dataloader that is 'loaded'
        show_img (bool): If True, show the original vs recon plot
        batch_max (int): Defines how many batches to show for plot
    """

    # Load test data
    total_ssim = 0.0
    min_ssim = 1.0
    max_ssim = 0.0
    num_batches = 0

    # Get a few batches for visualization
    with torch.no_grad():
        for batch_idx, x in enumerate(test_loader):
            x = x.to(DEVICE)
            pred, _ = model(x)

            # Compute SSIM for this batch
            ssim_score = ssim(
                torch.clamp(pred, -1, 1),
                torch.clamp(x, -1, 1),
                data_range=2.0,
            ).item()

            # Update total ssim
            total_ssim += ssim_score
            num_batches += 1
            
            # Update min and max ssim
            min_ssim = ssim_score if ssim_score < min_ssim else min_ssim
            max_ssim = ssim_score if ssim_score > max_ssim else max_ssim

            # Plot first few batches only
            if batch_idx < batch_max and show_img:
                visualize_batch(x, pred, batch_idx)

    avg_ssim = total_ssim / num_batches
    print(f"Minimum SSIM on test set: {min_ssim:.4f}")
    print(f"Average SSIM on test set: {avg_ssim:.4f}")
    print(f"Maximum SSIM on test set: {max_ssim:.4f}")


# ============================================================
# Visualization helper
# ============================================================
def visualize_batch(inputs, outputs, batch_idx, num_images=8):
    """
    Given original images and reconstructed images plots them where,
    First Row: Original
    Second Row: Reconstructed

    Args:
        inputs (torch.Tensor): A fully trained / loaded model
        outputs (torch.Tensor): If True, show the original vs recon plot
        batch_idx (int): Current Batch Number
        num_images (int): Number of image pairs to plot
    """
    inputs = inputs.cpu().numpy()
    outputs = outputs.cpu().numpy()

    # Safe guard against num_images > batch size
    batch_size = min(num_images, inputs.shape[0]) 
    plt.figure(figsize=(10, 5))

    # Plot original vs reconstruction 
    for i in range(batch_size):
        # Input
        plt.subplot(2, batch_size, i + 1)

        if i >= 4:
            plt.imshow(inputs[i, 0], cmap="plasma", vmin=-1, vmax=1)
        else:
            plt.imshow(inputs[i, 0], cmap="gray", vmin=-1, vmax=1)
        plt.axis("off")

        if i == 0:
            plt.ylabel("Input")

        # Reconstruction
        plt.subplot(2, batch_size, batch_size + i + 1)
        if i >= 4:
            plt.imshow(outputs[i, 0], cmap="plasma", vmin=-1, vmax=1)
        else:
            plt.imshow(outputs[i, 0], cmap="gray", vmin=-1, vmax=1)
        plt.axis("off")
        
        if i == 0:
            plt.ylabel("Reconstruction")

    plt.tight_layout()
    plt.show()

# ============================================================
# Visualize Codebook Usage
# ============================================================
def visualise_codebook_usage(model, dataloader, save=False, save_path="./"):
    """
    Visualises the distribution of codebook usage on the given dataloader

    Args:
        model (nn.Module): A fully trained / loaded model
        dataloader (torch.utils.data.DataLoader): A dataloader that is 'loaded'
        save (bool): If True, saves the plot as image instead of showing
        save_path (str): Path to save the images if save == True
    """

    # Initialise an empty usage table
    usage = torch.zeros(model.quantizer.embed_num, device=DEVICE)
    print("Calculating codebook usage frequency...")

    # For all data predict and append usages 
    with torch.no_grad():
        for x in tqdm(dataloader, desc="Usage"):
            x = x.to(DEVICE)
            z_e = model.pre_quant_conv(model.encoder(x))
            _, _, indices = model.quantizer(z_e)
            flat_idx = indices.view(-1)
            usage.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
    
    # Print the top 5 codes and their usagees
    top_k_counts, top_k_indices = torch.topk(usage, k=5)
    print("\n--- Top 5 Codebook Usages ---")
    for i in range(5):
        index = top_k_indices[i].item()
        count = int(top_k_counts[i].item()) # Convert to integer as it's a count
        print(f"Rank {i+1}: Index {index}, Count {count}")

    # Plot
    usage = usage.cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(usage)), usage, color='blue')
    plt.title("Codebook Usage Frequency")
    plt.xlabel("Code Index")
    plt.ylabel("Count")
    plt.ylim(0, 150000)

    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ============================================================
# Visualize Usage of Latent Space Codebook
# ============================================================
def visualize_input_latent_output(model, test_loader):
    """
    Generates a original -> latent index map -> reconstruction view
    plot of a single test sample.

    Args:
        model (nn.Module): Fully trained / loaded model
        dataloader (torch.utils.data.DataLoader): Loaded dataloader (train/val/test)
    """

    x = next(iter(test_loader)).to(DEVICE)
    with torch.no_grad():
        z_e = model.encoder(x)
        z_e = model.pre_quant_conv(z_e)
        z_q, _, indices = model.quantizer(z_e)
        z_q = model.post_quant_conv(z_q)
        recon = model.decoder(z_q)

    # Pick the first sample
    img_in = x[0, 0].cpu()
    img_latent = indices[0].cpu()
    img_out = recon[0, 0].cpu()

    # Display 3 columns
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_in, cmap="plasma", vmin=-1, vmax=1)
    plt.title("Original Image")
    plt.axis("off")

    # Latent index map
    plt.subplot(1, 3, 2)
    plt.imshow(img_latent, cmap="tab20")
    plt.title("Discrete Latent Index Map")
    plt.axis("off")

    # Reconstruction
    plt.subplot(1, 3, 3)
    plt.imshow(img_out, cmap="plasma", vmin=-1, vmax=1)
    plt.title("Reconstructed Output")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":

    # Load Model and Test Data
    model = load_model(CHECKPOINT_PATH)
    test_loader = get_dataloader("test", batch_size=BATCH_SIZE, workers=NUM_WORKERS, earlyStop=EARLY_STOP)

    # Plot loss graph
    # plot_loss_curves(CHECKPOINT_PATH) 

    # Evaluate model + visualise reconstructions
    # evaluate_model(model, test_loader)

    # visualise_codebook_usage(model, test_loader)

    visualize_input_latent_output(model, test_loader)