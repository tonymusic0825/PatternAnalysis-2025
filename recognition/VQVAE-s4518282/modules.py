"""
modules.py
-----------
Defines the neural architecture for a Vector Quantized Variational Autoencoder (VQ-VAE).

Modules include:
- Residual blocks and stacks 
- Encoder / Decoder with instance normalization
- Vector Quantizer 
- Full VQVAE model wrapper

Author: Youngsu Choi
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

# ============================================================
# *** Residual Structures
# ============================================================
class ResidualBlock(nn.Module):
    """
    Basic residual block with two convolutions and ReLU activation.
    """
    def __init__(self, channels):
        """
        Initialize a residual block.

        Args:
            channels (int): Number of input/output feature channels for both
                convolutions in the block.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        """Add skip connection to processed output."""
        return x + self.block(x)

class ResidualStack(nn.Module):
    """
    Stack multiple residual blocks sequentially.
    """
    def __init__(self, channels, n_blocks=2):
        """
        Initialize a stack of residual blocks.

        Args:
            channels (int): Number of channels for each residual block.
            n_blocks (int): How many residual blocks to stack.
        """
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(x)
    

# ============================================================
# *** Encoder
# ============================================================
class DownBlock(nn.Module):
    """
    Downsampling block: Conv → InstanceNorm → LeakyReLU.
    """
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1):
        """
        Initialize a downsampling block.

        Args:
            in_c (int): Number of input channels.
            out_c (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            stride (int): Convolution stride (controls downsampling rate).
            padding (int): Convolution padding.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """
    Encoder: progressively downsamples input image and applies residual stack.
    """
    def __init__(self, in_c=1, channels=(64, 128, 256), n_res=2):
        """
        Initialize the encoder network.

        Args:
            in_c (int): Number of input channels (e.g., 1 for grayscale).
            channels (tuple[int, ...]): Feature map sizes per stage. The last
                entry is the latent channel size before quantization.
            n_res (int): Number of residual blocks to apply at the end.
        """
        super().__init__()

        self.layers = []

        self.layers.append(nn.Conv2d(in_c, channels[0], kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(len(channels) - 1):
            self.layers.append(DownBlock(channels[i], channels[i + 1]))
        
        self.res_stack = ResidualStack(channels[-1], n_blocks=n_res)
        self.encoder = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_stack(x)

        return x

# ============================================================
# *** Decoder
# ============================================================
class UpBlock(nn.Module):
    """
    Upsampling block: ConvTranspose → InstanceNorm → LeakyReLU.
    """
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1):
        """
        Initialize an upsampling block.

        Args:
            in_c (int): Number of input channels.
            out_c (int): Number of output channels.
            kernel_size (int): Transposed convolution kernel size.
            stride (int): Transposed convolution stride (controls upsampling).
            padding (int): Transposed convolution padding.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    """
    Decoder: reconstructs image from quantized latent embeddings.
    """
    def __init__(self, out_c=1, channels=(256, 128, 64), n_res=2):
        """
        Initialize the decoder network.

        Args:
            out_c (int): Number of channels in the reconstructed output
                (e.g., 1 for grayscale reconstruction).
            channels (tuple[int, ...]): Channel sizes following the encoder's
                reverse order. The first item is the latent channel size.
            n_res (int): Number of residual blocks applied after the first
                upsampling stage (matching the latent resolution).
        """
        super().__init__()

        layers = []

        # Upsample stages (reverse of encoder)
        for i in range(len(channels) - 1):
            layers.append(UpBlock(channels[i], channels[i + 1]))

        self.res_stack = ResidualStack(channels[1], n_blocks=n_res)  # 2nd channel in list after first upsample
        self.decoder = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.Conv2d(channels[-1], out_c, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i == 0:
                x = self.res_stack(x)  
        return self.final(x)


# ============================================================
# *** Vector Quantizer
# ============================================================
class Quantizer(nn.Module):
    """ 
    Vector quantization layer for VQ-VAE.
    Converts continuous encoder outputs to discrete embeddings.

    REF: https://github.com/explainingai-code/VQVAE-Pytorch/blob/main/model/quantizer.py 
    """
    def __init__(self, embed_num=512, embed_dim=256, beta=0.25):
        """
        Initialize the vector quantizer.

        Args:
            embed_num (int): Size of the codebook (number of embeddings).
            embed_dim (int): Dimensionality of each code vector.
            beta (float): Weight for the commitment loss term.
        """
        super().__init__()
        self.beta = beta
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        
        # Create embedding
        self.embedding = nn.Embedding(embed_num, embed_dim)

        # Initialize codebook weights to small uniform values for stability
        # This helps prevent large distance magnitudes at the start of training
        # and avoids "dead" codebook entries early on.
        self.embedding.weight.data.uniform_(-1.0 / embed_num, 1.0 / embed_num)

    def forward(self, x):
        """
        Args:
            x (Tensor): Encoder output of shape [B, C, H, W].
        Returns:
            quantized (Tensor): Quantized latent map [B, C, H, W].
            vq_loss (Tensor): Codebook + commitment loss.
            indices (Tensor): Code indices for each spatial position.
        """

        # Flatten [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_perm.view(B, -1, C) 

        # Compute distances [B, H*W, codebook_size]
        dist = torch.cdist(x_flat, self.embedding.weight[None, :].expand(B, -1, -1)) 
        indices = torch.argmin(dist, dim=-1) # Pick the closest! [B, H*W]

        # Embedding lookup
        quantized = self.embedding(indices.view(-1))
        quantized = quantized.view(B, H, W, C) 

        # Compute codebook + commitment loss
        x_flat_all = x_flat.reshape(-1, C)      # [B*H*W, C]
        quant_flat = quantized.reshape(-1, C)   # [B*H*W, C]

        codebook_loss = F.mse_loss(quant_flat.detach(), x_flat_all)
        commit_loss = F.mse_loss(quant_flat, x_flat_all.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Enable backprop and reshape
        quantized = x_perm + (quantized - x_perm).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(B, H, W)

        return quantized, vq_loss, indices


# ============================================================
# *** VQ-VAE Model
# ============================================================
class VQVAE(nn.Module):
    """
    Full VQ-VAE model combining Encoder, Quantizer, and Decoder.
    """
    def __init__(self, in_c=1, out_c=1, channels=(64,128,256,256), embed_num=512, embed_dim=256, beta=0.25, n_res=2):
        """
        Initialize the full VQ-VAE.

        Args:
            in_c (int): Number of input channels to the encoder.
            out_c (int): Number of output channels from the decoder.
            channels (tuple[int, ...]): Encoder channel sizes; the decoder
                uses the reverse of this tuple.
            embed_num (int): Codebook size for the vector quantizer.
            embed_dim (int): Code (embedding) dimensionality.
            beta (float): Commitment loss weight.
            n_res (int): Number of residual blocks in encoder/decoder stacks.
        """
        super().__init__()
        self.encoder = Encoder(in_c=in_c, channels=channels, n_res=n_res)
        self.quantizer = Quantizer(embed_num=embed_num, embed_dim=embed_dim, beta=beta)
        self.decoder = Decoder(out_c=out_c, channels=channels[::-1])

        ch = channels[-1]

        # Project encoder output → embedding_dim
        self.pre_quant_conv = nn.Conv2d(ch, embed_dim, kernel_size=1)

        # Project embedding_dim → decoder input
        self.post_quant_conv = nn.Conv2d(embed_dim, ch, kernel_size=1)
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e) 
        z_q, vq_loss, indices = self.quantizer(z_e)
        z_q = self.post_quant_conv(z_q)  
        pred = self.decoder(z_q)

        return pred, vq_loss

# SANITY CHECK
if __name__ == "__main__":
    enc = Encoder()
    quant = Quantizer(embed_num=512, embed_dim=256, beta=0.25)
    dec = Decoder()

    # Dummy input
    x = torch.randn(16, 1, 256, 128)
    print(f"Input shape: {x.shape}")

    # Encoder forward
    z_e = enc(x)
    print("Output shape (Encoder):", z_e.shape)

    # Quantizer forward
    z_q, vq_loss, indices = quant(z_e)
    print("Output shape (Quantizer):", z_q.shape)
    print("Quantizer loss:", round(vq_loss.item(), 6))
    print("Codebook indices shape:", indices.shape)

    # Decoder forward
    x_hat = dec(z_q)
    print("Output shape (Decoder):", x_hat.shape)

    # Test VQVAE
    print("\n==== TEST VQVAE FULL ====")
    model = VQVAE(in_c=1, out_c=1, channels=(64,128,256,256), embed_num=512, embed_dim=256, beta=0.25)

    # Forward pass through entire model
    pred, vq_loss = model(x)

    print("Input:", x.shape)
    print("Reconstructed output:", pred.shape)
    print("VQ Loss:", round(vq_loss.item(), 6))

    # Simple reconstruction loss test
    recon_loss = torch.nn.functional.l1_loss(pred, x)
    total_loss = recon_loss + vq_loss
    print("Reconstruction loss:", round(recon_loss.item(), 6))
    print("Total loss:", round(total_loss.item(), 6))