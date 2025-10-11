import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper functions =======================================================

def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.2, inplace=True)

def conv3x3(in_c, out_c):
    """
    3x3 conv with padding. StyleGAN2 actually uses 
    equalized LR but we'll keep it simpler for now
    """
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)

def conv1x1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

# =========================================================================

# DISCRIMINATOR ===========================================================
class ResDown(nn.Module):
    """
    A Residual downsampling block:
        1) Main: Conv + Activation + Conv + Avg Pool
        2) Skip: Conv + Avg Pool
        3) Output = main + skip
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = conv3x3(in_c, in_c)
        self.conv2 = conv3x3(in_c, out_c)
        self.skip = conv1x1(in_c, out_c)
        self.avg = nn.AvgPool2d(2)
    
    def forward(self, x):
        skip = self.avg(self.skip(x))
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.avg(main)

        return main + skip

class MBStdDev(nn.Module):
    """
    StyleGAN Trick: Append a channel with per-group std dev to help Discriminator. 
    Helps detect similar fakes.

    1)  Calculate Standard Deviation: 
        For a given feature map (tensor) x of shape [B,C,H,W] the layer 
        computes the standard deviation for each feature element (pixel location and channel) 
        across all N images in the minibatch.

        This results in a tensor of shape [1,C,H,W]. The batch dimension is now 1 because 
        the statistic is taken across the batch.

    2)  Calculate Mean Standard Deviation: 
        Averages the standard deviation tensor over all channels (C) 
        and spatial locations (H,W) to get a single, scalar value.

    3)  Create a New Feature Map: 
        This single scalar value is replicated to create a new feature map of shape [B,1,H,W], 
        where every element in this new feature map has the same standard deviation value.

    4)  Concatenation: This new, single-channel feature map is concatenated with the original 
        feature map x along the channel dimension. 
        The discriminator now processes a feature map of shape [B,C+1,H,W].

    Required: Make sure that batch size is > 1 AND B % g == 0
    """
    def __init__(self, group_size=4, ep=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = ep

    def forward(self, x):
        B, C, H, W = x.shape
        g = min(self.group_size, B)
        
        std = x.view(g, -1, C, H, W)
        std = std - std.mean(dim=0, keepdim=True)
        std = torch.sqrt(std.pow(2).mean(dim=0) + self.eps)
        std = std.mean(dim=(1, 2, 3), keepdim=True)
        std = std.repeat(g, 1, H, W)  # [B, 1, H, W]

        return torch.cat([x, std], dim=1) # [B, 1 + C, H, W]


# =========================================================================

# GENERATOR ===============================================================

class Mapping(nn.Module):
    """
    MLP that takes the initial noise vector z (default 256 dim) and
    mapped to w (disentangled vector). Here we also inject the label embedding.
    """
    def __init__(self, z_dim=256, w_dim=256, num_layers=8, y_dim=64):
        super().__init__()
        layers, dim = [], (z_dim + y_dim)

        # Class embedding
        self.y_emb = nn.Embedding(2, y_dim)

        # Create MLP
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = w_dim
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, y):
        # Spherical normalization of z (as original paper states)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8) * math.sqrt(z.shape[1])
        y_v = self.y_emb(y).to(z.dtype)

        z_con = torch.cat([z, y_v], dim=1)

        return self.mlp(z_con)


