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
    mapped to w (disentangled vector). Here we also inject the label embedding
    so that w is 'style-aware'
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

class StyleAffine(nn.Module):
    """
    These are the 'A' blocks represented within the StyleGAN2 architesture diagram.
    Just a simple Linear layer to match each (de)mod/conv operations
    """
    def __init__(self, w_dim, in_c):
        super().__init__()
        self.fc = nn.Linear(w_dim, in_c)
    
    def forward(self, w):
        return self.fc(w)

class ModulatedConv2d(nn.Module):
    """
    This class implements both the style application (Modulation) and 
    the artifact-prevention step (Demodulation) before performing the convolution. 
    Many implementations allow to turn off 'demod' thus we add this functionality as well.
    """
    def __init__(self, in_c, out_c, k=3, demod=True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k # Kernal Size
        self.demod = demod
        self.eps = 1e-8

        # He/Kaiming Initialization
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k) / math.sqrt(in_c * k * k))
    
    def forward(self, x, style):
        B, C, H, W = x.shape

        # ! The following is inspired from ChatGPT:
        # ChatGPT 5 (2025-10-11, 12:49 PM) 
        # Prompt: What is an efficient way to perform mod/demod for Conditional StyleGAN2?
        #         I'm using the He/Kaiming Initialization for stability. 
        #         Here is my code as reference [Above code] and explain please.
        # Response: I will provide a full breakdown of the ModulatedConv2d class, explaning the role 
        #           of each line in implementing the Modulation and Demodulation steps, 
        #           and the final Grouped Convolution Trick for efficiency
        w = self.weight[None]
        s = style.view(B, 1, self.in_c, 1, 1)
        w = w * (s + 1.0)
        
        if self.demod:
            d = torch.rsqrt((w ** 2).sum(dim=[2,3,4]) + self.eps).view(B, self.out_ch, 1, 1, 1)
            w = w * d

        x = x.view(1, B*C, H, W)
        w = w.view(B*self.out_ch, self.in_ch, self.k, self.k)
        y = F.conv2d(x, w, padding=self.k//2, groups=B)

        return y.view(B, self.out_ch, H, W)


class NoiseInjection(nn.Module):
    """
    This class implements the noise injection behaviour within the StyleGAN architecture
    """
    def __init__(self, ch):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        
        return x + self.weight * noise



