"""
Heavily Inspired from the following sources:
https://blog.paperspace.com/implementation-stylegan2-from-scratch/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EqualizedWeight(nn.Module):
    """
    Implements the Equalized Learning Rate mechanism for weight initialization.

    The weight tensor is scaled by a constant 'c' derived from the 'fan-in'
    of the layer, which helps maintain signal variance during training.

    REF: https://blog.paperspace.com/implementation-stylegan2-from-scratch/ 
    """
    def __init__(self, shape):
        super().__init__()

        # Calculate the scaling constant 'c' based on the layer's fan-in.
        # This is equivalent to 1 / sqrt(fan_in) * gain (where gain is 1.0 here).
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        
        # Initialize the weight tensor with a standard normal distribution.
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

class EqualizedLinear(nn.Module):
    """
    A fully connected (Linear) layer using the EqualizedWeight mechanism.

    This module replaces nn.Linear to apply the StyleGAN-specific
    Weight Equalization technique.

    REF: https://blog.paperspace.com/implementation-stylegan2-from-scratch/
    """
    def __init__(self, in_c, out_c, bias = 0.0):
        super().__init__()
        
        # Instantiate the equalized weight tensor
        self.weight = EqualizedWeight([out_c, in_c])
        
        # Define the bias parameter
        self.bias = nn.Parameter(torch.ones(out_c) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedConv2d(nn.Module):
    """
    2D Convolutional Layer that incorporates Weight equalisation

    This module replaces nn.Conv2d to apply the StyleGAN-specific
    Weight Equalization technique.

    REF: https://blog.paperspace.com/implementation-stylegan2-from-scratch/
    """
    def __init__(self, in_c, out_c, k, padding=0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_c, in_c, k, k])
        self.bias = nn.Parameter(torch.ones(out_c))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class NoiseInjection(nn.Module):
    """
    Injects learnable, scaled noise into the feature map.
    """
    def __init__(self):
        super().__init__()
        self.scale_noise = nn.Parameter(torch.zeros(1)) 

    def forward(self, x, noise):
        if noise is not None:
            # Scale the noise using the learnable parameter
            # Self.scale_noise[None, :, None, None] broadcasts the [1] tensor to [1, 1, 1, 1]
            x = x + self.scale_noise[None, :, None, None] * noise
        return x

class Conv2dWeightModulate(nn.Module):
    """
    Style modulation + optional demodulation, then grouped conv.

    REF: https://blog.paperspace.com/implementation-stylegan2-from-scratch/ 
    """
    def __init__(self, in_c, out_c, k, demodulate=True, eps=1e-8):
        super().__init__()
        self.out_c = out_c
        self.demod = demodulate
        self.padding = (k - 1) // 2
        self.weight = EqualizedWeight([out_c, in_c, k, k])
        self.eps = eps
    
    def forward(self, x, s):
        B, C, H, W = x.shape
        w = self.weight()[None, ...] # [1, out_c, in_c, k, k]
        s = s[:, None, :, None, None]       # [B, 1, in_c, 1, 1]
        w = w * s                           # [B, out_c, in_c, k, k]

        if self.demod:
            d = torch.rsqrt((w ** 2).sum((2,3,4), keepdim=True) + self.eps)  # [B, out_c]
            w = w * d 

        x = x.reshape(1, B * x.size(1), H, W)
        w = w.reshape(B * self.out_c, -1, w.size(-2), w.size(-1))
        y = F.conv2d(x, w, padding=self.padding, groups=B)
        return y.reshape(B, self.out_c, H, W)

class ToRGB(nn.Module):
    """
    This class implements the final output of the generator.
    Takes high channel inputs and maps it to lower (gray-scale) channel output.
    (In the case of ADNI this is just mapping the channel to 1)

    REF: https://blog.paperspace.com/implementation-stylegan2-from-scratch/ 
    """
    def __init__(self, w_dim, in_c, out_c=1):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, in_c, bias=1.0)
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.conv = Conv2dWeightModulate(in_c, out_c, k=1, demodulate=False)

    def forward(self, x, w):
        s = self.to_style(w)
        x = self.conv(x, s)
        return F.leaky_relu(x + self.bias[None, :, None, None], 0.2, inplace=True)

class StyleBlock(nn.Module):
    """
    Implements one 'StyleBlock' which includes:

    One mod/demod conv with style affine + learnable noise + bias + LeakyReLU.
    """
    def __init__(self, w_dim, in_c, out_c):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, in_c, bias=1.0)        
        self.conv = Conv2dWeightModulate(in_c, out_c, k=3)   
        self.noise_inj = NoiseInjection() 
        self.bias = nn.Parameter(torch.zeros(out_c))

    def forward(self, x, w, noise):  
        # compute style vector
        s = self.to_style(w)  

        # Mod / Demod and inject noise                 
        x = self.conv(x, s)   
        x = self.noise_inj(x, noise)

        return F.leaky_relu(x + self.bias[None, :, None, None], 0.2, inplace=True)

class GenBlock(nn.Module):
    """
    This is the core block for the generation model. Build with two StyleBlocks
    """
    def __init__(self, w_dim, in_c, out_c):
        super().__init__()
        self.style1 = StyleBlock(w_dim, in_c, out_c)
        self.style2 = StyleBlock(w_dim, out_c, out_c)
        self.to_rgb = ToRGB(w_dim, out_c, out_c=1)   # grayscale

    def forward(self, x, w, noise_pair):
        # noise_pair: (n1, n2) each shape [B,1,H,W] or None
        x = self.style1(x, w, noise_pair[0])
        x = self.style2(x, w, noise_pair[1])
        rgb = self.to_rgb(x, w)
        return x, rgb

class DiscBlock(nn.Module):
    """
    This is the core block for the discriminative model.
    Implements two 3x3 convolutions with res connections
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res = nn.Sequential(
            nn.AvgPool2d(2),
            EqualizedConv2d(in_c, out_c, k=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_c, in_c, k=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedConv2d(in_c, out_c, k=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        res = self.res(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + res) * self.scale

class MBStdDev(nn.Module):
    """
    StyleGAN Trick: Append a channel with per-group std dev to help Discriminator. 
    Helps detect similar fakes.

    1)  Calculate Standard Deviation

    2)  Calculate Mean Standard Deviation: 

    3)  Create a New Feature Map

    4)  Concatenation
    """
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        g = min(self.group_size, B)

        # Add fall back just in case
        if B % g != 0:
            g = 1

        y = x.view(g, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + self.eps)
        y = y.mean(dim=(1,2,3), keepdim=True)  
        y = y.repeat(g, 1, H, W) # [B, 1, H, W]      
           
        return torch.cat([x, y], dim=1) # [B, 1 + C, H, W]


































































































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

# # DISCRIMINATOR ===========================================================
# class ResDown(nn.Module):
#     """
#     A Residual downsampling block:
#         1) Main: Conv + Activation + Conv + Avg Pool
#         2) Skip: Conv + Avg Pool
#         3) Output = main + skip
#     """

#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv1 = conv3x3(in_c, in_c)
#         self.conv2 = conv3x3(in_c, out_c)
#         self.skip = conv1x1(in_c, out_c)
#         self.avg = nn.AvgPool2d(2)
#         self.variance_scale = 1.0 / math.sqrt(2) 
    
#     def forward(self, x):
#         skip = self.avg(self.skip(x))
#         main = leaky_relu(self.conv1(x))
#         main = leaky_relu(self.conv2(main))
#         main = self.avg(main)

#         return (main + skip) * self.variance_scale 

# class MBStdDev(nn.Module):
#     """
#     StyleGAN Trick: Append a channel with per-group std dev to help Discriminator. 
#     Helps detect similar fakes.

#     1)  Calculate Standard Deviation: 
#         For a given feature map (tensor) x of shape [B,C,H,W] the layer 
#         computes the standard deviation for each feature element (pixel location and channel) 
#         across all N images in the minibatch.

#         This results in a tensor of shape [1,C,H,W]. The batch dimension is now 1 because 
#         the statistic is taken across the batch.

#     2)  Calculate Mean Standard Deviation: 
#         Averages the standard deviation tensor over all channels (C) 
#         and spatial locations (H,W) to get a single, scalar value.

#     3)  Create a New Feature Map: 
#         This single scalar value is replicated to create a new feature map of shape [B,1,H,W], 
#         where every element in this new feature map has the same standard deviation value.

#     4)  Concatenation: This new, single-channel feature map is concatenated with the original 
#         feature map x along the channel dimension. 
#         The discriminator now processes a feature map of shape [B,C+1,H,W].

#     Required: Make sure that batch size is > 1 AND B % g == 0
#     """
#     def __init__(self, group_size=4, ep=1e-8):
#         super().__init__()
#         self.group_size = group_size
#         self.eps = ep

#     def forward(self, x):
#         B, C, H, W = x.shape
#         g = min(self.group_size, B)

#         # Add fall back just in case
#         if B % g != 0:
#             g = 1
        
#         std = x.view(g, -1, C, H, W)
#         std = std - std.mean(dim=0, keepdim=True)
#         std = torch.sqrt(std.pow(2).mean(dim=0) + self.eps)
#         std = std.mean(dim=(1, 2, 3), keepdim=True)
#         std = std.repeat(g, 1, H, W)  # [B, 1, H, W]

#         return torch.cat([x, std], dim=1) # [B, 1 + C, H, W]


# # =========================================================================

# # GENERATOR ===============================================================

# class Mapping(nn.Module):
#     """
#     MLP that takes the initial noise vector z (default 256 dim) and
#     mapped to w (disentangled vector). Here we also inject the label embedding
#     so that w is 'style-aware'
#     """
#     def __init__(self, z_dim=256, w_dim=256, num_layers=8, y_dim=64):
#         super().__init__()
#         layers, dim = [], (z_dim + y_dim)

#         # Class embedding
#         self.y_emb = nn.Embedding(2, y_dim)

#         # Create MLP
#         for _ in range(num_layers):
#             layers.append(nn.Linear(dim, w_dim))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             dim = w_dim
        
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, z, y):
#         # Spherical normalization of z (as original paper states)
#         z = z / (z.norm(dim=1, keepdim=True) + 1e-8) * math.sqrt(z.shape[1])
#         y_v = self.y_emb(y).to(z.dtype)

#         z_con = torch.cat([z, y_v], dim=1)

#         return self.mlp(z_con)

# class StyleAffine(nn.Module):
#     """
#     These are the 'A' blocks represented within the StyleGAN2 architesture diagram.
#     Just a simple Linear layer to match each (de)mod/conv operations
#     """
#     def __init__(self, w_dim, in_c):
#         super().__init__()
#         self.fc = nn.Linear(w_dim, in_c)
    
#     def forward(self, w):
#         return self.fc(w)

# class ModulatedConv2d(nn.Module):
#     """
#     This class implements both the style application (Modulation) and 
#     the artifact-prevention step (Demodulation) before performing the convolution. 
#     Many implementations allow to turn off 'demod' thus we add this functionality as well.
#     """
#     def __init__(self, in_c, out_c, k=3, demod=True):
#         super().__init__()
#         self.in_c = in_c
#         self.out_c = out_c
#         self.k = k # Kernal Size
#         self.demod = demod
#         self.eps = 1e-8

#         # He/Kaiming Initialization
#         self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k) / math.sqrt(in_c * k * k))
    
#     def forward(self, x, style):
#         B, C, H, W = x.shape

#         # ! The following is inspired from ChatGPT:
#         # ChatGPT 5 (2025-10-11, 12:49 PM) 
#         # Prompt: What is an efficient way to perform mod/demod for Conditional StyleGAN2?
#         #         I'm using the He/Kaiming Initialization for stability. 
#         #         Here is my code as reference [Above code] and explain please.
#         # Response: I will provide a full breakdown of the ModulatedConv2d class, explaning the role 
#         #           of each line in implementing the Modulation and Demodulation steps, 
#         #           and the final Grouped Convolution Trick for efficiency
#         w = self.weight[None]
#         s = style.view(B, 1, self.in_c, 1, 1)
#         w = w * (s + 1.0)
        
#         if self.demod:
#             d = torch.rsqrt((w ** 2).sum(dim=[2,3,4]) + self.eps).view(B, self.out_c, 1, 1, 1)
#             w = w * d

#         x = x.view(1, B*C, H, W)
#         w = w.view(B*self.out_c, self.in_c, self.k, self.k)
#         y = F.conv2d(x, w, padding=self.k//2, groups=B)

#         return y.view(B, self.out_c, H, W)


# class NoiseInjection(nn.Module):
#     """
#     This class implements the noise injection behaviour within the StyleGAN architecture
#     """
#     def __init__(self, ch):
#         super().__init__()
#         self.weight = nn.Parameter(torch.zeros(1, ch, 1, 1))

#     def forward(self, x):
#         noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        
#         return x + self.weight * noise

# class ToRGB(nn.Module):
#     """
#     This class implements the final output of the generator.
#     Takes high channel inputs and maps it to lower (gray-scale) channel output.
#     """
#     def __init__(self, in_c, w_dim, out_c=1):
#         super().__init__()
#         self.affine = StyleAffine(w_dim, in_c)
#         self.conv   = ModulatedConv2d(in_c, out_c, k=1, demod=False)

#     def forward(self, x, w):
#         s = self.affine(w)
#         return self.conv(x, s)

# def upsample(x):
#     """ 
#     StyleGAN2 uses interpolation instead of deconvolution
#     """
#     return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

# class GenBlock(nn.Module):
#     """
#     This is the core block for the generation model combining all:
#     Affine, (de)mod convolution, toRGB and noise injections
#     """
#     def __init__(self, in_c, out_c, w_dim, is_first=False):
#         super().__init__()
        
#         self.is_first = is_first
#         self.affine1 = StyleAffine(w_dim, in_c)
#         self.mod1 = ModulatedConv2d(in_c, out_c, k=3)
#         self.noise1 = NoiseInjection(out_c)
#         self.affine2 = StyleAffine(w_dim, out_c)

#         self.mod2 = ModulatedConv2d(out_c, out_c, k=3)
#         self.noise2 = NoiseInjection(out_c)

#     def forward(self, x, w):
#         if not self.is_first: 
#             x = upsample(x)

#         s1 = self.affine1(w)
#         x = self.mod1(x, s1)
#         x = self.noise1(x)
#         x = leaky_relu(x)

#         s2 = self.affine2(w)
#         x = self.mod2(x, s2)
#         x = self.noise2(x)
#         x = leaky_relu(x)

#         return x

