"""
Heavily Inspired from the following sources:
https://blog.paperspace.com/implementation-stylegan2-from-scratch/
A Style-Based Generator Architecture for Generative Adversarial Networks https://arxiv.org/pdf/1812.04948
Analyzing and Improving the Image Quality of StyleGAN https://arxiv.org/pdf/1912.04958
StyleGan2 official pytorch implementation https://github.com/NVlabs/stylegan2-ada-pytorch
"""


from modules_helper import *
import torch
import torch.nn.functional as F
import math

class Generator(nn.Module):
    """
    StyleGAN2 Generator 
    """
    def __init__(
        self,
        log_res=8,   # 2^8 = 256 which is what we want
        w_dim=256,          
        n_features=32,
        max_features=256,
        z_dim=256,
        n_classes=2,
        y_dim=64,
        out_c=1,
    ):
        super().__init__()
        self.log_res = log_res
        self.w_dim = w_dim
        self.out_c = out_c

        # channel schedule (same pattern as test.py)
        feats = [min(max_features, n_features * (2 ** i)) for i in range(log_res - 2, -1, -1)]
        self.n_blocks = len(feats)

        self.mapping = Mapping(z_dim, w_dim, num_layers=8, n_classes=n_classes, y_dim=y_dim)

        self.initial_constant = nn.Parameter(torch.randn(1, feats[0], 4, 4))
        self.style = StyleBlock(w_dim, feats[0], feats[0])
        self.to_rgb = ToRGB(w_dim, feats[0], out_c=out_c)

        blocks = [GenBlock(w_dim, feats[i - 1], feats[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
    
    def get_noise(self, B, device):
        """
        Generates noise for each generator block
        """
        noise = []
        res = 4

        for i in range(self.log_res):
            n1 = None if i == 0 else torch.randn(B, 1, res, res, device=device)
            n2 = torch.randn(B, 1, res, res, device=device)
            noise.append((n1, n2))
            res *= 2

        return noise

    def forward(self, z, y, return_w):
        B, device = z.size(0), z.device # For noise
        w_single = self.mapping(z, y) 
        noise = self.make_noise(B, device)
        w = w_single[None, :, :].expand(self.n_blocks, -1, -1)  # [n_blocks,B,W_DIM]

        x = self.initial_constant.expand(B, -1, -1, -1)
        x = self.style(x, w[0], noise[0][1])   
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)

class Discriminator(nn.Module):
    """
    test.py-style D backbone (EqualizedConv2d + residual blocks + MBStdDev),
    with your *projection* term to keep it conditional.
    """
    def __init__(self, log_res=8, n_features=64, max_features=256, n_classes=2):
        super().__init__()

        feats = [min(max_features, n_features * (2 ** i)) for i in range(log_res - 1)]
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, k=1, padding=0),
            nn.LeakyReLU(0.2, True),
        )

        n_blocks = len(feats) - 1
        self.blocks = nn.Sequential(*[DiscBlock(feats[i], feats[i + 1]) for i in range(n_blocks)])

        self.mbsd = MBStdDev()
        final_c = feats[-1] + 1
        self.conv = EqualizedConv2d(final_c, final_c, k=3)
        self.final = EqualizedLinear(2 * 2 * final_c, 1)

        # produce a compact feature h, then rf -> logit
        # self.fc = EqualizedLinear(2 * 2 * final_c, 64)
        # self.out = EqualizedLinear(64, 1)

        # projection term (same dimensionality as h)
        # self.y_emb = nn.Embedding(n_classes, 64)

    def forward(self, x, y):
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.mbsd(x)

        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.out(x)























# class Discriminator(nn.Module):
#     """
#     Conditional (projection) discriminator for StyleGAN2
#     """
#     def __init__(self, n_classes=2, img_channels=1):
#         super().__init__()

#         self.orig = conv1x1(img_channels, 512)
#         self.down1 = ResDown(512, 512) # 256 -> 128 (These are image dimensions, H/W)
#         self.down2 = ResDown(512, 512) # 128 -> 64
#         self.down3 = ResDown(512, 256) # 64 -> 32
#         self.down4 = ResDown(256, 128) # 32 -> 16
#         self.down5 = ResDown(128, 64)  # 16 -> 8
#         self.down6 = ResDown(64, 64)   # 8 -> 4

#         # Fully connected layers + minibatch std dev
#         self.mbsd = MBStdDev(group_size=4)
#         self.conv4 = conv3x3(64 + 1, 64)
#         self.fc = nn.Linear(64 * 4 * 4, 64)
#         self.out = nn.Linear(64, 1) 

#         # small, stable init for the last layer
#         nn.init.zeros_(self.out.weight)
#         nn.init.zeros_(self.out.bias)
#         self.y_emb = nn.Embedding(n_classes, 64)
    
#     def forward(self, x, y):
#         y = y.long()
#         h = self.orig(x)   # [B,512,256,256]
#         h = self.down1(h)  # -> [B, 512, 128,128]
#         h = self.down2(h)  # -> [B, 512, 64, 64]
#         h = self.down3(h)  # -> [B, 256, 32, 32]
#         h = self.down4(h)  # -> [B, 128, 16, 16]
#         h = self.down5(h)  # -> [B, 64,  8,  8]
#         h = self.down6(h)  # -> [B, 64,  4,  4]
#         h = self.mbsd(h)
#         h = leaky_relu(self.conv4(h))
#         h = h.view(h.size(0), -1)    
#         h = leaky_relu(self.fc(h))   

#         rf = self.out(h)      
#         proj = (h * self.y_emb(y)).sum(dim=1, keepdim=True) 
#         logit = rf + proj
#         return logit, h


# class Generator(nn.Module):
#     def __init__(self, z_dim=256, w_dim=256, y_emb_dim=64):
#         super().__init__()
#         self.mapping = Mapping(z_dim, w_dim, 8, y_emb_dim)
#         self.const   = nn.Parameter(torch.randn(1, 256, 4, 4)) # learned 4×4

#         self.block1 = GenBlock(256, 256, w_dim, is_first=True) # 4 -> 4
#         self.block2 = GenBlock(256, 128, w_dim)                # 4 -> 8
#         self.block3 = GenBlock(128, 128, w_dim)                # 8 -> 16
#         self.block4 = GenBlock(128, 128, w_dim)                # 16 -> 32
#         self.block5 = GenBlock(128, 64, w_dim)                 # 32 -> 64
#         self.block6 = GenBlock(64, 64, w_dim)                  # 64 -> 128
#         self.block7 = GenBlock(64, 64, w_dim)                  # 128 -> 256
#         self.toRGB = ToRGB(64, w_dim, out_c=1)

#     def forward(self, z, y, return_w=False):
#         """
#         return_w is for t-sne plot
#         """
#         w = self.mapping(z, y)

#         x = self.const.repeat(z.shape[0], 1, 1, 1) # [B, 256, 4, 4]
#         x = self.block1(x, w) 
#         x = self.block2(x, w)
#         x = self.block3(x, w)
#         x = self.block4(x, w)
#         x = self.block5(x, w) 
#         x = self.block6(x, w)
#         x = self.block7(x, w)
#         img = self.toRGB(x, w)
#         img = torch.tanh(img)  # [-1,1] to match dataset norm

#         if return_w:
#             return (img, w)

#         return img

# * Sanity check Testing OPTIONAL
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sanity] using device = {device}")

    B = 7
    z_dim = 256
    img_ch = 1
    n_classes = 2

    G = Generator(z_dim=z_dim, w_dim=256, y_emb_dim=64).to(device).train()
    D = Discriminator(n_classes=n_classes, img_channels=img_ch).to(device).train()
    
    x_real = torch.randn(B, img_ch, 256, 256, device=device)
    y = torch.randint(0, n_classes, (B,), device=device)

    d_real, h_real = D(x_real, y)
    print("[sanity] D(real) logit:", tuple(d_real.shape), "  h:", tuple(h_real.shape))

    z = torch.randn(B, z_dim, device=device)
    out = G(z, y, return_w=True)
    x_fake, w = out
    print("[sanity] G(z,y) img:", tuple(x_fake.shape), "  w:", tuple(w.shape))

    d_fake, _ = D(x_fake.detach(), y)
    print("[sanity] D(fake) logit:", tuple(d_fake.shape))