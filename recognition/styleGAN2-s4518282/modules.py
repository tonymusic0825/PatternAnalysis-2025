from modules_helper import *
import torch
import torch.nn.functional as F
import math

class Discriminator(nn.Module):
    """
    Conditional (projection) discriminator for StyleGAN2
    """
    def __init__(self, n_classes=2, img_channels=1):
        super().__init__()

        self.orig = conv1x1(img_channels, 512)
        self.down1 = ResDown(512, 512) # 256 -> 128 (These are image dimensions, H/W)
        self.down2 = ResDown(512, 512) # 128 -> 64
        self.down3 = ResDown(512, 256) # 64 -> 32
        self.down4 = ResDown(256, 128) # 32 -> 16
        self.down5 = ResDown(128, 64)  # 16 -> 8
        self.down6 = ResDown(64, 64)   # 8 -> 4

        # Fully connected layers + minibatch std dev
        self.mbsd = MBStdDev(group_size=4)
        self.conv4 = conv3x3(64 + 1, 64)
        self.fc = nn.Linear(64 * 4 * 4, 64)
        self.out = nn.Linear(64, 1) 

        # small, stable init for the last layer
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.y_emb = nn.Embedding(n_classes, 64)
    
    def forward(self, x, y):
        y = y.long()
        h = self.orig(x)   # [B,512,256,256]
        print(f"This is after the original 1x1 conv {h.shape}")    
        h = self.down1(h)  # -> [B, 512, 128,128]
        h = self.down2(h)  # -> [B, 512, 64, 64]
        h = self.down3(h)  # -> [B, 256, 32, 32]
        h = self.down4(h)  # -> [B, 128, 16, 16]
        h = self.down5(h)  # -> [B, 64,  8,  8]
        h = self.down6(h)  # -> [B, 64,  4,  4]
        h = self.mbsd(h)
        h = leaky_relu(self.conv4(h))
        h = h.view(h.size(0), -1)    
        h = leaky_relu(self.fc(h))   

        rf = self.out(h)      
        proj = (h * self.y_emb(y)).sum(dim=1, keepdim=True) 
        logit = rf + proj
        return logit, h


class Generator(nn.Module):
    def __init__(self, z_dim=256, w_dim=256, y_emb_dim=64):
        super().__init__()
        self.mapping = Mapping(z_dim, w_dim, 8, y_emb_dim)
        self.const   = nn.Parameter(torch.randn(1, 256, 4, 4)) # learned 4×4

        self.block1 = GenBlock(256, 256, w_dim, is_first=True) # 4 -> 4
        self.block2 = GenBlock(256, 128, w_dim)                # 4 -> 8
        self.block3 = GenBlock(128, 128, w_dim)                # 8 -> 16
        self.block4 = GenBlock(128, 128, w_dim)                # 16 -> 32
        self.block5 = GenBlock(128, 64, w_dim)                 # 32 -> 64
        self.block6 = GenBlock(64, 64, w_dim)                  # 64 -> 128
        self.block7 = GenBlock(64, 64, w_dim)                  # 128 -> 256
        self.toRGB = ToRGB(64, w_dim, out_c=1)

    def forward(self, z, y, return_w=False):
        """
        return_w is for t-sne plot
        """
        w = self.mapping(z, y)

        x = self.const.repeat(z.shape[0], 1, 1, 1) # [B, 256, 4, 4]
        x = self.block1(x, w) 
        x = self.block2(x, w)
        x = self.block3(x, w)
        x = self.block4(x, w)
        x = self.block5(x, w) 
        x = self.block6(x, w)
        x = self.block7(x, w)
        img = self.toRGB(x, w)
        img = torch.tanh(img)  # [-1,1] to match dataset norm

        if return_w:
            return (img, w)

        return img

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