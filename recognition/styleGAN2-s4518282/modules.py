from modules_helper import *

class Disciminator(nn.Module):
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
        h = self.orig(x)   # [B,512,256,256]
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




        