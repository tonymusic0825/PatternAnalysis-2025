import torch
import torch.nn.functional as F
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_c=1, channels=(64, 128, 256, 256), n_res=2):
        super().__init__()

        self.layers = []

        self.layers.append(nn.Conv2d(in_c, channels[0], kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(len(channels) - 1):
            self.layers.append(DownBlock(channels[i], channels[i + 1]))
        
        self.layers.append(DownBlock(channels[-1], channels[-1]))
        self.encoder = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.encoder(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, out_c=1, channels=(256, 256, 128, 64)):
        super().__init__()

        layers = []

        # Upsample stages (reverse of encoder)
        for i in range(len(channels) - 1):
            layers.append(UpBlock(channels[i], channels[i + 1]))

        layers.append(nn.ConvTranspose2d(channels[-1], out_c, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    enc = Encoder()
    dec = Decoder()
    x = torch.randn(16, 1, 256, 128)
    z = enc(x)
    print("Output shape (Encoder):", z.shape)

    x = dec(z)
    print("Output shape (Decoder):", x.shape)