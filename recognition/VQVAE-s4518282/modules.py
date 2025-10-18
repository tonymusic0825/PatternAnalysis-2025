import torch
import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    def __init__(self, channels, n_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(x)

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
    def __init__(self, in_c=1, channels=(64, 128, 256), n_res=2):
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
    def __init__(self, out_c=1, channels=(256, 128, 64), n_res=2):
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
    
class Quantizer(nn.Module):
    """ 
    REF: https://github.com/explainingai-code/VQVAE-Pytorch/blob/main/model/quantizer.py 
    """
    def __init__(self, embed_num=512, embed_dim=256, beta=0.25):
        super().__init__()
        self.beta = beta
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        
        # Create embedding
        self.embedding = nn.Embedding(embed_num, embed_dim)

    def forward(self, x):

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

class VQVAE(nn.Module):
    def __init__(self, in_c=1, out_c=1, channels=(64,128,256,256), embed_num=512, embed_dim=256, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_c=in_c, channels=channels)
        self.quantizer = Quantizer(embed_num=embed_num, embed_dim=embed_dim, beta=beta)
        self.decoder = Decoder(out_c=out_c, channels=channels[::-1])
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        pred = self.decoder(z_q)

        return pred, vq_loss

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