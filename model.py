import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),   # 256 -> 128
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU() # 32 -> 16
        )

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # 16 -> 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # 32 -> 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 64 -> 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()  # 128 -> 256
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)                         # -> [B, 256, 16, 16]
        x = x.view(batch_size, -1)                  # -> [B, 256*16*16]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)         # -> [B, latent_dim]
        x = self.fc_decode(z).view(batch_size, 256, 16, 16)
        x = self.decoder(x)                         # -> [B, 3, 256, 256]
        return x
