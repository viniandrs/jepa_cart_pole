import torch
import torch.nn as nn
from jepacartpole.config import ConfigVAE

class VAE(nn.Module):
    """
    Convolutional VAE for 64x64 frames, following the ConvVAE of World Models
    (Ha & Schmidhuber, 2018). All convolutions use no padding, so spatial sizes
    are 64 -> 31 -> 14 -> 6 -> 2 in the encoder and 1 -> 5 -> 13 -> 30 -> 64
    in the decoder.
    """
    def __init__(self, config: ConfigVAE):
        super(VAE, self).__init__()

        latent_dim = config.latent_dim
        image_channels = config.image_channels

        self.beta = config.beta
        self.kl_tolerance = config.kl_tolerance
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Input: n_channels x 64 x 64
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), # output: (32, 31, 31)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2), # output: (64, 14, 14)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2), # output: (128, 6, 6)
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2), # output: (256, 2, 2)
            nn.ReLU(),

            nn.Flatten() # output: (1024)
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 1024)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (1024, 1, 1)), # output: (1024, 1, 1)

            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), # output: (128, 5, 5)
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), # output: (64, 13, 13)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), # output: (32, 30, 30)
            nn.ReLU(),

            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2), # output: (C, 64, 64)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon, x, mu, logvar):
        """
        World Models VAE loss: sum-of-squares reconstruction error plus KL
        divergence floored at kl_tolerance * latent_dim per sample.

        Returns:
            tuple: (total_loss, recon_loss, kl_loss), each averaged over the
                   batch. kl_loss is the raw KL (before the floor) for logging.
        """
        recon_loss = (recon - x).pow(2).sum(dim=(1, 2, 3)).mean()

        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_floor = self.kl_tolerance * self.latent_dim
        kl_clamped = torch.clamp(kl_per_sample, min=kl_floor).mean()

        total_loss = recon_loss + self.beta * kl_clamped

        return total_loss, recon_loss, kl_per_sample.mean()
