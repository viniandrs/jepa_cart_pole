import torch
import torch.nn as nn
import torch.nn.functional as F
from jepacartpole.config import ConfigVAE

class VAE(nn.Module):
    def __init__(self, config: ConfigVAE):
        super(VAE, self).__init__()

        hidden_size_enc = config.hidden_size_enc
        hidden_size_dec = config.hidden_size_dec
        latent_dim = config.latent_dim
        image_channels = config.image_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: n_channels x 400 x 600
            nn.Conv2d(image_channels, hidden_size_enc, kernel_size=4, stride=2, padding=1), # output: (32, 200, 300)
            nn.ReLU(),
            
            nn.Conv2d(hidden_size_enc, 2 * hidden_size_enc, kernel_size=4, stride=2, padding=1), # output: (64, 100, 150)
            nn.ReLU(),
            
            nn.Conv2d(2 * hidden_size_enc, 4 * hidden_size_enc, kernel_size=4, stride=2, padding=1), # output: (128, 50, 75)
            nn.ReLU(),
            
            nn.Conv2d(4 * hidden_size_enc, 8 * hidden_size_enc, kernel_size=4, stride=2, padding=1), # output: (256, 25, 37)
            nn.ReLU(),
            
            nn.Flatten() # output: (256 * 25 * 37)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(8 * hidden_size_enc * 25 * 37, latent_dim)
        self.fc_logvar = nn.Linear(8 * hidden_size_enc * 25 * 37, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 64 * 25 * 37)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (8 * hidden_size_dec, 25, 37)), # output: (64, 25, 37)
            
            nn.ConvTranspose2d(8 * hidden_size_dec, 4 * hidden_size_dec, kernel_size=4, stride=2, padding=1, output_padding=(0, 1)), # output: (32, 50, 75)
            nn.Dropout2d(0.3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(4 * hidden_size_dec, 2 * hidden_size_dec, kernel_size=4, stride=2, padding=1), # output: (16, 100, 150)
            nn.Dropout2d(0.3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(2 * hidden_size_dec, hidden_size_dec, kernel_size=4, stride=2, padding=1), # output: (8, 200, 300)
            nn.Dropout2d(0.3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_size_dec, image_channels, kernel_size=4, stride=2, padding=1), # output: (1, 400, 600)
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
    
    def loss(self, recon, x, mu, logvar, beta=2.0):
        batch_size = x.size(0)

        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum') / batch_size
        
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / batch_size

         
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss