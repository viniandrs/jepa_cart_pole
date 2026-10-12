import torch
import torch.nn as nn

from ..utils import get_checkpoint_dir, get_device, load_checkpoint
from ..vae.encoder import make_conv_decoder


class LatentDecoder(nn.Module):
    """
    World Models deconv decoder (same as the VAE decoder) mapping a frozen
    JEPA latent back to a 64x64 frame. Only a qualitative tool: it is trained
    after the JEPA and its gradients never reach the encoder.
    """
    def __init__(self, latent_dim=32, image_channels=3):
        super(LatentDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.decoder_input = nn.Linear(latent_dim, 1024)
        self.decoder = make_conv_decoder(image_channels)

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, z):
        return self.decode(z)


def load_latent_decoder(path=None, device=None):
    """
    Load a LatentDecoder checkpoint saved by train_latent_decoder.

    Returns:
        tuple: (decoder in eval mode, checkpoint dict)
    """
    device = device or get_device()
    path = path or get_checkpoint_dir() / 'latent_decoder.pth'
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    decoder = LatentDecoder(ckpt['latent_dim'], ckpt['image_channels'])
    checkpoint = load_checkpoint(decoder, path, device=device)
    return decoder.eval(), checkpoint
