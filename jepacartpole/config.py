from dataclasses import dataclass

@dataclass
class ConfigVAE:
    """Configuration for VAE encoder training (defaults follow World Models)."""
    latent_dim: int = 4  # v2; v1 used 32 (see registry/v1-vae-z32)
    beta: float = 1.0
    kl_tolerance: float = 0.5  # KL floor per latent dim (World Models)
    image_channels: int = 3
    image_size: int = 64
    batch_size: int = 100
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6

@dataclass
class ConfigJEPA:
    """Configuration for JEPA predictor training."""
    z_dim: int = 4  # must match ConfigVAE.latent_dim
    action_dim: int = 1
    context_len: int = 2  # number of past latents fed to the predictor
    hidden_dim: int = 128
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    sample_latents: bool = True  # train on z ~ N(mu, sigma^2) (World Models); False uses mu
    weight_decay: float = 0.0
    val_fraction: float = 0.1  # fraction of episodes held out for validation

@dataclass
class ConfigTransferLearning:
    """Configuration for transfer learning encoder."""
    backbone: str = 'resnet18'  # Options: 'resnet18', 'resnet34', 'resnet50', etc.
    pretrained: bool = True
    freeze_backbone: bool = True
    latent_dim: int = 16