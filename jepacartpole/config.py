from dataclasses import dataclass

@dataclass
class ConfigVAE:
    """Configuration for VAE encoder training."""
    latent_dim: int = 16
    hidden_size_enc: int = 16
    hidden_size_dec: int = 4
    beta: float = 2
    image_channels: int = 3
    batch_size: int = 128
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6

@dataclass
class ConfigJEPA:
    """Configuration for JEPA predictor training."""
    z_dim: int = 16
    action_dim: int = 1
    hidden_dim: int = 128
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3

@dataclass
class ConfigTransferLearning:
    """Configuration for transfer learning encoder."""
    backbone: str = 'resnet18'  # Options: 'resnet18', 'resnet34', 'resnet50', etc.
    pretrained: bool = True
    freeze_backbone: bool = True
    latent_dim: int = 16