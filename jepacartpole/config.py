from dataclasses import dataclass

@dataclass
class ConfigVAE:
    """Configuration for VAE encoder training (defaults follow World Models)."""
    latent_dim: int = 32
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
    z_dim: int = 32
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
@dataclass
class ConfigFullJEPA:
    """Configuration for the full JEPA (encoder + predictor trained jointly)."""
    latent_dim: int = 32
    image_channels: int = 3
    context_len: int = 2  # number of past latents fed to the predictor
    hidden_dim: int = 128  # predictor MLP width
    action_dim: int = 1
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    ema_momentum: float = 0.99  # target encoder momentum at the start (0 = no EMA, target = encoder)
    ema_momentum_end: float = 0.99  # momentum at the end (cosine schedule); 1.0 (I-JEPA) freezes the target too early here
    pred_weight: float = 25.0  # VICReg invariance weight, here the prediction loss
    var_weight: float = 25.0  # VICReg variance weight (0 disables)
    cov_weight: float = 1.0  # VICReg covariance weight (0 disables)
    var_gamma: float = 1.0  # target std per regularised dimension
    expander_dim: int = 256  # VICReg expander width; 0 = variance/covariance applied directly on z (fails here)
    val_fraction: float = 0.1  # fraction of episodes held out for validation

@dataclass
class ConfigLatentDecoder:
    """Configuration for the decoder trained on frozen JEPA latents (qualitative only)."""
    batch_size: int = 100
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
