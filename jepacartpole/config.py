from dataclasses import dataclass, field

@dataclass
class ConfigVAE:
    latent_dim: int = 16
    hidden_size_enc: int = 16
    hidden_size_dec: int = 4
    beta: float = 2
    image_channels: int = 3
    batch_size: int = 128
    epochs: int = 4

@dataclass
class ConfigJEPA:
    pass