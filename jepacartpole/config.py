from dataclasses import dataclass, field

@dataclass
class ConfigVAE:
    latent_dim: int = 8
    hidden_size_enc: int = 32
    hidden_size_dec: int = 8
    image_channels: int = 1
    batch_size: int = 256
    epochs: int = 4

@dataclass
class ConfigJEPA:
    pass