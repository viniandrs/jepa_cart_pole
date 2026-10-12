import torch
import torch.nn as nn
from jepacartpole.config import ConfigJEPA


class JEPAPredictor(nn.Module):
    """
    MLP predictor that maps (z_{t-k+1}, ..., z_t, action_t) -> z_{t+1},
    where k = config.context_len. With k = 1 this is the plain (z_t, a_t)
    predictor; k > 1 lets it infer velocities from consecutive latents.
    """
    def __init__(self, config: ConfigJEPA):
        super(JEPAPredictor, self).__init__()

        self.z_dim = config.z_dim
        self.action_dim = config.action_dim
        self.context_len = config.context_len
        hidden_dim = config.hidden_dim

        # Input size: flattened latent context + action
        input_dim = self.context_len * self.z_dim + self.action_dim

        # Simple 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.z_dim)
        )

    def forward(self, z_ctx, action):
        """
        Args:
            z_ctx: (B, context_len, z_dim) latent context, oldest first
            action: (B, action_dim) action taken at the last context step

        Returns:
            (B, z_dim) predicted next latent
        """
        x = torch.cat([z_ctx.flatten(1), action], dim=-1)
        return self.net(x)
