import torch
import torch.nn as nn
from ..config import ConfigJEPA

class JEPAPredictor(nn.Module):
    """
    Simple MLP predictor that maps (z_t, action_t) -> z_{t+1}
    """
    def __init__(self,  config: ConfigJEPA):
        super(JEPAPredictor, self).__init__()
        
        z_dim = config.latent_dim
        hidden_dim = config.hidden_dim
        action_dim = 1
        
        # Input size: latent + action
        input_dim = z_dim + action_dim
        
        # Simple 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
    
    def forward(self, z, action):
        # Concatenate z and action
        x = torch.cat([z, action], dim=-1)
        return self.net(x)