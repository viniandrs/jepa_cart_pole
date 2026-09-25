import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JEPAPredictor(nn.Module):
    """
    Simple MLP predictor that maps (z_t, action_t) -> z_{t+1}
    """
    def __init__(self, z_dim, action_dim, hidden_dim=128):
        super(JEPAPredictor, self).__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim
        
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