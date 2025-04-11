import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class VoxelEncoder(nn.Module):
    def __init__(self, voxel_grid_size, output_dim):
        super(VoxelEncoder, self).__init__()
        self.voxel_grid_size = voxel_grid_size
        self.output_dim = output_dim

        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),  # 32x32x32 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32x32x32 -> 16x16x16
            
            nn.Conv3d(32, 128, kernel_size=3, padding=1),  # 16x16x16 -> 16x16x16
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 16x16x16 -> 8x8x8
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),  # 8x8x8 -> 8x8x8
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8x8x8 -> 4x4x4
            
            nn.Conv3d(256, self.output_dim, kernel_size=4, stride=1, padding=0),  # 4x4x4 -> 1x1x1
            nn.Flatten()
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        # x shape: (batch_size, 1, voxel_grid_size, voxel_grid_size, voxel_grid_size)
        x = self.encoder(x)
        return x  # shape: (batch_size, output_dim)

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, voxel_latent_dim, n_layers=4, hidden_dim=512, skip_frequency=2):
        super(ODEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.voxel_latent_dim = voxel_latent_dim
        self.skip_frequency = skip_frequency
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the ODE function layers
        self.input_layer = nn.Linear(latent_dim + voxel_latent_dim + 1, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(n_layers):
            if i % skip_frequency == 1:
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim + voxel_latent_dim + 1, hidden_dim),
                    nn.ReLU()
                ))
            else:
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))
        self.output_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, t, latent, voxel_latent):
        # Concatenate the latent code and voxel latent
        batch_size = latent.size(0)
        x = torch.cat([latent, voxel_latent, t.unsqueeze(0).repeat(batch_size).unsqueeze(1)], dim=-1)
        
        # Pass through the input layer
        x = self.input_layer(x)
        x = F.relu(x)

        # Pass through the hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if i % self.skip_frequency == 1:
                x = torch.cat([x, voxel_latent, t.unsqueeze(0).repeat(batch_size).unsqueeze(1)], dim=-1)
            x = layer(x)

        # Pass through the output layer
        output = self.output_layer(x)
        return output

class VCCNF(nn.Module):
    def __init__(self, latent_dim, voxel_grid_size, voxel_latent_dim, n_layers=4, hidden_dim=512, skip_frequency=2):
        super(VCCNF, self).__init__()
        self.latent_dim = latent_dim
        self.voxel_grid_size = voxel_grid_size
        self.voxel_latent_dim = voxel_latent_dim

        # Define the encoder
        self.encoder = VoxelEncoder(voxel_grid_size, voxel_latent_dim)

        # Define the ODE function
        self.ode_func = ODEFunc(latent_dim, voxel_latent_dim, n_layers=n_layers, hidden_dim=hidden_dim, skip_frequency=skip_frequency)
        
        self.integrated_time = torch.tensor([0, 1]).float()  # Time points for integration
        
    def forward(self, voxel_grid, latent_code):
        # Encode the voxel grid
        voxel_latent = self.encoder(voxel_grid)
        
        def odeint_func(t, latent_code):
            # Reshape the latent code to match the expected input shape
            latent_code = latent_code.view(-1, self.latent_dim)
            return self.ode_func(t, latent_code, voxel_latent)
        
        # Integrate the ODE
        latent_code = odeint(odeint_func, latent_code, self.integrated_time, method="rk4")
        
        return latent_code
