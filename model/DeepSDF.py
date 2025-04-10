import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDF(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_hidden_layers, skip_layers=[3, 7]):
        super(DeepSDF, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.skip_layers = skip_layers

        # Define the input layer
        self.input_layer = nn.Linear(latent_dim + 3, hidden_dim)

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if i in skip_layers:
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim + latent_dim + 3, hidden_dim),
                    nn.ReLU()
                ))
            else:
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, latent_code, points):
        n_models, n_samples = points.shape[0:2]
        latent_code = latent_code.view(n_models, 1, self.latent_dim).repeat(1, n_samples, 1).view(-1, self.latent_dim)
        points = points.view(-1, 3)
        # Concatenate the latent code and points
        x = torch.cat([latent_code, points], dim=-1)

        # Pass through the input layer
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Pass through the hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if i in self.skip_layers:
                x = torch.cat([latent_code, points, x], dim=-1)
            x = layer(x)
            x = self.dropout(x)

        # Pass through the output layer
        sdf_value = self.output_layer(x)
        sdf_value = torch.tanh(sdf_value)
        return sdf_value
