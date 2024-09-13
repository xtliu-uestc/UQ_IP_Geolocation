import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MLPGeo(nn.Module):
    def __init__(self, opt, sizes, output_dim):
        super(MLPGeo, self).__init__()
        self.input_size = sizes[0]  # Input layer size (33)
        self.hidden_sizes = sizes[1:]  # List of hidden layer sizes (66, 33)
        self.output_size = output_dim  # Output layer size (equals 2 for outputting mean and variance)

        # Define network layers using nn.Sequential
        layers = []
        current_size = self.input_size

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            # First hidden layer: from 33 input dimensions to 66 output;
            # second hidden layer: from 66 input dimensions to 33 output
            layers.append(nn.ReLU())  # Add activation function
            current_size = hidden_size

        # # Final output layer
        self.layers = nn.Sequential(*layers)

        # Linear layer for reducing the aggregated features from 33 dimensions to 2 dimensions
        self.dimension_reducer = nn.Linear(current_size, 2)  

        self.output_layer = nn.Linear(current_size, self.output_size * 2)  # Output layer set to output_size * 2 to predict mean and variance

    def forward(self, lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y):
        lm_X = torch.tensor(lm_X).to(device)
        lm_Y = torch.tensor(lm_Y).to(device)
        lm_delay = torch.tensor(lm_delay).to(device)
        tg_X = torch.tensor(tg_X).to(device)
        tg_Y = torch.tensor(tg_Y).to(device)
        lm_delay = lm_delay.unsqueeze(1)  # (N1, 1)
        lm_features = torch.cat([lm_X, lm_Y, lm_delay], dim=1)  # (N1, 33)
        encoded_features = self.layers(lm_features)  # Encoded features (N1, 33)
        # Aggregate features and reduce dimensions
        aggregated_feature = encoded_features.mean(dim=0).unsqueeze(0)  # (1, 33)
        reduced_feature = self.dimension_reducer(aggregated_feature)  # (1, 30)
        reduced_feature = reduced_feature.repeat(tg_X.size(0), 1)  # (N2, 30)

        if tg_delay.dim() == 1:
            tg_delay = tg_delay.unsqueeze(1)  # (N2, 1)
        tg_delay = tg_delay.mean(dim=1).unsqueeze(1)
        tg_features = torch.cat([reduced_feature, tg_X, tg_delay], dim=1)
        output = self.output_layer(tg_features)  #  (N2, output_size * 2)
        pred_mean, raw_var = torch.chunk(output, chunks=2, dim=1)
        pred_mean = pred_mean.view(-1, self.output_size)  # (N2, 2)
        pred_var = F.softplus(raw_var.view(-1, self.output_size))
        return pred_mean, pred_var
    
    def get_uncertainty(self, pred_var):
        uncertainty = pred_var.mean(dim=1)  # Average the uncertainty for each sample
        return uncertainty

    def gaussian_nll(self, pred_mean, pred_var, tg_Y):
        y_diff = tg_Y - pred_mean
        scaled_y_diff = y_diff / torch.sqrt(pred_var +1e-6)
        weights = 0.5
        nll = 0.5 * torch.mean(weights * (torch.log(pred_var) + scaled_y_diff ** 2)) + 0.5 * torch.log(torch.tensor(2 * math.pi, device=pred_mean.device))
        return nll

    def compute_loss(self, lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y):
        lm_X = lm_X.float()
        lm_Y = lm_Y.float()
        tg_X = tg_X.float()
        tg_Y = tg_Y.float()
        lm_delay = lm_delay.float()
        tg_delay = tg_delay.float()
        mean, var = self(lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y)
        nll = self.gaussian_nll(mean, var, tg_Y)
        return nll

class DEGeo(nn.Module):
    def __init__(self, model_class, num_models, opt, sizes, output_dim):
        super(DEGeo, self).__init__()
        self.models = nn.ModuleList([model_class(opt, sizes, output_dim) for _ in range(num_models)])

    def forward(self, lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y):
        means = []
        variances = []
        for model in self.models:
            mean, var = model(lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        variances = torch.stack(variances)
        mean_ensemble = means.mean(dim=0)
        variance_ensemble = variances.mean(dim=0) + means.var(dim=0)
        return mean_ensemble, variance_ensemble

    def compute_loss(self, lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y, opt):
        total_loss = 0
        for model in self.models:
            loss = model.compute_loss(lm_X, lm_Y, lm_delay, tg_delay, tg_X, tg_Y)
            total_loss += loss
        return total_loss / len(self.models)