from __future__ import print_function
from distutils.version import LooseVersion
import numpy as np
import torch
import math
import warnings
import torch.nn as nn
import random
import copy
import torch.nn as nn
warnings.filterwarnings(action='once')
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MaxMinLogRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min + 1e-12)

class MaxMinRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        # data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min + 1e-12)

class MaxMinLogScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data[data != 0] = -np.log(data[data != 0] + 1)
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data[data != 0] = (data[data != 0] - min) / (max - min + 1e-12)
        return data

    def inverse_transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data = data * (max - min) + min
        return np.exp(data)

class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        data_o = np.array(data)
        self.max = data_o.max()
        self.min = data_o.min()

    def transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / (max - min + 1e-12)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


def graph_normal(graphs, normal=2):
    if normal == 2:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)

    elif normal == 1:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = [1, 1], [0, 0]

    return graphs


def get_data_generator(opt, data_train, data_test, normal=2):
    # load data
    data_train = data_train[np.array([graph["exist"] for graph in data_train])]
    data_test = data_test[np.array([graph["exist"] for graph in data_test])]

    data_train, data_test = graph_normal(data_train, normal=normal), graph_normal(data_test, normal=normal)

    random.seed(opt.seed)
    random.shuffle(data_train)
    random.seed(opt.seed)
    random.shuffle(data_test)

    return data_train, data_test

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

## Calculates the PDF of a multivariate Gaussian distribution.
def gauss_density_centered(x, std):
    result = torch.exp(-0.5 * x**2) / (math.sqrt(2 * math.pi) * std)
    return result

def gmm_density_centered(x, std):
    std = std.to(x.device)
    if x.dim() == std.dim() - 1:
        x_T = x.t()
        x = x_T.unsqueeze(0).to(std.device) #(1,N2，2)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    # Compute the Gaussian density centered at zero
    result = gauss_density_centered(x, std)
    return result.squeeze(0)  #Shape: (1, N2, 2)

def sample_gmm_centered(std, num_samples):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components
    std = std.view(1, num_dims, num_components)   #(1,N2,2)
    if torch.isnan(std).any() or torch.isinf(std).any():
        print("Error: std contains NaN or Inf values")
        print(std)
        exit(1)
    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()
    # Check for NaN or Inf in sampled std
    if torch.isnan(std_samp).any() or torch.isinf(std_samp).any():
        print("Error: std_samp contains NaN or Inf values")
        print(std_samp)
        exit(1)
    std_samp = std_samp.to(device) 
    x_centered = std_samp * torch.randn(num_samples, num_dims).to(device)
    if torch.isnan(x_centered).any() or torch.isinf(x_centered).any():
        print("Error: x_centered contains NaN or Inf values")
        print(x_centered)
        exit(1)
    prob_dens = gmm_density_centered(x_centered, std)   # Compute probability density of samples
    return x_centered.t(), prob_dens    #Return transposed samples (N2, 2) and probability density


def dis_loss(y, y_pred, max, min):
    if isinstance(max, list):
        max = torch.tensor(max)  
    if isinstance(min, list):
        min = torch.tensor(min)
    y = y.to(device)
    y_pred = y_pred.to(device)
    y[:, 0] = y[:, 0] * (max[0] - min[0])
    y[:, 1] = y[:, 1] * (max[1] - min[1])
    y_pred[:, 0] = y_pred[:, 0] * (max[0] - min[0])
    y_pred[:, 1] = y_pred[:, 1] * (max[1] - min[1])
    distance = torch.sqrt((((y - y_pred) * 100) * ((y - y_pred) * 100)).sum(dim=1))
    return distance

def gaussian_nll(pred_mean, pred_var, tg_Y_tensor):
        pred_mean, pred_var, tg_Y_tensor =  pred_mean.to(device), pred_var.to(device), tg_Y_tensor.to(device)
        y_diff = tg_Y_tensor - pred_mean
        nll = 0.5 * torch.mean((torch.log(pred_var) + y_diff ** 2)) + 0.5 * torch.log(torch.tensor(2 * math.pi, device=pred_mean.device))
        return nll

def save_cpt(model, optimizer, epoch, savepath):
    """
    Saved model checkpoint and restores model and optimizer states.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, savepath)
    print(f"Checkpoint saved at {savepath}")

def load_cpt(model, optimizer, checkpoint_path):
    """
    Loads a model checkpoint and restores model and optimizer states.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return None

def ause(y_true, y_pred, uncertainties, alpha_range=np.linspace(0, 1, 100)):
    errors = np.abs(y_true - y_pred) # Calculate prediction errors
    mae_S = np.mean(errors) # Compute the MAE for the entire set
    ause_values = [] # Initialize list to store AUSE values
    for alpha in alpha_range:
        # -----------  split based on prediction errors -----------
        n = len(errors)
        sorted_error_indices = np.argsort(errors)[::-1]  ## Sort by error in descending order
        sorted_errors = errors[sorted_error_indices]
        n_alpha = int(alpha * n) # Determine split point
        S_error_U_alpha = sorted_errors[:n_alpha] if n_alpha > 0 else np.array([])  #Top alpha% errors
        S_error_V_alpha = sorted_errors[n_alpha:] if n_alpha < n else np.array([])  # Remaining errors
        # ----------- Split based on uncertainties -----------
        sorted_uncertainty_indices = np.argsort(uncertainties)[::-1]  # Sort by uncertainty in descending order
        sorted_uncertainties = uncertainties[sorted_uncertainty_indices]
        S_uncertainty_U_alpha = sorted_uncertainties[:n_alpha] if n_alpha > 0 else np.array([]) # Top alpha% uncertainties
        S_uncertainty_V_alpha = sorted_uncertainties[n_alpha:] if n_alpha < n else np.array([])  # Remaining uncertainties
        # ----------- Compute AUSE-----------
        mae_S_U_alpha = np.mean(S_uncertainty_V_alpha) if len(S_uncertainty_V_alpha) > 0 else 0 ## MAE for remaining uncertainties
        mae_S_V_alpha = np.mean(S_error_V_alpha) if len(S_error_V_alpha) > 0 else 0 ## MAE for remaining errors
        # Normalize MAE values
        normalized_mae_S_U_alpha = mae_S_U_alpha / (mae_S + 1e-10)
        normalized_mae_S_V_alpha = mae_S_V_alpha / (mae_S + 1e-10)
        # Compute absolute difference for AUSE value
        ause_value = abs(normalized_mae_S_U_alpha - normalized_mae_S_V_alpha)
        ause_values.append(ause_value)
    #Calculate AUSE using the trapezoidal rule
    AUSE_trapezoidal = np.trapz(ause_values, alpha_range)
    #  Calculate AUSE using simple summation as an alternative
    AUSE_summation = np.sum(ause_values) / len(ause_values)
    return AUSE_trapezoidal, AUSE_summation

def calibration_error(y_true, y_pred, thresholds = np.linspace(0, 1, num=100)):   
    # Commented out normalization lines for rescaling y_true and y_pred
    N = len(y_true)
    p_hat = []
    # ----------- Calculate empirical frequency  -----------
    for p in thresholds:
        count = np.sum(y_pred < p) # Count predictions below threshold
        p_hat.append(count / N) # Compute empirical probability
    p_hat = np.array(p_hat)
    calibration_error = 0
    epsilon = 1e-10   # Small constant to prevent division by zero
    # -----------  Compute calibration error -----------
    for j, p in enumerate(thresholds):
        if p_hat[j] > 0:   # Ensure p_hat[j] is not zero
            w_j = 1.0 / (N * (p_hat[j] + epsilon))   # Weight factor
            calibration_error += w_j * (p - p_hat[j]) ** 2  # Add weighted squared difference
        else:
            # If p_hat[j] is zero, skip this term or handle it in a custom way
            continue
    return calibration_error

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def calculate_tg_Y_variance(data):
    """
    Calculate the variance of tg_Y for each dimension.

    Parameters:
    data: List of dictionaries containing tg_Y.

    Returns:
    tg_Y_var: Variance for each dimension (Tensor, shape [2]).
    """
    tg_Y_list = []

    # Iterate through the dataset and extract tg_Y
    for i in range(len(data)):
        tg_Y = data[i]["tg_Y"]
        tg_Y_list.append(tg_Y)
    # Concatenate all tg_Y samples into one NumPy array or PyTorch tensor
    tg_Y_tensor = torch.tensor(np.concatenate(tg_Y_list, axis=0), dtype=torch.float32)
    # Calculate variance for each dimension
    tg_Y_var = torch.var(tg_Y_tensor, dim=0, unbiased=False)
    return tg_Y_var


#Testing Phrase
def predict_y(model, x_feature, lm_Y_tensor, num_dims, num_samples, tg_X_tensor, num_initializations=20, num_iterations=10, step_size=5e-5):
    """
    Predict the most likely target value y_star using gradient ascent.

    Parameters:
    - model: Trained neural network model
    - x_feature: Input features
    - lm_Y_tensor: Landmark target tensor
    - num_dims: Dimensionality of the target variable
    - num_samples: Number of samples
    - tg_X_tensor: Target tensor for features
    - num_initializations: Number of random initializations
    - num_iterations: Number of gradient ascent iterations
    - step_size: Step size for gradient ascent

    Returns:
    - y_mean: Predicted mean target values
    - y_variance: Predicted variance of the target values
    """
    x_feature = x_feature.to(device)
    tg_X_tensor = tg_X_tensor.to(device)
    lm_Y_tensor = lm_Y_tensor.to(device)  # [N, 2]
    # Compute mean and std for the landmark tensor
    mean_lm_Y_per_dim = torch.mean(lm_Y_tensor, dim=0)
    std_lm_Y_per_dim = torch.std(lm_Y_tensor, dim=0)  #
    std_lm_Y_per_dim[torch.isnan(std_lm_Y_per_dim)] = 0.0  #  Handle NaN values
    std_lm_Y_per_dim = torch.clamp(std_lm_Y_per_dim, min=1e-6)  # Avoid zero std
    best_y_samples = []
    best_scores = []
    best_y = None
    best_score = -float('inf')
    
    for _ in range(num_initializations):
        # Randomly initialize y_samples with noise
        y_samples = torch.normal(mean=mean_lm_Y_per_dim, std=std_lm_Y_per_dim).unsqueeze(0).expand(tg_X_tensor.size(0), num_samples).clone()
        noise = torch.randn_like(y_samples) * 0.05  # Add noise to samples
        y_samples = y_samples + noise
        y_samples = y_samples.to(device)
        y_samples.requires_grad_(True)

        for _ in range(num_iterations):
            # Compute scores using model
            scores = model.predictor_net(x_feature, tg_X_tensor, y_samples)
            # Backpropagate scores to get gradients
            scores.sum().backward(retain_graph=True)

            # Update y_samples using gradient ascent
            with torch.no_grad():
                y_samples += step_size * y_samples.grad
                y_samples.grad.zero_()  # Clear gradients

            y_samples.requires_grad_(True)  # Enable gradients again
            # Update best_y if current score is better
            if scores.max().item() > best_score:
                best_score = scores.max().item()
                best_y = y_samples.clone().detach()
   
        best_y_samples.append(y_samples.detach().cpu())
        best_scores.append(scores.detach().cpu())
    # Stack best samples
    best_y_samples = torch.stack(best_y_samples, dim=0)
    best_scores = torch.stack(best_scores, dim=0)
    # Compute mean and variance of y_samples
    y_mean = best_y_samples.mean(dim=0)
    y_variance = best_y_samples.var(dim=0, unbiased=True)
    return y_mean, y_variance