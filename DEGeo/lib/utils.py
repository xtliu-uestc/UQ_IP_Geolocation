import numpy as np
import torch
import random
import warnings
import torch.nn as nn

warnings.filterwarnings(action='once')

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


def get_data_generator(opt, data_train, data_test, normal=2):
    data_train = data_train[np.array([graph["exist"] for graph in data_train])]
    data_test = data_test[np.array([graph["exist"] for graph in data_test])]

    data_train, data_test = graph_normal(data_train, normal=normal), graph_normal(data_test, normal=normal)

    random.seed(opt.seed)
    random.shuffle(data_train)
    random.seed(opt.seed)
    random.shuffle(data_test)

    return data_train, data_test

def graph_normal(graphs, normal=2):
    if normal == 2:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)
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
            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)
    elif normal == 1:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)
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

def dis_loss(y, y_pred, max, min):
    y[:, 0] = y[:, 0] * (max[0] - min[0])
    y[:, 1] = y[:, 1] * (max[1] - min[1])
    y_pred[:, 0] = y_pred[:, 0] * (max[0] - min[0])
    y_pred[:, 1] = y_pred[:, 1] * (max[1] - min[1])
    distance = torch.sqrt((((y - y_pred) * 100) * ((y - y_pred) * 100)).sum(dim=1))
    return distance

def ensemble_mean_var(ensemble, xs):
    en_mean = 0
    en_var = 0
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            mean, var = model(xs)
        en_mean += mean
        en_var += var + mean ** 2
    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean ** 2
    return en_mean, en_var

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

def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)

def save_cpt(ensemble, optimizer, epoch, save_path):
    # Switch all models in the ensemble to evaluation mode
    for model in ensemble.models:
        model.eval()
    state_dicts = {f'model_{i}': model.state_dict() for i, model in enumerate(ensemble.models)}
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': [model.state_dict() for model in ensemble.models],
            'optimizer_state_dict': optimizer.state_dict()
        },
        save_path
    )

def load_cpt(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dicts = checkpoint['model_state_dict']
    # # Load the state dictionary for each model
    for i, model in enumerate(model.models):
        model.load_state_dict(state_dicts[f'model_{i}'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def add_loss(nll_loss, mse_loss, coeffi):
    # our loss function
    lossr = coeffi * (mse_loss.mean())
    loss = nll_loss + lossr
    return loss