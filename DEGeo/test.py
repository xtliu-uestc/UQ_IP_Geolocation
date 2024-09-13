# -*- coding: utf-8 -*-
"""
    load checkpoint and then test
"""
import torch.nn
from lib.utils import *
import argparse
import numpy as np
import random
from lib.model_0811 import DeepEnsemble, MLPModel, MLPDropoutModel
import copy
import pandas as pd

parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=2022, help='manual seed')
parser.add_argument('--model_name', type=str, default='MLPGeo')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--max_epoch', type=int, default=2000)
parser.add_argument('--ensemble_size', type=int, default=5)
parser.add_argument('--harved_epoch', type=int, default=5)
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=200)
parser.add_argument('--load_epoch', type=int, default=5)
parser.add_argument('--lambda1', type=float, default=7e-3)

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")

opt = parser.parse_args()
print("Learning rate: ", opt.learning_rate)
print("Dataset: ", opt.dataset)

if opt.seed:
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''load data'''
train_data = np.load("./datasets/{}/Clustering_s1234_lm70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s1234_lm70_test.npz".format(opt.dataset),
                    allow_pickle=True)
train_data, test_data = train_data["data"], test_data["data"]
print("data loaded.")

if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)
    losses = [np.inf]

    save_type = "mse_nll_0811"
    checkpoint = torch.load(f"asset/model/{opt.model_name}_{opt.dataset}_{opt.load_epoch}.pth")
    print(f"Load model asset/model/{opt.model_name}_{opt.dataset}_{opt.load_epoch}.pth")
    
    if opt.model_name == 'MLPGeo':
        model_class = MLPGeo
    else:
        raise ValueError(f"Unknown model_name: {opt.model_name}")

    sizes = [opt.dim_in + 3, (opt.dim_in + 3) * 2, opt.dim_in + 3]
    ensemble = DeepEnsemble(model_class, opt.ensemble_size, opt, sizes, output_dim=2).to(device)
    state_dicts = checkpoint['model_state_dict'] # Load the state dictionary for each model
    for i, model in enumerate(ensemble.models):
        model.load_state_dict(state_dicts[i])
    if torch.cuda.is_available():
        ensemble.cuda()

    # test
    total_ause_trapezoidal, total_ause_trapezoidal, test_num, total_loss, total_mse, total_mae, total_nll = 0, 0, 0, 0, 0, 0, 0
    dislist = []
    model.eval()
    distance_all = []  
    y_true_list = []
    y_pred_list = []
    uncertainties_list = []

    with torch.no_grad():
        for i in range(len(test_data)):
            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                                                                           test_data[i][
                                                                               "tg_X"], test_data[i]["tg_Y"], \
                                                                           test_data[i][
                                                                               "lm_delay"], test_data[i]["tg_delay"], \
                                                                           test_data[i]["y_max"], test_data[i]["y_min"]

            lm_X_tensor = torch.tensor(lm_X).float().to(device)
            lm_Y_tensor = torch.tensor(lm_Y).float().to(device)
            lm_delay_tensor = torch.tensor(lm_delay).float().to(device)
            tg_X_tensor = torch.tensor(tg_X).float().to(device)
            tg_delay_tensor = torch.tensor(tg_delay).float().to(device)
            tg_Y_tensor = torch.tensor(tg_Y).float().to(device)
            y_max_tensor = torch.tensor(y_max).float().to(device)
            y_min_tensor = torch.tensor(y_min).float().to(device)
            # Get predictions and uncertainties from ensemble
            y_pred, uncertainty = ensemble(lm_X_tensor, lm_Y_tensor, lm_delay_tensor, tg_delay_tensor, tg_X_tensor, tg_Y_tensor)
            y_pred_tensor = torch.tensor(y_pred).float().to(device)
            # Calculate the  loss
            nll_loss = ensemble.compute_loss(lm_X_tensor, lm_Y_tensor, lm_delay_tensor, tg_delay_tensor, tg_X_tensor, tg_Y_tensor, opt)
            # Calculate the distance
            distance = dis_loss(tg_Y_tensor, y_pred_tensor, y_max_tensor, y_min_tensor)
            
            for i in range(len(distance.cpu().detach().numpy())):
                dislist.append(distance.cpu().detach().numpy()[i])
                distance_all.append(distance.cpu().detach().numpy()[i])
              
            total_mse += (distance * distance).sum()
            total_mae += distance.sum()
            test_num += len(tg_Y)
            total_nll += nll_loss.sum()
                        
            # Collect data for AUSE calculation
            y_true_list.extend(tg_Y_tensor.cpu().numpy())
            y_pred_list.extend(y_pred.cpu().detach().numpy())
            uncertainties_list.extend(uncertainty.cpu().detach().numpy())
        
        total_nll = total_nll / test_num
        total_mse = total_mse / test_num
        total_mae = total_mae / test_num
        print("test: mse: {:.4f}  mae: {:.4f}".format(total_mse, total_mae))
        dislist_sorted = sorted(dislist)
        print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])
        print("test: nll: {:.4f}".format(total_nll))

        # Calculate AUSE
        y_true_array = np.array(y_true_list)
        y_pred_array = np.array(y_pred_list)
        uncertainties_array = np.array(uncertainties_list)
        ause_trapezoidal, ause_summation = ause(y_true_array, y_pred_array, uncertainties_array)
        print(f'AUSE (Trapezoidal): {ause_trapezoidal:.4f}, AUSE (Summation): {ause_summation:.4f}')
        # Calculate CE    
        ce = calibration_error(y_true_array, y_pred_array)
        print(f'ce: {ce:.4f}')  