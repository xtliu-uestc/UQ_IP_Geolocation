import torch.nn
from lib.utils import *
import argparse
import numpy as np
import random
from lib.model import *
import copy
import pandas as pd


parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=2022, help='manual seed')
parser.add_argument('--model_name', type=str, default='EBGeo')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--harved_epoch', type=int, default=5) 
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=10)
parser.add_argument('--load_epoch', type=int, default=100)

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")

opt = parser.parse_args()
print("Learning rate: ", opt.lr)
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
train_data = np.load("./datasets/{}/Clustering_s2022_lm70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s2022_lm70_test.npz".format(opt.dataset),
                    allow_pickle=True)
train_data, test_data = train_data["data"], test_data["data"]
print("data loaded.")

if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)

    losses = [np.inf]
    checkpoint = torch.load(f"asset/model/EBGeo_{opt.dataset}_epoch_{opt.load_epoch}.pth")
    print(f"Load model asset/model/{opt.dataset}_{opt.load_epoch}.pth")
    model = eval("EBGeo")(opt.dim_in)
    model.load_state_dict(checkpoint['model_state_dict'])

    if cuda:
        model.cuda()
    # test
    total_mse, total_mae, test_num, total_nll = 0, 0, 0, 0
    dislist = []
    model.eval()
    distance_all = []  
    macs_list = []
    params_list = []

    y_true_list = []
    y_pred_list = []
    uncertainties_list = []

    for i in range(len(test_data)):
        lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                                                                           test_data[i][
                                                                               "tg_X"], test_data[i]["tg_Y"], \
                                                                           test_data[i][
                                                                               "lm_delay"], test_data[i]["tg_delay"], \
                                                                           test_data[i]["y_max"], test_data[i]["y_min"]           

        lm_X_tensor, lm_Y_tensor = torch.tensor(lm_X, dtype=torch.float32).to(device), torch.tensor(lm_Y, dtype=torch.float32).to(device)
        tg_X_tensor, tg_Y_tensor = torch.tensor(tg_X, dtype=torch.float32).to(device), torch.tensor(tg_Y, dtype=torch.float32).to(device)
        x_feature = model.feature_net(lm_X_tensor, lm_Y_tensor)  #(N2,64)
        num_samples = 2  #M
        num_dims = tg_X.shape[0]  #N2
        # 沿着第一个维度计算均值，返回 (2,)
        # # Predict the most likely y
        y_mean, y_variance = predict_y(model, x_feature, lm_Y_tensor, num_dims, num_samples, tg_X_tensor)

        distance = dis_loss(tg_Y_tensor, y_mean, y_max, y_min)

        for i in range(len(distance.cpu().detach().numpy())):
                dislist.append(distance.cpu().detach().numpy()[i])
                distance_all.append(distance.cpu().detach().numpy()[i])
                
        nll_loss = gaussian_nll(y_mean, y_variance, tg_Y_tensor)
        test_num += len(tg_Y)
        total_mse += (distance * distance).sum()
        total_mae += distance.sum()
        total_nll += nll_loss.sum()
        y_true_list.extend(tg_Y)
        y_pred_list.extend(y_mean.cpu().detach().numpy())
        uncertainties_list.extend(y_variance.cpu().detach().numpy())
    
    total_mse = total_mse / test_num
    total_mae = total_mae / test_num
    total_nll = total_nll / test_num     
    print("test: mse: {:.4f}  mae: {:.4f} nll: {:.4f}".format(total_mse, total_mae, total_nll))
    
    dislist_sorted = sorted(dislist)
    print('test median: {:.4f}'.format(dislist_sorted[int(len(dislist_sorted) / 2)]))

    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)
    uncertainties_array = np.array(uncertainties_list)

    ause_trapezoidal, ause_summation = ause(y_true_array, y_pred_array, uncertainties_array)
    print(f'AUSE (Trapezoidal): {ause_trapezoidal:.4f}, AUSE (Summation): {ause_summation:.4f}')
           
    ce = calibration_error(y_true_array, y_pred_array)
    print(f'ce: {ce:.4f}') 