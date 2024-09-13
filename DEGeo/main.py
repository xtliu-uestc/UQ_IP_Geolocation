import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import random
import pandas as pd
from lib.utils import *
from lib.model import DEGeo, MLPGeo

parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=2022, help='manual seed')
parser.add_argument('--model_name', type=str, default='MLPGeo')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--max_epoch', type=int, default=2000)
parser.add_argument('--lambda1', type=float, default=7e-3)
parser.add_argument('--ensemble_size', type=int, default=5)
parser.add_argument('--harved_epoch', type=int, default=5)
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=50)

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
train_data = np.load(f"./datasets/{opt.dataset}/Clustering_s{opt.seed}_lm70_train.npz", allow_pickle=True)["data"]
test_data = np.load(f"./datasets/{opt.dataset}/Clustering_s{opt.seed}_lm70_test.npz", allow_pickle=True)["data"]
train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)
print("data loaded.")

if __name__ == '__main__':
    
    log_path = f"asset/log"
    os.makedirs(log_path, exist_ok=True)  # Automatically create directories
    f = open(f"asset/log/{opt.dataset}.txt", 'a')
    f.write(f"*********{opt.dataset}*********\n")
    f.write("dim_in="+str(opt.dim_in)+", ")
    f.write("early_stop_epoch="+str(opt.early_stop_epoch)+", ")
    f.write("harved_epoch="+str(opt.harved_epoch)+", ")
    f.write("saved_epoch="+str(opt.saved_epoch)+", ")
    f.write("learning_rate="+str(opt.lr)+", ")
    f.write("model_name="+opt.model_name+", ")
    f.write("seed="+str(opt.seed)+",")
    f.write("\n")
    f.close()

    if opt.model_name == 'MLPGeo':
        model_class = MLPGeo
    else:
        raise ValueError(f"Unknown model_name: {opt.model_name}")
    lr = opt.lr
    sizes = [opt.dim_in + 3, (opt.dim_in + 3) * 2, opt.dim_in + 3]
    ensemble = DEGeo(model_class, opt.ensemble_size, opt, sizes, output_dim=2).to(device)
    optimizer = optim.Adam(ensemble.parameters(), lr=opt.lr)

    ensemble.train()
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0
    for epoch in range(opt.max_epoch):
        print("epoch {}.    ".format(epoch))
        total_loss, total_mse, total_mae, train_num, total_nll = 0, 0, 0, 0, 0
        for i in range(len(train_data)):
            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = train_data[i]["lm_X"], \
                                                                       train_data[i]["lm_Y"], \
                                                                       train_data[i]["tg_X"], \
                                                                       train_data[i]["tg_Y"], \
                                                                       train_data[i]["lm_delay"], \
                                                                       train_data[i]["tg_delay"], \
                                                                       train_data[i]["y_max"], \
                                                                       train_data[i]["y_min"]
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
            optimizer.zero_grad()
            # compute loss
            y_pred_tensor = torch.tensor(y_pred).float().to(device)
            nll_loss = ensemble.compute_loss(lm_X_tensor, lm_Y_tensor, lm_delay_tensor, tg_delay_tensor, tg_X_tensor, tg_Y_tensor, opt)
            
            # compute distance
            distance = dis_loss(tg_Y_tensor, y_pred_tensor, y_max_tensor, y_min_tensor)
            mse_loss = distance * distance  # mse loss
            loss = add_loss(nll_loss, mse_loss, coeffi=opt.lambda1)

            loss.backward()
            optimizer.step() 

            mse_loss = mse_loss.sum()
            total_mse += mse_loss
            total_mae += distance.sum()
            total_nll += nll_loss
            train_num += len(tg_Y)

        total_mse = total_mse / train_num
        total_mae = total_mae / train_num
        total_nll = total_nll / train_num
        print("train: nll: {:.4f}".format(total_nll))
        print("train: mse: {:.4f} mae: {:.4f} nll: {:.4f}".format(total_mse, total_mae, total_nll))

        #test
        total_mse, total_mae, total_ause_trapezoidal, total_ause_summation, test_num, total_ce = 0, 0, 0, 0, 0, 0 
        dislist = []

        distance_all = []
        y_true_list = []
        y_pred_list = []
        uncertainties_list = []

        y_max_list = []
        y_min_list = []

        ensemble.eval()
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

                # Get predictions and uncertainties from ensemble
                y_pred, uncertainty = ensemble(lm_X_tensor, lm_Y_tensor, lm_delay_tensor, tg_delay_tensor, tg_X_tensor, tg_Y_tensor)
                # Calculate the  loss
                nll_loss = ensemble.compute_loss(lm_X_tensor, lm_Y_tensor, lm_delay_tensor, tg_delay_tensor, tg_X_tensor, tg_Y_tensor, opt)

                y_pred_tensor = torch.tensor(y_pred).float().to(device)
                # distance
                distance = dis_loss(tg_Y_tensor, y_pred_tensor, y_max_tensor, y_min_tensor)
                for i in range(len(distance.cpu().detach().numpy())):
                    dislist.append(distance.cpu().detach().numpy()[i])
                    distance_all.append(distance.cpu().detach().numpy()[i])   
                
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()
                test_num += len(tg_Y)
                total_nll += nll_loss.sum()

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

            y_true_array = np.array(y_true_list)
            y_pred_array = np.array(y_pred_list)
            uncertainties_array = np.array(uncertainties_list)
            # Calculate AUSE
            ause_trapezoidal, ause_summation = ause(y_true_array, y_pred_array, uncertainties_array)
            # Calculate CE 
            ce = calibration_error(y_true_array, y_pred_array)
            print(f'AUSE (Trapezoidal): {ause_trapezoidal:.4f}, AUSE (Summation): {ause_summation:.4f}')
            print(f'ce: {ce:.4f}') 

            model_dir = "asset/model"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # Save checkpoint for each 200 epoch
            if epoch > 0 and epoch % opt.saved_epoch == 0 and epoch < 2100:
                # Distinguish save path based on the script name
                savepath = f"asset/model/{opt.model_name}_{opt.dataset}_{epoch}.pth"
                save_cpt(ensemble, optimizer, epoch, savepath)
                print("Save checkpoint!")

                f = open(f"asset/log/{opt.model_name}_{opt.dataset}.txt", 'a')
                f.write(f"\n*********epoch={epoch}*********\n")
                f.write("test: nll: {:.3f}".format(total_loss))
                
                f.write(f"\tAUSE (Trapezoidal): {ause_trapezoidal:.4f}, AUSE (Summation): {ause_summation:.4f}\n")
                f.close()
                
            batch_metric = total_nll.cpu().numpy()
            if batch_metric <= np.min(losses):
                no_better_epoch = 0 
                early_stop_epoch = 0
                print("Better nll in epoch {}: {:.4f}".format(epoch, batch_metric))
            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = 0

            losses.append(batch_metric)
                
            # Halve the learning rate
            if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
                lr /= 2
                print("learning rate changes to {}!\n".format(lr))
                optimizer = optim.Adam(ensemble.parameters(), lr=opt.lr)
                no_better_epoch = 0

            if early_stop_epoch == opt.early_stop_epoch:
                break