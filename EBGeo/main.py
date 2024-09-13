import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from lib.utils import *
from lib.model import *
import torch.distributions
import math
import pickle

parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=2022, help='manual seed')
parser.add_argument('--model_name', type=str, default='EBGeo')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--harved_epoch', type=int, default=5)
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=10)

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[30, 51], help="30 if Shanghai / 51 else")
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
train_data, test_data = get_data_generator(opt, train_data, test_data, normal=1)
print("data loaded.")
# Ensure that the directory for saving the model and the log directory exist
model_dir = "asset/model"
log_dir = "asset/log"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

model = EBGeo(opt.dim_in).cuda()  # Model initialization

model.apply(init_weights)
if cuda:
    model.cuda()

    '''initiate criteria and optimizer'''
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-5)  # weight_decay
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    if __name__ == '__main__':
        #train
        losses = [np.inf]
        no_better_epoch = 0
        early_stop_epoch = 0
        epoch_losses_train = []
        checkpoint_path = f"{model_dir}/EBGeo_{opt.dataset}_latest.pth"
        start_epoch = load_cpt(model, optimizer, checkpoint_path) if os.path.exists(checkpoint_path) else 0
        for epoch in range(opt.num_epochs):
            print ("###########################")
            print ("######## NEW EPOCH ########")
            print ("###########################")
            print ("epoch: %d/%d" % (epoch, opt.num_epochs))
            total_loss, total_mse, total_mae, train_num, total_nll = 0, 0, 0, 0, 0
            batch_losses = []
            model.train()
            for i in range(len(train_data)):
                lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = train_data[i]["lm_X"], \
                                                                       train_data[i]["lm_Y"], \
                                                                       train_data[i]["tg_X"], \
                                                                       train_data[i]["tg_Y"], \
                                                                       train_data[i]["lm_delay"], \
                                                                       train_data[i]["tg_delay"], \
                                                                       train_data[i]["y_max"], \
                                                                       train_data[i]["y_min"]   
                optimizer.zero_grad()

                #Convert numpy arrays to PyTorch tensors
                lm_X_tensor, lm_Y_tensor = torch.tensor(lm_X, dtype=torch.float32).to(device), torch.tensor(lm_Y, dtype=torch.float32).to(device)
                tg_X_tensor, tg_Y_tensor = torch.tensor(tg_X, dtype=torch.float32).to(device), torch.tensor(tg_Y, dtype=torch.float32).to(device)
                x_feature = model.feature_net(lm_X_tensor, lm_Y_tensor)  #(N1,64)
                num_dims = tg_X.shape[0]  #N2
                num_samples = 2  #2
                train_tg_Y_var = calculate_tg_Y_variance(train_data)  #[N,2]
                sigma = torch.sqrt(train_tg_Y_var)  # #[N,2]
                p_distr = torch.distributions.normal.Normal(loc=torch.tensor(0.0, device=device), scale=torch.tensor(sigma, device=device)) # #[N,2]
                stds= torch.zeros((num_dims, 2))  #(N,2)
                sigma_mean = torch.mean(sigma, dim=0, keepdim=True)
                stds [:, 0] = sigma_mean
                stds [:, 1] = sigma_mean + 0.02
                stds = torch.tensor(stds)

                y_samples_zero, q_y_samples = sample_gmm_centered(stds, num_samples=num_samples)  #Monte Carlo sampling
                y_samples_zero = y_samples_zero.cuda()  # Perturbation terms generated during the sampling process
                y_samples = tg_Y_tensor + y_samples_zero  #Add sampling perturbation to the true values

                scores_samples = model.predictor_net(x_feature, tg_X_tensor, y_samples) # # Energy function corresponding to f(x, y)

                q_y_samples = q_y_samples.to(device)
                p_y_samples = torch.exp(p_distr.log_prob(y_samples_zero))  # probability density of the perturbation

                max_scores = torch.max(scores_samples, dim=1, keepdim=True)[0] # ensures numerical stability by preventing overflow in the exponentials
                log_Z = torch.logsumexp(scores_samples - max_scores - torch.log(q_y_samples + 1e-12), dim=1) - math.log(num_samples)
                #compute nll_loss
                nll_loss = torch.mean(log_Z - torch.mean(scores_samples*(p_y_samples/q_y_samples + 1e-12), dim=1))

                # Backpropagation and optimization
                total_nll += nll_loss
                nll_loss.backward()
                optimizer.step()   
                train_num += len(tg_Y)
                
            total_nll = total_nll /train_num
            print("train: nll_loss: {:.4f} ".format(total_nll))

            if epoch >0 and epoch % opt.saved_epoch ==0 and epoch<1000:
                savepath = f"{model_dir}/EBGeo_{opt.dataset}_epoch_{epoch}.pth"
                save_cpt(model, optimizer, epoch, savepath)
                # Log the events
                log_path = f"{log_dir}/{opt.dataset}.txt"
                with open(log_path, 'a') as f:
                    f.write(f"\n********* Epoch={epoch} *********\n")
                    f.write(f"Train: nll: {total_nll:.3f}\n")
                print(f"Save checkpoint and log for epoch {epoch}")
            batch_metric = total_nll
            batch_metric_np = batch_metric.detach().cpu().numpy()
            losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses]
            if batch_metric_np <= np.min(losses):
                no_better_epoch = 0 
                early_stop_epoch = 0
                print("Better nll in epoch {}: {:.4f}".format(epoch, batch_metric))
            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = 0
            losses.append(batch_metric_np)
                
            # Halve the learning rate
            if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
                lr /= 2
                print("learning rate changes to {}!\n".format(lr))
                optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
                no_better_epoch = 0

            if early_stop_epoch == opt.early_stop_epoch:
                break