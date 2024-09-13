import torch
import torch.nn as nn
import torch.nn.functional as F
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class EBGeoPredictorNet(nn.Module):
    def __init__(self, dim_in):
        super(EBGeoPredictorNet, self).__init__()
        self.dim_in = dim_in
        self.dim_z = self.dim_in + 2
        self.dim_as = self.dim_z * 2
        # Linear layers for prediction network
        self.fc1_y = nn.Linear(self.dim_z, self.dim_as)  # First layer (32 -> 64)
        self.fc1_xy = nn.Linear(self.dim_as * 2, self.dim_as) # Second layer (128 -> 64)
        self.fc2_xy = nn.Linear(self.dim_as, self.dim_as)  # Third layer (64 -> 64)
        self.fc3_xy = nn.Linear(self.dim_as, self.dim_as)  # Fourth layer (64 -> 64)
        self.fc4_xy = nn.Linear(self.dim_as, 2) # Output layer (64 -> 2)
    def forward(self, x, tg_X, y):
        x_feature = x.mean(dim=0).unsqueeze(0)  # Compute feature vector from x
        tg_X_shape_0 = tg_X.shape[0]
        x_feature = x_feature.repeat(tg_X_shape_0, 1) # Repeat x feature to match tg_X size
        if y.shape[0] != tg_X_shape_0:
            y = y.repeat(tg_X_shape_0, 1)  # Repeat y to match tg_X
        tg_X_tensor = torch.tensor(tg_X, dtype=torch.float32).to(device) # Combine tg_X and y into a single tensor
        y_feature = torch.cat([tg_X_tensor, y], dim=1)  # (N2, 32)
        # Pass through layers with tanh activations
        y_feature = torch.tanh(self.fc1_y(y_feature))  # (N2, 32)->(N2, 64)
        xy_features_0 = torch.cat([x_feature, y_feature], 1)   # (N2, 64)->(N2, 128)
        xy_features_1 = torch.tanh(self.fc1_xy(xy_features_0)) # (N2,64)
        xy_features_2 = torch.tanh(self.fc2_xy(xy_features_1)) + xy_features_1  # Residual connection
        xy_features_3 = torch.tanh(self.fc3_xy(xy_features_2)) + xy_features_2  # Residual connection
        score = self.fc4_xy(xy_features_3) #(N2,2)
        return score

class EBGeoFeatureNet(nn.Module):
    def __init__(self, dim_in):
        super(EBGeoFeatureNet, self).__init__()
        self.dim_in = dim_in
        self.dim_z = self.dim_in + 2
        self.dim_as = self.dim_z * 2
        # Linear layers for feature extraction
        self.fc1_x = nn.Linear(self.dim_z, self.dim_as)   # First layer (32 -> 64)
        self.fc2_x = nn.Linear(self.dim_as, self.dim_as)  # Second layer (64 -> 64)
    def forward(self, lm_X, lm_Y):
        x = torch.cat((lm_X, lm_Y), dim=1)   # Combine lm_X and lm_Y into a single tensor
        x_feature = F.relu(self.fc1_x(x))     # Pass through layers with ReLU activations
        x_feature = F.relu(self.fc2_x(x_feature)) 
        return x_feature

class EBGeo(nn.Module):
    def __init__(self, dim_in):
        super(EBGeo, self).__init__()
        self.dim_in = dim_in
        self.input_dim = dim_in
        # Initialize feature extraction and prediction networks
        self.feature_net = EBGeoFeatureNet(self.input_dim)
        self.predictor_net = EBGeoPredictorNet(self.input_dim)

    def forward(self, lm_X, lm_Y, tg_X, y):
        x_feature = self.feature_net(lm_X, lm_Y) # # Extract features from lm_X and lm_Y
        return self.predictor_net(x_feature, tg_X, y)  # Predict scores for tg_X and y