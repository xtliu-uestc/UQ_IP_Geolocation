# Advancing IP Geolocation Robustness Through the Lens of Uncertainty Quantification

This repository provides the original PyTorch implementation of the UQ_IP_Geolocation framework.

# Basic Usage
## Environmental Requirements

The code was tested with `python 3.8.13`, `PyTorch 1.12.1`, `cudatoolkit 11.6.0`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```bash
# Create virtual environment
conda create --name UQGeo python=3.8.13

# Activate environment
conda activate UQGeo

# Install PyTorch & Cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# Install other requirements
conda install numpy pandas
pip install scikit-learn
```

## How to run our programs

### Run the code with DEGeo

```bash
# Open the "UQGeo/DEGeo" folder
cd UQGeo/DEGeo

# data preprocess (executing IP clustering). 
python preprocess.py --dataset "New_York"
python preprocess.py --dataset "Los_Angeles"
python preprocess.py --dataset "Shanghai"

# run the model DEGeo
python main.py --dataset "New_York" --lr 5e-3 --dim_in 30 --lambda1 7e-3
python main.py --dataset "Los_Angeles" --lr 3e-3 --dim_in 30 --lambda1 7e-3
python main.py --dataset "Shanghai" --lr 0.0015 --dim_in 51 --lambda1 1e-3

# load the checkpoint and then test
python test.py --dataset "New_York" --lr 5e-3 --dim_in 30 --lambda1 7e-3 --load_epoch 400
python test.py --dataset "Los_Angeles" --lr 3e-3 --dim_in 30 --lambda1 7e-3 --load_epoch 600
python test.py --dataset "Shanghai" --lr 0.0015 --dim_in 51 --lambda1 1e-3 --load_epoch 200
```

### Run the code with EBGeo

```bash
# Open the "UQGeo/EBGeo" folder
cd UQGeo/EBGeo

# data preprocess (executing IP clustering). 
python preprocess.py --dataset "New_York"
python preprocess.py --dataset "Los_Angeles"
python preprocess.py --dataset "Shanghai"

# run the model EBGeo
python main.py --dataset "New_York" --lr 1e-6 --dim_in 30
python main.py --dataset "Los_Angeles" --lr 1e-6 --dim_in 30 
python main.py --dataset "Shanghai" --lr 1.5e-3 --dim_in 51

# load the checkpoint and then test
python test.py --dataset "New_York" --lr 1e-6 --dim_in 30
python test.py --dataset "Los_Angeles" --lr 1e-6 --dim_in 30
python test.py --dataset "Shanghai" --lr 1.5e-6 --dim_in 51
```

## Folder Structure

```plaintext
DEGeo/EBGeo
├── datasets # Contains three large-scale real-world IP geolocation datasets.
│   ├── New_York # IP geolocation dataset collected from New York City including 12 landmarks.
│   ├── Los_Angeles # IP geolocation dataset collected from Los Angeles including 15 landmarks.
│   └── Shanghai # IP geolocation dataset collected from Shanghai including 12 landmarks.
├── lib # Contains model implementation files
│   ├── layers.py # Contains model implementation files.
│   ├── model.py # The core source code of the proposed EBGeo/DEGeo.
│   └── utils.py # Auxiliary functions, including the code of uncertainty quantification functions.
├── asset # Contains saved checkpoints and logs when running the model
│   ├── log # Contains logs when running the model.
│   └── model # Contains the saved checkpoints.
├── preprocess.py # Preprocess dataset and execute IP clustering for the model running.
├── main.py # Run model for training and testing.
├── test.py # Load checkpoint and then test the model.
```

## The description of hyperparameters used in main.py

| Hyperparameter    | Description                                                           |
|-------------------|-----------------------------------------------------------------------|
| `seed`            | The random number seed used for parameter initialization during training, which ensures reproducibility of the model's training process. |
| `model_name`      | The name of the model being trained or tested, used to identify different models. |
| `dataset`         | The dataset used by `main.py` for training and testing the model.      |
| `lr`              | The learning rate used during training to control the step size of the optimizer. |
| `harved_epoch`    | The number of consecutive epochs with no performance improvement after which the learning rate is halved to help the model converge. |
| `num_epochs`      | The total number of training epochs, defining how many times the model will iterate over the training dataset. |
| `early_stop_epoch`| The number of consecutive epochs with no performance improvement after which the training process will stop to prevent overfitting. |
| `saved_epoch`     | The interval (number of epochs) at which a checkpoint is saved, so the model can be tested or resumed later. |
| `dim_in`          | The dimension of the input data, defining the number of input features fed into the model. |
| `max_epoch`       | The maximum number of epochs allowed for training, regardless of other stopping criteria. |
| `ensemble_size`   | The number of models in the ensemble, if ensemble methods are being used to improve model performance. |
| `lambda_1`        | The trade-off coefficient in the loss function, balancing between different objectives or regularization terms in the loss function. |
| `load_epoch`      | The epoch number from which a model checkpoint is loaded to continue training or testing. |

## Dataset Information

The "datasets" folder contains three subfolders corresponding to three large-scale real-world IP geolocation datasets collected from New York City, Los Angeles, and Shanghai. There are three files in each subfolder:

- `data.csv` # features (including attribute knowledge and network measurements) and labels (longitude and latitude) for street-level IP geolocation.
- `ip.csv` # IP addresses.
- `last_traceroute.csv` # last four routers and corresponding delays for efficient IP host clustering.

The detailed columns and description of `data.csv` in the New York dataset are as follows:

### New York

| Column Name                       | Data Description                                                                 |
|------------------------------------|----------------------------------------------------------------------------------|
| `ip`                               | The IPv4 address                                                                 |
| `as_mult_info`                     | The ID of the autonomous system where IP locates                                 |
| `country`                          | The country where the IP locates                                                 |
| `prov_cn_name`                     | The state/province where the IP locates                                          |
| `city`                             | The city where the IP locates                                                    |
| `isp`                              | The Internet Service Provider of the IP                                          |
| `vp900/901/..._ping_delay_time`    | The ping delay from probing hosts "vp900/901/..." to the IP host                 |
| `vp900/901/..._trace`              | The traceroute list from probing hosts "vp900/901/..." to the IP host            |
| `vp900/901/..._tr_steps`           | #steps of the traceroute from probing hosts "vp900/901/..." to the IP host       |
| `vp900/901/..._last_router_delay`  | The delay from the last router to the IP host in the traceroute list from probing hosts "vp900/901/..." |
| `vp900/901/..._total_delay`        | The total delay from probing hosts "vp900/901/..." to the IP host                |
| `longitude`                        | The longitude of the IP (as the label)                                           |
| `latitude`                         | The latitude of the IP host (as the label)                                       |

PS: The detailed columns and description of `data.csv` in the other two datasets are similar to the New York dataset.
