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


## How to run our programs

### Run the code with DEGeo

```bash
# Open the "UQGeo/EBGeo" folder
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


## Citing TrustGeo

If you find this work helpful, please consider citing us:

```bibtex
@misc{trustgeo2024,
  author = {Author Name, Second Author, Third Author},
  title = {TrustGeo: Trustworthy Geolocation with Uncertainty Quantification},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/ICDM-UESTC/TrustGeo}},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### README 内容说明：
1. **项目简介**：开头简要描述了 TrustGeo 的目的和功能。
2. **特性**：列出了项目的一些关键功能，如不确定性量化、多个数据集支持等。
3. **环境要求**：使用 Anaconda 安装环境和依赖项的说明。
4. **安装步骤**：指导用户如何克隆仓库并安装所需的依赖。
5. **使用指南**：提供了训练、测试和可视化的示例命令，帮助用户快速上手使用模型。
6. **引用**：提供 BibTeX 引用条目，便于其他研究者在论文中引用该项目。
7. **许可**：简单说明项目的许可信息。

你可以将此内容复制到你的 `README.md` 文件中，并根据实际情况进行调整。如果需要进一步修改或有其他问题，请告诉我！
