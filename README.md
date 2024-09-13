# Advancing IP Geolocation Robustness Through the Lens of Uncertainty Quantification

This repository provides the original PyTorch implementation of the UQ_IP_Geolocation framework.

## Table of Contents
- [Features](#features)
- [Environmental Requirements](#environmental-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Citing TrustGeo](#citing-trustgeo)

## Features
- **Accurate Geolocation**: Implements advanced machine learning algorithms to predict the geographical location with high accuracy.
- **Uncertainty Quantification**: Incorporates methods to measure and quantify prediction uncertainty, providing confidence levels along with location predictions.
- **Support for Multiple Datasets**: Easily adaptable to various datasets including 'New_York', 'Los_Angeles', and 'Shanghai'.
- **Extensible Framework**: Designed to be extended with additional modules or modified to fit different geolocation requirements.

## Environmental Requirements

The code was tested with `python 3.8.13`, `PyTorch 1.12.1`, `cudatoolkit 11.6.0`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

### Create virtual environment

```bash
conda create --name UQGeo python=3.8.13
```

### Activate environment

```bash
conda activate UQGeo
```

### Install PyTorch & Cudatoolkit

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

### Install other requirements

```bash
conda install numpy pandas
pip install scikit-learn
```

## Folder Structure




## Usage

### Example Command to Train the Model

You can train the TrustGeo model on the 'New_York' dataset using the following command:

```bash
python train.py --dataset "New_York" --epochs 100 --batch_size 32 --learning_rate 0.001
```

### Testing the Model

To evaluate the model performance on the test set, use the following command:

```bash
python test.py --dataset "New_York" --load_model "best_model.pth"
```

### Visualize Results

To visualize the prediction results, run:

```bash
python visualize.py --dataset "New_York"
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
