# Enhanced Pix2Pix GAN for Paired Image Translation

This project implements an **Enhanced Pix2Pix GAN** model designed for translating edge maps to photographic images. The core of this project revolves around improving the original Pix2Pix GAN architecture by introducing a modified U-Net generator and a dual-critic system. Additionally, advanced loss functions, such as **Wasserstein loss** and **L1 loss**, are integrated to enhance the quality and stability of the generated images.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Results](#results)

## Project Overview

The aim of this project is to reimagine the **Pix2Pix GAN framework** for paired image translation tasks, such as translating edge maps to real images. The model has been built from the ground up and includes architectural enhancements to improve training stability and image fidelity. The **Edges2Shoes dataset** has been used for training and evaluation, translating edge maps of shoes into realistic photographs.

## Folder Structure

```plaintext
Enhanced-Pix2Pix-GAN/
│
├── configs/                           # Configuration files for training
│   └── train_config.json              # JSON configuration file for training parameters
│
├── data/                              # Dataset directory (place your dataset here)
│   ├── train/                         # Directory containing training data
│   └── val/                           # Directory containing validation data
│
├── model_logs/                        # Directory where model checkpoints and logs will be saved
│
├── notebooks/                         # Jupyter notebooks for data exploration, predictions and evaluation
│   ├── 01-look-at-prepared-data.ipynb # Notebook for inspecting the prepared data
│   ├── 02-look-at-predictions.ipynb   # Notebook for evaluating model predictions
│   └── 03-evaluate-model.ipynb        # Notebook for model evaluation metrics
│
├── report/                            # Report detailing project methodology and results
│   └── Enhanced Pix2Pix GAN.pdf       # Report detailing the methodology and results of the project
│
├── scripts/                           # Shell scripts for training and testing
│   └── train.sh                       # Shell script to train the model
│
├── unit_tests/                        # Unit tests for various components of the model
│   ├── test_gan_loss.py               # Unit test for GAN loss functions
│   ├── test_network_components.py     # Unit test for network components
│   └── test_networks.py               # Unit test for network architecture
│
├── arg_parse.py                       # Argument parsing for training script
├── data_preparation.py                # Data preparation and augmentation scripts
├── gan_loss.py                        # Implementation of loss functions used in GANs
├── main.py                            # Main entry point for training the model
├── metrics.py                         # Metrics for evaluating model performance
├── network_components.py              # Components used in the network architecture
├── networks.py                        # Definition of the full model architecture
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies required for the project
├── train.py                           # Model training loop and optimizer setup
└── utils.py                           # Utility functions used throughout the project

```

## Setup and Installation

### Requirements

- Python 3.8+
- PyTorch (with CUDA support)
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shahinntu/Enhanced-Pix2Pix-GAN.git
   cd Enhanced-Pix2Pix-GAN
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have access to a GPU for training (CUDA enabled).

## Data Preparation

The model uses the **Edges2Shoes dataset** for training, translating edge maps of shoes into photographs.

Place the dataset in the `data/` directory:
- Training data in `data/train/`
- Validation data in `data/val/`


## Usage

### Unit Tests

To run the unit tests, navigate to the `unit_tests/` directory by running the following command:

```bash
cd unit_tests
```

Once inside the directory, you can run the specific test files. For example:

```bash
python test_gan_loss.py
```

Repeat this command for other test files, such as:

```bash
python test_network_components.py
python test_networks.py
```

These tests are designed to validate the functionality of individual components like the GAN loss, network components, and overall network architecture.

### Training

To start training the Enhanced Pix2Pix GAN, use the following command:

```bash
bash scripts/train.sh
```

Make sure to modify the `train_config.json` file as needed for hyperparameters like learning rate, batch size, and number of epochs.

### Evaluation

To evaluate the model, use the provided notebooks in the `notebooks/` directory. Specifically:

- `03-evaluate-model.ipynb` allows you to evaluate the model on test data and generate metrics.
- `02-look-at-predictions.ipynb` helps in visualizing model predictions.

Make sure to place the validation data in the `data/val` directory before running the evaluation. 

Additionally, ensure that the correct model checkpoint version is specified in the notebook. You need to specify the correct version (e.g., 231119195749) from the model_logs/ directory to load the appropriate trained model for evaluation.

## Results

The model has been evaluated on the **Edges2Shoes dataset**, with the following key outcomes:
- **Frechet Inception Distance (FID)**: 80.71 (lower is better)
- The enhanced Pix2Pix GAN shows improved handling of complex edge maps compared to the original implementation.

Refer to the report `Enhanced Pix2Pix GAN.pdf` for a detailed discussion of results, methodology, and model architecture.
