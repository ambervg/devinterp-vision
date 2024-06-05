# LLC-estimation-vision

Repository for research on vision models performed together with Timaeus aspart of the Athena Research Program. This project focuses on the Developmental Interpretability of Deep Neural Networks (DNNs) for vision models, specifically examining the LLC estimation of these models over their training checkpoints.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contact](#contact)

## Introduction

This repository contains the code and resources for analyzing the developmental interpretability of vision models. The main goal is to estimate the Local Learning Coefficients (LLC) over various checkpoints created during the training process of these models.

## Getting Started

Clone this repository and set up a virtual environment. Then, install the required packages.

1. Clone the repository:
    ```bash
    git clone https://github.com/ambervg/LLC-estimation-vision.git
    cd LLC-estimation-vision
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -e .
    ```

## Usage

1. **Data Retrieval:** First we will need to retrieve the data for training the vision models. This could be (for example) data from ImageNet-1K or CIFAR-10.
    - Download ImageNet-1k (e.g. 11GB from [this website](https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k?resource=download)) and save it in the `data/imagenet` directory.
2. **Model Training and Checkpoint Generation:** Then, we will need to train a model and generate checkpoint during the training process. The scripts to do this are in the `scripts/vision_model` directory.
    ```bash
    python scripts/vision_model/train_resnet.py --model resnet18 --dataset imagenet --resume_from_checkpoint None
    ```
    -  Argument Options
        - `--model_architecture`:
            - Description: Specifies the architecture of the ResNet model to use. Choices: `resnet18`, `resnet50`
            - Example: `--model_architecture resnet18`
        - `--dataset`:
            - Description: Specifies the dataset to use for training. Choices: `imagenet1k`, `cifar10`
            - Example: `--dataset imagenet1k`
        - `--resume_from_checkpoint` (optional):
            - Description: Path to the checkpoint file to resume training from. If not specified, training will start from scratch.
            - Default: `None`
            - Example: `--resume_from_checkpoint checkpoints/checkpoint.pth.tar`

3. **Epsilon & Gamma Calibration:** Next, 
    ```bash
    python blabla.py 
    ```
4. **Plotting LLC Estimations:** Finally we can generate some plots 

## Project Structure

```
LLC-estimation-vision/
├── data/
│ ├── imagenet1k/ # Training data images from ImageNet
│ ├── cifar10-python/ # Training data images from Cifar10
│ └── checkpoints/  # Save generated checkpoints here unless other location specified
├── results/ # Output results and visualizations
│ ├── llc_estimations/
│ └── llc_plots/
├── scripts/
│ ├── LLC_estimation/ # Scripts related to estimating and plotting LLC
│ └── vision_model/ # Scripts related to training/evaluating models and creating checkpoints 
├── venv/
├── README.md # Project documentation
├── requirements.txt # Python dependencies
```

## Contact

For questions or inquiries, feel free to open up an issue. 😄👍