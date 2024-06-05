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

Here you can provide instructions on how to use your project, including examples and explanations of the main functionalities.

1. **Data Retrieval:** Describe how to prepare and format the data for the vision models.
2. **Model Training and Checkpoint Generation:**
3. **Epsilon & Gamma Calibration:**
    ```bash
    python blabla.py 
    ```
4. **Plotting LLC Estimations:**

## Project Structure

```
LLC-estimation-vision/
├── data/
│ ├── imagenet/ # Training data images from ImageNet
│ └── cifar10-python/ # Training data images from Cifar10
├── results/ # Output results and visualizations
│ ├── llc_estimations/
│ └── llc_plots/
├── scripts/
│ ├── LLC_estimation/ # All scripts related to estimating and plotting LLC
│ └── vision_model/ # All scripts related to training/evaluating models and creating checkpoints 
├── venv/
├── README.md # Project documentation
├── requirements.txt # Python dependencies
```

## Contact

For questions or inquiries, feel free to open up an issue.