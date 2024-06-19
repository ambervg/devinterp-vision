# LLC-estimation-vision
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

Repository for research on vision models performed together with [Timaeus](https://timaeus.co/) as part of the [Athena Research Program](https://researchathena.org/). This project focuses on the [Developmental Interpretability](https://medium.com/@groenestijnamber/12-developmental-interpretability-eca2f3d5ec80) of Deep Neural Networks (DNNs) for vision models, specifically examining the LLC estimation of these models over their training checkpoints.

This repository contains the code and resources for analyzing the developmental interpretability of vision models. The main goal is to estimate the Local Learning Coefficients (LLC) over various checkpoints created during the training process of these models.

## 📋 Table of Contents 

- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contact](#-contact)

## 🚀 Getting Started

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

## 🔍 Usage

1. **Data Retrieval:** First we will need to retrieve the data for training the vision models. This could be (for example) data from ImageNet-1K or CIFAR-10.
    - Download ImageNet-1k (e.g. 11GB from [this website](https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k?resource=download)) and save it in the `data/imagenet` directory.

2. **Model Training and Checkpoint Generation:** Then, we will need to train a model and save checkpoints during the training process. The scripts to do this are in the `scripts/vision_model` directory.
    ```bash
    python scripts/vision_model/train_resnet.py --model resnet18 --dataset imagenet --resume_from_checkpoint None \
        --model_architecture resnet18 \ # Specifies the architecture of the ResNet model to use (choices: resnet18, resnet50)
        --dataset imagenet1k \          # Specifies the dataset to use for training (choices: imagenet1k, cifar10)
        --resume_from_checkpoint None   # Path to the checkpoint file to resume training from (default: None, if not specified, training starts from scratch). Example: checkpoints/checkpoint.pth.tar
    ```
3. **Epsilon & Gamma Calibration:** 
    
    ⚖️ **Step 3A**: Compare the MALA Acceptence Rates for multiple epsilon-gamma combos
    ```bash
    python scripts/LLC_estimation/epsilon_gamma_calibration.py \
    --model resnet18 \
    --checkpoint data/checkpoints/checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar \
    --step 1
    ```
    🧹 **Step 3B**: Pick a single value for epsilon and gamma, based on:
        - MALA Acceptance Rate: Generally, the best option is to pick the highest one.
        - Calibration sweep showing the LLC over time for the trained ResNet18 model given different values for epsilon (learning rate) and gamma (localization): Ensure there are no large peaks.
    ```bash
    # Now, plot a curve of your favourite. Fill in the --epsilon and --gamma accordingly.
    
    python scripts/LLC_estimation/epsilon_gamma_calibration.py \
    --model resnet18 \
    --checkpoint data/checkpoints/checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar \
    --step 2 \
    --epsilon 0.01 \
    --gamma 100
    ```
    ⛓️ **Step 3C**: Do the same but now for multiple chains (default: 5 chains).
    ```bash
    python scripts/LLC_estimation/epsilon_gamma_calibration.py \
    --model resnet18 \
    --checkpoint data/checkpoints/checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar \
    --step 3 \
    --epsilon 0.01 \
    --gamma 100 \
    --checkpoints_dir data/checkpoints
    ```
    ℹ️ **Flag Information:** _click to expand_
    - <details>
      <summary><code>--model</code></summary>
      <ul>
        <li><strong>Description:</strong> Specifies the architecture of the ResNet model to use.</li>
        <li><strong>Choices:</strong> resnet18, resnet50</li>
        <li><strong>Example:</strong> <code>--model resnet18</code></li>
      </ul>
    </details>
    
    - <details>
      <summary><code>--checkpoint</code></summary>
      <ul>
        <li><strong>Description:</strong> Path to the checkpoint file to use for calibration.</li>
        <li><strong>Example:</strong> <code>--checkpoint checkpoints/checkpoint_YYYY-MM-DD-HHhMM_epoch_XX.pth</code></li>
      </ul>
    </details>
    
    - <details>
      <summary><code>--step</code></summary>
      <ul>
        <li><strong>Description:</strong> Step in the calibration process.</li>
        <li><strong>Choices:</strong> 1, 2, 3, 4</li>
        <li><strong>Example:</strong> <code>--step 1</code></li>
      </ul>
    </details>
    
    - <details>
      <summary><code>--epsilon</code> (relevant for steps 2, 3, 4)</summary>
      <ul>
        <li><strong>Description:</strong> Chosen epsilon value based on step 1.</li>
        <li><strong>Example:</strong> <code>--epsilon 0.01</code></li>
      </ul>
    </details>
    
    - <details>
      <summary><code>--gamma</code> (relevant for steps 2, 3, 4)</summary>
      <ul>
        <li><strong>Description:</strong> Chosen gamma value based on step 1.</li>
        <li><strong>Example:</strong> <code>--gamma 10</code></li>
      </ul>
    </details>
    
    - <details>
      <summary><code>--checkpoints_dir</code> (relevant for step 4)</summary>
      <ul>
        <li><strong>Description:</strong> Directory containing checkpoints to loop.</li>
        <li><strong>Default:</strong> data/checkpoints</li>
        <li><strong>Example:</strong> <code>--checkpoints_dir data/checkpoints</code></li>
      </ul>
    </details>
    

4. **Plotting LLC Estimations:** 
- Finally we can generate a plot showing the LLC values over all checkpoints. 
- For this we will loop over all the checkpoints in a specified directory. Because this operation can take a long time, intermediate results will be stored in `results/llc_estimations` and the script can be stopped and started at any moment. 

    ```bash
    python scripts/LLC_estimation/epsilon_gamma_calibration.py --model resnet18 --checkpoint data/checkpoints/checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar --step 4 --epsilon 0.01 --gamma 100
    ```
- When we have the results from all checkpoints, we can make the final plot.

    ```bash
    python scripts/LLC_estimation/plot_llc_from_results.py
    ```

## 📂 Project Structure

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

## 📬 Contact

For questions or inquiries, feel free to open up an issue. 😄👍
