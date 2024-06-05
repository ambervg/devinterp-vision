import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torchvision
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification

from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD
from devinterp.slt.mala import MalaAcceptanceRate
from devinterp.utils import optimal_temperature 
from devinterp.utils import plot_trace


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transformers_cross_entropy(inputs, outputs):
    """
    Computes the cross-entropy loss for transformer models.

    Args:
    - inputs (torch.Tensor): The model predictions.
    - outputs (torch.Tensor): The true labels.

    Returns:
    - torch.Tensor: The computed cross-entropy loss.
    """
    return torch.nn.functional.cross_entropy(inputs, outputs)


def loop_visualize_llc(dir, epsilon, gamma):
    """
    Loops over model checkpoints in a directory, performs LLCestimation, and saves the results.

    Args:
    - dir (str): Directory containing model checkpoints.
    - epsilon (float): Epsilon value for visualization (learning rate).
    - gamma (float): Gamma value for visualization (localization).
    """

    # Check which checkpoints are already plotted
    checkpoints_already_done = {os.path.splitext(file)[0] for file in os.listdir("results/llc_plots")}
    print(f"✅ checkpoints_already_done: {checkpoints_already_done}")

    # Loop over checkpoints
    for filename in os.listdir(dir):
        print(f"🐢 Filename: {filename}")  # turtles all the way down

        base_filename = filename.replace(".pth.tar", "")
        if base_filename in checkpoints_already_done:
            continue
        
        model = models.resnet18(weights="DEFAULT")  # initiate to empty resnet
        checkpoint = torch.load(os.path.join(dir, filename))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)

        result = visualize_llc_trace(model, epsilon, gamma)

        # Save the result to a JSON file
        save_dir = os.path.join(os.getcwd(),"results", "llc_estimations")
        save_path = os.path.join(save_dir, f"{base_filename}.json")
        with open(save_path, 'w') as fp:
            json.dump({k: v.tolist() for k, v in result.items()}, fp)  # Convert arrays to lists for JSON serialization


def visualize_llc_trace(model, epsilon, gamma, please_plot=False):
    """
    Visualizes the LLC trace for a given model.

    Args:
    - model (torch.nn.Module): The model to visualize.
    - epsilon (float): Epsilon value for visualization (learning rate).
    - gamma (float): Gamma value for visualization (localization).
    - please_plot (bool): Whether to plot the LLC trace. Default is False.

    Returns:
    - dict: The results of the LLC trace visualization.
    """

    # Define transformations for pre-processing images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    # Load the model
    checkpoint = torch.load(args.checkpoint)
    model = models.resnet18(weights="DEFAULT")  # Make sure to set this to the correct model
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    # Load the data
    dataset_path = os.path.join(os.getcwd(), 'data', 'imagenet1k')
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    result = estimate_learning_coeff_with_summary(
        model,
        loader,
        criterion=transformers_cross_entropy,
        optimizer_kwargs=dict(
            lr=epsilon,
            localization=gamma,
            temperature=optimal_temperature(loader),
        ),
        sampling_method=SGLD,
        num_chains=NUM_CHAINS,
        num_draws=NUM_DRAWS,
        device=DEVICE,
        online=True,
    )
    trace = result.pop("llc/trace")
    if please_plot:
        plot_trace(
            trace,
            "LLC",
            x_axis="Step",
            title=os.path.split(args.checkpoint)[-1].replace(".pth.tar", ""),
            plot_mean=False,
            plot_std=False,
            fig_size=(12, 9),
            true_lc=None,
        )
    return result


def estimate_mala_sweeper(model, epsilons, gammas):
    """
    Estimates the MALA acceptance rate for different epsilon and gamma values.

    Args:
    - model (torch.nn.Module): The model to estimate.

    Returns:
    - dict: The results of the MALA sweeper estimation.
    """

    # Define transformations for pre-processing images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the data
    dataset_path = os.path.join(os.getcwd(), 'data', 'imagenet1k')
    dataset = ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = {}
    for epsilon in epsilons: 
        for gamma in gammas:
            print(f"Running for epsilon: {epsilon}, gamma: {gamma}")
            mala_estimator = MalaAcceptanceRate(
                num_chains=NUM_CHAINS,
                num_draws=NUM_DRAWS,
                temperature=optimal_temperature(train_loader),
                learning_rate=epsilon,
                device=DEVICE,
            )
            result = estimate_learning_coeff_with_summary(
                model,
                train_loader,
                criterion=transformers_cross_entropy,
                optimizer_kwargs=dict(
                    lr=epsilon,
                    localization=gamma,
                    temperature=optimal_temperature(train_loader),
                ),
                sampling_method=SGLD,
                num_chains=NUM_CHAINS,
                num_draws=NUM_DRAWS,
                callbacks=[mala_estimator],
                device=DEVICE,
                online=True,
            )
            mala_acceptance_rate_mean = mala_estimator.sample()["mala_accept/mean"]
            results[(epsilon, gamma)] = result
            print(f"epsilon {epsilon}, gamma {gamma}, mala rate: {mala_acceptance_rate_mean}")
    return results


def plot_sweep_single_model(results, epsilons, gammas, **kwargs):
    """
    Plots the sweep results for a single model.

    Args:
    - results (dict): The results of the MALA sweeper estimation.
    - epsilons (list): List of epsilon values.
    - gammas (list): List of gamma values.
    - kwargs (dict): Additional keyword arguments for plotting.
    """

    llc_color = "teal"
    fig, axs = plt.subplots(len(epsilons), len(gammas))

    for i, epsilon in enumerate(epsilons):
        for j, gamma in enumerate(gammas):
            result = results[(epsilon, gamma)]
            # plot loss traces
            loss_traces = result["loss/trace"]
            for trace in loss_traces:
                init_loss = trace[0]
                zeroed_trace = trace - init_loss
                sgld_steps = list(range(len(trace)))
                axs[i, j].plot(sgld_steps, zeroed_trace)

            # plot llcs
            means = result["llc/means"]
            stds = result["llc/stds"]
            sgld_steps = list(range(len(means)))
            axs2 = axs[i, j].twinx()
            axs2.plot(
                sgld_steps,
                means,
                color=llc_color,
                linestyle="--",
                linewidth=2,
                label=f"llc",
                zorder=3,
            )
            axs2.fill_between(
                sgld_steps,
                means - stds,
                means + stds,
                color=llc_color,
                alpha=0.3,
                zorder=2,
            )

            # center zero, assume zero is in the range of both y axes already
            y1_min, y1_max = axs[i, j].get_ylim()
            y2_min, y2_max = axs2.get_ylim()
            y1_zero_ratio = abs(y1_min) / (abs(y1_min) + abs(y1_max))
            y2_zero_ratio = abs(y2_min) / (abs(y2_min) + abs(y2_max))
            percent_to_add = abs(y1_zero_ratio - y2_zero_ratio)
            y1_amt_to_add = (y1_max - y1_min) * percent_to_add
            y2_amt_to_add = (y2_max - y2_min) * percent_to_add
            if y1_zero_ratio < y2_zero_ratio:
                # add to bottom of y1 and top of y2
                y1_min -= y1_amt_to_add
                y2_max += y2_amt_to_add
            elif y2_zero_ratio < y1_zero_ratio:
                # add to bottom of y2 and top of y1
                y2_min -= y2_amt_to_add
                y1_max += y1_amt_to_add
            axs[i, j].set_ylim(y1_min, y1_max)
            axs2.set_ylim(y2_min, y2_max)

            axs[i, j].set_title(f"$\epsilon$ = {epsilon} : $\gamma$ = {gamma}")
            # only show x axis label on last row
            if i == len(epsilons) - 1:
                axs[i, j].set_xlabel("SGLD time step")
            axs[i, j].set_ylabel("loss")
            axs2.set_ylabel("llc", color=llc_color)
            axs2.tick_params(axis="y", labelcolor=llc_color)
    if kwargs["title"]:
        fig.suptitle(kwargs["title"], fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_single_graph(result, title=""):
    """
    Plots the results of a single model.

    Args:
    - result (dict): The results of the MALA sweeper estimation.
    - title (str): The title for the plot.
    """

    llc_color = "teal"
    fig, axs = plt.subplots(1, 1)
    # plot loss traces
    loss_traces = result["loss/trace"]
    for trace in loss_traces:
        init_loss = trace[0]
        zeroed_trace = trace - init_loss
        sgld_steps = list(range(len(trace)))
        axs.plot(sgld_steps, zeroed_trace)

    # plot llcs
    means = result["llc/means"]
    stds = result["llc/stds"]
    sgld_steps = list(range(len(means)))
    axs2 = axs.twinx()
    axs2.plot(
        sgld_steps,
        means,
        color=llc_color,
        linestyle="--",
        linewidth=2,
        label=f"llc",
        zorder=3,
    )
    axs2.fill_between(
        sgld_steps, means - stds, means + stds, color=llc_color, alpha=0.3, zorder=2
    )

    # center zero, assume zero is in the range of both y axes already
    y1_min, y1_max = axs.get_ylim()
    y2_min, y2_max = axs2.get_ylim()
    y1_zero_ratio = abs(y1_min) / (abs(y1_min) + abs(y1_max))
    y2_zero_ratio = abs(y2_min) / (abs(y2_min) + abs(y2_max))
    percent_to_add = abs(y1_zero_ratio - y2_zero_ratio)
    y1_amt_to_add = (y1_max - y1_min) * percent_to_add
    y2_amt_to_add = (y2_max - y2_min) * percent_to_add
    if y1_zero_ratio < y2_zero_ratio:
        # add to bottom of y1 and top of y2
        y1_min -= y1_amt_to_add
        y2_max += y2_amt_to_add
    elif y2_zero_ratio < y1_zero_ratio:
        # add to bottom of y2 and top of y1
        y2_min -= y2_amt_to_add
        y1_max += y1_amt_to_add
    axs.set_ylim(y1_min, y1_max)
    axs2.set_ylim(y2_min, y2_max)
    axs.set_xlabel("SGLD time step")
    axs.set_ylabel("loss")
    axs2.set_ylabel("llc", color=llc_color)
    axs2.tick_params(axis="y", labelcolor=llc_color)
    axs.axhline(color="black", linestyle=":")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    EPSILONS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    GAMMAS = [1, 10, 100]
    NUM_CHAINS = 1
    NUM_DRAWS = 50

    # Create the parser
    parser = argparse.ArgumentParser(description="Train a ResNet model on a selected dataset.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet50"], help="Model architecture to use: 'resnet18' or 'resnet50'")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Path to the checkpoint file to use for calibration")
    parser.add_argument("--step", type=int, required=True, choices=[1,2,3,4], help="Step in the calibration process: 1, 2, 3, or 4.")
    
    parser.add_argument("--epsilon", type=float, required=False, help="Chosen epsilon value based on step 1. Required for step 2/3/4.")
    parser.add_argument("--gamma", type=float, required=False, help="Chosen gamma value based on step 1. Required for step 2/3/4.")
    parser.add_argument("--checkpoints_dir", type=str, required=False, default="data/checkpoints", help="Dir containing checkpoints to loop. Required for step 4.")

    # Parse the arguments
    args = parser.parse_args()

    # Step 1: We will be determining the correct epsilon and gamma values
    checkpoint = torch.load(args.checkpoint)
    if args.model == "resnet18":
        model = models.resnet18(weights="DEFAULT")
    elif args.model == "resnet50":
        model = models.resnet50(weights="DEFAULT")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    if args.step == 1:
        results = estimate_mala_sweeper(model, EPSILONS, GAMMAS)  
        plot_sweep_single_model(results, EPSILONS, GAMMAS, title="Calibration sweep of model for lr ($\\epsilon$) and localization ($\\gamma$)")

    # Step 2: We will plot a single graph of our favourite.
    if args.step == 2:
        results = estimate_mala_sweeper(model, [args.epsilon], [args.gamma])  
        result = results[(args.epsilon, args.gamma)]
        plot_single_graph(result)

    # Step 3: Now, we plot a few chains of the trace of our favourite.
    NUM_CHAINS = 5
    if args.step == 3:
        result = visualize_llc_trace(model, args.epsilon, args.gamma, please_plot=True)

    # Step 4: If that looks good, we make that plot for all saved checkpoints
    if args.step == 4:
        loop_visualize_llc(args.checkpoints_dir, args.epsilon, args.gamma)

