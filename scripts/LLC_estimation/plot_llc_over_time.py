import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
    return torch.nn.functional.cross_entropy(
        inputs, outputs
    )  # transformers doesn't output a vector


def loop_visualize_llc(dir, epsilon, gamma):
    """
    Loops over model checkpoints in a directory, visualizes LLC, and saves the results.
    
    Args:
    - dir (str): Directory containing model checkpoints.
    - epsilon (float): Epsilon value for visualization. Learning Rate.
    - gamma (float): Gamma value for visualization. Localization.
    """

    # Check which checkpoints are already plotted
    checkpoints_already_done = {os.path.splitext(file)[0] for file in os.listdir("/home/amber/vision/vision/LLC_estimation/llc_plots")}
    print(f"✅ checkpoints_already_done: {checkpoints_already_done}")

    # Loop over checkpoints
    for filename in os.listdir(dir):
        print(f"🐢 Filename: {filename}")

        base_filename = filename.replace(".pth.tar", "")
        if base_filename in checkpoints_already_done:
            continue
        
        model = models.resnet18(weights="DEFAULT")  # initiate to empty resnet
        checkpoint = torch.load(os.path.join(dir, filename))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)

        result = visualize_llc_trace(model, epsilon, gamma)

        # Save the result to a JSON file
        save_dir = os.path.join(os.getcwd(),"vision", "LLC_estimation", "llc_plots")
        save_path = os.path.join(save_dir, f"{base_filename}.json")
        with open(save_path, 'w') as fp:
            json.dump({k: v.tolist() for k, v in result.items()}, fp)  # Convert arrays to lists for JSON serialization


def visualize_llc_trace(model, epsilon, gamma, please_plot=False):

    # Define transformations for pre-processing images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    # Load the model
    checkpoint = torch.load(filename)
    model = models.resnet18(weights="DEFAULT")  # Make sure to set this to the correct model
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    # Load the data
    dataset_path = os.path.join(os.getcwd(), 'vision', 'imagenet', 'imagenet1k')
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
            title=os.path.split(filename)[-1].replace(".pth.tar", ""),
            plot_mean=False,
            plot_std=False,
            fig_size=(12, 9),
            true_lc=None,
        )
    print(result)
    return result


def estimate_mala_sweeper(model):

    # Define transformations for pre-processing images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the data
    dataset_path = os.path.join(os.getcwd(), 'vision', 'imagenet', 'imagenet1k')
    dataset = ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = {}
    for epsilon in EPSILONS:
        for gamma in GAMMAS:
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
    EPSILONS = [0.01]  #[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    GAMMAS = [100]  #[1, 10, 100]
    NUM_CHAINS = 1
    NUM_DRAWS = 200

    # First, we will be determining the correct epsilon and gamma values
    filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet18-imagenet1k-v2/checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar")
    # filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet50-imagenet1k-v1/checkpoint_2024-03-26-04h12_epoch_27_val_400.pth.tar")
    checkpoint = torch.load(filename)
    model = models.resnet18(weights="DEFAULT")  # Make sure to set this to the correct model
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    if False:
        results = estimate_mala_sweeper(model)  
        plot_sweep_single_model(results, EPSILONS, GAMMAS, title="Calibration sweep of model for lr ($\\epsilon$) and localization ($\\gamma$)")

    ## Next, we will plot a single graph of our favourite.
    EPSILON = 0.01
    GAMMA = 100
    if False:
        result = results[(EPSILON, GAMMA)]
        plot_single_graph(result)

    ## Now, we plot a few chains of the trace of our favourite.
    NUM_CHAINS = 5
    if True:
        result = visualize_llc_trace(model, EPSILON, GAMMA, please_plot=True)

    # Finally, if that looks good, we make that plot for all saved checkpoints
    dir = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet18-imagenet1k-v2/")
    if False:
        loop_visualize_llc(dir, EPSILON, GAMMA)

