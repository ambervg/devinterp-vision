import os
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
# from devinterp.utils import plot_trace


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_trace(
    trace,
    y_axis,
    x_axis="step",
    title=None,
    plot_mean=True,
    plot_std=True,
    fig_size=(12, 9),
    true_lc=None,
):
    plt.figure(figsize=fig_size)
    num_chains, num_draws = trace.shape
    sgld_step = list(range(num_draws))
    if true_lc:
        plt.axhline(y=true_lc, color="r", linestyle="dashed")
    # trace
    for i in range(num_chains):
        draws = trace[i]
        plt.plot(sgld_step, draws, linewidth=1, label=f"chain {i}")

    # mean
    mean = np.mean(trace, axis=0)
    plt.plot(
        sgld_step,
        mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label="mean",
        zorder=3,
    )

    # std
    std = np.std(trace, axis=0)
    plt.fill_between(sgld_step, mean - std, mean + std, color="gray", alpha=0.3, zorder=2)

    if title is None:
        title = f"{y_axis} values over sampling draws"
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(),"vision", "LLC_estimation", "llc_plots", f"{title.replace(' ', '_')}.png"))
    plt.savefig(os.path.join(os.getcwd(),"vision", "LLC_estimation", "llc_plots", f"{title.replace(' ', '_')}.svg"))


def transformers_cross_entropy(inputs, outputs):
    return torch.nn.functional.cross_entropy(
        inputs, outputs
    )  # transformers doesn't output a vector


def loop_visualize_llc(dir):

    # Check which checkpoints are already plotted
    checkpoints_already_done = {os.path.splitext(file)[0] for file in os.listdir("/home/amber/vision/vision/LLC_estimation/llc_plots")}
    print(f"checkpoints_already_done: {checkpoints_already_done}")

    # Loop over checkpoints
    for filename in os.listdir(dir):
        print(f"Filename: {filename}")

        base_filename = filename.replace(".pth.tar", "")
        if base_filename in checkpoints_already_done:
            continue
        
        model = models.resnet18(weights="DEFAULT")  # initiate to empty resnet
        checkpoint = torch.load(os.path.join(dir, filename))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)

        visualize_llc(model)


def visualize_llc(model):

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

    learning_coeff_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        criterion=transformers_cross_entropy,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=0.001, num_samples=len(dataset)),
        num_chains=5,  # How many independent chains to run
        num_draws=200,  # How many samples to draw per chain
        num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=1,  # How many steps to take between each sample
        device=DEVICE,
        online=True
    )
    trace = learning_coeff_stats.pop("llc/trace")

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


if __name__ == "__main__":
    loop_over_dir = False

    # When you only want to make one plot
    filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet18-imagenet1k-v2/checkpoint_2024-03-16-01h24_epoch_39_val_200.pth.tar")
    # filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet18-imagenet1k-v3/checkpoint_2024-03-16-05h37_epoch_23_train_1200.pth.tar")
    # filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet50-imagenet1k-v1/checkpoint_2024-03-26-04h12_epoch_27_val_400.pth.tar")
    # filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-v1/checkpoint_2024-02-22-05h01_epoch_37_val_1.pth.tar")
    # filename = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-v2/checkpoint_2024-02-24-01h39_epoch_35_train_1000.pth.tar")
    
    if not loop_over_dir:
        visualize_llc(filename)
    
    # When you want to make plots for all chackpoints in a dir
    dir = os.path.join("/media/amber/Elements/2024/Athena Program - checkpoints/checkpoints-resnet18-imagenet1k-v2")
    if loop_over_dir:
        loop_visualize_llc(dir)


