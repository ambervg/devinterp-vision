import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_graph(dir, title=""):
    llc_color = "teal"
    fig, axs = plt.subplots(1, 1)
    mean_dict = {}

    # Loop over results
    for filename in os.listdir(dir):
        # We only want to check the json files in this dir
        if not filename.endswith(".json"):
            continue
        
        # read file
        with open(os.path.join(dir, filename), 'r') as file:
            result = json.load(file)

        # Get theavg of the final values of moving_avg 
        final_values = [value[-1] for value in result['llc/moving_avg']]
        average_final_value = np.mean(final_values)
        mean_dict[filename] = average_final_value
    
    print(f"{len(mean_dict)}")

    # Sort chronologically so it's in the order of training step
    sorted_dict = {key: mean_dict[key] for key in sorted(mean_dict)}
    print(f"Amount of checkpoints: {len(sorted_dict.values())}")

    stds = result["llc/stds"]
    sgld_steps = list(range(len(mean_dict)))
    axs.plot(
        sgld_steps,
        sorted_dict.values(),
        color=llc_color,
        linestyle="--",
        linewidth=2,
        label=f"llc",
        zorder=3,
    )
    # axs.fill_between(sgld_steps, means - stds, means + stds, color=llc_color, alpha=0.3, zorder=2)

    # Set x-axis to logarithmic scale
    # axs.set_xscale("log")

    # center zero, assume zero is in the range of both y axes already
    y_min, y_max = axs.get_ylim()
    y_min = 90
    y_max = 110
    axs.set_ylim(y_min, y_max)
    axs.set_xlabel("Training step")
    axs.set_ylabel("llc")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(os.path.join("results", "llc_plots", "llc_plot_over_checkpoints.png"))

    # Show the plot
    plt.show()


if __name__ == "__main__":
    dir = "results/llc_estimations"
    plot_graph(dir, title="LLC plot over checkpoints")
