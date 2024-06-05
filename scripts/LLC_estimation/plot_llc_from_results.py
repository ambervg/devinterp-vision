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
        print(filename)
        # We only want to check the json files in this dir
        if not filename.endswith(".json"):
            continue
        
        # read file
        with open(os.path.join(dir, filename), 'r') as file:
            result = json.load(file)
        
        # get the mean from the last 100 values, determined based on visual inspection
        means = result["llc/means"]
        mean_last_100 = np.mean(means[-100:])
        mean_dict[filename] = mean_last_100

        print(mean_last_100)
    
    print(f"{len(mean_dict)}")
    sorted_dict = {key: mean_dict[key] for key in sorted(mean_dict)}
    print(sorted_dict.keys())
    print(f"Amount of checkpoints: {len(sorted_dict.values())}")

    # # stds = result["llc/stds"]
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
    # # axs.fill_between(
    # #     sgld_steps, means - stds, means + stds, color=llc_color, alpha=0.3, zorder=2
    # # )

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
    plt.show()

if __name__ == "__main__":
    dir = "/media/amber/Elements/2024/Athena Program - checkpoints/llc_results-resnet18-imagenet1k-v2_epsilon001_gamma100_only_means"
    plot_graph(dir, title="")
