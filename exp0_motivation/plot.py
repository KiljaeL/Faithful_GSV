import numpy as np
import matplotlib.pyplot as plt

def plot_fgsv_gsv_split(fgsv, gsv, colors, save_path="figures/Fig1.png"):
    keys = sorted(fgsv.keys(), key=int)
    x = np.arange(len(keys))
    bar_width = 0.6

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, gridspec_kw={'wspace': 0.15})
    for ax, title, data in zip(axs, ['GSV', 'FGSV'], [gsv, fgsv]):
        for i, key in enumerate(keys):
            values = data[key]
            ax.bar(i, values[0], width=bar_width, color=colors[0])
            y_bottom = values[0]
            for j, val in enumerate(values[1:]):
                ax.bar(i, val, width=bar_width, bottom=y_bottom,
                       color=np.clip(np.array(colors[1]) * (1 - 0.06 * j), 0, 1))
                y_bottom += val
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=11)
        ax.set_xlabel("Number of Groups", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(y=0.2475, color='gray', linestyle='dashed', alpha=0.3)
        ax.axhline(y=0.495, color='gray', linestyle='dashed', alpha=0.3)
    axs[0].set_ylabel("GSV", fontsize=14)
    axs[1].set_ylabel("FGSV", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()