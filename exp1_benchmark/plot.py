import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.patches import Patch

def plot_results(results_list, ns, save_path=None):

    sns.set_style("ticks")
    method_labels = ['Permutation', 'Group Testing', 'Complement', 'One-for-All',
                     'KernelSHAP', 'Unbiased KernelSHAP', 'LeverageSHAP', 'FGSV']
    group_labels = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$"]
    colors = sns.color_palette("pastel", n_colors=len(method_labels))
    width = 0.12
    x = np.arange(len(group_labels))

    fig, axes = plt.subplots(2, 3, figsize=(21, 8), sharex=False)

    for col_idx, (ax, results, n_val) in enumerate(zip(axes[0], results_list, ns)):
        # AUCC data
        group_auccs = [
            [results[f"{key}_aucc"][:, i] for i in range(len(group_labels))]
            for key in ['perm', 'gt', 'complement', 'ofa', 'kernel', 'unbiased_kernel', 'leverage', 'gsv']
        ]

        for method_idx in range(len(method_labels)):
            box_data = [group_auccs[method_idx][i] for i in range(len(group_labels))]
            positions = x + (method_idx - 3.5) * width

            ax.boxplot(
                box_data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor=colors[method_idx], color=colors[method_idx]),
                capprops=dict(color=colors[method_idx]),
                whiskerprops=dict(color=colors[method_idx]),
                flierprops=dict(marker='o', markersize=4, alpha=0.3, markeredgecolor=colors[method_idx]),
                medianprops=dict(color='black', linewidth=1, alpha=0.75)
            )

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=20)
        ax.set_title(f"$n = {n_val}$", fontsize=22)
        ax.set_xlim(-0.5, len(group_labels) - 0.5)
        ax.set_yscale("log")
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    axes[0, 0].set_ylabel("AUCC", fontsize=22)

    for ax, results in zip(axes[1], results_list):
        times = [
            results[f"{key}_time"].mean()
            for key in ['perm', 'gt', 'complement', 'ofa', 'kernel', 'unbiased_kernel', 'leverage', 'gsv']
        ]
        ax.bar(np.arange(len(method_labels)), times, color=colors)
        ax.set_xticks(np.arange(len(method_labels)))
        ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.set_ylim(0, max(0.1, max(times) * 1.2))

    axes[1, 0].set_ylabel("Time (s)", fontsize=22)

    legend_handles = [Patch(facecolor=colors[i], label=method_labels[i]) for i in range(len(method_labels))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()