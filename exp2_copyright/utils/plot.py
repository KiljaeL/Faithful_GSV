import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

def plot_stacked_generated_images():
    brand_images = {
        "Google": [5, 10, 11, 15],
        "Sprite": [4, 7, 15, 18],
        "Starbucks": [5, 6, 10, 13],
        "Vodafone": [2, 9, 16, 19]
    }
    brands = list(brand_images.keys())
    fig, axs = plt.subplots(4, 4, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.0, hspace=0.05)

    for row, brand in enumerate(brands):
        nums = brand_images[brand]
        for col, num in enumerate(nums):
            ax = axs[row, col]
            path = f"save/generated/{brand.lower()}/a_logo_by_{brand.lower()}_{num}.png"
            img = Image.open(path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_position(ax.get_position().shrunk(0.97, 1))

    for i, brand in enumerate(brands):
        fig.text(0.5, 0.87 - i * 0.1965, f"Prompt: \"A logo by {brand}\"", ha='center', va='bottom', fontsize=24)
    fig.text(0.5, 0.05, " ", ha='center', va='bottom', fontsize=24)
    plt.savefig("figures/Fig3_left_panel.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_stacked_scores(grouped_dict, split_dict, brands, base_brands, ylabel, save_path, ylim):
    pastel_colors = sns.color_palette("pastel")
    base_color = np.array(pastel_colors[1])
    shade_color = tuple(np.clip(base_color * 0.85, 0, 1))
    bar_width = 0.35
    fig, axs = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    for idx, prompt in enumerate(brands):
        x = np.arange(len(base_brands))
        grouped_vals = [grouped_dict[prompt][b] for b in base_brands]
        axs[idx].bar(x - bar_width/2, grouped_vals, bar_width, color=pastel_colors[0], label='Unsplit')
        for i, b in enumerate(base_brands):
            large = split_dict[prompt].get(f"{b}Large", 0)
            small = split_dict[prompt].get(f"{b}Small", split_dict[prompt].get(b, 0))
            axs[idx].bar(x[i] + bar_width/2, small, bar_width, color=base_color)
            axs[idx].bar(x[i] + bar_width/2, large, bar_width, bottom=small, color=shade_color)
            if large > 0 and small > 0:
                axs[idx].hlines(y=small, xmin=x[i], xmax=x[i] + bar_width, color='gray', linestyle='dashed', linewidth=1, alpha=0.7)
        axs[idx].set_title(f"\"A logo by {prompt}\"", fontsize=17)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(base_brands, rotation=25, fontsize=17)
        axs[idx].tick_params(axis='y', labelsize=17)
        axs[idx].set_ylim(0, ylim)
    axs[0].set_ylabel(ylabel, fontsize=17)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compose_figure(left_path, top_path, bottom_path, save_path):
    fig = plt.figure(figsize=(26, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 4], height_ratios=[1, 1], wspace=0, hspace=0)
    ax_left = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_left.imshow(mpimg.imread(left_path))
    ax_top.imshow(mpimg.imread(top_path))
    ax_bottom.imshow(mpimg.imread(bottom_path))
    for ax in [ax_left, ax_top, ax_bottom]:
        ax.axis('off')
    top_pos = ax_top.get_position()
    bottom_pos = ax_bottom.get_position()
    ax_bottom.set_position([bottom_pos.x0, top_pos.y0 - top_pos.height, bottom_pos.width, top_pos.height])
    ax_top.set_position([top_pos.x0, top_pos.y0, top_pos.width, top_pos.height])
    fig.text(0.21, 0.11, "(a)", fontsize=20, fontweight='bold')
    fig.text(0.61, 0.495, "(b)", fontsize=20, fontweight='bold')
    fig.text(0.61, 0.11, "(c)", fontsize=20, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

