import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(
    gsv_sex_values, gsv_age_values, gsv_bmi_values,
    fgsv_sex_values_save, fgsv_age_values_save, fgsv_bmi_values_save,
    save_path="figures/Fig4.png"
):
    sns.set_style("ticks")

    fgsv_sex_mean = fgsv_sex_values_save.mean(axis=0)
    fgsv_sex_std = fgsv_sex_values_save.std(axis=0)
    fgsv_age_mean = fgsv_age_values_save.mean(axis=0)
    fgsv_age_std = fgsv_age_values_save.std(axis=0)
    fgsv_bmi_mean = fgsv_bmi_values_save.mean(axis=0)
    fgsv_bmi_std = fgsv_bmi_values_save.std(axis=0)

    x_labels_sex = ['Sex', 'Sex x Age', 'Sex x BMI', 'Sex x Age x BMI']
    x_labels_age = ['Age', 'Sex x Age', 'Age x BMI', 'Sex x Age x BMI']
    x_labels_bmi = ['BMI', 'Sex x BMI', 'Age x BMI', 'Sex x Age x BMI']
    x_labels = [x_labels_sex, x_labels_age, x_labels_bmi]

    sex_group_names = ['Female', 'Male']
    age_group_names = ['Young', 'Middle', 'Old']
    bmi_group_names = ['Low BMI', 'Medium BMI', 'High BMI']
    group_name_map = [sex_group_names, age_group_names, bmi_group_names,
                      sex_group_names, age_group_names, bmi_group_names]

    value_arrays = [
        gsv_sex_values, gsv_age_values, gsv_bmi_values,
        fgsv_sex_mean, fgsv_age_mean, fgsv_bmi_mean
    ]
    std_arrays = [None, None, None, fgsv_sex_std, fgsv_age_std, fgsv_bmi_std]

    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    marker_styles = ['o', '^', 's']

    for plot_idx, (ax, values, std, group_names) in enumerate(zip(axes.flat, value_arrays, std_arrays, group_name_map)):
        xlab = x_labels[plot_idx % 3]
        for i, group_label in enumerate(group_names):
            marker = marker_styles[i % len(marker_styles)]
            ax.plot(xlab, values[:, i], marker=marker, label=group_label)
            if std is not None:
                ax.fill_between(range(len(xlab)),
                                values[:, i] - std[:, i],
                                values[:, i] + std[:, i],
                                alpha=0.2)
        if plot_idx <= 2:
            ax.set_xticks(range(len(xlab)))
            ax.set_xticklabels([])
        else:
            ax.set_xticks(range(len(xlab)))
            ax.set_xticklabels(xlab, rotation=15, ha='right', fontsize=14)
        ax.set_xlim(-0.5, len(xlab) - 0.5)
        ax.margins(y=0.1)
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(True)

    axes[0, 0].set_ylabel("(Aggregated) GSV", fontsize=14)
    axes[1, 0].set_ylabel("(Aggregated) FGSV", fontsize=14)

    legend_titles = ['Sex-wise Values', 'Age-wise Values', 'BMI-wise Values']
    for col in range(3):
        handles, labels = axes[0, col].get_legend_handles_labels()
        legend = axes[0, col].legend(
            handles, labels,
            title=legend_titles[col], fontsize=12,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(labels),
            frameon=False
        )
        legend.get_title().set_fontsize(16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()