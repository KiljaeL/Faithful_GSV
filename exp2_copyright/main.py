import os
import pickle
from utils.fsrs import load_utility, compute_srs, compute_fsrs
from utils.plot import plot_stacked_generated_images, plot_stacked_scores, compose_figure

data_dir = "save/srs/utility"
gshap_dir = "save/fsrs/utility"
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

brand_filter = ["Google", "Sprite", "Starbucks", "Vodafone"]
players_grouped = ["GoogleLarge", "GoogleSmall", "SpriteLarge", "SpriteSmall", "Starbucks", "Vodafone"]
players_ungrouped = ["Google", "Sprite", "Starbucks", "Vodafone"]
n, m, h, num_gen = 120, 2, 1, 20

srs_grouped, srs_ungrouped = {}, {}
for brand in brand_filter:
    utility = load_utility(os.path.join(data_dir, f"log_likelihoods_{brand.lower()}.pkl"))
    srs_grouped[brand], _ = compute_srs(utility, players_grouped)
    srs_ungrouped[brand], _ = compute_srs(utility, players_ungrouped)

linear_grouped = {
    brand: pickle.load(open(os.path.join(gshap_dir, f"log_likelihoods_linear_{brand.lower()}.pkl"), "rb"))
    for brand in brand_filter
}
linear_ungrouped = linear_grouped.copy()

fsrs_grouped, _ = compute_fsrs(players_grouped, brand_filter, linear_grouped, gshap_dir, n=n, m=m, h=h, num_gen=num_gen)
fsrs_ungrouped, _ = compute_fsrs(players_ungrouped, brand_filter, linear_grouped, gshap_dir, n=n, m=m, h=h, num_gen=num_gen)

plot_stacked_generated_images()
plot_stacked_scores(srs_ungrouped, srs_grouped, brand_filter, players_ungrouped, ylabel="SRS", save_path=f"{fig_dir}/Fig3_topright_srs.png", ylim=0.5)
plot_stacked_scores(fsrs_ungrouped, fsrs_grouped, brand_filter, players_ungrouped, ylabel="FSRS", save_path=f"{fig_dir}/Fig3_bottomright_fsrs.png", ylim=0.43)

compose_figure(
    left_path=f"{fig_dir}/Fig3_left_panel.png",
    top_path=f"{fig_dir}/Fig3_topright_srs.png",
    bottom_path=f"{fig_dir}/Fig3_bottomright_fsrs.png",
    save_path=f"{fig_dir}/Fig3.png"
)
