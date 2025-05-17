import os
import pickle
import numpy as np
from itertools import combinations

def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def load_utility(file_path):
    with open(file_path, "rb") as f:
        utility = pickle.load(f)
    return {k: np.array(v) for k, v in utility.items()}

def compute_srs(utility, players):
    shapley_values = {}
    n = len(players)

    def get_key(subset):
        sorted_subset = sorted(subset)
        brands_with_both = {
            player.replace("Large", "").replace("Small", "")
            for player in sorted_subset
            if (player.replace("Large", "Small") in sorted_subset)
        }
        processed = []
        for p in sorted_subset:
            brand = p.replace("Large", "").replace("Small", "")
            if brand in brands_with_both and ("Large" in p or "Small" in p):
                if brand not in processed:
                    processed.append(brand)
            else:
                processed.append(p)
        base_all = {p.replace("Large", "").replace("Small", "") for p in players}
        base_subset = {p.replace("Large", "").replace("Small", "") for p in processed}
        return "full" if base_all == base_subset else "_".join(processed) if processed else "null"

    for player in players:
        SV = 0
        others = [p for p in players if p != player]
        for size in range(len(others) + 1):
            for coalition in combinations(others, size):
                coef = (factorial(size) * factorial(n - size - 1)) / factorial(n)
                c_with = list(coalition) + [player]
                marginal = utility[get_key(c_with)] - utility[get_key(coalition)]
                SV += coef * marginal
        SV = max(0, np.mean(SV))
        shapley_values[player] = SV

    total = sum(shapley_values.values())
    srs = {k: v / total for k, v in shapley_values.items()}
    return srs, shapley_values

def compute_fsrs(players, brand_filter, linear_dict, correction_dir, n=120, m=2, h=1, num_gen=20):
    correction_dict, fgsv, fsrs = {}, {}, {}
    for prompt in brand_filter:
        fgsv[prompt] = {}
        correction_dict[prompt] = {}
        for player in players:
            s0 = 20 if "Large" in player else 10 if "Small" in player else 30
            file_path = os.path.join(correction_dir, f"log_likelihoods_{player.lower()}_{prompt.lower()}.pkl")
            utility = load_utility(file_path)
            Ts = np.zeros((n - 1, num_gen))
            for s in range(1, n):
                Es1 = round(s0 * s / n)
                s1_min, s1_max = max(0, s0 + s - n), min(s0, s)
                ind_central = Es1 + h <= s1_max and Es1 - h >= s1_min
                for iter in range(m):
                    Ts[s - 1] += s0/n * (1 - s0/n) * (utility["upper"][s - 1, iter, :] - utility["lower"][s - 1, iter, :]) / (m * (ind_central + 1))
            correction_dict[prompt][player] = Ts

        for player in players:
            s0 = 20 if "Large" in player else 10 if "Small" in player else 30
            GSV = s0 / n * np.mean(linear_dict[prompt]["full"] - linear_dict[prompt]["null"])
            GSV += np.mean(correction_dict[prompt][player].sum(axis=0))
            fgsv[prompt][player] = max(0, GSV)

        fsrs[prompt] = {k: v / sum(fgsv[prompt].values()) for k, v in fgsv[prompt].items()}
    return fsrs, fgsv
