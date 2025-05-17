import numpy as np

def U_sou(S, S_list=None, alpha=None, d=None):
    if S.dtype == bool:
        S = np.where(S)[0]
    S_set = set(S)
    val = 0.0
    for j in range(d):
        if S_list[j].issubset(S_set):
            val += alpha[j]
    return val / d

def sv_sou_true(n, S_list, alpha):
    shapley = np.zeros(n)
    for j, Sj in enumerate(S_list):
        aj = alpha[j]
        size = len(Sj)
        if size == 0:
            continue
        for i in Sj:
            shapley[i] += aj / size
    return shapley / n**2

def generate_sou_game(n, d=0, seed=42):
    np.random.seed(seed)
    if d == 0:
        d = n**2

    def compute_alpha(S, group_weight):
        if not S:
            return 0.0
        weights = [group_weight[i % 4] for i in S]
        return np.mean(weights)

    group_weight = {0: 0, 1: 1, 2: 2, 3: 3}
    S_list = [set(np.random.choice(n, size=np.random.randint(1, n), replace=False)) for _ in range(d)]
    alpha = np.array([compute_alpha(S, group_weight) for S in S_list])
    return S_list, alpha