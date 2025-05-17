import numpy as np

def DGP(n, mu1, mu2, Sigma, p):
    y = np.random.choice([0, 1], size=n, p=[p, 1-p])
    X = np.array([
        np.random.multivariate_normal(mu1, Sigma) if yi == 0
        else np.random.multivariate_normal(mu2, Sigma) for yi in y
    ])
    return X, y

def split_class(D_train, num_groups=2, seed=1):
    X, y = D_train
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    proportions = np.array([0.5])
    rest = 0.5 / (num_groups - 1)
    proportions = np.concatenate((proportions, np.ones(num_groups - 1) * rest))
    sizes = (proportions * len(X)).astype(int)
    sizes[-1] += len(X) - sizes.sum()

    groups, start = [], 0
    for size in sizes:
        groups.append(idx[start:start + size])
        start += size
    return groups