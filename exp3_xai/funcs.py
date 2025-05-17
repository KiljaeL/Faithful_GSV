import numpy as np
from itertools import combinations
from math import factorial
from tqdm import tqdm
from scipy.stats import hypergeom
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def U_reg(S, D_train, D_test):
    model = Ridge(alpha=0.01)

    X_train, y_train = D_train
    X_test, y_test = D_test

    if S is not None:
        X_S, y_S = X_train[S], y_train[S]

    if len(y_S) == 0:
        return -np.var(y_train)
    
    model.fit(X_S, y_S)
    
    return -mean_squared_error(y_test, model.predict(X_test))


def GSV(U, S0, D_train, D_test, m=None):
    n = len(S0)
    gsv = np.zeros(n)

    if n <= 10:
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for s in range(n):
                for subset in combinations(others, s):
                    weight = (factorial(s) * factorial(n - s - 1)) / factorial(n)
                    before = np.concatenate([S0[j] for j in subset]) if subset else np.array([], dtype=int)
                    after = np.concatenate((before, S0[i]))
                    gsv[i] += weight * (U(after, D_train, D_test) - U(before, D_train, D_test))
    else:
        v_empty = U(np.array([], dtype=bool), D_train, D_test)
        v_full = U(np.concatenate(S0), D_train, D_test)
        v_singleton = np.array([U(S0[i], D_train, D_test) for i in range(n)])
        v_remove = np.array([U(np.concatenate([S0[j] for j in range(n) if j != i]), D_train, D_test) for i in range(n)])

        weights = 1 / np.sqrt(np.arange(2, n - 1) * np.arange(n - 2, 1, -1))
        weights /= weights.sum()

        estimates = np.zeros((n, n - 3, 2))
        counts = np.zeros((n, n - 3, 2), dtype=int)

        for _ in tqdm(range(m), desc="One-for-All GSV"):
            s = np.random.choice(np.arange(2, n - 1), p=weights)
            subset_inds = np.random.choice(n, size=s, replace=False)
            subset = np.concatenate([S0[i] for i in subset_inds])
            v_subset = U(subset, D_train, D_test)
            idx = s - 2

            for i in range(n):
                if i in subset_inds:
                    counts[i, idx, 0] += 1
                    estimates[i, idx, 0] += (v_subset - estimates[i, idx, 0]) / counts[i, idx, 0]
                else:
                    counts[i, idx, 1] += 1
                    estimates[i, idx, 1] += (v_subset - estimates[i, idx, 1]) / counts[i, idx, 1]

        for i in range(n):
            gsv[i] = (
                estimates[i, :, 0].sum() + v_full + v_singleton[i] + (v_remove.sum() - v_remove[i]) / (n - 1)
                - estimates[i, :, 1].sum() - v_empty - v_remove[i] - (v_singleton.sum() - v_singleton[i]) / (n - 1)
            ) / n

    return gsv


def FGSV(U, S0, D_train, D_test, m=10, thres = 5):
    n = len(D_train[0])
    s0 = len(S0)
    
    linear_term = s0/n * (U(np.arange(n), D_train, D_test) - U(np.array([], dtype=bool), D_train, D_test))
    
    Ts = np.zeros(n-1)
    
    for s in tqdm(range(1, n)):
        s1_min = max(0, s0 + s - n)
        s1_max = min(s0, s)
        Ts_temp = 0
        
        if s < thres:
            for s1 in range(s1_min, s1_max + 1):
                Ts1_temp = 0
                ngrid = s1_max - s1_min + 1
                m_exact = int(np.round(m/ngrid))
                for _ in range(m_exact):
                    S1 = np.random.choice(S0, s1, replace=False)
                    S_minus_S1 = np.random.choice(np.setdiff1d(np.arange(n), S0), s-s1, replace=False)
                    S = np.concatenate((S1, S_minus_S1))
                    
                    Ts1_temp += U(S, D_train, D_test) / m_exact
                    
                Ts_temp += hypergeom.pmf(s1, n, s0, s) * (s1 - s * s0 / n) * Ts1_temp
        
            Ts[s-1] = n / (s * (n - s)) * Ts_temp
            
        else:
            Es1 = round(s0 * s / n)
            
            if Es1 + 1 <= s1_max and Es1 - 1 >= s1_min: # Central difference
                for _ in range(m):
                    S1_temp = np.random.choice(S0, Es1 + 1, replace=False)
                    S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1 + 1, replace=False)
                    
                    S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1-1)]))
                    S_lower = np.concatenate((S1_temp[0:(Es1-1)], S_minus_S1_temp))
                    
                    Ts_temp += U(S_upper, D_train, D_test) - U(S_lower, D_train, D_test)
                    
                Ts_temp /= 2 * m
            
            elif Es1 + 1 <= s1_max: # Forward difference
                for _ in range(m):
                    S1_temp = np.random.choice(S0, Es1 + 1, replace=False)
                    S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1, replace=False)
                    
                    S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1-1)]))
                    S = np.concatenate((S1_temp[0:Es1], S_minus_S1_temp))
                    
                    Ts_temp += U(S_upper, D_train, D_test) - U(S, D_train, D_test)
                    
                Ts_temp /= m
            
            elif Es1 - 1 >= s1_min: # Backward difference
                for _ in range(m):
                    S1_temp = np.random.choice(S0, Es1, replace=False)
                    S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1 + 1, replace=False)
                    
                    S = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1)]))
                    S_lower = np.concatenate((S1_temp[0:(Es1-1)], S_minus_S1_temp))
                    
                    Ts_temp += U(S, D_train, D_test) - U(S_lower, D_train, D_test)
                    
                Ts_temp /= m
        
            Ts[s-1] = Ts_temp * s0/n * (1 - s0/n)
                    
    return linear_term + np.sum(Ts)

