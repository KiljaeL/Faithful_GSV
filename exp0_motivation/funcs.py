import numpy as np
from scipy.stats import hypergeom
import itertools
from math import factorial
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def U_cls(S, D_train, D_test):
    model = LogisticRegression()
    
    X_train, y_train = D_train
    X_test, y_test = D_test
    n_class = len(set(y_train))
    
    if S is not None:
        X_S, y_S = X_train[S], y_train[S]
            
    if len(set(y_S)) < 2:
        return 1/n_class
   
    model.fit(X_S, y_S)
    
    return accuracy_score(y_test, model.predict(X_test))

def GSV(U, groups, D_train, D_test):
    n_groups = len(groups)
    gsv = np.zeros(n_groups)
    
    for i in range(n_groups):
        other_group_indices = list(range(n_groups))
        other_group_indices.remove(i)
        
        for subset_size in range(n_groups):
            for subset in itertools.combinations(other_group_indices, subset_size):

                weight = (factorial(subset_size) * factorial(n_groups - subset_size - 1)) / factorial(n_groups)
                
                before_group = [groups[j] for j in subset]
                if len(before_group) == 0:
                    before_group = np.array([])
                    value_before = 0.5
                else:
                    if len(before_group) == 1:
                        before_group = before_group[0]
                    else:
                        before_group = np.concatenate(before_group)
                    value_before = U(before_group, D_train, D_test)
                after_group = np.concatenate((before_group, groups[i])).astype(int)
                value_after = U(after_group, D_train, D_test)

                gsv[i] += weight * (value_after - value_before)

    return gsv

def FGSV(U, S0, D_train, D_test, m, thres):
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
            
            if Es1 + 1 <= s1_max and Es1 - 1 >= s1_min:
                for _ in range(m):
                    S1_temp = np.random.choice(S0, Es1 + 1, replace=False)
                    S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1 + 1, replace=False)
                    
                    S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1-1)]))
                    S_lower = np.concatenate((S1_temp[0:(Es1-1)], S_minus_S1_temp))
                    
                    Ts_temp += U(S_upper, D_train, D_test) - U(S_lower, D_train, D_test)
                    
                Ts_temp /= 2 * m
            
            elif Es1 + 1 <= s1_max:
                for _ in range(m):
                    S1_temp = np.random.choice(S0, Es1 + 1, replace=False)
                    S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1, replace=False)
                    
                    S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1-1)]))
                    S = np.concatenate((S1_temp[0:Es1], S_minus_S1_temp))
                    
                    Ts_temp += U(S_upper, D_train, D_test) - U(S, D_train, D_test)
                    
                Ts_temp /= m
            
            elif Es1 - 1 >= s1_min:
                for _ in range(m):
                    S1_temp = np.random.choice(S0, Es1, replace=False)
                    S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1 + 1, replace=False)
                    
                    S = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1)]))
                    S_lower = np.concatenate((S1_temp[0:(Es1-1)], S_minus_S1_temp))
                    
                    Ts_temp += U(S, D_train, D_test) - U(S_lower, D_train, D_test)
                    
                Ts_temp /= m
        
            Ts[s-1] = Ts_temp * s0/n * (1 - s0/n)
                    
    return linear_term + np.sum(Ts)