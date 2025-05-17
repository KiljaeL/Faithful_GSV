import numpy as np
from scipy.stats import hypergeom

def permutation(U, n, S_list, alpha, d, neval, nsave):
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    sv_hat = np.zeros(n)
    num_eval = 0
    t = 0
    
    while num_eval < neval:
        t += 1
        perm = np.random.permutation(n)

        V_prev = U(np.array([]), S_list, alpha, d)
        num_eval += 1
        
        for j in range(n):
            S_j = perm[:j+1]
            V_curr = U(S_j, S_list, alpha, d)
            num_eval += 1
            
            if abs(V_curr - V_prev) < 1e-4:
                V_curr = V_prev 
        
            i = perm[j]
            sv_hat[i] = (t - 1) / t * sv_hat[i] + (1 / t) * (V_curr - V_prev)
            
            V_prev = V_curr

            if isave < nsave and num_eval // (neval // nsave) == isave + 1:
                sv_hat_save[:, isave] = sv_hat
                isave += 1

    return sv_hat_save

def group_testing(U, n, S_list, alpha, d, neval, nsave):
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    
    weights = 1 / np.arange(1, n + 1, dtype=np.float64)
    weights = weights + weights[::-1]
    Z = weights.sum()
    weights = weights / Z
    
    sv_hat = np.zeros(n)
    B = np.zeros((neval, n + 1))
    
    for num_eval in range(neval):
        s = np.random.choice(np.arange(1, n + 1), p=weights)
        
        S = np.random.choice(np.arange(n + 1), size=s, replace=False)
        S_prime = np.setdiff1d(S, n)
        
        u = U(S_prime, S_list, alpha, d)
        B[num_eval, S] = u
        
        
        if (num_eval + 1) // (neval // nsave) == isave + 1:
            s_hat = np.zeros(n)
            
            baseline = np.mean(B[:num_eval, -1])
            
            for i in range(n):
                s_hat[i] = np.mean(B[:num_eval, i]) - baseline
            
            sv_hat_save[:, isave] = sv_hat * Z
            isave += 1
        
    return sv_hat_save 

def complement(U, n, S_list, alpha, d, neval, nsave):
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    num_eval = 0

    results_aggregate = np.zeros((2, n + 1, n), dtype=np.float64)

    while num_eval < neval:
        idxs = np.random.permutation(n)
        j = np.random.randint(1, n + 1)
        subset = idxs[:j]
        complement = idxs[j:]

        u_1 = U(subset, S_list, alpha, d)
        u_2 = U(complement, S_list, alpha, d)
        num_eval += 2

        temp = np.zeros(n)
        temp[subset] = 1
        results_aggregate[0, j, :] += temp * (u_1 - u_2)
        results_aggregate[1, j, :] += temp

        temp = np.zeros(n)
        temp[complement] = 1
        results_aggregate[0, n - j, :] += temp * (u_2 - u_1)
        results_aggregate[1, n - j, :] += temp
        
        if (num_eval // (neval // nsave)) == isave + 1:
            sv_hat = np.zeros(n)
            for i in range(n + 1):
                for j in range(n):
                    if results_aggregate[1, i, j] > 0:
                        sv_hat[j] += results_aggregate[0, i, j] / results_aggregate[1, i, j]
            sv_hat /= n
            sv_hat_save[:, isave] = sv_hat
            isave += 1

    return sv_hat_save

def one_for_all(U, n, S_list, alpha, d, neval, nsave):
    num_eval = 0
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    
    v_empty = U(np.zeros(n, dtype=bool), S_list, alpha, d)
    v_full = U(np.ones(n, dtype=bool), S_list, alpha, d)
    
    v_singleton = np.zeros(n)
    subset = np.zeros(n, dtype=bool)
    for i in range(n):
        subset[i] = True
        v_singleton[i] = U(subset, S_list, alpha, d)
        subset[i] = False
    
    v_remove = np.zeros(n)
    subset = np.ones(n, dtype=bool)
    for i in range(n):
        subset[i] = False
        v_remove[i] = U(subset, S_list, alpha, d)
        subset[i] = True
    
    tmp = np.arange(2, n-1, dtype=np.float64)
    weights = 1 / np.sqrt(np.multiply(tmp, tmp[::-1]))
    weights = weights / weights.sum()
    
    estimates = np.zeros((n, n-3, 2))
    counts = np.zeros((n, n-3, 2), dtype=int)
    
    while num_eval < neval:
        
        s = np.random.choice(np.arange(2, n-1), p=weights)
        
        pos = np.random.choice(np.arange(n), size=s, replace=False)
        subset = np.zeros(n, dtype=bool)
        subset[pos] = True
        
        v_subset = U(subset, S_list, alpha, d)
        num_eval += 1
        
        idx = s - 2
        
        for i in range(n):
            if i in pos:
                counts[i, idx, 0] += 1
                estimates[i, idx, 0] *= (counts[i, idx, 0] - 1) / counts[i, idx, 0]
                estimates[i, idx, 0] += v_subset / counts[i, idx, 0]
            else:
                counts[i, idx, 1] += 1
                estimates[i, idx, 1] *= (counts[i, idx, 1] - 1) / counts[i, idx, 1]
                estimates[i, idx, 1] += v_subset / counts[i, idx, 1]

        if num_eval // (neval // nsave) == isave + 1:
            sv_hat = np.zeros(n)
            for i in range(n):
                sv_hat[i] += estimates[i, :, 0].sum()
                sv_hat[i] += v_full
                sv_hat[i] += v_singleton[i]
                sv_hat[i] += (v_remove.sum() - v_remove[i]) / (n-1)
                
                sv_hat[i] -= estimates[i, :, 1].sum()
                sv_hat[i] -= v_empty
                sv_hat[i] -= v_remove[i]
                sv_hat[i] -= (v_singleton.sum() - v_singleton[i]) / (n-1)
                
            sv_hat /= n
            sv_hat_save[:, isave] = sv_hat
            isave += 1
    
    return sv_hat_save

def kernel_shap(U, n, S_list, alpha, d, neval, nsave, var_red = True):
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    
    if var_red:
        neval /= 2
    
    u_empty = U(np.zeros(n, dtype=bool), S_list, alpha, d)
    u_full = U(np.ones(n, dtype=bool), S_list, alpha, d)

    s_range = np.arange(1, n)
    weight_kernel = 1.0 / (s_range * (n - s_range))
    prob_s = weight_kernel / weight_kernel.sum()

    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    num_eval = 0
    while num_eval < neval:
        num_eval += 1
        s = np.random.choice(s_range, p=prob_s)
        S = np.random.choice(n, size=s, replace=False)

        z = np.zeros(n, dtype=bool)
        z[S] = True


        if var_red:
            u_S = U(z, S_list, alpha, d)
            u_Sc = U(~z, S_list, alpha, d)
            z = z.astype(float)
            z_bar = 1-z
            
            A_sample = (np.outer(z, z) + np.outer(z_bar, z_bar))/2
            b_sample = (z * (u_S - u_empty) + z_bar * (u_Sc - u_empty))/2
        else:
            u_S = U(z, S_list, alpha, d)
            z = z.astype(float)

            A_sample = np.outer(z, z)
            b_sample = z * (u_S - u_empty)
            
        
        A += (A_sample - A)/num_eval
        b += (b_sample - b)/num_eval

        if num_eval // (neval // nsave) == isave + 1:
            Ainv = np.linalg.pinv(A)
            sv_hat = Ainv @ (b - (np.ones(n) @ Ainv @ b - u_full + u_empty) / (np.ones(n) @ Ainv @ np.ones(n)) * np.ones(n))
            sv_hat_save[:, isave] = sv_hat
            isave += 1
            
    return sv_hat_save

def unbiased_kernel_shap(U, n, S_list, alpha, d, neval, nsave):
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    
    def compute_kernelshap_A(n):
        A = np.full((n, n), fill_value=np.nan, dtype=np.float64)
        
        denom_terms = [1 / (k * (n - k)) for k in range(1, n)]
        
        np.fill_diagonal(A, 0.5)
        
        for i in range(n):
            for j in range(i + 1, n):
                num = sum((k - 1) / (n - k) for k in range(2, n))
                Aij = num / (n * (n - 1) * sum(denom_terms))
                A[i, j] = A[j, i] = Aij
        return A


    A = compute_kernelshap_A(n)
    Ainv = np.linalg.pinv(A)
    
    u_empty = U(np.zeros(n, dtype=bool), S_list, alpha, d)
    u_full = U(np.ones(n, dtype=bool), S_list, alpha, d)

    s_range = np.arange(1, n)
    kernel_weights = 1.0 / (s_range * (n - s_range))
    prob_s = kernel_weights / kernel_weights.sum()

    b = np.zeros(n, dtype=np.float64)

    num_eval = 0
    while num_eval < neval:
        num_eval += 1
        s = np.random.choice(s_range, p=prob_s)
        S = np.random.choice(n, size=s, replace=False)

        z = np.zeros(n, dtype=bool)
        z[S] = True

        u_S = U(z, S_list, alpha, d)
        z = z.astype(float)

        b_sample = z * u_S - u_empty/2
        b += (b_sample - b)/num_eval

        if num_eval // (neval // nsave) == isave + 1:
            sv_hat = Ainv @ (b - (np.ones(n) @ Ainv @ b - u_full + u_empty) / (np.ones(n) @ Ainv @ np.ones(n)) * np.ones(n))
            sv_hat_save[:, isave] = sv_hat
            isave += 1

    return sv_hat_save

def leverage_shap(U, n, S_list, alpha, d, neval, nsave, var_red = True):
    isave = 0
    sv_hat_save = np.zeros((n, nsave))
    
    if var_red:
        neval /= 2
    
    u_empty = U(np.zeros(n, dtype=bool), S_list, alpha, d)
    u_full = U(np.ones(n, dtype=bool), S_list, alpha, d)

    s_range = np.arange(1, n)
    weight_kernel = 1.0 / (s_range * (n - s_range))
    prob_s = weight_kernel / weight_kernel.sum()

    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    num_eval = 0
    while num_eval < neval:
        num_eval += 1
        s = np.random.choice(s_range)
        S = np.random.choice(n, size=s, replace=False)

        z = np.zeros(n, dtype=bool)
        z[S] = True


        if var_red:
            u_S = U(z, S_list, alpha, d)
            u_Sc = U(~z, S_list, alpha, d)
            z = z.astype(float)
            z_bar = 1-z
            
            A_sample = np.sqrt(prob_s[s-1])*(np.outer(z, z) + np.outer(z_bar, z_bar))/2
            b_sample = np.sqrt(prob_s[s-1])*(z * (u_S - u_empty) + z_bar * (u_Sc - u_empty))/2
        else:
            u_S = U(z, S_list, alpha, d)
            z = z.astype(float)

            A_sample = np.outer(z, z)
            b_sample = z * (u_S - u_empty)
            
        
        A += (A_sample - A)/num_eval
        b += (b_sample - b)/num_eval

        if num_eval // (neval // nsave) == isave + 1:
            Ainv = np.linalg.pinv(A)
            sv_hat = Ainv @ (b - (np.ones(n) @ Ainv @ b - u_full + u_empty) / (np.ones(n) @ Ainv @ np.ones(n)) * np.ones(n))
            sv_hat_save[:, isave] = sv_hat
            isave += 1
            
    return sv_hat_save

def FGSV(U, n, S_list, alpha, d, S0, neval, nsave, thres = 5):
    s0 = len(S0)
    isave = 0

    linear_term = s0/n * (U(np.arange(n), S_list, alpha, d) - U(np.array([]), S_list, alpha, d))
    num_eval = 2
    
    Ts = np.zeros(n-1)
    counts = np.zeros(n-1)
    fgsv_hat_save = np.zeros(nsave)

    
    while num_eval < neval:
        s = np.random.randint(1, n)
        s1_min = max(0, s0 + s - n)
        s1_max = min(s0, s)
        Es1 = round(s0 * s / n)
            
        if s < thres:
            Ts_temp = 0
            for s1 in range(s1_min, s1_max + 1):
                S1 = np.random.choice(S0, s1, replace=False)
                S_minus_S1 = np.random.choice(np.setdiff1d(np.arange(n), S0), s-s1, replace=False)
                S = np.concatenate((S1, S_minus_S1))
                Ts_temp += hypergeom.pmf(s1, n, s0, s) * (s1 - s * s0 / n) * U(S, S_list, alpha, d)
                num_eval += 1
                
            Ts[s-1] = (counts[s-1] / (counts[s-1] + 1)) * Ts[s-1] + Ts_temp / (counts[s-1] + 1)
            counts[s-1] += 1
            
        else:
            if Es1 + 1 <= s1_max and Es1 - 1 >= s1_min:
                central_ind = True
                S1_temp = np.random.choice(S0, Es1 + 1, replace=False)
                S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1 + 1, replace=False)
                S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1-1)]))
                S_lower = np.concatenate((S1_temp[0:(Es1-1)], S_minus_S1_temp))
                
            elif Es1 + 1 <= s1_max:
                central_ind = False
                S1_temp = np.random.choice(S0, Es1 + 1, replace=False)
                S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1, replace=False)
                S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1-1)]))
                S_lower = np.concatenate((S1_temp[0:Es1], S_minus_S1_temp))
                
            elif Es1 - 1 >= s1_min:
                central_ind = False
                S1_temp = np.random.choice(S0, Es1, replace=False)
                S_minus_S1_temp = np.random.choice(np.setdiff1d(np.arange(n), S0), s - Es1 + 1, replace=False)
                S_upper = np.concatenate((S1_temp, S_minus_S1_temp[0:(s-Es1)]))
                S_lower = np.concatenate((S1_temp[0:(Es1-1)], S_minus_S1_temp))
                
            Ts[s-1] = (counts[s-1] / (counts[s-1] + 1)) * Ts[s-1] + s0/n * (1-s0/n) * (U(S_upper, S_list, alpha, d) - U(S_lower, S_list, alpha, d)) / (central_ind + 1) / (counts[s-1] + 1)
            counts[s-1] += 1
            num_eval += 2
            central_ind = False
        
        if num_eval // (neval // nsave) == isave + 1:
            fgsv_hat_save[isave] = linear_term + np.sum(Ts)
            isave += 1

    return fgsv_hat_save