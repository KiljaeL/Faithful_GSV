import numpy as np
import time
from tqdm import tqdm
from estimators import FGSV, permutation, one_for_all, group_testing, complement, kernel_shap, unbiased_kernel_shap, leverage_shap
from utils import U_sou, sv_sou_true, generate_sou_game


def run_experiment(n, neval, niter, nsave):
    d = n**2
    S_list, alpha = generate_sou_game(n, d)

    sv_true = sv_sou_true(n, S_list, alpha)
    S0_list = [[i for i in range(n) if i % 4 == j] for j in range(4)]
    sv_true_sums = [sv_true[S0].sum() for S0 in S0_list]

    methods = ['perm', 'ofa', 'gt', 'complement', 'kernel', 'unbiased_kernel', 'leverage']
    method_funcs = {
        'perm': permutation,
        'ofa': one_for_all,
        'gt': group_testing,
        'complement': complement,
        'kernel': kernel_shap,
        'unbiased_kernel': unbiased_kernel_shap,
        'leverage': leverage_shap
    }

    gsv_group_aucc = np.zeros((niter, len(S0_list)))
    gsv_indiv_aucc = np.zeros(niter)
    gsv_time = np.zeros(niter)

    method_group_aucc = {m: np.zeros((niter, len(S0_list))) for m in methods}
    method_indiv_aucc = {m: np.zeros(niter) for m in methods}
    method_time = {m: np.zeros(niter) for m in methods}

    for iter in tqdm(range(niter)):
        start_time = time.time()
        gsv_estimates = np.zeros((len(S0_list), nsave))
        indiv_estimates = np.zeros((n, nsave))

        for i, S0 in enumerate(S0_list):
            gsv_estimates[i, :] = FGSV(U_sou, n, S_list, alpha, d, S0, neval // 4, nsave, thres=10)
            gsv_group_aucc[iter, i] = np.mean(np.abs(gsv_estimates[i, :] - sv_true_sums[i]) / abs(sv_true_sums[i]))
            indiv_estimates[S0, :] = gsv_estimates[i, :] / len(S0)

        gsv_indiv_aucc[iter] = np.mean([
            np.sqrt(np.sum((indiv_estimates[:, isave] - sv_true) ** 2)) / np.sqrt(np.sum(sv_true ** 2))
            for isave in range(nsave)
        ])
        gsv_time[iter] = time.time() - start_time

        for m in methods:
            start_time = time.time()
            sv_estimates = method_funcs[m](U_sou, n, S_list, alpha, d, neval, nsave)
            for i, S0 in enumerate(S0_list):
                method_group_aucc[m][iter, i] = np.mean(np.abs(sv_estimates[S0].sum(axis=0) - sv_true_sums[i]) / abs(sv_true_sums[i]))
            method_indiv_aucc[m][iter] = np.mean([
                np.sqrt(np.sum((sv_estimates[:, isave] - sv_true) ** 2)) / np.sqrt(np.sum(sv_true ** 2))
                for isave in range(nsave)
            ])
            method_time[m][iter] = time.time() - start_time

    results_dict = {
        'sv_true': sv_true,
        'gsv_aucc': gsv_group_aucc,
        'gsv_sv_aucc': gsv_indiv_aucc,
        'gsv_time': gsv_time,
    }
    for m in methods:
        results_dict[f'{m}_aucc'] = method_group_aucc[m]
        results_dict[f'{m}_sv_aucc'] = method_indiv_aucc[m]
        results_dict[f'{m}_time'] = method_time[m]

    return results_dict