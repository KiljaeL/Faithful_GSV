from experiment import run_experiment
from plot import plot_results
import os

os.makedirs("figures", exist_ok=True)

neval = 20000
niter = 30
nsave = 100
ns = [64, 128, 256]

results_64 = run_experiment(n=64, neval=neval, niter=niter, nsave=nsave)
results_128 = run_experiment(n=128, neval=neval, niter=niter, nsave=nsave)
results_256 = run_experiment(n=256, neval=neval, niter=niter, nsave=nsave)

plot_results([results_64, results_128, results_256], ns, save_path="figures/Fig2.png")
