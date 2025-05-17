from data import DGP, split_class
from funcs import U_cls, FGSV, GSV
from plot import plot_fgsv_gsv_split
import numpy as np
import seaborn as sns

np.random.seed(42)

n = 200
mu1 = np.array([-3, 0])
mu2 = np.array([3, 0])
Sigma = np.eye(2)
p = 0.5

m = 2000
thres = 20

D_train = DGP(n, mu1, mu2, Sigma, p)
D_test = DGP(n, mu1, mu2, Sigma, p)

groups, fgsv, gsv = {}, {}, {}

for num_groups in range(2, 6):
    name = str(num_groups)
    groups[name] = split_class(D_train, num_groups)
    gsv[name] = GSV(U_cls, groups[name], D_train, D_test)
    fgsv[name] = np.array([FGSV(U_cls, g, D_train, D_test, m, thres) for g in groups[name]])

sns.set_style("ticks")
pastel_colors = sns.color_palette("pastel")
plot_fgsv_gsv_split(fgsv, gsv, pastel_colors, save_path="figures/Fig1.png")