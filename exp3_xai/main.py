import numpy as np
from utils import get_data, get_S0, suppress_warnings, aggregate_values
from funcs import U_reg, GSV, FGSV
from plot import plot_results
from joblib import Parallel, delayed

np.random.seed(42)

num_rep = 30

num_group_sex = 2
num_group_age = 3
num_group_bmi = 3

S0_sex = get_S0(sex=True, age=False, bmi=False, num_group_age=num_group_age, num_group_bmi=num_group_bmi)
S0_age = get_S0(sex=False, age=True, bmi=False, num_group_age=num_group_age, num_group_bmi=num_group_bmi)
S0_bmi = get_S0(sex=False, age=False, bmi=True, num_group_age=num_group_age, num_group_bmi=num_group_bmi)
S0_sex_age = get_S0(sex=True, age=True, bmi=False, num_group_age=num_group_age, num_group_bmi=num_group_bmi)
S0_sex_bmi = get_S0(sex=True, age=False, bmi=True, num_group_age=num_group_age, num_group_bmi=num_group_bmi)
S0_age_bmi = get_S0(sex=False, age=True, bmi=True, num_group_age=num_group_age, num_group_bmi=num_group_bmi)
S0_sex_age_bmi = get_S0(sex=True, age=True, bmi=True, num_group_age=num_group_age, num_group_bmi=num_group_bmi)

S0_list = [S0_sex, S0_age, S0_bmi, S0_sex_age, S0_sex_bmi, S0_age_bmi, S0_sex_age_bmi]
names = ['sex', 'age', 'bmi', 'sex_age', 'sex_bmi', 'age_bmi', 'sex_age_bmi']
lengths = list(map(len, S0_list))
cumulative = np.cumsum([0] + lengths)
S0_all = sum(S0_list, [])

D_train, D_test = get_data(num_group_age, num_group_bmi, ind_array=True)

gsv_sex = GSV(U_reg, S0_sex, D_train, D_test)
gsv_age = GSV(U_reg, S0_age, D_train, D_test)
gsv_bmi = GSV(U_reg, S0_bmi, D_train, D_test)
gsv_sex_age = GSV(U_reg, S0_sex_age, D_train, D_test).reshape(num_group_sex, num_group_age)
gsv_sex_bmi = GSV(U_reg, S0_sex_bmi, D_train, D_test).reshape(num_group_sex, num_group_bmi)
gsv_age_bmi = GSV(U_reg, S0_age_bmi, D_train, D_test).reshape(num_group_age, num_group_bmi)
gsv_sex_age_bmi = GSV(U_reg, S0_sex_age_bmi, D_train, D_test, m=5).reshape(num_group_sex, num_group_age, num_group_bmi)

gsv_sex_values, gsv_age_values, gsv_bmi_values = aggregate_values(
    gsv_sex, gsv_age, gsv_bmi, gsv_sex_age, gsv_sex_bmi, gsv_age_bmi, gsv_sex_age_bmi
)

fgsv_sex_values_save = np.zeros((num_rep, 4, num_group_sex))
fgsv_age_values_save = np.zeros((num_rep, 4, num_group_age))
fgsv_bmi_values_save = np.zeros((num_rep, 4, num_group_bmi))

for seed in range(num_rep):
    np.random.seed(seed)

    fgsv_all = Parallel(n_jobs=-1)(
        delayed(suppress_warnings(FGSV))(U_reg, S0_curr, D_train, D_test, m=1000, thres=35)
        for S0_curr in S0_all
    )

    fgsv_parts = [
        np.array(fgsv_all[cumulative[i]:cumulative[i+1]])
        for i in range(len(S0_list))
    ]

    fgsv_sex, fgsv_age, fgsv_bmi, fgsv_sex_age, fgsv_sex_bmi, fgsv_age_bmi, fgsv_sex_age_bmi = tuple(fgsv_parts)
    fgsv_sex_age = fgsv_sex_age.reshape(num_group_sex, num_group_age)
    fgsv_sex_bmi = fgsv_sex_bmi.reshape(num_group_sex, num_group_bmi)
    fgsv_age_bmi = fgsv_age_bmi.reshape(num_group_age, num_group_bmi)
    fgsv_sex_age_bmi = fgsv_sex_age_bmi.reshape(num_group_sex, num_group_age, num_group_bmi)

    fgsv_sex_values, fgsv_age_values, fgsv_bmi_values = aggregate_values(
        fgsv_sex, fgsv_age, fgsv_bmi, fgsv_sex_age, fgsv_sex_bmi, fgsv_age_bmi, fgsv_sex_age_bmi
    )

    fgsv_sex_values_save[seed] = fgsv_sex_values
    fgsv_age_values_save[seed] = fgsv_age_values
    fgsv_bmi_values_save[seed] = fgsv_bmi_values


plot_results(gsv_sex_values, gsv_age_values, gsv_bmi_values,
                fgsv_sex_values_save, fgsv_age_values_save, fgsv_bmi_values_save,
                save_path="figures/Fig4.png")