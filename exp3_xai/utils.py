import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from itertools import product
import warnings

def get_data(num_group_age, num_group_bmi, ind_array=False):
    data = load_diabetes()
    y = pd.Series(data.target)
    y = (y - y.mean()) / y.std()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    X['sex'] = (X['sex'] == X['sex'][0]).astype(int)
    X['age_group'] = pd.qcut(X['age'], q=num_group_age, labels=False)
    X['bmi_group'] = pd.qcut(X['bmi'], q=num_group_bmi, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=350, random_state=42)

    if ind_array:
        return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))
    else:
        return (X_train, y_train), (X_test, y_test)

def get_S0(sex=True, age=True, bmi=True, num_group_sex=2, num_group_age=3, num_group_bmi=3):
    D_train, _ = get_data(num_group_age, num_group_bmi)
    X_train, _ = D_train

    group_vars = []
    group_sizes = {}

    if sex:
        group_vars.append("sex")
        group_sizes["sex"] = num_group_sex
    if age:
        group_vars.append("age_group")
        group_sizes["age_group"] = num_group_age
    if bmi:
        group_vars.append("bmi_group")
        group_sizes["bmi_group"] = num_group_bmi

    value_ranges = [range(group_sizes[var]) for var in group_vars]
    S0 = []
    for combination in product(*value_ranges):
        cond = np.ones(len(X_train), dtype=bool)
        for var, val in zip(group_vars, combination):
            cond &= (X_train[var] == val)
        S0.append(np.where(cond)[0])
    return S0

def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

def aggregate_values(gsv_sex, gsv_age, gsv_bmi, gsv_sex_age, gsv_sex_bmi, gsv_age_bmi, gsv_sex_age_bmi):
    sex_values = np.stack([
        gsv_sex,
        gsv_sex_age.sum(axis=1),
        gsv_sex_bmi.sum(axis=1),
        gsv_sex_age_bmi.sum(axis=(1, 2))
    ], axis=0)

    age_values = np.stack([
        gsv_age,
        gsv_sex_age.sum(axis=0),
        gsv_age_bmi.sum(axis=1),
        gsv_sex_age_bmi.sum(axis=(0, 2))
    ], axis=0)

    bmi_values = np.stack([
        gsv_bmi,
        gsv_sex_bmi.sum(axis=0),
        gsv_age_bmi.sum(axis=0),
        gsv_sex_age_bmi.sum(axis=(0, 1))
    ], axis=0)

    return sex_values, age_values, bmi_values