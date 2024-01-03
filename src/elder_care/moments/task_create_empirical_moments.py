"""Create empirical moments for MSM estimation."""
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import pandas as pd
from elder_care.config import BLD
from pytask import Product

MIN_AGE = 50
MAX_AGE = 65

AGE_50 = 50
AGE_53 = 53
AGE_56 = 56
AGE_59 = 59
AGE_62 = 62

AGE_55 = 55
AGE_60 = 60
AGE_65 = 65

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_moments(
    path_to_estimation_data: Path = BLD / "data" / "estimation_data.csv",
    path_to_design_weight: Path = BLD / "data" / "estimation_data_design_weight.csv",
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_ind_weight: Path = BLD / "data" / "estimation_data_ind_weight.csv",
    path_to_parent_child_data: Path = BLD / "data" / "parent_child_data.csv",
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "empirical_moments.csv",
) -> None:
    dat_un = pd.read_csv(path_to_estimation_data)
    dat_design_weight = pd.read_csv(path_to_design_weight)
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    dat_ind_weight = pd.read_csv(path_to_ind_weight)

    parent = pd.read_csv(path_to_parent_child_data)

    # _observed_any_care = dat["any_care"].mean()
    # _observed_no_care = 1 - _observed_any_care

    # _expected_any_care = 0.2
    # _expected_no_care = 1 - _expected_any_care

    # dat["weight"] = _expected_no_care / _observed_no_care
    # dat.loc[dat["any_care"] == True, "weight"] = _expected_any_care / _observed_any_care

    # dat_weighted = multiply_rows_with_weight(dat, weight="weight")

    # dat_weighted.loc[dat_weighted["care"] == 0, "working_part_or_full_time"].mean()

    # (Pdb++) dat_weighted.loc[dat_weighted["care"] == 0, "working_part_or_full_time"].mean()
    # 0.5934259207037689
    # (Pdb++) dat_weighted.loc[dat_weighted["care"] == 1, "working_part_or_full_time"].mean()
    # 0.416571343065207
    # (Pdb++) dat_weighted.loc[dat_weighted["care"] == 1, "part_time"].mean()
    # 0.18438429857310548
    # (Pdb++) dat_weighted.loc[dat_weighted["care"] == 1, "full_time"].mean()
    # 0.4141292587641083
    # (Pdb++) dat_weighted.loc[dat_weighted["care"] == 0, "full_time"].mean()
    # 0.5909413925944842
    # (Pdb++) dat_weighted.loc[dat_weighted["care"] == 0, "part_time"].mean()
    # 0.12982559906302918
    # (Pdb++)

    dat = dat_hh_weight.copy()
    dat["care_weighted"] = dat["care"] * dat["hh_weight"]

    weight = "hh_weight"
    moments = []

    # 1. Share working by age bin
    age_bins_coarse = [(AGE_50, AGE_55), (AGE_55, AGE_60), (AGE_60, AGE_65)]
    age_bins_fine = [
        (AGE_50, AGE_53),
        (AGE_53, AGE_56),
        (AGE_56, AGE_59),
        (AGE_59, AGE_62),
        (AGE_62, AGE_65),
    ]
    age_bins = [
        (AGE_50, AGE_53),
        (AGE_53, AGE_56),
        (AGE_56, AGE_59),
        (AGE_59, AGE_62),
        (AGE_62, AGE_65),
    ]

    # working
    moments += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "working_part_or_full_time",
        ].sum()
        / dat.loc[(dat["age"] >= age_bin[0]) & (dat["age"] <= age_bin[1]), weight].sum()
        for age_bin in age_bins
    ]

    moments += [
        dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "working_part_or_full_time",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    moments += [
        dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "working_part_or_full_time",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    # full-time
    full_time = []
    full_time += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "full_time",
        ].sum()
        / dat.loc[(dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]), weight].sum()
        for age_bin in age_bins
    ]

    full_time += [
        dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "full_time",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    full_time += [
        dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "full_time",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    # part-time
    part_time = []
    part_time += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "part_time",
        ].sum()
        / dat.loc[(dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]), weight].sum()
        for age_bin in age_bins
    ]

    part_time += [
        dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "part_time",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    part_time += [
        dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "part_time",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    age_moments = []
    age_moments += [
        dat.loc[(dat["age"] == age), "working_part_or_full_time"].sum()
        / dat.loc[(dat["age"] == age), weight].sum()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    age_moments += [
        dat.loc[(dat["age"] == age), "full_time"].sum()
        / dat.loc[(dat["age"] == age), weight].sum()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    age_moments += [
        dat.loc[(dat["age"] == age), "part_time"].sum()
        / dat.loc[(dat["age"] == age), weight].sum()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    # age labor income
    age_labor_income = []
    age_labor_income += [
        dat.loc[(dat["age"] == age), "labor_income"].sum()
        / dat.loc[(dat["age"] == age), weight].sum()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    # labor income
    age_bins = [(AGE_50, AGE_55), (AGE_55, AGE_60), (AGE_60, AGE_65)]
    labor_income = []
    labor_income += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "labor_income",
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]
    labor_income += [
        dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "labor_income",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == False)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]
    labor_income += [
        dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            "labor_income",
        ].sum()
        / dat.loc[
            (dat["intensive_care"] == True)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]

    age_bins = [(AGE_50, AGE_55), (AGE_55, AGE_60), (AGE_60, AGE_65)]
    net_wealth_fine_bins = []

    net_wealth_fine_bins += get_moments_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="hnetw",
        is_caregiver="all",
    )
    net_wealth_fine_bins += get_moments_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="hnetw",
        is_caregiver=False,
    )
    net_wealth_fine_bins += get_moments_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="hnetw",
        is_caregiver=True,
    )

    net_wealth_coarse_bins = get_moments_by_age_bin(
        dat,
        age_bins=age_bins_coarse,
        moment="hnetw",
        is_caregiver="all",
    )

    net_wealth_by_age = get_moments_by_age(dat, moment="hnetw", is_caregiver="all")
    breakpoint()

    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == True), "full_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == True), "hh_weight"].sum()
    # 0.15722689712906093
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == True), "part_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == True), "hh_weight"].sum()
    # 0.4247275301275734
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == False), "full_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == False), "hh_weight"].sum()
    # 0.3383581652731629
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == False), "part_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["intensive_care"] == False), "hh_weight"].sum()
    # 0.2682820942232801
    # (Pdb++)

    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == True), "full_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == True), "hh_weight"].sum()
    # 0.3977131496092772
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == True), "part_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == True), "hh_weight"].sum()
    # 0.32929151561913556
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == False), "full_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == False), "hh_weight"].sum()
    # 0.19270118187121696
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) * (dat["intensive_care"] == False), "full_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) & (dat["intensive_care"] == False), "hh_weight"].sum()
    # 0.2758954640216199
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) * (dat["intensive_care"] == False), "part_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) & (dat["intensive_care"] == False), "hh_weight"].sum()
    # 0.4479437717450064
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) & (dat["intensive_care"] == False), "part_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) & (dat["intensive_care"] == False), "hh_weight"].sum()
    # 0.4479437717450064
    # (Pdb++) dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) & (dat["intensive_care"] == False), "full_time"].sum() / dat.loc[(dat["age"] < 61) & (dat["light_care"] == False) & (dat["intensive_care"] == False), "hh_weight"].sum()
    # 0.2758954640216199
    # (Pdb++) dat.loc[(dat["care"] == True, "light_care"].mean()
    # *** SyntaxError: closing parenthesis ']' does not match opening parenthesis '('
    # (Pdb++) dat.loc[(dat["care"] == True), "light_care"].mean()
    # 0.9558823529411765
    # (Pdb++) dat.loc[(dat["care"] == True), "intensive_care"].mean()
    # 0.7097701149425287
    # (Pdb++) pd.crosstab(dat["light_care"], dat["intensive_care"])
    # intensive_care  0.0  1.0
    # light_care
    # 0.0              59   18
    # 1.0             101    0
    # (Pdb++) dat["intensive_care"].sum()
    # 247.0
    # (Pdb++) dat["light_care"].sum()
    # 390.0
    # (Pdb++) dat["care"].sum()
    # 637
    # (Pdb++) table(dat["light_care"].sum()
    # *** SyntaxError: '(' was never closed
    # (Pdb++) table(dat["light_care"])
    # light_care
    # 0.0    198
    # 1.0    390
    # Name: Count, dtype: int64
    # (Pdb++) table(dat["intensive_care"])
    # intensive_care
    # 0.0    1092
    # 1.0     247
    # Name: Count, dtype: int64
    # (Pdb++) 247 / 637
    # 0.3877551020408163
    # (Pdb++) 390 / 637
    # 0.6122448979591837

    dat.loc[(dat["any_care"] == True), "care"].mean()

    dat.loc[(dat["any_care"] == True), "care_weighted"].sum() / dat.loc[
        dat["any_care"] == True,
        "hh_weight",
    ].sum()

    moments = []

    moments += [
        dat.loc[dat["age"] == age, "working"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    moments += [
        dat.loc[dat["age"] == age, "full_time"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    # wealth by age bin

    # group by age bins?

    moments += [
        dat.loc[(dat["care"] == False) & (dat["age"] == age), "working"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    moments += [
        dat.loc[(dat["care"] == False) & (dat["age"] == age), "full_time"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    moments += [
        dat.loc[(dat["care"] == True) & (dat["age"] == age), "working"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    moments += [
        dat.loc[(dat["care"] == True) & (dat["age"] == age), "full_time"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    moments += [
        dat.loc[(dat["intensive_care"] == True) & (dat["age"] == age), "working"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    moments += [
        dat.loc[
            (dat["intensive_care"] == True) & (dat["age"] == age),
            "full_time",
        ].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    # share giving care
    moments += [
        dat.loc[dat["age"] == age, "care"].mean() for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    moments += [
        dat.loc[dat["age"] == age, "intensive_care"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    # share intensive care givers among informal care givers
    moments += [
        dat.loc[(dat["care"] == True) & (dat["age"] == age), "intensive_care"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    # parent child data
    moments += [
        parent.loc[parent["health"] == health, "only_informal"].mean()
        for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
    ]
    moments += [
        parent.loc[parent["health"] == health, "combination_care"].mean()
        for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
    ]
    # only informal
    moments += [
        parent.loc[parent["health"] == health, "only_home_care"].mean()
        for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
    ]

    filtered_dat = dat[(dat["age"] >= MIN_AGE) & (dat["age"] <= MAX_AGE)]
    grouped = filtered_dat.groupby("age")["working"].mean()

    moments_arr = jnp.array(moments)
    pd.Series(moments).to_csv(path_to_save, index=False)


def get_moments_by_age(
    dat,
    moment,
    is_caregiver,
    care_type="intensive_care",
    weight="hh_weight",
):
    if is_caregiver in [None, "all"]:
        care_cond = True
    else:
        care_cond = dat[care_type] == is_caregiver

    return [
        dat.loc[
            care_cond & (dat["age"] == age),
            moment,
        ].sum()
        / dat.loc[
            care_cond & (dat["age"] == age),
            weight,
        ].sum()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]


def get_moments_by_age_bin(
    dat,
    age_bins,
    moment,
    is_caregiver,
    care_type="intensive_care",
    weight="hh_weight",
):
    if is_caregiver in [None, "all"]:
        care_cond = True
    elif not is_caregiver:
        care_cond = dat[care_type] == is_caregiver
    else:
        care_cond = dat[care_type] == is_caregiver

    return [
        dat.loc[
            care_cond & (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            moment,
        ].sum()
        / dat.loc[
            care_cond & (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    ]


def multiply_rows_with_weight(dat, weight):
    # Create a DataFrame of weights with the same shape as dat
    weights = dat[weight].values.reshape(-1, 1)

    static_cols = [
        "mergeid",
        "int_year",
        "int_month",
        "age",
        "care",
        "any_care",
        "light_care",
        "intensive_care",
        weight,
    ]
    data_columns = dat.drop(columns=static_cols).values

    result = data_columns * weights

    dat_weighted = pd.DataFrame(
        result,
        columns=[col for col in dat.columns if col not in static_cols],
    )
    dat_weighted.insert(0, "mergeid", dat["mergeid"])
    dat_weighted.insert(1, "int_year", dat["int_year"])
    dat_weighted.insert(2, "int_month", dat["int_month"])
    dat_weighted.insert(3, "age", dat["age"])
    dat_weighted.insert(4, weight, dat[weight])
    dat_weighted.insert(5, "care", dat["care"])
    dat_weighted.insert(6, "any_care", dat["any_care"])
    dat_weighted.insert(7, "light_care", dat["light_care"])
    dat_weighted.insert(8, "intensive_care", dat["intensive_care"])

    # data['design_weight_avg'] = data.groupby('mergeid')['design_weight'].transform('mean')
    dat_weighted[f"{weight}_avg"] = dat_weighted.groupby("mergeid")[weight].transform(
        "mean",
    )

    return dat_weighted
