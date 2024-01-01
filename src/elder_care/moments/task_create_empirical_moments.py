"""Create empirical moments for MSM estimation."""
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import pandas as pd
from elder_care.config import BLD
from pytask import Product

MIN_AGE = 55
MAX_AGE = 68

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


def task_create_empirical_moments(
    path_to_estimation_data: Path = BLD / "data" / "estimation_data.csv",
    path_to_estimation_data_design_weight: Path = BLD
    / "data"
    / "estimation_data_design_weight.csv",
    path_to_estimation_data_hh_weight: Path = BLD
    / "data"
    / "estimation_data_hh_weight.csv",
    path_to_estimation_data_ind_weight: Path = BLD
    / "data"
    / "estimation_data_ind_weight.csv",
    path_to_parent_child_data: Path = BLD / "data" / "parent_child_data.csv",
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "empirical_moments.csv",
) -> None:
    data = pd.read_csv(path_to_estimation_data)
    dat_design_weight = pd.read_csv(path_to_estimation_data_design_weight)
    dat_hh_weight = pd.read_csv(path_to_estimation_data_hh_weight)
    dat_ind_weight = pd.read_csv(path_to_estimation_data_ind_weight)

    parent = pd.read_csv(path_to_parent_child_data)

    dat = dat_design_weight.copy()

    moments = []

    moments += [
        dat.loc[dat["age"] == age, "working"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    moments += [
        dat.loc[dat["age"] == age, "full_time"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]
    breakpoint()
    dat.loc[
        (dat["intensive_care"] == 0) & (dat["age"] < 62), "working_part_or_full_time",
    ].sum() / dat["design_weight"].sum()

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

    # Fix working variable = part time + full time
    moments += [dat.loc[dat["intensive_care"] == True, "working"].mean()]
    moments += [dat.loc[dat["intensive_care"] == True, "full_time"].mean()]
    moments += [dat.loc[dat["intensive_care"] == True, "part_time"].mean()]
    moments += [dat.loc[dat["light_care"] == True, "working"].mean()]
    moments += [dat.loc[dat["light_care"] == True, "full_time"].mean()]
    moments += [dat.loc[dat["light_care"] == True, "part_time"].mean()]
    moments += [dat.loc[dat["care"] == False, "working"].mean()]
    moments += [dat.loc[dat["care"] == False, "full_time"].mean()]
    moments += [dat.loc[dat["care"] == False, "part_time"].mean()]
    # drop sick people?

    # (Pdb++) dat["working"].mean()
    # 0.3457186286100321
    # (Pdb++) dat.loc[(dat["age"] < 60) & (dat["care"] == False), "working"].mean()
    # 0.6021563342318059
    # (Pdb++) dat.loc[(dat["age"] < 60) & (dat["care"] == True), "working"].mean()
    # 0.6963696369636964

    # (Pdb++) dat.loc[(dat["retired"] == False), "working"].mean()
    # 0.6193645990922844
    # (Pdb++) dat.loc[(dat["retired"] == False) & (dat["care"] == False), "working"].mean()
    # 0.6059867734075879
    # (Pdb++) dat.loc[(dat["retired"] == False) & (dat["care"] == True), "working"].mean()
    # 0.7083333333333334

    # --> Use SOEP dat (Korfhage) instead?

    breakpoint()

    # moments += [
    #     dat.loc[(dat["intensive_care"] == True) & (dat["age"] == age), "working"].mean()
    #     for age in range(MIN_AGE, MAX_AGE + 1)
    # ]
    # moments += [
    #     dat.loc[
    #         (dat["intensive_care"] == True) & (dat["age"] == age),

    #         "full_time",
    #     ].mean()
    #     for age in range(MIN_AGE, MAX_AGE + 1)
    # ]

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

    # transition from full to part time. no work

    # transition from no care to informal care

    # transitions BFischer table 26
    breakpoint()

    filtered_dat = dat[(dat["age"] >= MIN_AGE) & (dat["age"] <= MAX_AGE)]
    grouped = filtered_dat.groupby("age")["working"].mean()

    moments_arr = jnp.array(moments)
    pd.Series(moments).to_csv(path_to_save, index=False)
