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


def task_create_moments(
    path_to_estimation_data: Path = BLD / "data" / "estimation_data.csv",
    path_to_parent_child_data: Path = BLD / "data" / "parent_child_data.csv",
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "empirical_moments.csv",
) -> None:
    dat = pd.read_csv(path_to_estimation_data)
    parent = pd.read_csv(path_to_parent_child_data)

    moments = []

    moments += [
        dat.loc[dat["age"] == age, "working"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

    moments += [
        dat.loc[dat["age"] == age, "full_time"].mean()
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]

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
