"""Descriptives from SHARE data."""
from pathlib import Path
from typing import Annotated
import numpy as np

import pandas as pd
from elder_care.config import BLD
from pytask import Product

from elder_care.moments.task_create_empirical_moments import (
    deflate_income_and_wealth,
)

BASE_YEAR = 2015

MALE = 1
FEMALE = 2

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


def task_create_descriptives(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    # path_to_save: Annotated[Path, Product] = BLD / "descri" / "empirical_moments.csv",
) -> None:
    """Create empirical moments for SHARE data."""
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    dat = deflate_income_and_wealth(dat, cpi_data)

    weight = "hh_weight"
    parent = parent_hh_weight.copy()

    parent["only_home_care_weighted"] = parent["only_home_care"] * parent[weight]
    parent["home_care_weighted"] = parent["home_care"] * parent[weight]
    parent["only_informal_care_weighted"] = parent["only_informal"] * parent[weight]
    parent["combination_care_child_weighted"] = (
        parent["combination_care_child"] * parent[weight]
    )
    parent["informal_care_weighted"] = parent["informal_care_child"] * parent[weight]
    parent["informal_care_general_weighted"] = (
        parent["informal_care_general"] * parent[weight]
    )
    parent["combination_care_weighted"] = parent["combination_care"] * parent[weight]

    parent["home_care"].mean()
    # parent["informal_care_general_weighted"].sum() / parent[weight].sum()
    # parent["informal_care_weighted"].sum() / parent[weight].sum()

    si = parent.loc[
        ((parent["home_care"] == True) | (parent["informal_care_child"] == True))
        & (parent["home_care"].notna())
        & (parent["informal_care_child"].notna()),
        "informal_care_child",
    ].mean()

    sh = parent.loc[
        ((parent["home_care"] == True) | (parent["informal_care_child"] == True))
        & (parent["home_care"].notna())
        & (parent["informal_care_child"].notna()),
        "home_care",
    ].mean()

    sc = parent.loc[
        ((parent["home_care"] == True) | (parent["informal_care_child"] == True))
        & (parent["home_care"].notna())
        & (parent["informal_care_child"].notna()),
        "combination_care_child",
    ].mean()

    assert np.round(sh + si - sc, 7) == 1

    age_upper = 90

    informal_care_by_age_weighted = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "informal_care_general_weighted",
        ].sum()
        / parent.loc[(parent["age"] == age), weight].sum()
        for age in range(70, age_upper)
    ]
    home_care_by_age_weighted = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "home_care_weighted",
        ].sum()
        / parent.loc[(parent["age"] == age), weight].sum()
        for age in range(70, age_upper)
    ]
    informal_care_by_age_child_weighted = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "informal_care_weighted",
        ].sum()
        / parent.loc[(parent["age"] == age), weight].sum()
        for age in range(70, age_upper)
    ]
    combination_care_by_age_weighted = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "combination_care_weighted",
        ].sum()
        / parent.loc[(parent["age"] == age), weight].sum()
        for age in range(70, age_upper)
    ]

    # by age
    informal_care_by_age_cond_care = [
        parent.loc[
            (parent["age"] == age)
            & (
                (parent["home_care"] == True)
                | (parent["informal_care_general"] == True)
            )
            & (parent["home_care"].notna())
            & (parent["informal_care_general"].notna()),
            "informal_care_child",
        ].mean()
        for age in range(70, age_upper)
    ]
    informal_care_by_age = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "informal_care_general",
        ].mean()
        for age in range(70, age_upper)
    ]
    informal_care_by_age_child = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "informal_care_child",
        ].mean()
        for age in range(70, age_upper)
    ]
    home_care_by_age = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "home_care",
        ].mean()
        for age in range(70, age_upper)
    ]
    combination_care_by_age = [
        parent.loc[
            (parent["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            & (parent["home_care"].notna()) & (parent["informal_care_general"].notna()),
            "combination_care",
        ].mean()
        for age in range(70, age_upper)
    ]

    #
    moment = "informal_care_child"
    share_informal_care = (
        parent.loc[(parent[moment] == True), "informal_care_weighted"].sum()
        / parent.loc[
            ((parent["home_care"] == True) | (parent["informal_care_child"] == True))
            & (parent["home_care"].notna())
            & (parent["informal_care_child"].notna()),
            # (parent["home_care"] == True) | (parent["informal_care_child"] == True),
            weight,
        ].sum()
    )

    moment = "home_care"
    share_home_care = (
        parent.loc[(parent[moment] == True), "home_care_weighted"].sum()
        / parent.loc[
            ((parent["home_care"] == True) | (parent["informal_care_child"] == True))
            & (parent["home_care"].notna())
            & (parent["informal_care_child"].notna()),
            # (parent["home_care"] == True) | (parent["informal_care_child"] == True),
            weight,
        ].sum()
    )

    moment = "only_home_care"
    share_only_home_care = (
        parent.loc[(parent[moment] == True), "only_home_care_weighted"].sum()
        / parent.loc[
            ((parent["home_care"] == True) | (parent["informal_care_child"] == True))
            & (parent["home_care"].notna())
            & (parent["informal_care_child"].notna()),
            # (parent["home_care"] == True) | (parent["informal_care_child"] == True),
            weight,
        ].sum()
    )

    moment = "only_informal"
    share_only_informal_care = (
        parent.loc[(parent[moment] == True), "only_informal_care_weighted"].sum()
        / parent.loc[
            (parent["home_care"] == True)
            | (parent["informal_care_child"] == True)
            & (parent["home_care"].notna())
            & (parent["informal_care_child"].notna()),
            # (parent["home_care"] == True) | (parent["informal_care_child"] == True),
            weight,
        ].sum()
    )

    moment = "combination_care_child"
    share_combination_care = (
        parent.loc[(parent[moment] == True), "combination_care_child_weighted"].sum()
        / parent.loc[
            (parent["home_care"] == True)
            | (parent["informal_care_child"] == True)
            & (parent["home_care"].notna())
            & (parent["informal_care_child"].notna()),
            # (parent["home_care"] == True) | (parent["informal_care_child"] == True),
            weight,
        ].sum()
    )

    # assert (
    #     np.round(
    #         share_only_informal_care + share_only_home_care + share_combination_care, 2
    #     )
    #     == 1
    # )
    breakpoint()
