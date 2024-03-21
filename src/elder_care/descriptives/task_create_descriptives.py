"""Descriptives from SHARE data."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from elder_care.config import BLD
from elder_care.moments.task_create_empirical_moments import deflate_income_and_wealth

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
AGE_70 = 70
AGE_75 = 75
AGE_80 = 80
AGE_90 = 90
AGE_100 = 100

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2

AGE_BINS_FINE = [
    (AGE_50, AGE_53),
    (AGE_53, AGE_56),
    (AGE_56, AGE_59),
    (AGE_59, AGE_62),
    (AGE_62, AGE_65),
]

AGE_BINS_COARSE = [
    (AGE_50, AGE_55),
    (AGE_55, AGE_60),
    (AGE_60, AGE_65),
    (AGE_65, AGE_70),
    # (AGE_70, AGE_75),
    (AGE_70, AGE_100),
    # (AGE_75, AGE_100),
]


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_ltc_by_children(
    path_to_hh_weight: Path = BLD / "data" / "data_hh_weight_all.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    path_to_save_females_normalized: Annotated[Path, Product] = BLD
    / "descriptives"
    / "informal_caregiving_by_age_group_females_normalized.png",
) -> None:
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    # dat = dat[dat["gender"] == FEMALE]

    weight = "hh_weight"
    dat = deflate_income_and_wealth(dat, cpi_data)

    dat["intensive_care_general_weighted"] = (
        dat["intensive_care_general"] * dat["hh_weight"]
    )
    dat["intensive_care_all_parents_weighted"] = (
        dat["intensive_care_all_parents"] * dat["hh_weight"]
    )
    dat["intensive_care_new_weighted"] = dat["intensive_care_new"] * dat["hh_weight"]
    dat["intensive_care_spouse_weighted"] = (
        dat["intensive_care_spouse"] * dat["hh_weight"]
    )
    dat["intensive_care_child_weighted"] = (
        dat["intensive_care_child"] * dat["hh_weight"]
    )
    dat["intensive_care_neighbor_weighted"] = (
        dat["intensive_care_neighbor"] * dat["hh_weight"]
    )

    share_intensive_informal_own_parents_by_age = _get_share_by_age(
        dat,
        moment="intensive_care_new_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
    )
    share_intensive_informal_parents_by_age = _get_share_by_age(
        dat,
        moment="intensive_care_all_parents_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
    )
    share_intensive_informal_spouse_by_age = _get_share_by_age(
        dat,
        moment="intensive_care_spouse_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
    )
    share_intensive_informal_general_by_age = _get_share_by_age(
        dat,
        moment="intensive_care_general_weighted",
        age_lower=50,
        age_upper=90,
        weight=weight,
    )

    share_intensive_informal_own_parents_by_age_bin = _get_share_by_age_bin(
        dat,
        moment="intensive_care_new_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_FINE,
    )
    share_intensive_informal_parents_by_age_bin = _get_share_by_age_bin(
        dat,
        moment="intensive_care_all_parents_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_FINE,
    )
    share_intensive_informal_general_by_age_bin = _get_share_by_age_bin(
        dat,
        moment="intensive_care_general_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_FINE,
    )

    share_intensive_informal_own_parents_by_age_bin_coarse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_new_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    share_intensive_informal_parents_by_age_bin_coarse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_all_parents_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    share_intensive_informal_spouse_by_age_bin_coarse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_spouse_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    share_intensive_informal_child_by_age_bin_coarse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_child_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    share_intensive_informal_neighbor_by_age_bin_coarse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_neighbor_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    share_intensive_informal_general_by_age_bin_coarse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_general_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )

    share_intensive_informal_parents_in_law_by_age_bin_coarse = np.array(
        share_intensive_informal_parents_by_age_bin_coarse,
    ) - np.array(share_intensive_informal_own_parents_by_age_bin_coarse)
    share_intensive_informal_other_by_age_bin_coarse = (
        np.array(share_intensive_informal_general_by_age_bin_coarse)
        - np.array(share_intensive_informal_parents_by_age_bin_coarse)
        - np.array(share_intensive_informal_spouse_by_age_bin_coarse)
        - np.array(share_intensive_informal_child_by_age_bin_coarse)
        - np.array(share_intensive_informal_neighbor_by_age_bin_coarse)
    )
    share_intensive_informal_own_parents_by_age_bin_coarse = np.array(
        share_intensive_informal_own_parents_by_age_bin_coarse,
    )
    share_intensive_informal_child_by_age_bin_coarse = np.array(
        share_intensive_informal_child_by_age_bin_coarse,
    )
    share_intensive_informal_spouse_by_age_bin_coarse = np.array(
        share_intensive_informal_spouse_by_age_bin_coarse,
    )
    share_intensive_informal_neighbor_by_age_bin_coarse = np.array(
        share_intensive_informal_neighbor_by_age_bin_coarse,
    )

    # mean_share_of_own_parental_intensive = np.mean(
    #     np.array(share_intensive_informal_own_parents_by_age)
    #     / np.array(share_intensive_informal_general_by_age),
    # )
    # mean_share_of_parental_intensive = np.mean(
    #     np.array(share_intensive_informal_parents_by_age)
    #     / np.array(share_intensive_informal_general_by_age),
    # )
    # mean_share_of_pspousal_intensive = np.mean(
    #     np.array(share_intensive_informal_spouse_by_age)
    #     / np.array(share_intensive_informal_general_by_age),
    # )

    l = [
        share_intensive_informal_general_by_age_bin_coarse,
        share_intensive_informal_parents_by_age_bin_coarse,
        share_intensive_informal_spouse_by_age_bin_coarse,
    ]

    # Example usage
    general_shares = [
        0.07782676002330877,
        0.09567270550608412,
        0.11456777273811151,
        0.09863404007600894,
        0.0719608965446451,
        0.05082212348295812,
    ]
    parents_shares = [
        0.04939677264811303,
        0.05236983799966206,
        0.05127086046728292,
        0.030544438280519625,
        0.00194934100365271,
        0.0,
    ]
    spouse_shares = [
        0.01010368,
        0.01420626,
        0.03107896,
        0.03445891,
        0.03640086,
        0.0288907,
    ]

    create_and_save_caregiving_plot_normalized(
        dat,
        gender=FEMALE,
        save_path=path_to_save_females_normalized,
    )


def task_create_ltc_by_parent_age(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    # path_to_save: Annotated[Path, Product] = BLD / "descri" / "empirical_moments.csv",
) -> None:
    """Create share care dependent by parental age."""
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


def _get_share_by_age(dat, moment, age_lower, age_upper, weight):
    return [
        dat.loc[
            (dat["age"] == age)
            # & (
            #     (parent["home_care"] == True)
            #     | (parent["informal_care_general"] == True)
            # )
            # & (dat["home_care"].notna()) & (dat["informal_care_general"].notna()),
            & (dat[moment].notna()),
            moment,
        ].sum()
        / dat.loc[(dat["age"] == age) & (dat[moment].notna()), weight].sum()
        for age in range(age_lower, age_upper)
    ]


def _get_share_by_age_bin(dat, moment, age_lower, age_upper, weight, age_bins):
    return [
        dat.loc[
            (dat["age"] >= age_bin[0]) & (dat["age"] < age_bin[1]),
            # & (dat[moment].notna()),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] >= age_bin[0]) & (dat["age"] < age_bin[1]),
            # & (dat[moment].notna()),
            weight,
        ].sum()
        for age_bin in age_bins
    ]


def _get_share_by_age_bin_unweighted(
    dat,
    moment,
    age_bins,
):
    return [
        dat.loc[
            (dat["age"] >= age_bin[0]) & (dat["age"] < age_bin[1]),
            # & (dat[moment].notna()),
            moment,
        ].mean()
        for age_bin in age_bins
    ]


def create_and_save_caregiving_plot(dat, gender, save_path):
    """Creates and saves a stacked bar chart of intensive informal caregiving shares.

    Parameters:
    general (list): Share of general intensive informal caregiving by age bin.
    parents (list): Share of intensive informal caregiving to parents by age bin.
    spouse (list): Share of intensive informal caregiving to spouse by age bin.
    save_path (str): Path to save the generated plot.

    """
    dat = dat[dat["gender"] == gender]
    weight = "hh_weight"

    dat["intensive_care_general_weighted"] = (
        dat["intensive_care_general"] * dat["hh_weight"]
    )
    dat["intensive_care_all_parents_weighted"] = (
        dat["intensive_care_all_parents"] * dat["hh_weight"]
    )
    dat["intensive_care_in_laws_weighted"] = (
        dat["intensive_care_in_laws"] * dat["hh_weight"]
    )
    dat["intensive_care_new_weighted"] = dat["intensive_care_new"] * dat["hh_weight"]
    dat["intensive_care_spouse_weighted"] = (
        dat["intensive_care_spouse"] * dat["hh_weight"]
    )

    # general

    all_parents = _get_share_by_age_bin(
        dat,
        moment="intensive_care_all_parents_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    in_laws = _get_share_by_age_bin(
        dat,
        moment="intensive_care_in_laws_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    spouse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_spouse_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )

    general = _get_share_by_age_bin(
        dat,
        moment="intensive_care_general_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    own_parents = (np.asarray(all_parents) - np.asarray(in_laws)).tolist()

    if gender == FEMALE:
        share_gender = 0.071
    elif gender == MALE:
        share_gender = 0.065

    _sum = general[-1] + own_parents[-1] + in_laws[-1] + spouse[-1]
    general[-1] = share_gender
    own_parents[-1] = (own_parents[-1] / _sum) * share_gender
    in_laws[-1] = (in_laws[-1] / _sum) * share_gender
    spouse[-1] = (spouse[-1] / _sum) * share_gender

    # Calculate the remaining share for 'other' caregiving
    other = [
        g - p - i - s
        for g, p, i, s in zip(general, own_parents, in_laws, spouse, strict=False)
    ]

    # Age bins
    age_bins_coarse = ["50-54", "55-59", "60-64", "65-69", "70+"]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(
        age_bins_coarse,
        own_parents,
        label="Own Parents",
        # color="sandybrown",
        color="palegreen",
        # color="moccasin",
    )
    plt.bar(
        age_bins_coarse,
        in_laws,
        bottom=own_parents,
        label="Parents In-Law",
        # color="sandybrown",
        # color="green",
        # color="moccasin",
        color="khaki",
    )
    plt.bar(
        age_bins_coarse,
        spouse,
        bottom=[i + j for i, j in zip(own_parents, in_laws, strict=False)],
        label="Spouse",
        color="lightcoral",
    )
    plt.bar(
        age_bins_coarse,
        other,
        bottom=[
            i + j + k for i, j, k in zip(own_parents, in_laws, spouse, strict=False)
        ],
        label="Other",
        color="lightblue",
    )

    plt.xlabel("Age Bins")
    plt.ylabel("Share of (Daily) Informal Caregivers")
    plt.ylim(0, 0.13)  # Set y-axis range from 0 to 15%
    plt.grid(axis="y")

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [3, 2, 1, 0]

    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Save plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent display in the notebook


def create_and_save_caregiving_plot_normalized(dat, gender, save_path):
    """Creates and saves a stacked bar chart of intensive informal caregiving shares.

    Parameters:
    general (list): Share of general intensive informal caregiving by age bin.
    parents (list): Share of intensive informal caregiving to parents by age bin.
    spouse (list): Share of intensive informal caregiving to spouse by age bin.
    save_path (str): Path to save the generated plot.

    """
    dat = dat[dat["gender"] == gender]
    weight = "hh_weight"

    dat["intensive_care_general_weighted"] = (
        dat["intensive_care_general"] * dat["hh_weight"]
    )
    dat["intensive_care_all_parents_weighted"] = (
        dat["intensive_care_all_parents"] * dat["hh_weight"]
    )
    dat["intensive_care_in_laws_weighted"] = (
        dat["intensive_care_in_laws"] * dat["hh_weight"]
    )
    dat["intensive_care_new_weighted"] = dat["intensive_care_new"] * dat["hh_weight"]
    dat["intensive_care_spouse_weighted"] = (
        dat["intensive_care_spouse"] * dat["hh_weight"]
    )

    # general

    all_parents = _get_share_by_age_bin(
        dat,
        moment="intensive_care_all_parents_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    in_laws = _get_share_by_age_bin(
        dat,
        moment="intensive_care_in_laws_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    spouse = _get_share_by_age_bin(
        dat,
        moment="intensive_care_spouse_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )

    general = _get_share_by_age_bin(
        dat,
        moment="intensive_care_general_weighted",
        age_lower=50,
        age_upper=65,
        weight=weight,
        age_bins=AGE_BINS_COARSE,
    )
    own_parents = (np.asarray(all_parents) - np.asarray(in_laws)).tolist()

    if gender == FEMALE:
        share_gender = 0.071
    elif gender == MALE:
        share_gender = 0.065

    _sum = (
        np.asarray(general)
        # + np.asarray(own_parents)
        # + np.asarray(in_laws)
        # + np.asarray(spouse)
    )
    general = np.array([1, 1, 1, 1, 1])  # * share_gender
    own_parents = np.asarray(own_parents) / _sum  # * share_gender
    in_laws = np.asarray(in_laws) / _sum  # * share_gender
    spouse = np.asarray(spouse) / _sum  # * share_gender

    # Calculate the remaining share for 'other' caregiving
    other = [
        g - p - i - s
        for g, p, i, s in zip(general, own_parents, in_laws, spouse, strict=False)
    ]

    # Age bins
    age_bins_coarse = ["50-54", "55-59", "60-64", "65-69", "70+"]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(
        age_bins_coarse,
        own_parents,
        label="Own Parents",
        # color="sandybrown",
        color="palegreen",
        # color="moccasin",
    )
    plt.bar(
        age_bins_coarse,
        in_laws,
        bottom=own_parents,
        label="Parents In-Law",
        # color="sandybrown",
        # color="green",
        # color="moccasin",
        color="khaki",
    )
    plt.bar(
        age_bins_coarse,
        spouse,
        bottom=[i + j for i, j in zip(own_parents, in_laws, strict=False)],
        label="Spouse",
        color="lightcoral",
    )
    plt.bar(
        age_bins_coarse,
        other,
        bottom=[
            i + j + k for i, j, k in zip(own_parents, in_laws, spouse, strict=False)
        ],
        label="Other",
        color="lightblue",
    )

    plt.xlabel("Age Bins")
    plt.ylabel("Share of (Daily) Informal Caregivers")
    plt.ylim(0, 1)  # Set y-axis range from 0 to 15%
    plt.grid(axis="y")

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [3, 2, 1, 0]

    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Save plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent display in the notebook


# ==============================================================================


def plot_share_by_age():
    # Adjusting the colors for better differentiation
    # Data
    age_bins_5yr = ["50-54", "55-59", "60-64", "65-69", "70-74", "75-79"]
    share_own_parents = [
        0.05085612876092149,
        0.034542708584536244,
        0.06490798521935749,
        0.08613743226007055,
        0.08818484444915196,
        0.10235000831551055,
    ]
    share_parents_in_law = [0.02477087, 0.01900799, 0.0225679, 0.01033598, 0.0, 0.0]
    share_spouse = [
        0.02016726,
        0.03061945,
        0.06738196,
        0.08341278,
        0.10401785,
        0.13471133,
    ]
    share_child = [0.00250696, 0.00027805, 0.00193487, 0.00152451, 0.0, 0.0]
    share_neighbor = [
        0.00631942,
        0.00937135,
        0.01010362,
        0.0253278,
        0.01518971,
        0.0218772,
    ]
    share_other = [
        -0.00036443,
        0.01074745,
        0.00565645,
        0.0027085,
        0.0362054,
        0.01460215,
    ]

    # Stacking the shares
    shares_stacked = np.vstack(
        (
            share_own_parents,
            share_parents_in_law,
            share_spouse,
            share_child,
            share_neighbor,
            share_other,
        ),
    )

    # Colors for each category
    colors = [
        "steelblue",
        "seagreen",
        "goldenrod",
        "lightcoral",
        "mediumpurple",
        "sandybrown",
    ]

    # Plotting stacked bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(age_bins_5yr, shares_stacked[0], label="Own Parents", color=colors[0])
    plt.bar(
        age_bins_5yr,
        shares_stacked[1],
        bottom=shares_stacked[0],
        label="Parents in Law",
        color=colors[1],
    )
    bottom_for_next = shares_stacked[0] + shares_stacked[1]

    for i, (category, color) in enumerate(
        zip(["Spouse", "Child", "Neighbor", "Other"], colors[2:], strict=False),
        start=2,
    ):
        plt.bar(
            age_bins_5yr,
            shares_stacked[i],
            bottom=bottom_for_next,
            label=category,
            color=color,
        )
        bottom_for_next += shares_stacked[i]

    plt.xlabel("Age Bins")
    plt.ylabel("Share of Intensive Informal Care")
    plt.legend()
    plt.title("Share of Intensive Informal Care by Relationship and Age Bin")
    plt.grid(axis="y")
    plt.show()
