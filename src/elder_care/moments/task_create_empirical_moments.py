"""Create empirical moments for MSM estimation."""
from pathlib import Path
from typing import Annotated

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
    path_to_parent_child_design_weight: Path = BLD
    / "data"
    / "parent_child_data_design_weight.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_parent_child_ind_weight: Path = BLD
    / "data"
    / "parent_child_data_ind_weight.csv",
    #
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "empirical_moments.csv",
) -> None:
    dat_un = pd.read_csv(path_to_estimation_data)
    dat_design_weight = pd.read_csv(path_to_design_weight)
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    dat_ind_weight = pd.read_csv(path_to_ind_weight)

    parent_un = pd.read_csv(path_to_parent_child_data)
    parent_design_weight = pd.read_csv(path_to_parent_child_design_weight)
    parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    parent_ind_weight = pd.read_csv(path_to_parent_child_ind_weight)

    dat = dat_hh_weight.copy()
    dat["care_weighted"] = dat["care"] * dat["hh_weight"]

    weight = "hh_weight"
    intensive_care_var = "intensive_care_no_other"

    age_bins_coarse = [(AGE_50, AGE_55), (AGE_55, AGE_60), (AGE_60, AGE_65)]
    age_bins_fine = [
        (AGE_50, AGE_53),
        (AGE_53, AGE_56),
        (AGE_56, AGE_59),
        (AGE_59, AGE_62),
        (AGE_62, AGE_65),
    ]
    age_bins = age_bins_fine

    # share working by age
    share_working_by_age = get_share_by_age(
        dat, moment="working_part_or_full_time", weight=weight
    )
    share_working_full_time_by_age = get_share_by_age(
        dat, moment="full_time", weight=weight
    )

    # income by age, working and non-working?
    net_income_by_age = get_income_by_age(dat, moment="labor_income", weight=weight)

    # total household NET wealth by age bin
    # We calculate wealth using the HILDA wealth model, which is based on variables collected
    # in the special module
    # included in waves 2, 6, 10 and 14. We calculate wealth at the household level,
    # which includes the wealth of the spouse. Our
    # measure of wealth is financial wealth plus non-financial wealth minus household debt
    # minus combined household super,
    # where the components are defined as in Summerfield et al. (2013, pp. 71–75).
    # We deflate wealth by the consumer price
    wealth_by_age_bin = get_wealth_by_age_bin(
        dat, age_bins, moment="hnetw", weight=weight
    )

    # share working by caregiving type (and age bin) --> to be checked

    share_working_informal_care_by_age_bin = get_share_by_informal_care_type_by_age_bin(
        dat,
        age_bins,
        moment="working_part_or_full_time",
        is_caregiver=True,
        care_type=intensive_care_var,
        weight=weight,
    )
    share_working_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins,
            moment="working_part_or_full_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    share_working_full_time_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins,
            moment="full_time",
            is_caregiver=True,
            care_type=intensive_care_var,
            weight=weight,
        )
    )
    share_working_full_time_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins,
            moment="full_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    share_working_part_time_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins,
            moment="part_time",
            is_caregiver=True,
            care_type=intensive_care_var,
            weight=weight,
        )
    )
    share_working_part_time_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins,
            moment="part_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    # net_wealth_by_age = get_moments_by_age(dat, moment="hnetw", is_caregiver="all")

    # ================================================================================

    dat["intensive_care_weighted"] = dat[intensive_care_var] * dat[weight]

    share_intensive_care = []
    share_intensive_care += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "intensive_care_weighted",
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins_fine
    ]

    # share intensive care givers among informal care givers

    # ================================================================================
    # PARENT CHILD DATA
    # ================================================================================

    # weighted parent child moments
    parent["informal_care_weighted"] = parent["informal_care_child"] * parent[weight]
    parent["only_informal_weighted"] = parent["only_informal"] * parent[weight]
    parent["combination_care_weighted"] = parent["combination_care"] * parent[weight]
    parent["only_home_care_weighted"] = parent["only_home_care"] * parent[weight]
    parent = parent_hh_weight.copy()

    # care mix by health status of parent

    # (Pdb++) parent_child
    # [0.1103448275862069, 0.14909303686366296, 0.12862190812720847, 0.006269592476489028, 0.042832065535400816, 0.1674911660777385, 0.013166144200626959, 0.038853130485664134, 0.11448763250883393]

    # parent child: mother
    informal_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        parent,
        moment="informal_care_child_weighted",
        parent="mother",
        is_other_parent_alive=True,
        weight=weight,
    )
    informal_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        parent,
        moment="informal_care_child_weighted",
        parent="mother",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_informal_care_by_mother_health_couple = (
        get_caregiving_status_by_parental_health(
            parent,
            moment="only_informal_weighted",
            parent="mother",
            is_other_parent_alive=True,
            weight=weight,
        )
    )
    only_informal_care_by_mother_health_single = (
        get_caregiving_status_by_parental_health(
            parent,
            moment="only_informal_weighted",
            parent="mother",
            is_other_parent_alive=False,
            weight=weight,
        )
    )

    combination_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        parent,
        moment="combination_care_weighted",
        parent="mother",
        is_other_parent_alive=True,
        weight=weight,
    )
    combination_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        parent,
        moment="combination_care_weighted",
        parent="mother",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_home_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        parent,
        moment="only_home_care_weighted",
        parent="mother",
        is_other_parent_alive=True,
        weight=weight,
    )
    only_home_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        parent,
        moment="only_home_care_weighted",
        parent="mother",
        is_other_parent_alive=False,
        weight=weight,
    )

    # parent child: father
    informal_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        parent,
        moment="informal_care_child_weighted",
        parent="father",
        is_other_parent_alive=True,
        weight=weight,
    )
    informal_care_by_father_health_single = get_caregiving_status_by_parental_health(
        parent,
        moment="informal_care_child_weighted",
        parent="father",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_informal_care_by_father_health_couple = (
        get_caregiving_status_by_parental_health(
            parent,
            moment="only_informal_weighted",
            parent="father",
            is_other_parent_alive=True,
            weight=weight,
        )
    )
    only_informal_care_by_father_health_single = (
        get_caregiving_status_by_parental_health(
            parent,
            moment="only_informal_weighted",
            parent="father",
            is_other_parent_alive=False,
            weight=weight,
        )
    )

    combination_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        parent,
        moment="combination_care_weighted",
        parent="father",
        is_other_parent_alive=True,
        weight=weight,
    )
    combination_care_by_father_health_single = get_caregiving_status_by_parental_health(
        parent,
        moment="combination_care_weighted",
        parent="father",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_home_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        parent,
        moment="only_home_care_weighted",
        parent="father",
        is_other_parent_alive=True,
        weight=weight,
    )
    only_home_care_by_father_health_single = get_caregiving_status_by_parental_health(
        parent,
        moment="only_home_care_weighted",
        parent="father",
        is_other_parent_alive=False,
        weight=weight,
    )

    # TODO: labor and caregiving transitions


def get_share_by_age(
    dat,
    moment,
    weight="hh_weight",
):
    return {
        f"{moment}_{age}": dat.loc[
            (dat["age"] == age),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] == age),
            weight,
        ].sum()
        for age in range(MIN_AGE + 1, MAX_AGE + 1)
    }


def get_share_by_age_bin(
    dat,
    age_bins,
    moment,
    weight="hh_weight",
):
    return {
        f"{moment}_{age_bin[0]}_{age_bin[1]}": dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    }


def get_share_by_informal_care_type_by_age_bin(
    dat,
    age_bins,
    moment,
    is_caregiver,
    care_type,
    weight="hh_weight",
):
    is_care = (1 - is_caregiver) * "no" + "informal_care"

    return {
        f"{moment}_{is_care}_{age_bin[0]}_{age_bin[1]}": dat.loc[
            (dat[care_type] == is_caregiver)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            moment,
        ].sum()
        / dat.loc[
            (dat[care_type] == is_caregiver)
            & (dat["age"] > age_bin[0])
            & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    }


def get_share_by_informal_care_type(dat, moment, is_caregiver, care_type, weight):
    is_care = (1 - is_caregiver) * "no" + "informal_care"

    return {
        f"{moment}_{is_care}": dat.loc[
            (dat[care_type] == is_caregiver),
            moment,
        ].sum()
        / dat.loc[
            (dat[care_type] == is_caregiver),
            weight,
        ].sum(),
    }


def get_income_by_age(dat, moment, weight):
    return {
        f"{moment}_": dat.loc[
            (dat["age"] == age),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] == age),
            weight,
        ].sum()
        for age in range(MIN_AGE + 1, MAX_AGE + 1)
    }


def get_wealth_by_age_bin(dat, age_bins, moment, weight):
    return {
        f"{moment}_{age_bin[0]}_{age_bin[1]}": dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    }


def get_wealth_by_age_bin(dat, age_bins, moment, weight):
    return {
        f"{moment}_{age_bin[0]}_{age_bin[1]}": dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            moment,
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins
    }


# def get_share_by_informal_care_type_by_age_bin(
#     dat, age_bins, moment, is_caregiver, care_type, weight
# ):
#     is_care = (1 - is_caregiver) * "no" + "informal_care"

#     return {
#         f"{moment}_{is_care}_{age_bin[0]}_{age_bin[1]}": dat.loc[
#             (dat[care_type] == is_caregiver)
#             & (dat["age"] > age_bin[0])
#             & (dat["age"] <= age_bin[1]),
#             moment,
#         ].sum()
#         / dat.loc[
#             (dat[care_type] == is_caregiver)
#             & (dat["age"] > age_bin[0])
#             & (dat["age"] <= age_bin[1]),
#             weight,
#         ].sum()
#         for age_bin in age_bins
#     }


def get_caregiving_status_by_parental_health(
    dat, moment, parent, is_other_parent_alive, weight,
):
    parent_status = (
        is_other_parent_alive * "couple" + (1 - is_other_parent_alive) * "single"
    )

    return {
        f"{moment}_{parent}_{parent_status}_health_{0}": dat.loc[
            # (dat[f"{other_parent}_alive"] == is_other_parent_alive),
            (dat["married"] == is_other_parent_alive)
            & (dat[f"{parent}_health"] == health),
            moment,
        ].sum()
        / dat.loc[
            (dat[f"{parent}_health"] == health)
            # & (dat[f"{other_parent}_alive"] == is_other_parent_alive),
            & (dat["married"] == is_other_parent_alive),
            weight,
        ].sum()
        for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
    }


# ================================================================================


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
