"""Create empirical moments for MSM estimation.

The SHARE data sets on
- children
- parent child combinations

are used.

"""
from pathlib import Path
from typing import Annotated

import pandas as pd
from elder_care.config import BLD
from pytask import Product

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


def task_create_moments(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "empirical_moments.csv",
) -> None:
    """Create empirical moments for SHARE data."""
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    dat = deflate_income_and_wealth(dat, cpi_data)

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
    share_not_working_by_age = get_share_by_age(
        dat,
        moment="not_working_part_or_full_time",
        weight=weight,
    )
    share_working_by_age = get_share_by_age(
        dat,
        moment="working_part_or_full_time",
        weight=weight,
    )
    share_working_full_time_by_age = get_share_by_age(
        dat,
        moment="full_time",
        weight=weight,
    )
    share_working_part_time_by_age = get_share_by_age(
        dat,
        moment="part_time",
        weight=weight,
    )

    employment_by_age = pd.concat(
        [
            share_working_by_age,
            share_not_working_by_age,
            share_working_part_time_by_age,
            share_working_full_time_by_age,
        ],
        ignore_index=False,
        axis=0,
    )

    # share working by age bin
    share_not_working_by_age_bin = get_share_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="not_working_part_or_full_time",
        weight=weight,
    )
    share_working_by_age_bin = get_share_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="working_part_or_full_time",
        weight=weight,
    )
    share_working_full_time_by_age_bin = get_share_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="full_time",
        weight=weight,
    )
    share_working_part_time_by_age_bin = get_share_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="part_time",
        weight=weight,
    )

    employment_by_age_bin = pd.concat(
        [
            share_working_by_age_bin,
            share_not_working_by_age_bin,
            share_working_part_time_by_age_bin,
            share_working_full_time_by_age_bin,
        ],
        ignore_index=False,
        axis=0,
    )

    # income by age, working and non-working?
    net_income_by_age_bin = get_income_by_age_bin(
        dat,
        age_bins=age_bins_fine,
        moment="real_labor_income",
        weight=weight,
    )

    # total household NET wealth by age bin
    # We calculate wealth using the HILDA wealth model, which is based on
    # variables collected in the special module
    # included in waves 2, 6, 10 and 14. We calculate wealth at the household level,
    # which includes the wealth of the spouse. Our
    # measure of wealth is financial wealth plus non-financial wealth minus household
    # debt minus combined household super,
    # where the components are defined as in Summerfield et al. (2013)
    # We deflate wealth by the consumer price
    wealth_by_age_bin = get_wealth_by_age_bin(
        dat,
        age_bins,
        moment="real_hnetw",
        weight=weight,
    )

    income_and_wealth = pd.concat(
        [net_income_by_age_bin, wealth_by_age_bin],
        ignore_index=False,
        axis=0,
    )

    # share working by caregiving type (and age bin) --> to be checked
    employment_by_informal_caregiving_status_by_age_bin = (
        get_employment_by_caregiving_status_by_age_bin(
            dat,
            age_bins_coarse,
            intensive_care_var,
            weight=weight,
        )
    )

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

    # ================================================================================
    # PARENT CHILD DATA
    # ================================================================================

    # weighted parent child moments
    parent = parent_hh_weight.copy()
    parent["informal_care_child_weighted"] = (
        parent["informal_care_child"] * parent[weight]
    )
    parent["home_care_weighted"] = parent["home_care"] * parent[weight]
    parent["combination_care_weighted"] = parent["combination_care"] * parent[weight]

    parent["only_informal_weighted"] = parent["only_informal"] * parent[weight]
    parent["only_home_care_weighted"] = parent["only_home_care"] * parent[weight]

    dat["no_intensive_informal_weighted"] = dat["no_intensive_informal"] * dat[weight]
    dat["intensive_care_no_other_weighted"] = (
        dat["intensive_care_no_other"] * dat[weight]
    )
    intensive_care_var_weighted = "intensive_care_no_other_weighted"

    mother_couple = parent[(parent["gender"] == FEMALE) & (parent["married"] == True)]
    mother_single = parent[(parent["gender"] == FEMALE) & (parent["married"] == False)]

    father_couple = parent[(parent["gender"] == MALE) & (parent["married"] == True)]
    father_single = parent[(parent["gender"] == MALE) & (parent["married"] == False)]

    parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    parent["no_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )

    # care mix by health status of parent
    caregiving_by_mother_health = (
        get_caregiving_status_by_mother_health_and_marital_status(
            mother_couple,
            mother_single,
            weight=weight,
        )
    )
    caregiving_by_father_health = (
        get_caregiving_status_by_father_health_and_marital_status(
            father_couple,
            father_single,
            weight=weight,
        )
    )

    # labor and caregiving transitions
    # work transitions
    employment_transitions_soep = get_employment_transitions_soep()

    # caregiving transitions
    care_transitions_estimation_data = (
        get_care_transitions_from_estimation_data_weighted(
            dat,
            intensive_care_var=intensive_care_var,
            intensive_care_var_weighted=intensive_care_var_weighted,
            weight=weight,
        )
    )

    # formal care transitions from parent child data set
    care_transitions_parent_child_data = (
        get_care_transitions_from_parent_child_data_weighted(
            parent,
            weight=weight,
        )
    )

    all_moments = pd.concat(
        [
            employment_by_age,
            employment_by_age_bin,
            income_and_wealth,
            employment_by_informal_caregiving_status_by_age_bin,
            caregiving_by_mother_health,
            caregiving_by_father_health,
            employment_transitions_soep,
            care_transitions_estimation_data,
            care_transitions_parent_child_data,
        ],
        ignore_index=False,
        axis=0,
    )

    all_moments.to_csv(path_to_save)


# ================================================================================
# Transitions
# ================================================================================


def get_care_transitions_from_parent_child_data_weighted(parent, weight):
    """Get care transitions from parent child data set using survey weights."""
    # weighted
    no_formal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_home_care",
        current_choice="no_home_care_weighted",
        weight=weight,
    )
    no_formal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_home_care",
        current_choice="home_care_weighted",
        weight=weight,
    )

    formal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="home_care",
        current_choice="no_home_care_weighted",
        weight=weight,
    )
    formal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="home_care",
        current_choice="home_care_weighted",
        weight=weight,
    )

    no_formal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_home_care",
        current_choice="no_informal_care_child_weighted",
        weight=weight,
    )
    no_formal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_home_care",
        current_choice="informal_care_child_weighted",
        weight=weight,
    )

    formal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="home_care",
        current_choice="no_informal_care_child_weighted",
        weight=weight,
    )
    formal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="home_care",
        current_choice="informal_care_child_weighted",
        weight=weight,
    )

    # informal care transitions from parent child data set
    no_informal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="no_home_care_weighted",
        weight=weight,
    )
    no_informal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="home_care_weighted",
        weight=weight,
    )

    informal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="no_home_care_weighted",
        weight=weight,
    )
    informal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="home_care_weighted",
        weight=weight,
    )

    no_informal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="no_informal_care_child_weighted",
        weight=weight,
    )
    no_informal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="informal_care_child_weighted",
        weight=weight,
    )

    informal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="no_informal_care_child_weighted",
        weight=weight,
    )
    informal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="informal_care_child_weighted",
        weight=weight,
    )

    return pd.concat(
        [
            no_formal_to_no_formal_weighted,
            no_formal_to_formal_weighted,
            formal_to_no_formal_weighted,
            formal_to_formal_weighted,
            #
            no_formal_to_no_informal_weighted,
            no_formal_to_informal_weighted,
            formal_to_no_informal_weighted,
            formal_to_informal_weighted,
            #
            no_informal_to_no_formal_weighted,
            no_informal_to_formal_weighted,
            informal_to_no_formal_weighted,
            informal_to_formal_weighted,
            # ?
            no_informal_to_no_informal_weighted,
            no_informal_to_informal_weighted,
            informal_to_no_informal_weighted,
            informal_to_informal_weighted,
        ],
        ignore_index=False,
        axis=0,
    )


def get_care_transitions_from_parent_child_data_unweighted(parent):
    no_formal_to_no_formal = get_care_transition_unweighted(
        parent,
        previous_choice="no_home_care",
        current_choice="no_home_care",
    )
    no_formal_to_formal = get_care_transition_unweighted(
        parent,
        previous_choice="no_home_care",
        current_choice="home_care",
    )

    formal_to_no_formal = get_care_transition_unweighted(
        parent,
        previous_choice="home_care",
        current_choice="no_home_care",
    )
    formal_to_formal = get_care_transition_unweighted(
        parent,
        previous_choice="home_care",
        current_choice="home_care",
    )

    no_formal_to_no_informal = get_care_transition_unweighted(
        parent,
        previous_choice="no_home_care",
        current_choice="no_informal_care_child",
    )
    no_formal_to_informal = get_care_transition_unweighted(
        parent,
        previous_choice="no_home_care",
        current_choice="informal_care_child",
    )

    formal_to_no_informal = get_care_transition_unweighted(
        parent,
        previous_choice="home_care",
        current_choice="no_informal_care_child",
    )
    formal_to_informal = get_care_transition_unweighted(
        parent,
        previous_choice="home_care",
        current_choice="informal_care_child",
    )

    return pd.concat(
        [
            no_formal_to_no_formal,
            no_formal_to_formal,
            formal_to_no_formal,
            formal_to_formal,
            #
            no_formal_to_no_informal,
            no_formal_to_informal,
            formal_to_no_informal,
            formal_to_informal,
        ],
        ignore_index=False,
        axis=0,
    )


def get_care_transitions_from_estimation_data_weighted(
    dat,
    weight,
    intensive_care_var,
    intensive_care_var_weighted,
):
    """Get care transitions from estimation data using survey weights."""
    no_care_to_no_informal_care_weighted = get_care_transition_weighted(
        dat,
        previous_choice="no_intensive_informal",
        current_choice="no_intensive_informal_weighted",
        weight=weight,
    )
    no_care_to_informal_care_weighted = get_care_transition_weighted(
        dat,
        previous_choice="no_intensive_informal",
        current_choice=intensive_care_var_weighted,
        weight=weight,
    )

    informal_care_to_no_informal_care_weighted = get_care_transition_weighted(
        dat,
        previous_choice=intensive_care_var,
        current_choice="no_intensive_informal_weighted",
        weight=weight,
    )
    informal_care_to_informal_care_weighted = get_care_transition_weighted(
        dat,
        previous_choice=intensive_care_var,
        current_choice=intensive_care_var_weighted,
        weight=weight,
    )

    return pd.concat(
        [
            no_care_to_no_informal_care_weighted,
            no_care_to_informal_care_weighted,
            informal_care_to_no_informal_care_weighted,
            informal_care_to_informal_care_weighted,
        ],
        ignore_index=False,
        axis=0,
    )


def get_care_transitions_from_estimation_data_unweighted(
    dat,
    intensive_care_var,
):
    """Get care transitions from estimation data."""
    no_care_to_no_informal_care = get_care_transition_unweighted(
        dat,
        previous_choice="no_intensive_informal",
        current_choice="no_intensive_informal",
    )
    no_care_to_informal_care = get_care_transition_unweighted(
        dat,
        previous_choice="no_intensive_informal",
        current_choice=intensive_care_var,
    )

    informal_care_to_no_informal_care = get_care_transition_unweighted(
        dat,
        previous_choice=intensive_care_var,
        current_choice="no_intensive_informal",
    )
    informal_care_to_informal_care = get_care_transition_unweighted(
        dat,
        previous_choice=intensive_care_var,
        current_choice=intensive_care_var,
    )

    return pd.concat(
        [
            no_care_to_no_informal_care,
            no_care_to_informal_care,
            informal_care_to_no_informal_care,
            informal_care_to_informal_care,
        ],
        ignore_index=False,
        axis=0,
    )


def get_emplyoment_transitions(dat, weight):
    """Get employment transitions."""
    dat["not_working_part_or_full_time_weighted"] = (
        dat["not_working_part_or_full_time"] * dat[weight]
    )
    dat["part_time_weighted"] = dat["part_time"] * dat[weight]
    dat["full_time_weighted"] = dat["full_time"] * dat[weight]

    no_work_to_no_work = get_work_transition_unweighted(
        dat,
        previous_choice="not_working_part_or_full_time",
        current_choice="not_working_part_or_full_time",
    )
    no_work_to_part_time = get_work_transition_unweighted(
        dat,
        previous_choice="not_working_part_or_full_time",
        current_choice="part_time",
    )
    no_work_to_full_time = get_work_transition_unweighted(
        dat,
        previous_choice="not_working_part_or_full_time",
        current_choice="full_time",
    )

    part_time_to_no_work = get_work_transition_unweighted(
        dat,
        previous_choice="part_time",
        current_choice="not_working_part_or_full_time",
    )
    part_time_to_part_time = get_work_transition_unweighted(
        dat,
        previous_choice="part_time",
        current_choice="part_time",
    )
    part_time_to_full_time = get_work_transition_unweighted(
        dat,
        previous_choice="part_time",
        current_choice="full_time",
    )

    full_time_to_no_work = get_work_transition_unweighted(
        dat,
        previous_choice="full_time",
        current_choice="not_working_part_or_full_time",
    )
    full_time_to_part_time = get_work_transition_unweighted(
        dat,
        previous_choice="full_time",
        current_choice="part_time",
    )
    full_time_to_full_time = get_work_transition_unweighted(
        dat,
        previous_choice="full_time",
        current_choice="full_time",
    )

    return pd.concat(
        [
            no_work_to_no_work,
            no_work_to_part_time,
            no_work_to_full_time,
            part_time_to_no_work,
            part_time_to_part_time,
            part_time_to_full_time,
            full_time_to_no_work,
            full_time_to_part_time,
            full_time_to_full_time,
        ],
        ignore_index=False,
        axis=0,
    )


def get_caregiving_status_by_mother_health_and_marital_status(
    mother_couple,
    mother_single,
    weight,
):
    """Get caregiving status by mother's health and marital status."""
    informal_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        mother_couple,
        parent="mother",
        moment="informal_care_child_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    informal_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        mother_single,
        parent="mother",
        moment="informal_care_child_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    home_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        mother_couple,
        parent="mother",
        moment="home_care_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    home_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        mother_single,
        parent="mother",
        moment="home_care_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_informal_care_by_mother_health_couple = (
        get_caregiving_status_by_parental_health(
            mother_couple,
            parent="mother",
            moment="only_informal_weighted",
            is_other_parent_alive=True,
            weight=weight,
        )
    )
    only_informal_care_by_mother_health_single = (
        get_caregiving_status_by_parental_health(
            mother_single,
            parent="mother",
            moment="only_informal_weighted",
            is_other_parent_alive=False,
            weight=weight,
        )
    )

    combination_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        mother_couple,
        parent="mother",
        moment="combination_care_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    combination_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        mother_single,
        parent="mother",
        moment="combination_care_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_home_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        mother_couple,
        parent="mother",
        moment="only_home_care_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    only_home_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        mother_single,
        parent="mother",
        moment="only_home_care_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    return pd.concat(
        [
            informal_care_by_mother_health_couple,
            informal_care_by_mother_health_single,
            home_care_by_mother_health_couple,
            home_care_by_mother_health_single,
            only_informal_care_by_mother_health_couple,
            only_informal_care_by_mother_health_single,
            combination_care_by_mother_health_couple,
            combination_care_by_mother_health_single,
            only_home_care_by_mother_health_couple,
            only_home_care_by_mother_health_single,
        ],
        ignore_index=False,
        axis=0,
    )


def get_caregiving_status_by_father_health_and_marital_status(
    father_couple,
    father_single,
    weight,
):
    """Get informal care by father's health and marital status."""
    informal_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        father_couple,
        parent="father",
        moment="informal_care_child_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    informal_care_by_father_health_single = get_caregiving_status_by_parental_health(
        father_single,
        parent="father",
        moment="informal_care_child_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    home_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        father_couple,
        parent="father",
        moment="home_care_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    home_care_by_father_health_single = get_caregiving_status_by_parental_health(
        father_single,
        parent="father",
        moment="home_care_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_informal_care_by_father_health_couple = (
        get_caregiving_status_by_parental_health(
            father_couple,
            parent="father",
            moment="only_informal_weighted",
            is_other_parent_alive=True,
            weight=weight,
        )
    )
    only_informal_care_by_father_health_single = (
        get_caregiving_status_by_parental_health(
            father_single,
            parent="father",
            moment="only_informal_weighted",
            is_other_parent_alive=False,
            weight=weight,
        )
    )

    combination_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        father_couple,
        parent="father",
        moment="combination_care_weighted",
        is_other_parent_alive=True,
        weight=weight,
    )
    combination_care_by_father_health_single = get_caregiving_status_by_parental_health(
        father_single,
        parent="father",
        moment="combination_care_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    only_home_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        father_couple,
        moment="only_home_care_weighted",
        parent="father",
        is_other_parent_alive=True,
        weight=weight,
    )
    only_home_care_by_father_health_single = get_caregiving_status_by_parental_health(
        father_single,
        parent="father",
        moment="only_home_care_weighted",
        is_other_parent_alive=False,
        weight=weight,
    )

    return pd.concat(
        [
            informal_care_by_father_health_couple,
            informal_care_by_father_health_single,
            home_care_by_father_health_couple,
            home_care_by_father_health_single,
            only_informal_care_by_father_health_couple,
            only_informal_care_by_father_health_single,
            combination_care_by_father_health_couple,
            combination_care_by_father_health_single,
            only_home_care_by_father_health_couple,
            only_home_care_by_father_health_single,
        ],
        ignore_index=False,
        axis=0,
    )


def get_employment_by_caregiving_status_by_age_bin(
    dat,
    age_bins_coarse,
    intensive_care_var,
    weight,
):
    share_working_informal_care_by_age_bin = get_share_by_informal_care_type_by_age_bin(
        dat,
        age_bins=age_bins_coarse,
        moment="working_part_or_full_time",
        is_caregiver=True,
        care_type=intensive_care_var,
        weight=weight,
    )
    share_working_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="working_part_or_full_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    share_not_working_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="not_working_part_or_full_time",
            is_caregiver=True,
            care_type=intensive_care_var,
            weight=weight,
        )
    )
    share_not_working_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="not_working_part_or_full_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    share_working_full_time_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="full_time",
            is_caregiver=True,
            care_type=intensive_care_var,
            weight=weight,
        )
    )
    share_working_full_time_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="full_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    share_working_part_time_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="part_time",
            is_caregiver=True,
            care_type=intensive_care_var,
            weight=weight,
        )
    )
    share_working_part_time_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            dat,
            age_bins=age_bins_coarse,
            moment="part_time",
            is_caregiver=False,
            care_type=intensive_care_var,
            weight=weight,
        )
    )

    return pd.concat(
        [
            1 - share_working_informal_care_by_age_bin,
            share_not_working_informal_care_by_age_bin,
            share_working_part_time_informal_care_by_age_bin,
            share_working_full_time_informal_care_by_age_bin,
            #
            1 - share_working_no_informal_care_by_age_bin,
            share_not_working_no_informal_care_by_age_bin,
            share_working_part_time_no_informal_care_by_age_bin,
            share_working_full_time_no_informal_care_by_age_bin,
        ],
        ignore_index=False,
        axis=0,
    )


# ================================================================================
# Auxiliary functions
# ================================================================================


def get_work_transition_unweighted(dat, previous_choice, current_choice):
    """Get transition from previous to current work status."""
    return pd.Series(
        {
            f"transition_from_{previous_choice}_to_{current_choice}": len(
                dat[
                    (dat[f"lagged_{previous_choice}"] == True)
                    & (dat[current_choice] == True)
                ],
            )
            / len(
                dat[
                    (dat[f"lagged_{previous_choice}"] == True)
                    & (dat[current_choice].notna())
                ],
            ),
        },
    )


def get_work_transition_weighted(dat, previous_choice, current_choice, weight):
    """Get transition from previous to current work status using survey weights."""
    return pd.Series(
        {
            f"transition_weighted_from_{previous_choice}_to_{current_choice}": dat.loc[
                (dat[f"lagged_{previous_choice}"] == True)
                & (dat[current_choice] == True),
                current_choice,
            ].sum()
            / dat.loc[
                (dat[f"lagged_{previous_choice}"] == True)
                & (dat[current_choice].notna()),
                weight,
            ].sum(),
        },
    )


def get_care_transition_unweighted(dat, previous_choice, current_choice):
    """Get transition from previous to current caregiving status."""
    return pd.Series(
        {
            f"transition_from_{previous_choice}_to_{current_choice}": len(
                dat[
                    (dat[f"lagged_{previous_choice}"] == True)
                    & (dat[current_choice] == True)
                ],
            )
            / len(
                dat[
                    (dat[f"lagged_{previous_choice}"] == True)
                    & (dat[current_choice].notna())
                ],
            ),
        },
    )


def get_care_transition_weighted(dat, previous_choice, current_choice, weight):
    """Get transition from previous to current caregiving status using weights."""
    return pd.Series(
        {
            f"transition_weighted_from_{previous_choice}_to_{current_choice}": dat.loc[
                dat[f"lagged_{previous_choice}"] == True,
                current_choice,
            ].sum()
            / dat.loc[
                (dat[f"lagged_{previous_choice}"] == True)
                & (dat[current_choice].notna()),
                weight,
            ].sum(),
        },
    )


# ================================================================================
# Moment functions
# ================================================================================


def get_share_by_age(
    dat,
    moment,
    weight="hh_weight",
):
    """Get share of people by age."""
    return pd.Series(
        {
            f"{moment}_{age}": dat.loc[
                (dat["age"] == age),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] == age),
                weight,
            ].sum()
            for age in range(MIN_AGE + 1, MAX_AGE + 1)
        },
    )


def get_share_by_age_bin(
    dat,
    age_bins,
    moment,
    weight="hh_weight",
):
    """Get share of people by age bin."""
    return pd.Series(
        {
            f"{moment}_{age_bin[0]}_{age_bin[1]}": dat.loc[
                (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
                weight,
            ].sum()
            for age_bin in age_bins
        },
    )


def get_share_by_informal_care_type_by_age_bin(
    dat,
    age_bins,
    moment,
    is_caregiver,
    care_type,
    weight="hh_weight",
):
    """Get share of people by informal care type and age bin."""
    is_care = (1 - is_caregiver) * "no_" + "informal_care"

    return pd.Series(
        {
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
        },
    )


def get_share_by_informal_care_type(dat, moment, is_caregiver, care_type, weight):
    """Get share of people by informal care type."""
    is_care = (1 - is_caregiver) * "no" + "informal_care"

    return pd.Series(
        {
            f"{moment}_{is_care}": dat.loc[
                (dat[care_type] == is_caregiver),
                moment,
            ].sum()
            / dat.loc[
                (dat[care_type] == is_caregiver),
                weight,
            ].sum(),
        },
    )


def get_income_by_age(dat, moment, weight):
    """Calculate mean income by age."""
    return pd.Series(
        {
            f"{moment}_{age}": dat.loc[
                (dat["age"] == age),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] == age),
                weight,
            ].sum()
            for age in range(MIN_AGE + 1, MAX_AGE + 1)
        },
    )


def get_income_by_age_bin(dat, age_bins, moment, weight):
    """Calculate mean income by age bin."""
    return pd.Series(
        {
            f"{moment}_{age_bin[0]}_{age_bin[1]}": dat.loc[
                (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
                weight,
            ].sum()
            for age_bin in age_bins
        },
    )


def get_wealth_by_age_bin(dat, age_bins, moment, weight):
    """Calculate mean wealth by age bin."""
    return pd.Series(
        {
            f"{moment}_{age_bin[0]}_{age_bin[1]}": dat.loc[
                (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
                weight,
            ].sum()
            for age_bin in age_bins
        },
    )


def get_caregiving_status_by_parental_health(
    dat,
    moment,
    parent,
    is_other_parent_alive,
    weight,
):
    """Get caregiving status by health and marital status of parent."""
    parent_status = (
        is_other_parent_alive * "couple" + (1 - is_other_parent_alive) * "single"
    )

    return pd.Series(
        {
            f"{moment}_{parent}_{parent_status}_health_{health}": dat.loc[
                (dat["married"] == is_other_parent_alive) & (dat["health"] == health),
                moment,
            ].sum()
            / dat.loc[
                (dat["health"] == health) & (dat["married"] == is_other_parent_alive),
                weight,
            ].sum()
            for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
        },
    )


def get_employment_transitions_soep():
    """Get employment transitions of females age 51-65 from SOEP."""
    return pd.Series(
        {
            "not_working_to_not_working": 0.93383000,
            "not_working_to_part_time": 0.05226726,
            "not_working_to_full_time": 0.01390274,
            #
            "part_time_to_not_working": 0.1275953,
            "part_time_to_part_time": 0.80193001,
            "part_time_to_full_time": 0.07047471,
            #
            "full_time_to_not_working": 0.06666667,
            "full_time_to_part_time": 0.06812715,
            "full_time_to_full_time": 0.86520619,
        },
    )


# ================================================================================


def multiply_rows_with_weight(dat, weight):
    """Weight each row of a DataFrame by survey weight."""
    # Create a DataFrame of weights with the same shape as dat
    weights = dat[weight].to_numpy().reshape(-1, 1)

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
    data_columns = dat.drop(columns=static_cols).to_numpy()

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

    dat_weighted[f"{weight}_avg"] = dat_weighted.groupby("mergeid")[weight].transform(
        "mean",
    )

    return dat_weighted


# ================================================================================
# Deflate
# ================================================================================


def deflate_income_and_wealth(dat, cpi):
    """Deflate income and wealth data."""
    dat = dat.copy()

    base_year_cpi = cpi[cpi["int_year"] == BASE_YEAR]["cpi"].iloc[0]
    cpi["normalized_cpi"] = (cpi["cpi"] / base_year_cpi) * 100

    dat_with_cpi = dat.merge(cpi, on="int_year")

    vars_to_deflate = ["hnetw", "thinc", "thinc2", "ydip", "yind", "labor_income"]

    for var in vars_to_deflate:
        dat_with_cpi[f"real_{var}"] = (
            dat_with_cpi[var] * 100 / dat_with_cpi["normalized_cpi"]
        )

    return dat_with_cpi
