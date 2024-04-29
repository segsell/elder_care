"""Create empirical moments for MSM estimation.

The SHARE data sets on
- children
- parent child combinations

are used.

"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from elder_care.config import BLD
from elder_care.model.shared import (
    BAD_HEALTH,
    BASE_YEAR,
    FEMALE,
    GOOD_HEALTH,
    MAX_AGE,
    MIN_AGE,
)


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


# @pytask.mark.skip(reason="Respecifying moments.")
def task_create_moments(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_parent_child_hh_weight: Path = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    path_to_save: Annotated[Path, Product] = BLD / "moments" / "empirical_moments.csv",
) -> None:
    """Create empirical moments for SHARE data.

    age_bins_coarse = [
        (AGE_40, AGE_45),
        (AGE_45, AGE_60),
        (AGE_50, AGE_55),
        (AGE_55, AGE_60),
        (AGE_60, AGE_65),
        (AGE_65, AGE_70),
    ]

    net_income_by_age_bin_part_time = get_income_by_employment_by_age_bin(
        dat,
        age_bins_coarse,
        employment_status="part_time",
        moment="real_labor_income",
        weight=weight,
    )
    net_income_by_age_bin_full_time = get_income_by_employment_by_age_bin(
        dat,
        age_bins_coarse,
        employment_status="full_time",
        moment="real_labor_income",
        weight=weight,
    )

    # income by age, working and non-working?
    # must condition on working

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
        age_bins_coarse,
        moment="real_hnetw",
        weight=weight,
    )

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

    caregiving_by_mother_couple_health_sample_sizes = table(mother_couple["health"])
    caregiving_by_mother_single_health_sample_sizes = table(mother_single["health"])

    caregiving_by_father_couple_health_sample_sizes = table(father_couple["health"])
    caregiving_by_father_single_health_sample_sizes = table(father_single["health"])


    parent["health_weighted"] = parent["health"] * parent[weight]
    data = parent.loc[parent["gender"] == FEMALE]
    moment = "health_weighted"
    initial_mother_health = pd.Series(
        {
            f"mother_health_{health}": data.loc[
                (data["health"] == health) & (data["age"] >= 65) & (data["age"] <= 68),
                moment,
            ].sum()
            / data.loc[
                data["health"].notna() & (data["age"] >= 65) & (data["age"] <= 68),
                weight,
            ].sum()
            for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
        },
    )


    care_mix_moments = pd.read_csv(path_to_care_mix_moments, index_col=0)["0"]


    dat["mother_age_unweighted"] = dat["mother_age"] / dat[weight]
    dat["mother_alive_unweighted"] = dat["mother_alive"] / dat[weight]
    # len(dat.loc[dat["age"] == 44])
    mother_mean_age_at_start = np.mean(
        dat.loc[
            (dat["age"] >= MIN_AGE - 3) & (dat["age"] <= MIN_AGE + 2),
            "mother_age_unweighted",
        ]
    )
    mother_var_age_at_start = np.mean(
        dat.loc[
            (dat["age"] >= MIN_AGE - 3) & (dat["age"] <= MIN_AGE + 2),
            "mother_age_unweighted",
        ]
    )

    mother_mean_alive_at_start = np.mean(
        dat.loc[
            (dat["age"] >= MIN_AGE - 3) & (dat["age"] <= MIN_AGE + 2),
            "mother_alive_unweighted",
        ]
    )


    # 10.04.2024

    dat["mother_age_unweighted"] = dat["mother_age"] / dat[weight]
    dat["mother_alive_unweighted"] = dat["mother_alive"] / dat[weight]
    # len(dat.loc[dat["age"] == 44])
    mother_mean_age_at_start = np.mean(
        dat.loc[
            (dat["age"] >= MIN_AGE - 3) & (dat["age"] <= MIN_AGE + 2),
            "mother_age_unweighted",
        ],
    )
    mother_std_age_at_start = np.std(
        dat.loc[
            (dat["age"] >= MIN_AGE - 3) & (dat["age"] <= MIN_AGE + 2),
            "mother_age_unweighted",
        ],
    )

    mother_mean_alive_at_start = np.mean(
        dat.loc[
            (dat["age"] >= MIN_AGE - 3) & (dat["age"] <= MIN_AGE + 2),
            "mother_alive_unweighted",
        ],
    )

    mother_health_good = len(
        mother.loc[(mother["age"] == 76) & (mother["health"] == GOOD_HEALTH)],
    ) / len(mother.loc[mother["age"] == 76])
    mother_health_medium = len(
        mother.loc[(mother["age"] == 76) & (mother["health"] == MEDIUM_HEALTH)],
    ) / len(mother.loc[mother["age"] == 76])
    mother_health_bad = len(
        mother.loc[(mother["age"] == 76) & (mother["health"] == BAD_HEALTH)],
    ) / len(mother.loc[mother["age"] == 76])

    _total = mother_health_good + mother_health_medium + mother_health_bad

    """
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    dat = deflate_income_and_wealth(dat, cpi_data)

    weight = "hh_weight"
    intensive_care_var = "intensive_care_no_other"

    dat = dat.copy()
    dat = dat.loc[(dat["age"] >= MIN_AGE) & (dat["age"] < MAX_AGE)]

    share_informal_care_by_age_bin = get_share_informal_maternal_care_by_age_bin_soep()

    # ================================================================================
    # Parent child data (mother)
    # ================================================================================

    parent = parent_hh_weight.copy()

    parent["no_care_weighted"] = parent["no_care"] * parent[weight]
    parent["informal_care_child_weighted"] = (
        parent["informal_care_child"] * parent[weight]
    )
    parent["home_care_weighted"] = parent["home_care"] * parent[weight]
    parent["formal_care_weighted"] = parent["formal_care"] * parent[weight]
    parent["combination_care_weighted"] = parent["combination_care"] * parent[weight]
    parent["no_combination_care_weighted"] = (
        parent["no_combination_care"] * parent[weight]
    )

    parent["informal_care_child_no_comb_weighted"] = (
        parent["informal_care_child_no_comb"] * parent[weight]
    )
    parent["formal_care_no_comb_weighted"] = (
        parent["formal_care_no_comb"] * parent[weight]
    )

    dat["no_intensive_informal_weighted"] = dat["no_intensive_informal"] * dat[weight]
    dat["intensive_care_no_other_weighted"] = (
        dat["intensive_care_no_other"] * dat[weight]
    )
    intensive_care_var_weighted = "intensive_care_no_other_weighted"

    parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    parent["no_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )
    parent["only_formal_weighted"] = parent["only_formal"] * parent[weight]
    parent["no_only_formal_weighted"] = parent["no_only_formal"] * parent[weight]
    parent["no_only_informal_weighted"] = parent["no_only_informal"] * parent[weight]

    mother = parent[(parent["gender"] == FEMALE)]

    parent["only_informal_care_child_weighted"] = (
        parent["informal_care_child"] * parent[weight]
    )
    parent["no_only_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )
    parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    parent["no_formal_care_weighted"] = parent["no_formal_care"] * parent[weight]
    parent["no_informal_care_child_weighted"] = (
        parent["no_informal_care_child"] * parent[weight]
    )

    # caregiving_by_mother_health_and_presence_of_sibling = (
    #     get_caregiving_status_by_mother_health_and_presence_of_sibling(
    #         mother,
    #         sibling_var="has_two_daughters",
    #         weight=weight,
    #     )
    # )

    # first_half = caregiving_by_mother_health_and_presence_of_sibling[:4]
    # second_half = caregiving_by_mother_health_and_presence_of_sibling[4:]

    # normalized_first_half = first_half / first_half.sum()
    # normalized_second_half = second_half / second_half.sum()

    # caregiving_by_mother_health_and_presence_of_sibling_normalized = pd.concat(
    #     [normalized_first_half, normalized_second_half],
    # )

    # ================================================================================
    # Labor and caregiving transitions
    # ================================================================================

    employment_transitions_soep = get_employment_transitions_soep()

    # care_transitions_estimation_data = (
    #     get_care_transitions_from_estimation_data_weighted(
    #         dat,
    #         intensive_care_var=intensive_care_var,
    #         intensive_care_var_weighted=intensive_care_var_weighted,
    #         weight=weight,
    #     )
    # )

    # care_transitions_parent_child_data = (
    #     get_care_transitions_from_parent_child_data_weighted(
    #         parent,
    #         weight=weight,
    #     )
    # )

    employment_by_age_soep = get_employment_by_age_soep()
    # employment_by_age_bin_caregivers_soep = (
    #     get_employment_by_age_bin_informal_parental_caregivers_soep()
    # )
    # employment_by_age_bin_non_caregivers_soep = (
    #     get_employment_by_age_bin_non_informal_caregivers_soep()
    # )

    ols_coeffs_savings_rate = get_coefficients_savings_rate_regression()

    all_moments = pd.concat(
        [
            employment_by_age_soep,
            ols_coeffs_savings_rate,
            # employment_by_age_bin_non_caregivers_soep,
            # employment_by_age_bin_caregivers_soep,
            # #
            # share_informal_care_by_age_bin,
            # caregiving_by_mother_health_and_presence_of_sibling_normalized,
            # #
            employment_transitions_soep,
            # care_transitions_estimation_data,
            # care_transitions_parent_child_data,
        ],
        ignore_index=False,
        axis=0,
    )

    all_moments.to_csv(path_to_save)


# ================================================================================
# Transitions
# ================================================================================


def get_care_transitions_from_parent_child_data_weighted(parent, weight):
    """Get care transitions from parent child data set using survey weights.

    # informal no_informal_to_no_informal_weighted = get_care_transition_weighted(
    parent,     previous_choice="no_informal_care_child",
    current_choice="no_informal_care_child_weighted",     weight=weight, )
    no_informal_to_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="no_informal_care_child",
    current_choice="informal_care_child_weighted",     weight=weight, )

    informal_to_no_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="informal_care_child",
    current_choice="no_informal_care_child_weighted",     weight=weight, )
    informal_to_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="informal_care_child",
    current_choice="informal_care_child_weighted",     weight=weight, )


    no_informal_to_no_combination_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="no_combination_care_weighted",
        weight=weight,
    )
    no_informal_to_combination_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="combination_care_weighted",
        weight=weight,
    )
    informal_to_no_combination_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="no_combination_care_weighted",
        weight=weight,
    )
    informal_to_combination_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="combination_care_weighted",
        weight=weight,
    )

    """
    # ================================================================================
    # Informal care
    # ================================================================================

    no_informal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="no_formal_care_weighted",
        weight=weight,
    )
    no_informal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_informal_care_child",
        current_choice="formal_care_weighted",
        weight=weight,
    )
    informal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="no_formal_care_weighted",
        weight=weight,
    )
    informal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="informal_care_child",
        current_choice="formal_care_weighted",
        weight=weight,
    )

    # ================================================================================
    # Formal Care
    # ================================================================================

    no_formal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_formal_care",
        current_choice="no_formal_care_weighted",
        weight=weight,
    )
    no_formal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_formal_care",
        current_choice="formal_care_weighted",
        weight=weight,
    )

    formal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="formal_care",
        current_choice="no_formal_care_weighted",
        weight=weight,
    )
    formal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="formal_care",
        current_choice="formal_care_weighted",
        weight=weight,
    )

    no_formal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_formal_care",
        current_choice="no_informal_care_child_weighted",
        weight=weight,
    )
    no_formal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_formal_care",
        current_choice="informal_care_child_weighted",
        weight=weight,
    )
    formal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="formal_care",
        current_choice="no_informal_care_child_weighted",
        weight=weight,
    )
    formal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="formal_care",
        current_choice="informal_care_child_weighted",
        weight=weight,
    )

    return pd.concat(
        [
            # Informal care
            # to (no) formal
            no_informal_to_no_formal_weighted,
            no_informal_to_formal_weighted,
            informal_to_no_formal_weighted,
            informal_to_formal_weighted,
            # Formal care
            # to (no) informal
            no_formal_to_no_informal_weighted,
            no_formal_to_informal_weighted,
            formal_to_no_informal_weighted,
            formal_to_informal_weighted,
            # to (no) formal
            no_formal_to_no_formal_weighted,
            no_formal_to_formal_weighted,
            formal_to_no_formal_weighted,
            formal_to_formal_weighted,
        ],
        ignore_index=False,
        axis=0,
    )


def _get_care_transitions_from_parent_child_data_weighted_only(parent, weight):
    """Get care transitions from parent child data set using survey weights.

    # informal no_informal_to_no_informal_weighted = get_care_transition_weighted(
    parent,     previous_choice="no_informal_care_child",
    current_choice="no_informal_care_child_weighted",     weight=weight, )
    no_informal_to_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="no_informal_care_child",
    current_choice="informal_care_child_weighted",     weight=weight, )

    informal_to_no_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="informal_care_child",
    current_choice="no_informal_care_child_weighted",     weight=weight, )
    informal_to_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="informal_care_child",
    current_choice="informal_care_child_weighted",     weight=weight, )

    """
    no_formal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_only_formal",
        current_choice="no_only_formal_weighted",
        weight=weight,
    )
    no_formal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_only_formal",
        current_choice="only_formal_weighted",
        weight=weight,
    )

    formal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="only_formal",
        current_choice="no_only_formal_weighted",
        weight=weight,
    )
    formal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="only_formal",
        current_choice="only_formal_weighted",
        weight=weight,
    )

    no_formal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_only_formal",
        current_choice="no_only_informal_weighted",
        weight=weight,
    )
    no_formal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_only_formal",
        current_choice="only_informal_weighted",
        weight=weight,
    )

    formal_to_no_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="only_formal",
        current_choice="no_only_informal_weighted",
        weight=weight,
    )
    formal_to_informal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="only_formal",
        current_choice="only_informal_weighted",
        weight=weight,
    )

    # informal care transitions from parent child data set
    no_informal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_only_informal",
        current_choice="no_only_formal_weighted",
        weight=weight,
    )
    no_informal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="no_only_informal",
        current_choice="only_formal_weighted",
        weight=weight,
    )

    informal_to_no_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="only_informal",
        current_choice="no_only_formal_weighted",
        weight=weight,
    )
    informal_to_formal_weighted = get_care_transition_weighted(
        parent,
        previous_choice="only_informal",
        current_choice="only_formal_weighted",
        weight=weight,
    )

    return pd.concat(
        [
            no_informal_to_no_formal_weighted,
            no_informal_to_formal_weighted,
            informal_to_no_formal_weighted,
            informal_to_formal_weighted,
            #
            no_formal_to_no_informal_weighted,
            no_formal_to_informal_weighted,
            formal_to_no_informal_weighted,
            formal_to_informal_weighted,
            #
            no_formal_to_no_formal_weighted,
            no_formal_to_formal_weighted,
            formal_to_no_formal_weighted,
            formal_to_formal_weighted,
        ],
        ignore_index=False,
        axis=0,
    )


def get_care_transitions_from_parent_child_data_weighted_home_care(parent, weight):
    """Get care transitions from parent child data set using survey weights.

    # informal no_informal_to_no_informal_weighted = get_care_transition_weighted(
    parent,     previous_choice="no_informal_care_child",
    current_choice="no_informal_care_child_weighted",     weight=weight, )
    no_informal_to_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="no_informal_care_child",
    current_choice="informal_care_child_weighted",     weight=weight, )

    informal_to_no_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="informal_care_child",
    current_choice="no_informal_care_child_weighted",     weight=weight, )
    informal_to_informal_weighted = get_care_transition_weighted(     parent,
    previous_choice="informal_care_child",
    current_choice="informal_care_child_weighted",     weight=weight, )

    """
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

    return pd.concat(
        [
            no_informal_to_no_formal_weighted,
            no_informal_to_formal_weighted,
            informal_to_no_formal_weighted,
            informal_to_formal_weighted,
            #
            no_formal_to_no_informal_weighted,
            no_formal_to_informal_weighted,
            formal_to_no_informal_weighted,
            formal_to_informal_weighted,
            #
            no_formal_to_no_formal_weighted,
            no_formal_to_formal_weighted,
            formal_to_no_formal_weighted,
            formal_to_formal_weighted,
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


def get_emplyoment_transitions_share(dat, weight):
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


# ================================================================================
# Caregiving status by parental health
# ================================================================================


def get_caregiving_status_by_mother_health_and_presence_of_sibling(
    mother,
    sibling_var,
    weight,
):
    """Get caregiving status by mother's health and marital status.

    home_care_by_mother_health = get_caregiving_status_by_parental_health_any_sibling(
        mother,
        parent="mother",
        moment="home_care_weighted",
        weight=weight,
    )

    home_care_by_mother_health_sibling = (
        get_caregiving_status_by_parental_health_and_presence_of_sibling(
            mother,
            parent="mother",
            moment="home_care_weighted",
            has_other_sibling=True,
            sibling_var=sibling_var,
            weight=weight,
        )
    )


    only_informal_mother_health_1,0.05254345856715846
    only_informal_mother_health_2,0.07798704406329442
    only_formal_mother_health_1,0.024764440332745708
    only_formal_mother_health_2,0.10506268806066696
    combination_care_mother_health_1,0.018357843911438215
    combination_care_mother_health_2,0.07055417366795928
    only_informal_mother_sibling_health_1,0.09563346857355969
    only_informal_mother_sibling_health_2,0.14147469097360235
    only_formal_mother_sibling_health_1,0.04490369113141575
    only_formal_mother_sibling_health_2,0.07847892386314938
    combination_care_mother_sibling_health_1,0.036819814566593945
    combination_care_mother_sibling_health_2,0.06878949679959141

    informal_care_mother_health_1,0.07299073869631172
    informal_care_mother_health_2,0.1590767936814578
    formal_care_mother_health_1,0.13534639478066254
    formal_care_mother_health_2,0.36428609226949604
    combination_care_mother_health_1,0.018357843911438215
    combination_care_mother_health_2,0.07055417366795928
    informal_care_mother_sibling_health_1,0.13980470689107538
    informal_care_mother_sibling_health_2,0.2213289679890795
    formal_care_mother_sibling_health_1,0.14468126887776186
    formal_care_mother_sibling_health_2,0.3652255363660134
    combination_care_mother_sibling_health_1,0.036819814566593945
    combination_care_mother_sibling_health_2,0.06878949679959141

    """
    no_care_by_mother_health = get_caregiving_status_by_parental_health_any_sibling(
        mother,
        parent="mother",
        moment="no_care_weighted",
        weight=weight,
    )
    informal_care_by_mother_health = (
        get_caregiving_status_by_parental_health_any_sibling(
            mother,
            parent="mother",
            moment="informal_care_child_no_comb_weighted",
            weight=weight,
        )
    )
    formal_care_by_mother_health = get_caregiving_status_by_parental_health_any_sibling(
        mother,
        parent="mother",
        moment="formal_care_no_comb_weighted",
        weight=weight,
    )

    combination_care_by_mother_health = (
        get_caregiving_status_by_parental_health_any_sibling(
            mother,
            parent="mother",
            moment="combination_care_weighted",
            weight=weight,
        )
    )

    no_care_by_mother_health_sibling = (
        get_caregiving_status_by_parental_health_and_presence_of_sibling(
            mother,
            parent="mother",
            moment="no_care_weighted",
            has_other_sibling=True,
            sibling_var=sibling_var,
            weight=weight,
        )
    )
    informal_care_by_mother_health_sibling = (
        get_caregiving_status_by_parental_health_and_presence_of_sibling(
            mother,
            parent="mother",
            moment="informal_care_child_no_comb_weighted",
            has_other_sibling=True,
            sibling_var=sibling_var,
            weight=weight,
        )
    )
    formal_care_by_mother_health_sibling = (
        get_caregiving_status_by_parental_health_and_presence_of_sibling(
            mother,
            parent="mother",
            moment="formal_care_no_comb_weighted",
            has_other_sibling=True,
            sibling_var=sibling_var,
            weight=weight,
        )
    )
    combination_care_by_mother_health_sibling = (
        get_caregiving_status_by_parental_health_and_presence_of_sibling(
            mother,
            parent="mother",
            moment="combination_care_weighted",
            has_other_sibling=True,
            sibling_var=sibling_var,
            weight=weight,
        )
    )

    return pd.concat(
        [
            no_care_by_mother_health,
            informal_care_by_mother_health,
            formal_care_by_mother_health,
            combination_care_by_mother_health,
            no_care_by_mother_health_sibling,
            informal_care_by_mother_health_sibling,
            formal_care_by_mother_health_sibling,
            combination_care_by_mother_health_sibling,
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

    return pd.concat(
        [
            informal_care_by_mother_health_couple,
            home_care_by_mother_health_couple,
            combination_care_by_mother_health_couple,
            # only_informal_care_by_mother_health_couple,
            # only_informal_care_by_mother_health_single,
            informal_care_by_mother_health_single,
            home_care_by_mother_health_single,
            combination_care_by_mother_health_single,
            # only_home_care_by_mother_health_couple,
            # only_home_care_by_mother_health_single,
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

    return pd.concat(
        [
            informal_care_by_father_health_couple,
            combination_care_by_father_health_couple,
            home_care_by_father_health_couple,
            informal_care_by_father_health_single,
            home_care_by_father_health_single,
            combination_care_by_father_health_single,
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


def get_employment_share_by_age_share(dat, weight):
    """Get share working by age from SHARE data."""
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

    return pd.concat(
        [
            share_working_by_age,
            share_not_working_by_age,
            share_working_part_time_by_age,
            share_working_full_time_by_age,
        ],
        ignore_index=False,
        axis=0,
    )


def get_employment_share_by_age_bin_share(dat, age_bins_fine, weight):
    """Get employment share by age bin from SHARE data."""
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

    return pd.concat(
        [
            share_working_by_age_bin,
            share_not_working_by_age_bin,
            share_working_part_time_by_age_bin,
            share_working_full_time_by_age_bin,
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
    _previous_choice_string = (
        str(previous_choice)
        .replace("intensive_care_no_other", "informal_care")
        .replace("_weighted", "")
        .replace("_intensive", "")
        .replace("_child", "")
    )

    _current_choice_string = (
        str(current_choice)
        .replace("intensive_care_no_other", "informal_care")
        .replace("_weighted", "")
        .replace("_intensive", "")
        .replace("_child", "")
    )

    return pd.Series(
        {
            f"{_previous_choice_string}_to_{_current_choice_string}": dat.loc[
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
            for age in range(MIN_AGE, MAX_AGE)
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
            for age in range(MIN_AGE, MAX_AGE)
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


def get_income_by_employment_by_age_bin(
    dat,
    age_bins,
    employment_status,
    moment,
    weight,
):
    """Calculate mean income by age bin."""
    return pd.Series(
        {
            f"{moment}_{employment_status}_{age_bin[0]}_{age_bin[1]}": dat.loc[
                (dat[employment_status] > 0)
                & (dat["age"] > age_bin[0])
                & (dat["age"] <= age_bin[1]),
                moment,
            ].sum()
            / dat.loc[
                (dat[employment_status] > 0)
                & (dat["age"] > age_bin[0])
                & (dat["age"] <= age_bin[1]),
                weight,
            ].sum()
            for age_bin in age_bins
        },
    )


def get_wealth_by_age_bin(dat, age_bins, moment, weight):
    """Calculate mean wealth by age bin."""
    moment_string = moment.replace("hnetw", "wealth")
    return pd.Series(
        {
            f"{moment_string}_{age_bin[0]}_{age_bin[1]}": dat.loc[
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


def get_income_by_caregiving_status_and_age_bin(
    dat,
    age_bins,
    moment,
    is_caregiver,
    care_type,
    weight,
):
    """Calculate mean income by caregiving status and age bin."""
    is_care = (1 - is_caregiver) * "no" + "informal_care"

    return pd.Series(
        {
            f"{moment}_{is_care}_{age_bin[0]}_{age_bin[1]}": dat.loc[
                (dat["age"] > age_bin[0])
                & (dat["age"] <= age_bin[1])
                & (dat[care_type] == is_caregiver),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] > age_bin[0])
                & (dat["age"] <= age_bin[1])
                & (dat[care_type] == is_caregiver),
                weight,
            ].sum()
            for age_bin in age_bins
        },
    )


def get_wealth_by_caregiving_status_and_age_bin(
    dat,
    age_bins,
    moment,
    is_caregiver,
    care_type,
    weight,
):
    """Calculate mean wealth by caregiving status and age bin."""
    is_care = (1 - is_caregiver) * "no" + "informal_care"

    return pd.Series(
        {
            f"{moment}_{is_care}_{age_bin[0]}_{age_bin[1]}": dat.loc[
                (dat["age"] > age_bin[0])
                & (dat["age"] <= age_bin[1])
                & (dat[care_type] == is_caregiver),
                moment,
            ].sum()
            / dat.loc[
                (dat["age"] > age_bin[0])
                & (dat["age"] <= age_bin[1])
                & (dat[care_type] == is_caregiver),
                weight,
            ].sum()
            for age_bin in age_bins
        },
    )


# ================================================================================
# Auxiliary?
# ================================================================================


def get_share_informal_care_by_age_bin(
    dat,
    intensive_care_var,
    weight,
    age_bins_coarse,
):
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
        for age_bin in age_bins_coarse
    ]
    return pd.Series(
        {
            f"share_informal_care_{age_bin[0]}_{age_bin[1]}": share_intensive_care[i]
            for i, age_bin in enumerate(age_bins_coarse)
        },
    )


def get_share_informal_care_to_mother_by_age_bin(dat, weight, age_bins_coarse):
    dat["care_to_mother_intensive_weighted"] = dat["care_to_mother_intensive"]

    share_intensive_care = []
    share_intensive_care += [
        dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            "care_to_mother_intensive_weighted",
        ].sum()
        / dat.loc[
            (dat["age"] > age_bin[0]) & (dat["age"] <= age_bin[1]),
            weight,
        ].sum()
        for age_bin in age_bins_coarse
    ]

    return pd.Series(
        {
            f"share_informal_care_{age_bin[0]}_{age_bin[1]}": share_intensive_care[i]
            for i, age_bin in enumerate(age_bins_coarse)
        },
    )


# ================================================================================
# Parent-child sample
# ================================================================================


def get_caregiving_status_by_parental_health_any_sibling(
    dat,
    moment,
    parent,
    weight,
):
    """Get caregiving status by health and presence of sibling.

    Share of people providing informal care by health and presence of sibling out of all
    people with parents of that health category and sibling status.

    """
    moment_string = moment.replace("_weighted", "").replace("_child", "")

    return pd.Series(
        {
            f"{moment_string}_{parent}_health_{health}": dat.loc[
                dat["health"] == health,
                moment,
            ].sum()
            / dat.loc[
                dat["health"] == health,
                weight,
            ].sum()
            for health in [BAD_HEALTH]
        },
    )


def get_caregiving_status_by_parental_health_and_presence_of_sibling(
    dat,
    moment,
    parent,
    has_other_sibling,
    sibling_var,
    weight,
):
    """Get caregiving status by health and presence of sibling.

    Share of people providing informal care by health and presence of sibling out of all
    people with parents of that health category and sibling status.

    """
    moment_string = moment.replace("_weighted", "").replace("_child", "")

    sibling_status = (
        has_other_sibling * "sibling" + (1 - has_other_sibling) * "no_sibling"
    )

    return pd.Series(
        {
            f"{moment_string}_{sibling_status}_{parent}_health_{health}": dat.loc[
                (dat[sibling_var] == has_other_sibling) & (dat["health"] == health),
                moment,
            ].sum()
            / dat.loc[
                (dat[sibling_var] == has_other_sibling) & (dat["health"] == health),
                weight,
            ].sum()
            for health in [BAD_HEALTH]
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
    moment_string = moment.replace("_weighted", "").replace("_child", "")
    parent_status = (
        is_other_parent_alive * "couple" + (1 - is_other_parent_alive) * "single"
    )

    return pd.Series(
        {
            f"{moment_string}_{parent}_{parent_status}_health_{health}": dat.loc[
                (dat["married"] == is_other_parent_alive) & (dat["health"] == health),
                moment,
            ].sum()
            / dat.loc[
                (dat["health"] == health) & (dat["married"] == is_other_parent_alive),
                weight,
            ].sum()
            for health in [GOOD_HEALTH, BAD_HEALTH]
        },
    )


# ================================================================================
# SOEP
# ================================================================================


def get_coefficients_savings_rate_regression():
    """Get coefficients of savings rate regression.

    The coefficients are estimated using the SOEP data. The wealth variables used in the
    calculation of the savings rate are deflated by the consumer price index (CPI) to
    2015 prices.

    """
    return pd.Series(
        {
            "savings_rate_constant": 1.929170502311,
            "savings_rate_age": -0.0755207021,
            "savings_rate_age_squared": 0.0007550297,
            "savings_rate_high_education": 0.0050211845,
            "savings_rate_part_time": 0.0809547016,
            "savings_rate_full_time": 0.1041300926,
            # "savings_rate_informal_care": -0.0339010984,
        },
    )


def get_employment_transitions_soep():
    """Get employment transitions of females age 39-70 from SOEP."""
    return pd.Series(
        {
            "not_working_to_not_working": 0.9081446,
            "not_working_to_part_time": 0.07185538,
            "not_working_to_full_time": 0.02,
            #
            "part_time_to_not_working": 0.1070033,
            "part_time_to_part_time": 0.8167875,
            "part_time_to_full_time": 0.07620923,
            #
            "full_time_to_not_working": 0.0558282,
            "full_time_to_part_time": 0.07111375,
            "full_time_to_full_time": 0.873058,
        },
    )


def get_var_employment_transitions_soep():
    """Get variance of employment transitions."""
    return pd.Series(
        {
            "not_working_to_not_working": 0.0000014030153746,
            "not_working_to_part_time": 0.0000011223523576,
            "not_working_to_full_time": 0.0000003261830112,
            #
            "part_time_to_not_working": 0.0000026726781986,
            "part_time_to_part_time": 0.0000041901074618,
            "part_time_to_full_time": 0.0000019742078027,
            #
            "full_time_to_not_working": 0.0000015082989226,
            "full_time_to_part_time": 0.0000019227102296,
            "full_time_to_full_time": 0.0000032040667450,
        },
    )


def get_employment_transitions_soep_51_to_65():
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


def get_employment_by_caregiving_status_soep():
    """Get employment by caregiving status of females age 51-65."""
    return pd.Series(
        {
            "not_working_no_informal_care": 0.4313806,
            "part_time_no_informal_care": 0.2589699,
            "full_time_no_informal_care": 0.3096495,
            #
            "not_working_informal_care": 0.4925068,
            "part_time_informal_care": 0.2897366,
            "full_time_informal_care": 0.2177566,
        },
    )


def get_employment_by_age_soep():
    """Get employment shares by age of females age 39-70.

    "not_working_age_39": 0.3873923, "part_time_age_39": 0.35843870, "full_time_age_39":
    0.254168957,

    "not_working_age_70": 0.9805623,
    "part_time_age_70": 0.01388407,
    "full_time_age_70": 0.005553627,

    """
    return pd.Series(
        {
            # not working
            "not_working_age_40": 0.3624795,
            "not_working_age_41": 0.3429091,
            "not_working_age_42": 0.3217345,
            "not_working_age_43": 0.3128681,
            "not_working_age_44": 0.3001430,
            "not_working_age_45": 0.2968833,
            "not_working_age_46": 0.2921024,
            "not_working_age_47": 0.2930369,
            "not_working_age_48": 0.2827195,
            "not_working_age_49": 0.2772086,
            "not_working_age_50": 0.2789733,
            "not_working_age_51": 0.2890753,
            "not_working_age_52": 0.2981818,
            "not_working_age_53": 0.3093333,
            "not_working_age_54": 0.3253968,
            "not_working_age_55": 0.3312008,
            "not_working_age_56": 0.3494705,
            "not_working_age_57": 0.3658537,
            "not_working_age_58": 0.4085761,
            "not_working_age_59": 0.4484105,
            "not_working_age_60": 0.5026279,
            "not_working_age_61": 0.6012961,
            "not_working_age_62": 0.6930100,
            "not_working_age_63": 0.7615607,
            "not_working_age_64": 0.8543746,
            "not_working_age_65": 0.8963161,
            "not_working_age_66": 0.9447926,
            "not_working_age_67": 0.9586930,
            "not_working_age_68": 0.9703130,
            "not_working_age_69": 0.9747814,
            # part-time
            "part_time_age_40": 0.37338666,
            "part_time_age_41": 0.37436364,
            "part_time_age_42": 0.38436831,
            "part_time_age_43": 0.38283063,
            "part_time_age_44": 0.38827315,
            "part_time_age_45": 0.38005343,
            "part_time_age_46": 0.37396322,
            "part_time_age_47": 0.36248392,
            "part_time_age_48": 0.35486308,
            "part_time_age_49": 0.36013918,
            "part_time_age_50": 0.34748272,
            "part_time_age_51": 0.33326514,
            "part_time_age_52": 0.32491979,
            "part_time_age_53": 0.31377778,
            "part_time_age_54": 0.30905696,
            "part_time_age_55": 0.31471457,
            "part_time_age_56": 0.30509329,
            "part_time_age_57": 0.29766588,
            "part_time_age_58": 0.27831715,
            "part_time_age_59": 0.25571668,
            "part_time_age_60": 0.23430152,
            "part_time_age_61": 0.18230487,
            "part_time_age_62": 0.14037090,
            "part_time_age_63": 0.11734104,
            "part_time_age_64": 0.07310628,
            "part_time_age_65": 0.05763518,
            "part_time_age_66": 0.03640704,
            "part_time_age_67": 0.02897657,
            "part_time_age_68": 0.02065182,
            "part_time_age_69": 0.01782112,
            # full-time
            "full_time_age_40": 0.264133794,
            "full_time_age_41": 0.282727273,
            "full_time_age_42": 0.293897216,
            "full_time_age_43": 0.304301267,
            "full_time_age_44": 0.311583840,
            "full_time_age_45": 0.323063224,
            "full_time_age_46": 0.333934367,
            "full_time_age_47": 0.344479148,
            "full_time_age_48": 0.362417375,
            "full_time_age_49": 0.362652233,
            "full_time_age_50": 0.373543929,
            "full_time_age_51": 0.377659574,
            "full_time_age_52": 0.376898396,
            "full_time_age_53": 0.376888889,
            "full_time_age_54": 0.365546218,
            "full_time_age_55": 0.354084646,
            "full_time_age_56": 0.345436208,
            "full_time_age_57": 0.336480462,
            "full_time_age_58": 0.313106796,
            "full_time_age_59": 0.295872839,
            "full_time_age_60": 0.263070539,
            "full_time_age_61": 0.216398986,
            "full_time_age_62": 0.166619116,
            "full_time_age_63": 0.121098266,
            "full_time_age_64": 0.072519084,
            "full_time_age_65": 0.046048723,
            "full_time_age_66": 0.018800358,
            "full_time_age_67": 0.012330456,
            "full_time_age_68": 0.009035173,
            "full_time_age_69": 0.007397445,
        },
    )


def get_employment_by_age_bin_informal_parental_caregivers_soep():
    """Get employment shares by age bin of informal parental caregivers.

    return pd.Series({
        "not_working_age_40_45": 0.4051896,
        "part_time_age_40_45": 0.3702595,
        "full_time_age_40_45": 0.2245509,
        #
        "not_working_age_45_50": 0.3209970,
        "part_time_age_45_50": 0.3655589,
        "full_time_age_45_50": 0.3134441,
        #
        "not_working_age_50_55": 0.3302812,
        "part_time_age_50_55": 0.3871812,
        "full_time_age_50_55": 0.2825376,
        #
        "not_working_age_55_60": 0.4206566,
        "part_time_age_55_60": 0.3201094,
        "full_time_age_55_60": 0.2592339,
        #
        "not_working_age_60_65": 0.7387153,
        "part_time_age_60_65": 0.1336806,
        "full_time_age_60_65": 0.1276042,
        #
        "not_working_age_65_70": 0.9545455,
        "part_time_age_65_70": 0.0275974,
        "full_time_age_65_70": 0.0178571,
    })

    """
    return pd.Series(
        {
            "not_working_age_40_45": 0.4051896,
            "not_working_age_45_50": 0.3209970,
            "not_working_age_50_55": 0.3302812,
            "not_working_age_55_60": 0.4206566,
            "not_working_age_60_65": 0.7387153,
            "not_working_age_65_70": 0.9545455,
            #
            "part_time_age_40_45": 0.3702595,
            "part_time_age_45_50": 0.3655589,
            "part_time_age_50_55": 0.3871812,
            "part_time_age_55_60": 0.3201094,
            "part_time_age_60_65": 0.1336806,
            "part_time_age_65_70": 0.0275974,
            #
            "full_time_age_40_45": 0.2245509,
            "full_time_age_45_50": 0.3134441,
            "full_time_age_50_55": 0.2825376,
            "full_time_age_55_60": 0.2592339,
            "full_time_age_60_65": 0.1276042,
            "full_time_age_65_70": 0.0178571,
        },
    )


def get_employment_by_age_bin_non_informal_caregivers_soep():
    """Get employment shares by age bin of non-informal caregivers."""
    return pd.Series(
        {
            "not_working_age_40_45": 0.3199924,
            "not_working_age_45_50": 0.2835182,
            "not_working_age_50_55": 0.2951276,
            "not_working_age_55_60": 0.3752067,
            "not_working_age_60_65": 0.6762942,
            "not_working_age_65_70": 0.9477137,
            #
            "part_time_age_40_45": 0.38325688,
            "part_time_age_45_50": 0.36695721,
            "part_time_age_50_55": 0.32171694,
            "part_time_age_55_60": 0.28790557,
            "part_time_age_60_65": 0.15151144,
            "part_time_age_65_70": 0.03306054,
            #
            "full_time_age_40_45": 0.296750765,
            "full_time_age_45_50": 0.349524564,
            "full_time_age_50_55": 0.383155452,
            "full_time_age_55_60": 0.336887723,
            "full_time_age_60_65": 0.172194346,
            "full_time_age_65_70": 0.019225773,
        },
    )


def get_share_informal_maternal_care_by_age_bin_soep():

    return pd.Series(
        {
            "share_informal_care_40_45": 0.02980982,
            "share_informal_care_45_50": 0.04036255,
            "share_informal_care_50_55": 0.05350986,
            "share_informal_care_55_60": 0.06193384,
            "share_informal_care_60_65": 0.05304824,
            "share_informal_care_65_70": 0.03079298,
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

    vars_to_deflate = [
        "hnetw",
        "thinc",
        "thinc2",
        "ydip",
        "yind",
        "labor_income",
        "labor_income_monthly",
        "hourly_wage",
    ]

    for var in vars_to_deflate:
        dat_with_cpi[f"real_{var}"] = (
            dat_with_cpi[var] * 100 / dat_with_cpi["normalized_cpi"]
        )

    return dat_with_cpi
