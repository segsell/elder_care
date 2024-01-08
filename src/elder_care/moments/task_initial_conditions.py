"""Initial conditions."""
from pathlib import Path

import jax
import numpy as np
import pandas as pd
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

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_initial_conditions(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
) -> None:
    """Create initial conditions at age 50.

    State variables:
    - age
    - married
    - has_sister or has_sibling
    - health mother = good
    - health father = good
    - mother alive
    - father alive

    Choices
    - share not working part time, full time
    - share informal care
    - share formal care

    """
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    dat = deflate_income_and_wealth(dat, cpi_data)

    weight = "hh_weight"
    intensive_care_var = "intensive_care_no_other"

    dat["has_sister_weighted"] = dat["has_sister"] * dat[weight]
    dat["has_sibling_weighted"] = dat["has_sibling"] * dat[weight]
    dat["married_unweighted"] = dat["married"] / dat[weight]
    dat["informal_care_weighted"] = dat[intensive_care_var] * dat[weight]
    dat["mother_health_unweighted"] = dat["mother_health"] / dat[weight]
    dat["father_health_unweighted"] = dat["father_health"] / dat[weight]
    dat["mother_alive_unweighted"] = dat["mother_alive"] / dat[weight]
    dat["father_alive_unweighted"] = dat["father_alive"] / dat[weight]

    dat["father_health_good"] = dat["father_health_unweighted"] == GOOD_HEALTH
    dat["father_health_medium"] = dat["father_health_unweighted"] == MEDIUM_HEALTH
    dat["father_health_bad"] = dat["father_health_unweighted"] == BAD_HEALTH
    dat["father_health_good_weighted"] = dat["father_health_good"] * dat[weight]
    dat["father_health_medium_weighted"] = dat["father_health_medium"] * dat[weight]
    dat["father_health_bad_weighted"] = dat["father_health_bad"] * dat[weight]

    dat["mother_health_good"] = dat["mother_health_unweighted"] == GOOD_HEALTH
    dat["mother_health_medium"] = dat["mother_health_unweighted"] == MEDIUM_HEALTH
    dat["mother_health_bad"] = dat["mother_health_unweighted"] == BAD_HEALTH
    dat["mother_health_good_weighted"] = dat["mother_health_good"] * dat[weight]
    dat["mother_health_medium_weighted"] = dat["mother_health_medium"] * dat[weight]
    dat["mother_health_bad_weighted"] = dat["mother_health_bad"] * dat[weight]

    employment_shares_age_50 = get_employment_shares_age_50_soep()

    share_has_sister = _get_share_at_min_age(dat, weight, moment="has_sister_weighted")
    share_has_sibling = _get_share_at_min_age(
        dat,
        weight,
        moment="has_sibling_weighted",
    )
    share_married = _get_share_at_min_age(dat, weight, moment="married")
    share_informal_care = _get_share_at_min_age(
        dat,
        weight,
        moment="informal_care_weighted",
    )

    share_mother_alive = _get_share_at_min_age(dat, weight, moment="mother_alive")
    share_father_alive = _get_share_at_min_age(dat, weight, moment="father_alive")

    share_mother_good_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="mother_health_good_weighted",
    )
    share_mother_medium_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="mother_health_medium_weighted",
    )
    share_mother_bad_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="mother_health_bad_weighted",
    )

    share_father_good_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="father_health_good_weighted",
    )
    share_father_medium_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="father_health_medium_weighted",
    )
    share_father_bad_health = _get_share_parental_health_at_min_age(
        dat,
        weight,
        moment="father_health_bad_weighted",
    )

    sum_mother_health = (
        share_mother_good_health + share_mother_medium_health + share_mother_bad_health
    )
    sum_father_health = (
        share_father_good_health + share_father_medium_health + share_father_bad_health
    )

    mother_good_health = (
        share_mother_good_health / sum_mother_health
    ) * share_mother_alive
    mother_medium_health = (
        share_mother_medium_health / sum_mother_health
    ) * share_mother_alive
    mother_bad_health = (
        share_mother_bad_health / sum_mother_health
    ) * share_mother_alive

    father_good_health = (
        share_father_good_health / sum_father_health
    ) * share_father_alive
    father_medium_health = (
        share_father_medium_health / sum_father_health
    ) * share_father_alive
    father_bad_health = (
        share_father_bad_health / sum_father_health
    ) * share_father_alive

    assert np.allclose(
        mother_good_health + mother_medium_health + mother_bad_health,
        share_mother_alive,
    )
    assert np.allclose(
        father_good_health + father_medium_health + father_bad_health,
        share_father_alive,
    )

    return pd.Series(
        {
            "share_has_sister": share_has_sister,
            "share_has_sibling": share_has_sibling,
            "share_married": share_married,
            "share_not_working": employment_shares_age_50["not_working"],
            "share_part_time": employment_shares_age_50["part_time"],
            "share_full_time": employment_shares_age_50["full_time"],
            "share_informal_care": share_informal_care,
            "share_mother_alive": share_mother_alive,
            "share_father_alive": share_father_alive,
            "mother_good_health": mother_good_health,
            "mother_medium_health": mother_medium_health,
            "mother_bad_health": mother_bad_health,
            "father_good_health": father_good_health,
            "father_medium_health": father_medium_health,
            "father_bad_health": father_bad_health,
        },
    )


# ==============================================================================


def draw_random_array(seed, n_agents, values, probabilities):
    """Draw a random array with given probabilities.

    Usage:

    seed = 2024
    n_agents = 10_000

    # Parameters
    values = jnp.array([-1, 0, 1, 2])  # Values to choose from
    probabilities = jnp.array([0.3, 0.3, 0.2, 0.2])  # Corresponding probabilities

    table(pd.DataFrame(random_array)[0]) / 1000

    """
    key = jax.random.PRNGKey(seed)  # Initialize a random key
    return jax.random.choice(key, values, shape=(n_agents,), p=probabilities)


def get_share_age_50_to_55(dat, weight, moment):
    return (
        dat.loc[(dat["age"] >= MIN_AGE) & (dat["age"] < AGE_55), moment].sum()
        / dat.loc[(dat["age"] >= MIN_AGE) & (dat["age"] < AGE_55), weight].sum()
    )


def _get_share_at_min_age(dat, weight, moment):
    return (
        dat.loc[(dat["age"] == MIN_AGE), moment].sum()
        / dat.loc[(dat["age"] == MIN_AGE), weight].sum()
    )


def _get_share_parental_health_at_min_age(dat, weight, moment):
    return (
        dat.loc[(dat["age"] == MIN_AGE), moment].sum()
        / dat.loc[
            (dat["age"] == MIN_AGE) & (dat[moment].notna()),
            weight,
        ].sum()
    )


def _get_share_parental_health_at_min_age_unweighted(dat, moment, outcome):
    return (
        dat[(dat["age"] == MIN_AGE) & (dat[moment] == outcome)].shape[0]
        / dat[(dat["age"] == MIN_AGE) & (dat[moment].notna())].shape[0]
    )


def get_share_at_min_age(dat, weight, moment):
    return (
        dat[(dat["age"] == MIN_AGE) & (dat[moment] == True)].shape[0]
        / dat.loc[(dat["age"] == MIN_AGE), weight].shape[0]
    )


def get_employment_shares_age_50_soep():
    """Get employment shares at age 50 from SOEP.

    (Pdb++)  0.2782375 + 0.35092820 + 0.37083427
    0.9999999700000001

    """
    return pd.Series(
        {
            "not_working": 0.2782375,
            "part_time": 0.35092820,
            "full_time": 0.37083427,
        },
    )
