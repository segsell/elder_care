"""Simulate moments for estimation."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from elder_care.model.shared import (
    AGE_BINS_SIM,
    BAD_HEALTH,
    FULL_TIME,
    INFORMAL_CARE,
    MAX_AGE_SIM,
    MIN_AGE_SIM,
    OUT_OF_LABOR,
    PART_TIME,
    RETIREMENT,
)


def simulate_moments(arr, idx):
    """Df has multiindex ["period", "agent"] necessary?

    column_indices = {col: idx for idx, col in enumerate(sim.columns)} idx =
    column_indices.copy() arr = jnp.asarray(sim)

     # share_informal_care_by_age_bin = get_share_by_type_by_age_bin( #     arr, #
    ind=idx, #     care_type=ALL, #     lagged_choice=INFORMAL_CARE, # )


    #
    share_not_working_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        choice=NO_WORK,
        care_type=NO_INFORMAL_CARE,
    )
    share_part_time_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        choice=PART_TIME,
        care_type=NO_INFORMAL_CARE,
    )
    share_full_time_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        choice=FULL_TIME,
        care_type=NO_INFORMAL_CARE,
    )

    share_not_working_informal_care = get_share_by_type(
        arr,
        ind=idx,
        choice=NO_WORK,
        care_type=INFORMAL_CARE,
    )
    share_part_time_informal_care = get_share_by_type(
        arr,
        ind=idx,
        choice=PART_TIME,
        care_type=INFORMAL_CARE,
    )
    share_full_time_informal_care = get_share_by_type(
        arr,
        ind=idx,
        choice=FULL_TIME,
        care_type=INFORMAL_CARE,
    )


    [share_not_working_no_informal_care]
        + [share_part_time_no_informal_care]
        + [share_full_time_no_informal_care]
        + [share_not_working_informal_care]
        + [share_part_time_informal_care]
        + [share_full_time_informal_care]
        +



    + care_mix_informal_by_mother_age_bin
        # + care_mix_formal_by_mother_age_bin
        # + care_mix_combination_by_mother_age_bi

    share_not_working_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=NO_WORK,
    )
    share_working_part_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=PART_TIME,
    )
    share_working_full_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=FULL_TIME,
    )

    """
    # ================================================================================
    # Employment by age
    # ================================================================================

    share_out_of_labor_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=OUT_OF_LABOR,
    )
    share_working_part_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=PART_TIME,
    )
    share_working_full_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=FULL_TIME,
    )

    # ================================================================================
    # Savings rate
    # ================================================================================

    savings_rate_coeffs = fit_ols(
        x=arr[
            :,
            [
                idx["age"],
                idx["age_squared"],
                idx["high_educ"],
                idx["choice_part_time"],
                idx["choice_full_time"],
                # idx["choice_informal_care"],
            ],
        ],
        y=arr[:, idx["savings_rate"]],
    )

    # ================================================================================
    # Labor shares by informal caregiving status
    # ================================================================================

    # share_not_working_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=NO_WORK,
    #     care_type=NO_INFORMAL_CARE,
    #     age_bins=AGE_BINS_SIM,
    # )
    # share_part_time_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=PART_TIME,
    #     care_type=INFORMAL_CARE,
    #     age_bins=AGE_BINS_SIM,
    # )
    # share_full_time_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=FULL_TIME,
    #     care_type=NO_INFORMAL_CARE,
    #     age_bins=AGE_BINS_SIM,
    # )

    # share_not_working_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=NO_WORK,
    #     care_type=INFORMAL_CARE,
    #     age_bins=AGE_BINS_SIM,
    # )
    # share_part_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=PART_TIME,
    #     care_type=INFORMAL_CARE,
    #     age_bins=AGE_BINS_SIM,
    # )
    # share_full_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=FULL_TIME,
    #     care_type=INFORMAL_CARE,
    #     age_bins=AGE_BINS_SIM,
    # )

    # ================================================================================
    # Share caregiving by age bin
    # ================================================================================

    # share_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=INFORMAL_CARE,
    #     care_type=ALL,
    #     age_bins=AGE_BINS_SIM,
    # )

    # ================================================================================
    # Share care by mother's health
    # ================================================================================

    # no_care_mother_health = get_share_care_by_parental_health(
    #     arr,
    #     ind=idx,
    #     care_choice=NO_CARE,
    #     parent="mother",
    # )
    # informal_care_mother_health = get_share_care_by_parental_health(
    #     arr,
    #     ind=idx,
    #     care_choice=PURE_INFORMAL_CARE,
    #     parent="mother",
    # )
    # formal_care_mother_health = get_share_care_by_parental_health(
    #     arr,
    #     ind=idx,
    #     care_choice=PURE_FORMAL_CARE,
    #     parent="mother",
    # )
    # combination_care_mother_health = get_share_care_by_parental_health(
    #     arr,
    #     ind=idx,
    #     care_choice=COMBINATION_CARE,
    #     parent="mother",
    # )

    # no_care_mother_health_has_sibling = (
    #     get_share_care_by_parental_health_and_presence_of_sibling(
    #         arr,
    #         ind=idx,
    #         care_choice=NO_CARE,
    #         has_sibling=True,
    #         parent="mother",
    #     )
    # )
    # informal_care_mother_health_has_sibling = (
    #     get_share_care_by_parental_health_and_presence_of_sibling(
    #         arr,
    #         ind=idx,
    #         care_choice=PURE_INFORMAL_CARE,
    #         has_sibling=True,
    #         parent="mother",
    #     )
    # )
    # formal_care_mother_health_has_sibling = (
    #     get_share_care_by_parental_health_and_presence_of_sibling(
    #         arr,
    #         ind=idx,
    #         care_choice=PURE_FORMAL_CARE,
    #         has_sibling=True,
    #         parent="mother",
    #     )
    # )
    # combination_care_mother_health_has_sibling = (
    #     get_share_care_by_parental_health_and_presence_of_sibling(
    #         arr,
    #         ind=idx,
    #         care_choice=COMBINATION_CARE,
    #         has_sibling=True,
    #         parent="mother",
    #     )
    # )

    # ================================================================================
    # Employment transitions
    # ================================================================================

    no_work_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=OUT_OF_LABOR,
        current_choice=OUT_OF_LABOR,
    )
    no_work_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=OUT_OF_LABOR,
        current_choice=PART_TIME,
    )
    no_work_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=OUT_OF_LABOR,
        current_choice=FULL_TIME,
    )

    part_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=OUT_OF_LABOR,
    )
    part_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=PART_TIME,
    )
    part_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=FULL_TIME,
    )

    full_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=OUT_OF_LABOR,
    )
    full_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=PART_TIME,
    )
    full_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=FULL_TIME,
    )

    # ================================================================================
    # Caregiving transitions
    # ================================================================================

    # no_informal_care_to_no_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_INFORMAL_CARE,
    #     current_choice=NO_INFORMAL_CARE,
    # )
    # no_informal_care_to_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_INFORMAL_CARE,
    #     current_choice=INFORMAL_CARE,
    # )

    # informal_care_to_no_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=INFORMAL_CARE,
    #     current_choice=NO_INFORMAL_CARE,
    # )
    # informal_care_to_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=INFORMAL_CARE,
    #     current_choice=INFORMAL_CARE,
    # )

    # no_informal_care_to_no_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_INFORMAL_CARE,
    #     current_choice=NO_FORMAL_CARE,
    # )
    # no_informal_care_to_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_INFORMAL_CARE,
    #     current_choice=FORMAL_CARE,
    # )

    # informal_care_to_no_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=INFORMAL_CARE,
    #     current_choice=NO_FORMAL_CARE,
    # )
    # informal_care_to_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=INFORMAL_CARE,
    #     current_choice=FORMAL_CARE,
    # )

    # no_formal_care_to_no_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_FORMAL_CARE,
    #     current_choice=NO_INFORMAL_CARE,
    # )
    # no_formal_care_to_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_FORMAL_CARE,
    #     current_choice=INFORMAL_CARE,
    # )

    # formal_care_to_no_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FORMAL_CARE,
    #     current_choice=NO_INFORMAL_CARE,
    # )
    # formal_care_to_informal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FORMAL_CARE,
    #     current_choice=INFORMAL_CARE,
    # )

    # no_formal_care_to_no_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_FORMAL_CARE,
    #     current_choice=NO_FORMAL_CARE,
    # )
    # no_formal_care_to_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=NO_FORMAL_CARE,
    #     current_choice=FORMAL_CARE,
    # )

    # formal_care_to_no_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FORMAL_CARE,
    #     current_choice=NO_FORMAL_CARE,
    # )
    # formal_care_to_formal_care = get_transition(
    #     arr,
    #     ind=idx,
    #     lagged_choice=FORMAL_CARE,
    #     current_choice=FORMAL_CARE,
    # )

    # ================================================================================
    # Moments matrix
    # ================================================================================

    return jnp.asarray(
        # employment shares
        share_out_of_labor_by_age
        + share_working_part_time_by_age
        + share_working_full_time_by_age
        # assets and savings
        + savings_rate_coeffs.tolist()
        # employment shares by caregiving status
        # no informal care
        # + share_not_working_no_informal_care_by_age_bin
        # + share_part_time_no_informal_care_by_age_bin
        # + share_full_time_no_informal_care_by_age_bin
        # # informal care
        # + share_not_working_informal_care_by_age_bin
        # + share_part_time_informal_care_by_age_bin
        # + share_full_time_informal_care_by_age_bin
        #
        # share of informal care in total population by age bin
        # + share_informal_care_by_age_bin
        # # Care by mother's health by presence of sister
        # + no_care_mother_health
        # + informal_care_mother_health
        # + formal_care_mother_health
        # + combination_care_mother_health
        # + no_care_mother_health_has_sibling
        # + informal_care_mother_health_has_sibling
        # + formal_care_mother_health_has_sibling
        # + combination_care_mother_health_has_sibling
        #
        # Employment transitions
        + no_work_to_no_work
        + no_work_to_part_time
        + no_work_to_full_time
        + part_time_to_no_work
        + part_time_to_part_time
        + part_time_to_full_time
        + full_time_to_no_work
        + full_time_to_part_time
        + full_time_to_full_time,
        # +
        # Caregiving transitions
        # no_informal_care_to_no_informal_care
        # + no_informal_care_to_informal_care
        # + informal_care_to_no_informal_care
        # + informal_care_to_informal_care
        # + no_informal_care_to_no_formal_care
        # + no_informal_care_to_formal_care
        # + informal_care_to_no_formal_care
        # + informal_care_to_formal_care
        # +
        # #
        # no_formal_care_to_no_informal_care
        # + no_formal_care_to_informal_care
        # + formal_care_to_no_informal_care
        # + formal_care_to_informal_care
        # + no_formal_care_to_no_formal_care
        # + no_formal_care_to_formal_care
        # + formal_care_to_no_formal_care
        # + formal_care_to_formal_care,
    )


def fit_ols(x, y):
    """Fit a linear model using least-squares.

    coef = coef.T


    Args:
        x (jnp.ndarray): Array of shape (n, p) of x-values.
        y (jnp.ndarray): Array of shape (n, k) of y-values.

    Returns:
        coef (np.ndarray): Array of shape (p, k) of coefficients.

    """
    intercept = jnp.ones((len(x), 1))
    features = jnp.concatenate((intercept, jnp.atleast_2d(x)), axis=1)

    coef, *_ = jnp.linalg.lstsq(features, y, rcond=None)

    return coef


def get_share_by_age(df_arr, ind, choice):
    """Get share of agents choosing lagged choice by age bin."""
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    shares = []

    for age in range(MIN_AGE_SIM, MAX_AGE_SIM):
        age_mask = df_arr[:, ind["age"]] == age

        share = jnp.sum(age_mask & choice_mask) / jnp.sum(age_mask)
        # period count is always larger than 0! otherwise error
        shares.append(share)

    return shares


def get_share_by_type(df_arr, ind, choice, care_type):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

    return jnp.sum(choice_mask & care_type_mask) / jnp.sum(care_type_mask)


def get_transition(df_arr, ind, lagged_choice, current_choice):
    """Get transition probability from lagged choice to current choice."""
    return [
        jnp.sum(
            jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
            & jnp.isin(df_arr[:, ind["choice"]], current_choice),
        )
        / jnp.sum(jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)),
    ]


def get_share_by_type_by_age_bin(df_arr, ind, choice, care_type, age_bins):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

    shares = []
    for age_bin in age_bins:
        age_bin_mask = (df_arr[:, ind["age"]] >= age_bin[0]) & (
            df_arr[:, ind["age"]] < age_bin[1]
        )
        share = jnp.sum(choice_mask & type_mask & age_bin_mask) / jnp.sum(
            type_mask & age_bin_mask,
        )
        shares.append(share)

    return shares


def get_savings_rate_by_age_bin(arr, ind, care_type):
    """Get savings rate of given care type by age bin."""
    care_type_mask = jnp.isin(arr[:, ind["choice"]], care_type)
    means = []

    for age_bin in AGE_BINS_SIM:
        age_bin_mask = (arr[:, ind["age"]] >= age_bin[0]) & (
            arr[:, ind["age"]] < age_bin[1]
        )
        selected_values = jnp.take(
            arr[:, ind["savings_rate"]],
            jnp.nonzero(care_type_mask & age_bin_mask)[0],
        )
        mean = jnp.sum(selected_values) / jnp.sum(care_type_mask & age_bin_mask)

        means.append(mean)

    return means


def get_share_care_by_parental_health_old(
    df_arr,
    ind,
    care_choice,
    parent="mother",
):
    """Get share of agents choosing given care choice by parental health."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["choice"]], care_choice)
            & (df_arr[:, ind[f"{parent}_health"]], health),
        )
        for health in (BAD_HEALTH)
    ]


def get_share_care_by_parental_health_and_presence_of_sibling_old(
    df_arr,
    ind,
    care_choice,
    has_sibling,
    parent="mother",
):
    """Get share of agents choosing given care choice by parental health."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["choice"]], care_choice)
            & (df_arr[:, ind["has_sibling"]] == has_sibling)
            & (df_arr[:, ind[f"{parent}_health"]], health),
        )
        for health in (BAD_HEALTH)
    ]


def get_share_care_by_parental_health_and_presence_of_sibling(
    df_arr,
    ind,
    care_choice,
    has_sibling,
    parent="mother",
):
    """Get share of agents choosing given care choice by parental health."""
    care_choice_mask = jnp.isin(df_arr[:, ind["choice"]], care_choice)
    sibling_mask = df_arr[:, ind["has_sibling"]] == has_sibling

    shares = []
    for health in [BAD_HEALTH]:
        health_mask = df_arr[:, ind[f"{parent}_health"]] == health

        share = jnp.sum(care_choice_mask & sibling_mask & health_mask) / jnp.sum(
            sibling_mask & health_mask,
        )
        shares.append(share)

    return shares


def get_share_care_by_parental_health(
    df_arr,
    ind,
    care_choice,
    parent="mother",
):
    """Get share of agents choosing given care choice by parental health."""
    care_choice_mask = jnp.isin(df_arr[:, ind["choice"]], care_choice)

    shares = []
    for health in [BAD_HEALTH]:
        health_mask = df_arr[:, ind[f"{parent}_health"]] == health

        share = jnp.sum(care_choice_mask & health_mask) / jnp.sum(health_mask)
        shares.append(share)

    return shares


def get_care_mix_by_mother_age_bin(df_arr, ind, choice, care_type, age_bins):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    choice_mask = jnp.isin(df_arr[:, ind["choice"]], choice)
    care_type_mask = jnp.isin(df_arr[:, ind["choice"]], care_type)

    shares = []
    for age_bin in age_bins:
        age_bin_mask = (df_arr[:, ind["mother_age"]] >= age_bin[0]) & (
            df_arr[:, ind["mother_age"]] < age_bin[1]
        )
        share = jnp.sum(choice_mask & care_type_mask & age_bin_mask) / jnp.sum(
            care_type_mask & age_bin_mask,
        )
        shares.append(share)

    return shares


# ==============================================================================
# Prepare simulation array
# ==============================================================================


# Assuming _assign_working_hours is adapted for JAX, using vmap for batch operations
# if necessary.
# Since JAX does not support functions like .apply() directly,
# you'll likely need to vectorize this logic.


def create_simulation_array_from_df(data, options):
    """Create simulation array from dict."""
    data = data.copy()  # Make a copy to avoid modifying a slice

    options = options["model_params"]
    n_agents = options["n_agents"]
    n_periods = options["n_periods"]

    # Assigning the 'agent' and age-related calculations
    data.loc[:, "agent"] = jnp.tile(jnp.arange(n_agents), n_periods)
    period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()

    data.loc[:, "age"] = options["start_age"] + period_indices
    data.loc[:, "age_squared"] = data["age"] ** 2
    # data.loc[:, "mother_age"] = options["mother_start_age"] + period_indices

    # Financial calculations
    data.loc[:, "wealth"] = data["savings"] + data["consumption"]
    data.loc[:, "savings_rate"] = jnp.where(
        jnp.array(data["wealth"]) > 0,
        jnp.divide(jnp.array(data["savings"]), jnp.array(data["wealth"])),
        0,
    )

    # Squared experience
    data.loc[:, "experience"] = data["experience"] / 2
    data.loc[:, "experience_squared"] = data["experience"] ** 2

    # Employment status
    data.loc[:, "lagged_part_time"] = jnp.isin(
        jnp.array(data["lagged_choice"]),
        PART_TIME,
    ).astype(np.int8)

    data.loc[:, "choice_retired"] = jnp.isin(
        jnp.array(data["choice"]),
        RETIREMENT,
    ).astype(np.int8)
    data.loc[:, "choice_part_time"] = jnp.isin(
        jnp.array(data["choice"]),
        PART_TIME,
    ).astype(np.int8)
    data.loc[:, "choice_full_time"] = jnp.isin(
        jnp.array(data["choice"]),
        FULL_TIME,
    ).astype(np.int8)
    # data.loc[:, "choice_informal_care"] = jnp.isin(
    #     jnp.array(data["choice"]),
    #     INFORMAL_CARE,
    # ).astype(np.int8)

    # Wage calculations
    data.loc[:, "log_wage"] = (
        options["wage_constant"]
        + options["wage_age"] * data["age"]
        + options["wage_age_squared"] * data["age_squared"]
        + options["wage_experience"] * data["experience"]
        + options["wage_experience_squared"] * data["experience_squared"]
        + options["wage_high_education"] * data["high_educ"]
        + options["wage_part_time"] * data["lagged_part_time"]
    )

    data.loc[:, "wage"] = jnp.exp(
        jnp.array(data["log_wage"]) + jnp.array(data["income_shock"]),
    )

    # Working hours and income calculation
    data.loc[:, "working_hours"] = jax.vmap(_assign_working_hours_vectorized)(
        data["lagged_choice"].values,
    )

    data.loc[:, "income"] = data["working_hours"] * data["wage"]

    # Create a mapping of column indices
    column_indices = {col: idx for idx, col in enumerate(data.columns)}

    data = data.dropna()

    return jnp.array(data), column_indices


def create_simulation_array(sim_dict, options):
    """Create simulation array from dict."""
    options = options["model_params"]
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    agent = jnp.tile(jnp.arange(n_agents), n_periods)
    period = sim_dict["period"].ravel()
    savings = sim_dict["savings"].ravel()
    consumption = sim_dict["consumption"].ravel()
    lagged_choice = sim_dict["lagged_choice"].ravel()
    choice = sim_dict["choice"].ravel()
    high_educ = sim_dict["high_educ"].ravel()
    experience = sim_dict["experience"].ravel()
    income_shock = sim_dict["income_shock"].ravel()
    mother_health = sim_dict["mother_health"].ravel()
    has_sibling = sim_dict["has_sibling"].ravel()

    wealth = savings + consumption
    savings_rate = jnp.where(wealth > 0, jnp.divide(savings, wealth), 0)
    period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()

    age = options["start_age"] + period_indices
    age_squared = age**2
    mother_age = options["mother_start_age"] + period_indices

    experience_squared = experience**2

    # Adjusting the logic for PART_TIME and FULL_TIME checks
    lagged_part_time = jnp.isin(lagged_choice, PART_TIME)

    is_part_time = jnp.isin(choice, PART_TIME)
    is_full_time = jnp.isin(choice, FULL_TIME)
    is_informal_care = jnp.isin(choice, INFORMAL_CARE)

    log_wage = (
        options["wage_constant"]
        + options["wage_age"] * age
        + options["wage_age_squared"] * age_squared
        + options["wage_experience"] * experience
        + options["wage_experience_squared"] * experience_squared
        + options["wage_high_education"] * high_educ
        + options["wage_part_time"] * lagged_part_time
    )

    wage = jnp.exp(log_wage + income_shock)

    # Adapt _assign_working_hours to work with vectorized operations in JAX
    # Example stub for _assign_working_hours function
    # Vectorize the _assign_working_hours function if it's not already suitable
    # for vector operations
    working_hours = jax.vmap(_assign_working_hours_vectorized)(lagged_choice)

    income = working_hours * wage

    arr = jnp.column_stack(
        (
            agent,
            period,
            lagged_choice,
            choice,
            age,
            age_squared,
            mother_age,
            wealth,
            savings_rate,
            wage,
            working_hours,
            income,
            is_part_time,
            is_full_time,
            is_informal_care,
            experience,
            experience_squared,
            high_educ,
            mother_health,
            has_sibling,
        ),
    )

    idx = {
        "agent": 0,
        "period": 1,
        "lagged_choice": 2,
        "choice": 3,
        "age": 4,
        "age_squared": 5,
        "mother_age": 6,
        "wealth": 7,
        "savings_rate": 8,
        "wage": 9,
        "working_hours": 10,
        "income": 11,
        "choice_part_time": 12,
        "choice_full_time": 13,
        "choice_informal_care": 14,
        "experience": 15,
        "experience_squared": 16,
        "high_educ": 17,
        "mother_health": 18,
        "has_sibling": 19,
    }

    return arr, idx


def create_simulation_df_from_dict(sim_dict):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    return pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )


def _assign_working_hours_vectorized(choices):
    no_work_mask = jnp.isin(choices, OUT_OF_LABOR)
    part_time_mask = jnp.isin(choices, PART_TIME)
    full_time_mask = jnp.isin(choices, FULL_TIME)

    return jnp.where(
        no_work_mask,
        0,
        jnp.where(part_time_mask, 20, jnp.where(full_time_mask, 40, jnp.nan)),
    )
