from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods_for_model
from dcegm.solve import get_solve_func_for_model
from elder_care.config import BLD
from elder_care.model.budget import budget_constraint, create_savings_grid
from elder_care.model.state_space import (
    create_state_space_functions,
)
from elder_care.model.task_specify_model import get_options_dict
from elder_care.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)

jax.config.update("jax_enable_x64", True)


PARAMS = {
    "beta": 0.95,
    "rho": 1.95,
    "lambda": 1,
    "sigma": 1,
    "interest_rate": 0.04,
    #
    "utility_leisure_constant": 2,
    "utility_leisure_age": 1,
    #
    "disutility_part_time": -3,
    "disutility_full_time": -5,
    # caregiving
    "utility_informal_care_parent_medium_health": 2,
    "utility_informal_care_parent_bad_health": 1,
    "utility_formal_care_parent_medium_health": 0.7,
    "utility_formal_care_parent_bad_health": 1,
    "utility_combination_care_parent_medium_health": -0.8,
    "utility_combination_care_parent_bad_health": -1.5,
    # caregiving if sibling present
    "utility_informal_care_medium_health_sibling": 2.5,
    "utility_informal_care_bad_health_sibling": 2,
    "utility_formal_care_medium_health_sibling": 1,
    "utility_formal_care_bad_health_sibling": 1,
    "utility_combination_care_medium_health_sibling": -0.2,
    "utility_combination_care_bad_health_sibling": -0.4,
}


def task_debug(path_to_model: Path = BLD / "model" / "model.pkl"):

    options = get_options_dict()

    model_loaded = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
    )

    exog_savings_grid = create_savings_grid()

    func = get_solve_func_for_model(
        model=model_loaded,
        exog_savings_grid=exog_savings_grid,
        options=options,
    )

    results = func(PARAMS)

    n_agents = 100_000
    seed = 2024

    path = f"{BLD}/moments/initial_wealth_at_age_50.csv"
    initial_wealth_empirical = jnp.asarray(pd.read_csv(path)).ravel() * 0.8

    path = f"{BLD}/moments/initial_discrete_conditions_at_age_40.csv"
    initial_conditions = pd.read_csv(path, index_col=0)

    initial_resources, initial_states = draw_initial_states(
        initial_conditions, initial_wealth_empirical, n_agents, seed=seed,
    )

    entries_to_remove = ("married", "care_demand", "father_alive")
    for k in entries_to_remove:
        initial_states.pop(k, None)

    # Fix in dcegm?
    model_loaded["state_space_names"] = model_loaded["model_structure"][
        "state_space_names"
    ]
    model_loaded["map_state_choice_to_index"] = model_loaded["model_structure"][
        "map_state_choice_to_index"
    ]

    model_loaded["exog_mapping"] = model_loaded["model_funcs"]["exog_mapping"]
    model_loaded["get_next_period_state"] = model_loaded["model_funcs"][
        "get_next_period_state"
    ]

    sim_dict = simulate_all_periods_for_model(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["model_params"]["n_periods"],
        params=PARAMS,
        seed=seed,
        endog_grid_solved=results[3],
        value_solved=results[0],
        policy_left_solved=results[1],
        policy_right_solved=results[2],
        choice_range=jnp.arange(options["model_params"]["n_choices"], dtype=jnp.int16),
        model=model_loaded,
    )

    df = create_simulation_df(sim_dict)

    breakpoint()

    return df


# def simulate_local():


# ==============================================================================
# Temporary debugging
# ==============================================================================


def draw_initial_states(initial_conditions, initial_wealth, n_agents, seed):
    """Draw initial resources and states for the simulation.

    mother_health = get_initial_share_three(     initial_conditions,
    ["mother_good_health", "mother_medium_health", "mother_bad_health"], ) father_health
    = get_initial_share_three(     initial_conditions,     ["father_good_health",
    "father_medium_health", "father_bad_health"], )

    informal_care = get_initial_share_two(initial_conditions, "share_informal_care")

    """
    n_choices = 12

    employment = get_initial_share_three(
        initial_conditions,
        ["share_not_working", "share_part_time", "share_full_time"],
    )

    informal_care = jnp.array([1, 0])  # agents start with no informal care provided
    formal_care = jnp.array([1, 0])  # agents start with no formal care provided

    informal_formal = jnp.outer(informal_care, formal_care)
    lagged_choice_probs = jnp.outer(employment, informal_formal).flatten()

    high_educ = get_initial_share_two(initial_conditions, "share_high_educ")
    has_sibling = get_initial_share_two(initial_conditions, "share_has_sister")

    mother_alive = get_initial_share_two(initial_conditions, "share_mother_alive")

    initial_resources = draw_random_sequence_from_array(
        seed,
        initial_wealth,
        n_agents=n_agents,
    )

    initial_states = {
        "period": jnp.zeros(n_agents, dtype=np.int16),
        "lagged_choice": draw_random_array(
            seed=seed - 1,
            n_agents=n_agents,
            values=jnp.arange(n_choices),
            probabilities=lagged_choice_probs,
        ).astype(np.int16),
        "high_educ": draw_random_array(
            seed=seed - 2,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=high_educ,
        ).astype(np.int16),
        "has_sibling": draw_random_array(
            seed=seed - 3,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=has_sibling,
        ).astype(np.int16),
        "experience": draw_from_discrete_normal(
            seed=seed - 4,
            n_agents=n_agents,
            mean=jnp.ones(n_agents, dtype=np.uint16)
            * float(initial_conditions.loc["experience_mean"]),
            std_dev=jnp.ones(n_agents, dtype=np.uint16)
            * float(initial_conditions.loc["experience_std"]),
        ),
        "part_time_offer": jnp.zeros(n_agents, dtype=np.int16),
        "full_time_offer": jnp.zeros(n_agents, dtype=np.int16),
        "care_demand": jnp.zeros(n_agents, dtype=np.int16),
        "mother_health": draw_random_array(
            seed=seed - 5,
            n_agents=n_agents,
            values=jnp.array([0, 1, 2]),
            probabilities=jnp.array([0.188007, 0.743285, 0.068707]),
        ).astype(np.int16),
        "mother_alive": draw_random_array(
            seed=seed - 6,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=mother_alive,
        ).astype(np.int16),
    }

    return initial_resources, initial_states


def get_initial_share_three(initial_conditions, shares):
    return jnp.asarray(initial_conditions.loc[shares]).ravel()


def get_initial_share_two(initial_conditions, var):
    share_yes = initial_conditions.loc[var]
    return jnp.asarray([1 - share_yes, share_yes]).ravel()


def draw_random_sequence_from_array(seed, arr, n_agents):
    """Draw a random sequence from an array.

    rand = draw_random_sequence_from_array(     seed=2024,     n_agents=10_000,
    arr=jnp.array(initial_wealth), )

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, arr, shape=(n_agents,), replace=True)


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
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, values, shape=(n_agents,), p=probabilities)


def draw_from_discrete_normal(seed, n_agents, mean, std_dev):
    """Draw from discrete normal distribution."""
    key = jax.random.PRNGKey(seed)

    sample_standard_normal = jax.random.normal(key, (n_agents,))

    # Scaling and shifting to get the desired mean and standard deviation, then rounding
    return jnp.round(mean + std_dev * sample_standard_normal).astype(jnp.int16)
