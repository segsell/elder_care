"""Debugging tasks."""

from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods_for_model
from dcegm.solve import get_solve_func_for_model
from elder_care.config import BLD
from elder_care.model.budget import budget_constraint, create_savings_grid
from elder_care.model.shared import (
    AGE_BINS_SIM,
    ALL,
    FORMAL_CARE,
    FULL_TIME,
    INFORMAL_CARE,
    NO_WORK,
    PART_TIME,
)
from elder_care.model.state_space import create_state_space_functions
from elder_care.model.task_specify_model import get_options_dict
from elder_care.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)
from elder_care.simulation.initial_conditions import draw_initial_states
from elder_care.simulation.simulate import (
    create_simulation_array_from_df,
    create_simulation_df_from_dict,
    get_share_by_age,
    get_share_by_type_by_age_bin,
    simulate_moments,
)
from elder_care.utils import load_dict_from_pickle, save_dict_to_pickle

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


PARAMS = {
    "beta": 0.959,
    "rho": 0.8,
    "lambda": 1,
    "sigma": 0.5364562201,
    "interest_rate": 0.04,
    #
    "utility_leisure_constant": 1,
    "utility_leisure_age": 1,
    #
    "disutility_part_time": -5,
    "disutility_full_time": -6,
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
    # part-time job offer
    "part_time_constant": -2.568584,
    "part_time_not_working_last_period": 0.3201395,
    "part_time_high_education": 0.1691369,
    "part_time_above_retirement_age": -1.9976496,
    # full-time job offer
    "full_time_constant": -2.445238,
    "full_time_not_working_last_period": -0.9964007,
    "full_time_high_education": 0.3019138,
    "full_time_above_retirement_age": -2.6571659,
}


PROGRESS = {
    "rho": 0.7,
    "beta": 0.959,
    "sigma": 0.5364562201,
    "lambda": 1.0,
    "interest_rate": 0.04,
    "utility_leisure_constant": 3.2194001905695693,
    "utility_leisure_age": 0.046916363865977045,
    "utility_leisure_age_squared": -0.006495755962584588,
    "disutility_part_time": -1.9337796413959372,
    "disutility_full_time": -5.282394222692581,
    "utility_informal_care_parent_medium_health": -0.4089331199248291,
    "utility_informal_care_parent_bad_health": -0.3851096018741167,
    "utility_formal_care_parent_medium_health": 0.4045081430627899,
    "utility_formal_care_parent_bad_health": -0.4575685368898893,
    "utility_combination_care_parent_medium_health": -4.112870982901066,
    "utility_combination_care_parent_bad_health": -2.6130289452563393,
    "utility_informal_care_medium_health_sibling": 1.815903823857372,
    "utility_informal_care_bad_health_sibling": 2.439680402899742,
    "utility_formal_care_medium_health_sibling": 0.6975043998652197,
    "utility_formal_care_bad_health_sibling": 0.9263483706654374,
    "utility_combination_care_medium_health_sibling": -1.6125665883276101,
    "utility_combination_care_bad_health_sibling": -1.6952982282449929,
    # part-time job offer
    "part_time_constant": -2.568584,
    "part_time_not_working_last_period": 0.3201395,
    "part_time_high_education": 0.1691369,
    "part_time_above_retirement_age": -1.9976496,
    # full-time job offer
    "full_time_constant": -2.445238,
    "full_time_not_working_last_period": -0.9964007,
    "full_time_high_education": 0.3019138,
    "full_time_above_retirement_age": -2.6571659,
}


@pytask.mark.skip()
def task_debugging(
    path_to_save_result: Annotated[Path, Product] = BLD / "debugging" / "result.pkl",
    path_to_save_sim_dict: Annotated[Path, Product] = BLD
    / "debugging"
    / "sim_dict.pkl",
):
    """Debugging task.

    path_to_model: Path = BLD / "model" / "model.pkl",

    results = load_dict_from_pickle(BLD / "debugging" / "result.pkl")


    # check initial labor shares
    jnp.unique(initial_states["lagged_choice"], return_counts=True)


    _mother_health_probs = initial_conditions.loc[
        ["mother_good_health", "mother_medium_health", "mother_bad_health"]
    ].to_numpy()
    mother_health_probs = jnp.array(_mother_health_probs).ravel()

    """
    path_to_model = BLD / "model" / "model.pkl"
    options = get_options_dict()

    params = PROGRESS

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
    results = func(params)

    save_dict_to_pickle(results, path_to_save_result)

    n_agents = 100_000
    seed = 2024

    path_high_educ = f"{BLD}/moments/real_wealth_age_39_high_educ.csv"
    initial_wealth_high_educ = jnp.asarray(pd.read_csv(path_high_educ)).ravel()

    path_low_educ = f"{BLD}/moments/real_wealth_age_39_low_educ.csv"
    initial_wealth_low_educ = jnp.asarray(pd.read_csv(path_low_educ)).ravel()

    path = f"{BLD}/moments/initial_discrete_conditions_at_age_40.csv"
    initial_conditions = pd.read_csv(path, index_col=0)

    initial_resources, initial_states = draw_initial_states(
        initial_conditions,
        initial_wealth_low_educ=initial_wealth_low_educ,
        initial_wealth_high_educ=initial_wealth_high_educ,
        n_agents=n_agents,
        seed=seed,
    )

    sim_dict = simulate_all_periods_for_model(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=results[3],
        value_solved=results[0],
        policy_left_solved=results[1],
        policy_right_solved=results[2],
        choice_range=jnp.arange(options["model_params"]["n_choices"], dtype=jnp.int16),
        model=model_loaded,
    )
    save_dict_to_pickle(sim_dict, path_to_save_sim_dict)

    df_raw = create_simulation_df(sim_dict)
    data = df_raw.notna()

    #

    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    data = pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )
    data["age"] = data["period"] + options["model_params"]["start_age"]

    data_clean = data.dropna()
    column_indices = {col: idx for idx, col in enumerate(data_clean.columns)}
    idx = column_indices.copy()
    arr = jnp.asarray(data_clean)

    share_not_working_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=NO_WORK,
    )  # 15

    share_part_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=PART_TIME,
    )  # 15

    share_full_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=FULL_TIME,
    )  # 15

    share_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        choice=INFORMAL_CARE,
        care_type=ALL,
        age_bins=AGE_BINS_SIM,
    )

    share_formal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        choice=FORMAL_CARE,
        care_type=ALL,
        age_bins=AGE_BINS_SIM,
    )

    share_not_working_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        choice=NO_WORK,
        care_type=INFORMAL_CARE,
        age_bins=AGE_BINS_SIM,
    )
    share_part_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        choice=PART_TIME,
        care_type=INFORMAL_CARE,
        age_bins=AGE_BINS_SIM,
    )
    share_full_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        choice=FULL_TIME,
        care_type=INFORMAL_CARE,
        age_bins=AGE_BINS_SIM,
    )

    return (
        share_not_working_by_age,
        share_part_time_by_age,
        share_full_time_by_age,
        share_informal_care_by_age_bin,
        share_formal_care_by_age_bin,
        share_not_working_informal_care_by_age_bin,
        share_part_time_informal_care_by_age_bin,
        share_full_time_informal_care_by_age_bin,
    )


@pytask.mark.skip()
def task_debug_simulate():
    """Debug simulate.

    path_to_sim_dict: Path = BLD / "debugging" / "sim_dict.pkl",

    """
    path_to_sim_dict = BLD / "debugging" / "sim_dict.pkl"

    options = get_options_dict()
    sim_dict = load_dict_from_pickle(path_to_sim_dict)

    data = create_simulation_df_from_dict(sim_dict)

    arr, idx = create_simulation_array_from_df(data=data, options=options)
    out = simulate_moments(arr, idx)

    return out, arr
