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
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model
from elder_care.config import BLD
from elder_care.model.budget import budget_constraint, create_savings_grid
from elder_care.model.shared import (
    FULL_TIME,
    NO_WORK,
    OUT_OF_LABOR,
    PART_TIME,
    RETIREMENT,
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
    simulate_moments,
)
from elder_care.utils import load_dict_from_pickle, save_dict_to_pickle

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


PARAMS_OLD = {
    "rho": 0.8,
    "beta": 0.959,
    "sigma": 0.5364562201,
    "lambda": 0.9864699097918321,
    "interest_rate": 0.04,
    #
    "disutility_part_time_constant": 0.33354121247199703,
    "disutility_part_time_age": -0.12100801003524632,
    "disutility_part_time_age_squared": 0.0007139083714654349,
    "disutility_full_time_constant": 0.08529730099536248,
    "disutility_full_time_age": -0.054504780075805004,
    "disutility_full_time_age_squared": -0.0022061388612220744,
    "disutility_part_time_informal_care_constant": 0.33354121247199703,
    "disutility_part_time_informal_care_age": -0.12100801003524632,
    "disutility_part_time_informal_care_age_squared": 0.0007139083714654349,
    "disutility_full_time_informal_care_constant": 0.08529730099536248,
    "disutility_full_time_informal_care_age": -0.054504780075805004,
    "disutility_full_time_informal_care_age_squared": -0.0022061388612220744,
    #
    "part_time_constant": -2.102635900186225,
    "part_time_not_working_last_period": -1.0115255914421664,
    "part_time_high_education": 0.48013160890989515,
    "part_time_above_retirement_age": -2.110713962590601,
    "full_time_constant": -1.9425261133765783,
    "full_time_not_working_last_period": -2.097935912953995,
    "full_time_high_education": 0.8921957457184644,
    "full_time_above_retirement_age": -3.1212459549307496,
    #
    "utility_no_care_parent_bad_health": -1,
    "utility_informal_care_parent_bad_health": 0.5,
    "utility_formal_care_parent_bad_health": 0.2,
    "utility_combination_care_parent_bad_health": 0.4,
}

PARAMS = {
    "rho": 0.8,
    "beta": 0.959,
    "sigma": 0.5364562201,
    "lambda": 0.9864699097918321,
    "interest_rate": 0.04,
    #
    "disutility_part_time_constant": 0.33354121247199703,
    "disutility_full_time_constant": 0.08529730099536248,
    "disutility_part_time_age_40_50": -0.12100801003524632,
    "disutility_full_time_age_40_50": -0.054504780075805004,
    "disutility_part_time_age_50_plus": 0.0007139083714654349,
    "disutility_full_time_age_50_plus": -0.0022061388612220744,
    "disutility_part_time_age_squared_50_plus": 0.0007139083714654349,
    "disutility_full_time_age_squared_50_plus": -0.0022061388612220744,
    #
    "part_time_constant": -2.102635900186225,
    "part_time_not_working_last_period": -1.0115255914421664,
    "part_time_high_education": 0.48013160890989515,
    "part_time_above_retirement_age": -2.110713962590601,
    "full_time_constant": -1.9425261133765783,
    "full_time_not_working_last_period": -2.097935912953995,
    "full_time_high_education": 0.8921957457184644,
    "full_time_above_retirement_age": -3.1212459549307496,
    #
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

    params = PARAMS

    model_loaded = load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
    )

    exog_savings_grid = create_savings_grid()

    # results = load_dict_from_pickle(BLD / "debugging" / "result.pkl")
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

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        value_solved=results[0],
        policy_solved=results[1],
        endog_grid_solved=results[2],
        # choice_range=jnp.arange(options["model_params"]["n_choices"], dtype=jnp.int16),
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
    share_retired_by_age = get_share_by_age(
        arr,
        ind=idx,
        choice=RETIREMENT,
    )  # 15
    share_out_of_labor_force = get_share_by_age(
        arr,
        ind=idx,
        choice=OUT_OF_LABOR,
    )  # 15

    # share_informal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=INFORMAL_CARE,
    #     care_type=ALL,
    #     age_bins=AGE_BINS_SIM,
    # )

    # share_formal_care_by_age_bin = get_share_by_type_by_age_bin(
    #     arr,
    #     ind=idx,
    #     choice=FORMAL_CARE,
    #     care_type=ALL,
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

    # breakpoint()

    return (
        share_not_working_by_age,
        share_part_time_by_age,
        share_full_time_by_age,
        share_retired_by_age,
        share_out_of_labor_force,
        # share_informal_care_by_age_bin,
        # share_formal_care_by_age_bin,
        # share_not_working_informal_care_by_age_bin,
        # share_part_time_informal_care_by_age_bin,
        # share_full_time_informal_care_by_age_bin,
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

    # breakpoint()

    return out, arr
