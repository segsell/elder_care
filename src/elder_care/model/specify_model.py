import numpy as np
from dcegm.pre_processing.setup_model import load_and_setup_model, setup_and_save_model

from elder_care.model.budget import budget_constraint
from elder_care.model.state_space import (
    create_state_space_functions,
    sparsity_condition,
)
from elder_care.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)


def specify_model(
    path_dict,
    update_spec_for_policy_state,
    policy_state_trans_func,
    params,
    *,
    load_model=False,
):
    """Generate model and options dictionaries."""
    # Generate model_specs

    specs = {}
    # Execute load first step estimation data
    specs = update_spec_for_policy_state(
        specs=specs,
        path_dict=path_dict,
    )

    # Load specifications
    n_periods = specs["n_periods"]
    n_possible_ret_ages = specs["n_possible_ret_ages"]
    n_policy_states = specs["n_policy_states"]
    choices = np.arange(specs["n_choices"], dtype=int)

    options = {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "retirement_age_id": np.arange(n_possible_ret_ages, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": {
                "policy_state": {
                    "transition": policy_state_trans_func,
                    "states": np.arange(n_policy_states, dtype=int),
                },
            },
        },
        "model_params": specs,
    }

    if load_model:
        model = load_and_setup_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path=path_dict["intermediate_data"] + "model.pkl",
        )

    else:
        model = setup_and_save_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path=path_dict["intermediate_data"] + "model.pkl",
        )

    return model, options, params
