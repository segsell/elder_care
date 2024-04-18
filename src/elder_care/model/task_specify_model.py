"""Specify model for estimation and simulation."""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pytask
import yaml
from dcegm.pre_processing.setup_model import setup_and_save_model
from pytask import Product

from elder_care.config import BLD, SRC
from elder_care.exogenous_processes.task_create_exog_processes_soep import (
    task_create_exog_wage,
)
from elder_care.model.budget import budget_constraint
from elder_care.model.exogenous_processes import (
    exog_health_transition_mother_with_survival,
    prob_full_time_offer,
    prob_part_time_offer,
)
from elder_care.model.state_space import (
    create_state_space_functions,
    sparsity_condition,
)
from elder_care.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)
from elder_care.utils import load_dict_from_pickle


@pytask.mark.skip(reason="Respecifying model.")
def task_specify_and_setup_model(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
    path_to_save: Annotated[Path, Product] = BLD / "model" / "model_short_exp.pkl",
) -> dict[str, Any]:
    """Generate options and setup model.

    start_params["sigma"] = specs["income_shock_scale"]

    """
    options = get_options_dict(path_to_specs, path_to_exog)

    return setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_save,
    )


def get_options_dict(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
):

    specs, wage_params = load_specs(path_to_specs)

    exog_params = load_dict_from_pickle(path_to_exog)

    n_periods = specs["n_periods"]
    choices = np.arange(specs["n_choices"], dtype=np.int8)

    exog_processes = {
        "part_time_offer": {
            "states": np.arange(2, dtype=np.int8),
            "transition": prob_part_time_offer,
        },
        "full_time_offer": {
            "states": np.arange(2, dtype=np.int8),
            "transition": prob_full_time_offer,
        },
        "mother_health": {
            "states": np.arange(4, dtype=np.int8),
            "transition": exog_health_transition_mother_with_survival,
        },
    }

    return {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "income_shock_scale": specs["income_shock_scale"],
            "taste_shock_scale": specs["lambda"],
            "endogenous_states": {
                "high_educ": np.arange(2, dtype=np.uint8),
                "has_sibling": np.arange(2, dtype=np.uint8),
                "experience": np.arange(
                    stop=specs["experience_cap"] + 1,
                    dtype=np.uint8,
                ),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": exog_processes,
        },
        "model_params": specs
        | wage_params
        | exog_params
        | {"interest_rate": 0.04, "bequest_scale": 1.3},
    }


def load_specs(path_to_specs):
    specs = yaml.safe_load(Path.open(path_to_specs))

    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1

    wage_params = task_create_exog_wage()

    specs["income_shock_scale"] = wage_params.pop("wage_std_regression_residual")

    return specs, wage_params
