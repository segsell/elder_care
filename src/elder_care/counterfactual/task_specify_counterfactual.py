from pathlib import Path
from typing import Annotated, Any

import pytask

import numpy as np
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
from elder_care.model.shared import ALL
from elder_care.counterfactual.state_space_counterfactual import (
    create_state_space_functions_no_informal_care,
    create_state_space_functions_only_informal_care,
)
from elder_care.model.utility_functions import (
    create_final_period_utility_functions,
    create_utility_functions,
)

from elder_care.model.task_specify_model import get_options_dict


@pytask.mark.skip(reason="Respecifying model.")
def task_specify_and_setup_model_only_informal_care(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "model"
    / "counterfactual_only_informal_care.pkl",
) -> dict[str, Any]:
    """Generate options and setup model.

    start_params["sigma"] = specs["income_shock_scale"]

    """
    options = get_options_dict(path_to_specs, path_to_exog)

    return setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_functions_only_informal_care(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_save,
    )


@pytask.mark.skip(reason="Respecifying model.")
def task_specify_and_setup_model_no_informal_care(
    path_to_specs: Path = SRC / "model" / "specs.yaml",
    path_to_exog: Path = BLD / "model" / "exog_processes.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "model"
    / "counterfactual_no_informal_care.pkl",
) -> dict[str, Any]:
    """Generate options and setup model.

    start_params["sigma"] = specs["income_shock_scale"]

    """
    options = get_options_dict(path_to_specs, path_to_exog)

    return setup_and_save_model(
        options=options,
        state_space_functions=create_state_space_functions_no_informal_care(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_save,
    )
