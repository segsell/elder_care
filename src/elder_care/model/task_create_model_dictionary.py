"""Assemble model dictionaries."""

from pathlib import Path
from typing import Annotated

from pytask import Product

from elder_care.config import BLD
from elder_care.utils import load_dict_from_pickle, save_dict_to_pickle


def task_assemble_exog_processes(
    path_to_survival_prob_female: Path = BLD
    / "model"
    / "exog_survival_prob_female.pkl",
    path_to_health_transition_female: Path = BLD
    / "model"
    / "exog_health_transition_female.pkl",
    path_to_save: Annotated[Path, Product] = BLD / "model" / "exog_processes.pkl",
):
    """Assemble exogenous processes dictionaries.

    Exogenous processes:
    - survival probabilities
    - health transition

    No care demand atm! Implicit by medium and bad health of mother.

    """
    exog_survival_prob_female = load_dict_from_pickle(path_to_survival_prob_female)

    exog_health_transition_female = load_dict_from_pickle(
        path_to_health_transition_female,
    )

    exog_all = exog_survival_prob_female | exog_health_transition_female

    save_dict_to_pickle(exog_all, path_to_save)

    return exog_all
