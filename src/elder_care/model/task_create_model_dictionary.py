"""Assemble model dictionaries."""

from pathlib import Path

from elder_care.config import BLD
from elder_care.utils import load_dict_from_pickle


def task_assemble_exog_processes(
    path_to_care_single_mother: Path = BLD / "model" / "exog_care_single_mother.pkl",
    path_to_care_single_father: Path = BLD / "model" / "exog_care_single_father.pkl",
    path_to_care_couple: Path = BLD / "model" / "exog_care_couple.pkl",
):
    """Assemble exogenous processes dictionaries.

    Exogenous processes:
    - survival probabilities
    - health transition
    - care demand

    """
    exog_care_single_mother = load_dict_from_pickle(path_to_care_single_mother)
    exog_care_single_father = load_dict_from_pickle(path_to_care_single_father)
    exog_care_couple = load_dict_from_pickle(path_to_care_couple)

    return exog_care_single_mother | exog_care_single_father | exog_care_couple
