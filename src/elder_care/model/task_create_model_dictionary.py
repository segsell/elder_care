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
    path_to_survival_prob_male: Path = BLD / "model" / "exog_survival_prob_male.pkl",
    path_to_health_transition_female: Path = BLD
    / "model"
    / "exog_health_transition_female.pkl",
    path_to_health_transition_male: Path = BLD
    / "model"
    / "exog_health_transition_male.pkl",
    path_to_care_single_mother: Path = BLD / "model" / "exog_care_single_mother.pkl",
    path_to_care_single_father: Path = BLD / "model" / "exog_care_single_father.pkl",
    path_to_care_couple: Path = BLD / "model" / "exog_care_couple.pkl",
    path_to_save: Annotated[Path, Product] = BLD / "model" / "exog_processes.pkl",
):
    """Assemble exogenous processes dictionaries.

    Exogenous processes:
    - survival probabilities
    - health transition
    - care demand

    """
    exog_survival_prob_female = load_dict_from_pickle(path_to_survival_prob_female)
    exog_survival_prob_male = load_dict_from_pickle(path_to_survival_prob_male)
    _exog_survival_prob = exog_survival_prob_female | exog_survival_prob_male

    exog_health_transition_female = load_dict_from_pickle(
        path_to_health_transition_female,
    )
    exog_health_transition_male = load_dict_from_pickle(path_to_health_transition_male)
    _exog_health_transition = (
        exog_health_transition_female | exog_health_transition_male
    )

    exog_care_single_mother = load_dict_from_pickle(path_to_care_single_mother)
    exog_care_single_father = load_dict_from_pickle(path_to_care_single_father)
    exog_care_couple = load_dict_from_pickle(path_to_care_couple)
    _exog_care = exog_care_single_mother | exog_care_single_father | exog_care_couple

    exog_all = _exog_survival_prob | _exog_health_transition | _exog_care

    save_dict_to_pickle(exog_all, path_to_save)

    return exog_all
