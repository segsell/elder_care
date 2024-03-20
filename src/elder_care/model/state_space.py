import numpy as np

from elder_care.model.shared import (
    CARE,
    FULL_TIME,
    NO_CARE,
    NO_WORK,
    PART_TIME,
    WORK,
    is_working_full_time,
    is_working_part_time,
)


def create_state_space_functions():
    return {
        "get_next_period_state": update_endog_state,
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
    }


# ==============================================================================
# State transition
# ==============================================================================


def update_endog_state(period, choice, experience, married, has_sibling, options):
    """Update endogenous state variables.

    next_state["mother_age"] = options["mother_min_age"] + mother_age + 1
    next_state["father_age"] = options["father_min_age"] + father_age + 1

    alive based on exog state health based on exog state

    experience_cap: 44 # maximum of exp accumulated

    """
    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    below_exp_cap = experience < options["experience_cap"]
    experience_part_time = below_exp_cap * is_working_part_time(choice)
    experience_full_time = below_exp_cap * is_working_full_time(choice)
    next_state["experience"] = experience + experience_part_time + experience_full_time

    next_state["married"] = married
    next_state["has_sibling"] = has_sibling

    return next_state


def get_state_specific_feasible_choice_set(
    experience,
    part_time_offer,
    full_time_offer,
    care_demand,
    options,
):
    # state_vec including exog?
    feasible_choice_set = list(np.arange(options["n_choices"]))

    if care_demand:
        feasible_choice_set = [i for i in feasible_choice_set if i in CARE]
    else:
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_CARE]

    if experience == options["experience_cap"]:
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]
    elif (full_time_offer == True) | (part_time_offer == True):
        feasible_choice_set = [i for i in feasible_choice_set if i in WORK]
    elif (full_time_offer == False) & (part_time_offer == True):
        feasible_choice_set = [i for i in feasible_choice_set if i in PART_TIME]
    elif (full_time_offer == False) & (part_time_offer == False):
        feasible_choice_set = [i for i in feasible_choice_set if i in FULL_TIME]
    else:
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

    return np.array(feasible_choice_set)


def sparsity_condition(period, lagged_choice, experience, options):

    max_init_experience = options["max_init_experience"]

    cond = True

    # If you have not worked last period, you can't have worked all your live
    if (
        (
            (
                (is_working_full_time(lagged_choice) is False)
                & (is_working_part_time(lagged_choice) is False)
            )
            & (period + max_init_experience == experience)
            & (period > 0)
        )
        or experience > period + max_init_experience
        or experience > options["experience_cap"]
    ):
        cond = False

    return cond
