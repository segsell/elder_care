import numpy as np

from elder_care.model.shared import (
    CARE,
    FULL_TIME,
    NO_CARE,
    NO_WORK,
    PART_TIME,
    WORK,
    GOOD_HEALTH,
    MEDIUM_HEALTH,
    BAD_HEALTH,
    is_full_time,
    is_part_time,
)


def create_state_space_functions():
    return {
        "get_next_period_state": update_endog_state,
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
    }


# ==============================================================================
# State transition
# ==============================================================================


def update_endog_state(
    period,
    choice,
    #    experience,
    has_sibling,
    high_educ,
    # married,
    options,
):
    """Update endogenous state variables.

    next_state["mother_age"] = options["mother_min_age"] + mother_age + 1
    next_state["father_age"] = options["father_min_age"] + father_age + 1

    alive based on exog state health based on exog state

    experience_cap: 44 # maximum of exp accumulated

    """
    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    # below_exp_cap = experience < options["experience_cap"]
    # experience_part_time = below_exp_cap * is_part_time(choice)
    # experience_full_time = below_exp_cap * is_full_time(choice)
    # next_state["experience"] = experience + experience_part_time + experience_full_time

    # next_state["married"] = married
    next_state["has_sibling"] = has_sibling
    next_state["high_educ"] = high_educ

    return next_state


def get_state_specific_feasible_choice_set(
    period,
    # experience,
    part_time_offer,
    full_time_offer,
    mother_alive,
    father_alive,
    mother_health,
    father_health,
    options,
):
    _feasible_choice_set_all = list(np.arange(options["n_choices"]))

    if ((mother_alive == 1) & (mother_health in [MEDIUM_HEALTH, BAD_HEALTH])) | (
        (father_alive == 1) & (father_health in [MEDIUM_HEALTH, BAD_HEALTH])
    ):
        feasible_choice_set = [i for i in _feasible_choice_set_all if i in CARE]
    else:
        feasible_choice_set = [i for i in _feasible_choice_set_all if i in NO_CARE]

    # if experience == options["experience_cap"]:
    #     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]
    # elif period + options["start_age"] > options["retirement_age"]:
    #     feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]
    if (full_time_offer == False) & (part_time_offer == True):
        feasible_choice_set = [i for i in feasible_choice_set if i in PART_TIME]
    elif (full_time_offer == True) & (part_time_offer == False):
        feasible_choice_set = [i for i in feasible_choice_set if i in FULL_TIME]
    elif (full_time_offer == True) & (part_time_offer == True):
        feasible_choice_set = [i for i in feasible_choice_set if i in WORK]
    else:
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

    return np.array(feasible_choice_set)


def sparsity_condition(
    period,
    lagged_choice,
    options,
):

    max_init_experience = options["max_init_experience"]

    cond = True

    # If you have not worked last period, you can't have worked all your live
    # if (
    #     (
    #         (
    #             (is_full_time(lagged_choice) is False)
    #             & (is_part_time(lagged_choice) is False)
    #         )
    #         & (period + max_init_experience == experience)
    #         & (period > 0)
    #     )
    #     or experience > period + max_init_experience
    #     or experience > options["experience_cap"]
    # ):
    #     cond = False
    # if (period + options["start_age"] > options["retirement_age"] + 1) & (
    #     (is_full_time(lagged_choice) is True) & (is_part_time(lagged_choice) is True)
    # ):
    #     cond = False

    # if (mother_alive == 0) & (father_alive == 0) & (care_demand == 1):
    #     cond = False

    return cond
