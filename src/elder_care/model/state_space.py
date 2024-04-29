import numpy as np

from elder_care.model.shared import (
    FULL_TIME_AND_NO_WORK,
    NO_RETIREMENT,
    OUT_OF_LABOR,
    PART_TIME_AND_NO_WORK,
    RETIREMENT,
    WORK_AND_NO_WORK,
    is_full_time,
    is_part_time,
    is_retired,
)


def create_state_space_functions():
    return {
        "get_next_period_state": update_endog_state,
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
    }


# ==============================================================================
# State transition
# ==============================================================================


def get_state_specific_feasible_choice_set(
    period,
    lagged_choice,
    part_time_offer,
    full_time_offer,
    # mother_health,
    options,
):
    """Get feasible choice set for current parent state.

    # if ((mother_alive == 1) & (mother_health in [MEDIUM_HEALTH, BAD_HEALTH])) | ( #
    (father_alive == 1) & (father_health in [MEDIUM_HEALTH, BAD_HEALTH]) # ):

    if experience == options["experience_cap"]:     feasible_choice_set = [i for i in
    feasible_choice_set if i in NO_WORK]

    # elif period + options["start_age"] > options["retirement_age"]: #
    feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

    # elif (age > EARLY_RETIREMENT_AGE) & lagged_choice in NO_WORK: #
    feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

    """
    age = options["start_age"] + period

    feasible_choice_set = np.arange(options["n_choices"])

    # if mother_health == BAD_HEALTH:
    #     feasible_choice_set = [i for i in _feasible_choice_set_all if i in CARE]
    # else:
    #     feasible_choice_set = [i for i in _feasible_choice_set_all if i in NO_CARE]

    if age < options["min_ret_age"]:
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_RETIREMENT]

    if age >= options["max_ret_age"]:
        feasible_choice_set = [RETIREMENT]
    elif is_retired(lagged_choice):
        feasible_choice_set = [i for i in feasible_choice_set if i in RETIREMENT]
    elif (full_time_offer == 0) & (part_time_offer == 1):
        feasible_choice_set = [
            i for i in feasible_choice_set if i in PART_TIME_AND_NO_WORK
        ]
    elif (full_time_offer == 1) & (part_time_offer == 0):
        feasible_choice_set = [
            i for i in feasible_choice_set if i in FULL_TIME_AND_NO_WORK
        ]
    elif (full_time_offer == 1) & (part_time_offer == 1):
        feasible_choice_set = [i for i in feasible_choice_set if i in WORK_AND_NO_WORK]
    else:
        feasible_choice_set = [i for i in feasible_choice_set if i in OUT_OF_LABOR]

    return np.array(feasible_choice_set)


def update_endog_state(
    period,
    choice,
    experience,
    has_sibling,
    high_educ,
    options,
):
    """Update endogenous state variables.

    next_state["mother_age"] = options["mother_min_age"] + mother_age + 1
    next_state["father_age"] = options["father_min_age"] + father_age + 1


    experience_cap: 15 # maximum of exp accumulated, see Adda et al (2017)
    Returns to experience are flat after 15 years of experience.

    below_exp_cap = experience < options["experience_cap"]
    experience_current = below_exp_cap * is_working(choice)
    next_state["experience"] = experience + experience_current

    """
    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    below_exp_cap_part = experience + 1 < options["experience_cap"]
    below_exp_cap_full = experience + 2 < options["experience_cap"]
    experience_part_time = 1 * below_exp_cap_part * is_part_time(choice)
    experience_full_time = 2 * below_exp_cap_full * is_full_time(choice)
    next_state["experience"] = experience + experience_part_time + experience_full_time

    next_state["has_sibling"] = has_sibling
    next_state["high_educ"] = high_educ

    return next_state


def sparsity_condition(
    period,
    lagged_choice,
    experience,
    options,
):
    age = options["start_age"] + period

    max_init_experience = options["max_init_experience"]

    cond = True

    # You cannot retire before the earliest retirement age
    if (
        (age <= options["min_ret_age"]) & is_retired(lagged_choice)
        or (age > options["max_ret_age"]) & (is_retired(lagged_choice) is False)
        or (
            (is_full_time(lagged_choice) is False)
            & (is_part_time(lagged_choice) is False)
        )
        & (period + max_init_experience == experience)
        & (period > 0)
        | (experience > options["experience_cap"])
    ):
        cond = False

    if (age >= options["max_ret_age"] + 1) & (is_retired(lagged_choice) is False):
        cond = False

    return cond
