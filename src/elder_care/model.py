from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


MIN_AGE = 51
MAX_AGE = 65

AGE_50 = 50
AGE_53 = 53
AGE_56 = 56
AGE_59 = 59
AGE_62 = 62

AGE_55 = 55
AGE_60 = 60
AGE_65 = 65


AGE_BINS = [(AGE_50, AGE_55), (AGE_55, AGE_60), (AGE_60, AGE_65)]

PARENT_MIN_AGE = 68
PARENT_MAX_AGE = 98

RETIREMENT_AGE = 62

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2

ALL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


NO_WORK = [0, 1, 2, 3]
PART_TIME = [4, 5, 6, 7]
FULL_TIME = [8, 9, 10, 11]
WORK = PART_TIME + FULL_TIME

NO_CARE = [0, 4, 8]
FORMAL_CARE = [1, 3, 5, 7, 9, 11]  # % 2 == 1
INFORMAL_CARE = [2, 3, 6, 7, 10, 11]
CARE = FORMAL_CARE + INFORMAL_CARE

COMBINATION_CARE = [3, 7, 11]

# NO_INFORMAL_CARE = [0, 1, 4, 5, 8, 9] # check!
# NO_FORMAL_CARE = [0, 2, 4, 6, 8, 10]
NO_INFORMAL_CARE = list(set(CARE) - (set(INFORMAL_CARE)))
NO_FORMAL_CARE = list(set(ALL) - (set(FORMAL_CARE)))

TOTAL_WEEKLY_HOURS = 80
WEEKLY_HOURS_PART_TIME = 20
WEEKLY_HOURS_FULL_TIME = 40
WEEKLY_INTENSIVE_INFORMAL_HOURS = 14  # (21 + 7) / 2

N_MONTHS = 12
N_WEEKS = 4.33

PART_TIME_HOURS = 20 * N_WEEKS * N_MONTHS
FULL_TIME_HOURS = 40 * N_WEEKS * N_MONTHS


# ==============================================================================
# Model
# ==============================================================================


def is_not_working(lagged_choice):
    return lagged_choice in NO_WORK


def is_part_time(lagged_choice):
    return lagged_choice in PART_TIME


def is_full_time(lagged_choice):
    return lagged_choice in FULL_TIME


def is_informal_care(lagged_choice):
    # intensive only here
    return lagged_choice in INFORMAL_CARE


def is_no_informal_care(lagged_choice):
    # intensive only here
    return lagged_choice not in INFORMAL_CARE


def is_formal_care(lagged_choice):
    return lagged_choice in FORMAL_CARE


def is_no_formal_care(lagged_choice):
    return lagged_choice not in FORMAL_CARE


# ==============================================================================
# Exogenous processes
# ==============================================================================
def prob_part_time_offer(period, lagged_choice, options, params):
    """Compute logit probability of part time offer."""
    logit = (
        params["part_time_constant"]
        + params["part_time_not_working_last_period"] * is_not_working(lagged_choice)
        + params["part_time_working_full_time_last_period"]
        * is_full_time(lagged_choice)
        + params["part_time_above_retirement_age"]
        * (period + options["min_age"] >= RETIREMENT_AGE)
        # + params["part_time_high_education"] * high_educ
    )

    prob_logit = 1 / (1 + jnp.exp(-logit))

    return (
        is_part_time(lagged_choice) * 1 + (1 - is_part_time(lagged_choice)) * prob_logit
    )


def prob_full_time_offer(period, lagged_choice, options, params):
    """Compute logit probability of full time offer."""
    logit = (
        params["full_time_constant"]
        + params["full_time_not_working_last_period"] * is_not_working(lagged_choice)
        + params["full_time_working_part_time_last_period"]
        * is_part_time(lagged_choice)
        + params["full_time_above_retirement_age"]
        * (period + options["min_age"] >= RETIREMENT_AGE)
        # + params["full_time_high_education"] * high_educ
    )

    # _prob = jnp.exp(logit) / (1 + jnp.exp(logit))
    prob_logit = 1 / (1 + jnp.exp(-logit))

    return (
        is_full_time(lagged_choice) * 1 + (1 - is_full_time(lagged_choice)) * prob_logit
    )


def prob_survival_mother(period, options):
    """Predicts the survival probability based on logit parameters.

    coefs_male = np.array(
        [11.561515476144223, -0.11058331994203506, -1.0998977981246952e-05],
    )
    coefs_female = np.array(
        [17.01934835131644, -0.21245937682111807, 0.00047537366767865137],
    )

    if sex.lower() == "male":
        coefs = coefs_male
    elif sex.lower() == "female":
        coefs = coefs_female

    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2)

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    mother_age = period + options["mother_min_age"]

    logit = (
        options["survival_probability_mother_constant"]
        + options["survival_probability_mother_age"] * mother_age
        + options["survival_probability_mother_age_squared"] * (mother_age**2)
    )

    return 1 / (1 + jnp.exp(-logit))


def prob_survival_father(period, options):
    """Predicts the survival probability based on logit parameters.

    coefs_male = np.array(
        [11.561515476144223, -0.11058331994203506, -1.0998977981246952e-05],
    )
    coefs_female = np.array(
        [17.01934835131644, -0.21245937682111807, 0.00047537366767865137],
    )

    if sex.lower() == "male":
        coefs = coefs_male
    elif sex.lower() == "female":
        coefs = coefs_female

    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2)

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    father_age = period + options["father_min_age"]

    logit = (
        options["survival_probability_father_constant"]
        + options["survival_probability_father_age"] * father_age
        + options["survival_probability_father_age_squared"] * (father_age**2)
    )

    return 1 / (1 + jnp.exp(-logit))


def exog_health_transition_mother(period, mother_health, options):
    """Compute exogenous health transition probabilities.

    Multinomial logit model with three health states: good, medium, bad.

    This function computes the transition probabilities for an individual's health
    state based on their current age, squared age, and lagged health states.
    It uses a set of predefined parameters for medium and bad health states to
    calculate linear combinations, and then applies the softmax function to these
    linear combinations to get the transition probabilities.


    Returns:
        jnp.ndarray: Array of shape (3,) representing the probabilities of
            transitioning to good, medium, and bad health states, respectively.

    """
    mother_age = period + options["mother_min_age"]
    mother_age_squared = mother_age**2

    good_health = mother_health == GOOD_HEALTH
    medium_health = mother_health == MEDIUM_HEALTH
    bad_health = mother_health == BAD_HEALTH

    # Linear combination for medium health
    lc_medium_health = (
        options["mother_medium_health"]["medium_health_age"] * mother_age
        + options["mother_medium_health"]["medium_health_age_squared"]
        * mother_age_squared
        + options["mother_medium_health"]["medium_health_lagged_good_health"]
        * good_health
        + options["mother_medium_health"]["medium_health_lagged_medium_health"]
        * medium_health
        + options["mother_medium_health"]["medium_health_lagged_bad_health"]
        * bad_health
        + options["mother_medium_health"]["medium_health_constant"]
    )

    # Linear combination for bad health
    lc_bad_health = (
        options["mother_bad_health"]["bad_health_age"] * mother_age
        + options["mother_bad_health"]["bad_health_age_squared"] * mother_age_squared
        + options["mother_bad_health"]["bad_health_lagged_good_health"] * good_health
        + options["mother_bad_health"]["bad_health_lagged_medium_health"]
        * medium_health
        + options["mother_bad_health"]["bad_health_lagged_bad_health"] * bad_health
        + options["mother_bad_health"]["bad_health_constant"]
    )

    linear_comb = np.array([0, lc_medium_health, lc_bad_health])
    transition_probs = _softmax(linear_comb)

    return jnp.array([transition_probs[0], transition_probs[1], transition_probs[2]])


def exog_health_transition_father(period, father_health, options):
    """Compute exogenous health transition probabilities.

    Multinomial logit model with three health states: good, medium, bad.

    This function computes the transition probabilities for an individual's health
    state based on their current age, squared age, and lagged health states.
    It uses a set of predefined parameters for medium and bad health states to
    calculate linear combinations, and then applies the softmax function to these
    linear combinations to get the transition probabilities.


    Returns:
        jnp.ndarray: Array of shape (3,) representing the probabilities of
            transitioning to good, medium, and bad health states, respectively.

    """
    father_age = period + options["father_min_age"]
    father_age_squared = father_age**2

    good_health = father_health == GOOD_HEALTH
    medium_health = father_health == MEDIUM_HEALTH
    bad_health = father_health == BAD_HEALTH

    # Linear combination for medium health
    lc_medium_health = (
        options["father_medium_health"]["medium_health_age"] * father_age
        + options["father_medium_health"]["medium_health_age_squared"]
        * father_age_squared
        + options["father_medium_health"]["medium_health_lagged_good_health"]
        * good_health
        + options["father_medium_health"]["medium_health_lagged_medium_health"]
        * medium_health
        + options["father_medium_health"]["medium_health_lagged_bad_health"]
        * bad_health
        + options["father_medium_health"]["medium_health_constant"]
    )

    # Linear combination for bad health
    lc_bad_health = (
        options["father_bad_health"]["bad_health_age"] * father_age
        + options["father_bad_health"]["bad_health_age_squared"] * father_age_squared
        + options["father_bad_health"]["bad_health_lagged_good_health"] * good_health
        + options["father_bad_health"]["bad_health_lagged_medium_health"]
        * medium_health
        + options["father_bad_health"]["bad_health_lagged_bad_health"] * bad_health
        + options["father_bad_health"]["bad_health_constant"]
    )

    linear_comb = np.array([0, lc_medium_health, lc_bad_health])
    transition_probs = _softmax(linear_comb)

    return jnp.array([transition_probs[0], transition_probs[1], transition_probs[2]])


def _softmax(lc):
    """Compute the softmax of each element in an array of linear combinations.

    The softmax function is applied to an array of linear combination values (lc)
    to calculate the probabilities of each class in a multinomial logistic
    regression model.
    This function is typically used for multi-class classification problems.

    Args:
        lc (np.ndarray): An array of linear combination values. This can be a 1D array
            representing linear combinations for each class in a single data point,
            or a 2D array representing multiple data points.

    Returns:
        np.ndarray: An array of the same shape as `lc` where each value is transformed
            into the probability of the corresponding class, ensuring that the sum of
            probabilities across classes (for each data point if 2D) equals 1.

    Example:
    >>> lc = np.array([0, 1, 2])
    >>> softmax(lc)
    array([0.09003057, 0.24472847, 0.66524096])

    Note:
    - The function applies np.exp to each element in `lc` and then normalizes so that
      the sum of these exponentials is 1.
    - For numerical stability, the maximum value in each set of linear combinations
      is subtracted from each linear combination before exponentiation.

    """
    e_lc = np.exp(lc - np.max(lc))  # Subtract max for numerical stability
    return e_lc / e_lc.sum(axis=0)


def prob_exog_care_demand(
    period,
    mother_alive,
    mother_health,
    father_alive,
    father_health,
    options,
):
    """Create nested exogenous care demand probabilities.

    Compute based on parent alive. Otherwise zero.
    Done outside?!

    Nested exogenous transitions:
    - First, a parent's health state is determined by their age and lagged health state.

    Args:
        parental_age (int): Age of parent.
        parent_alive (int): Binary indicator of whether parent is alive.
        good_health (int): Binary indicator of good health.
        medium_health (int): Binary indicator of medium health.
        bad_health (int): Binary indicator of bad health.
        params (dict): Dictionary of parameters.

    Returns:
        jnp.ndarray: Array of shape (2,) representing the probabilities of
            no care demand and care demand, respectively.

    """
    mother_survival_prob = prob_survival_mother(period, options)
    father_survival_prob = prob_survival_father(period, options)

    mother_trans_probs_health = exog_health_transition_mother(
        period,
        mother_health,
        options,
    )
    father_trans_probs_health = exog_health_transition_father(
        period,
        father_health,
        options,
    )

    # ===============================================================

    # single mother
    mother_prob_care_good = _exog_care_demand_mother(
        period=period, mother_health=0, options=options,
    )
    mother_prob_care_medium = _exog_care_demand_mother(
        period=period, mother_health=1, options=options,
    )
    mother_prob_care_bad = _exog_care_demand_mother(
        period=period, mother_health=2, options=options,
    )

    _mother_trans_probs_care_demand = jnp.array(
        [mother_prob_care_good, mother_prob_care_medium, mother_prob_care_bad],
    )

    # single father
    father_prob_care_good = _exog_care_demand_father(
        period=period, father_health=0, options=options,
    )
    father_prob_care_medium = _exog_care_demand_father(
        period=period, father_health=1, options=options,
    )
    father_prob_care_bad = _exog_care_demand_father(
        period=period, father_health=2, options=options,
    )

    _father_trans_probs_care_demand = jnp.array(
        [father_prob_care_good, father_prob_care_medium, father_prob_care_bad],
    )

    # couple
    prob_care_mother_good_father_good = _exog_care_demand_couple(
        period=period,
        mother_health=0,
        father_health=0,
        options=options,
    )
    prob_care_mother_good_father_medium = _exog_care_demand_couple(
        mother_health=0,
        father_health=1,
        options=options,
    )
    prob_care_mother_good_father_bad = _exog_care_demand_couple(
        period=period,
        mother_health=0,
        father_health=2,
        options=options,
    )

    prob_care_mother_medium_father_good = _exog_care_demand_couple(
        period=period,
        mother_health=1,
        father_health=0,
        options=options,
    )
    prob_care_mother_medium_father_medium = _exog_care_demand_couple(
        period=period,
        mother_health=1,
        father_health=1,
        options=options,
    )
    prob_care_mother_medium_father_bad = _exog_care_demand_couple(
        period=period,
        mother_health=1,
        father_health=2,
        options=options,
    )

    prob_care_mother_bad_father_good = _exog_care_demand_couple(
        period=period,
        mother_health=2,
        father_health=0,
        options=options,
    )
    prob_care_mother_bad_father_medium = _exog_care_demand_couple(
        period=period,
        mother_health=2,
        father_health=1,
        options=options,
    )
    prob_care_mother_bad_father_bad = _exog_care_demand_couple(
        period=period,
        mother_health=2,
        father_health=2,
        options=options,
    )

    _couple_trans_probs_care_demand = jnp.array(
        [
            prob_care_mother_good_father_good,
            prob_care_mother_good_father_medium,
            prob_care_mother_good_father_bad,
            #
            prob_care_mother_medium_father_good,
            prob_care_mother_medium_father_medium,
            prob_care_mother_medium_father_bad,
            #
            prob_care_mother_bad_father_good,
            prob_care_mother_bad_father_medium,
            prob_care_mother_bad_father_bad,
        ],
    )

    # Non-zero probability of care demand only if parent is alive,
    # weighted by the parent's survival probability
    mother_single_prob_care_demand = (
        mother_survival_prob * mother_alive * (1 - father_alive)
    ) * (mother_trans_probs_health @ _mother_trans_probs_care_demand)

    father_single_prob_care_demand = (
        father_survival_prob * father_alive * (1 - mother_alive)
    ) * (father_trans_probs_health @ _father_trans_probs_care_demand)

    couple_prob_care_demand = (
        father_survival_prob * father_alive * mother_survival_prob * mother_alive
    ) * (
        jnp.outer(mother_trans_probs_health, father_trans_probs_health).flatten()
        @ _couple_trans_probs_care_demand
    )

    prob_care_demand = (
        mother_single_prob_care_demand
        + father_single_prob_care_demand
        + couple_prob_care_demand
    )

    return jnp.array([1 - prob_care_demand, prob_care_demand])


#


def _exog_care_demand_mother(period, mother_health, options):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    mother_age = period + options["mother_min_age"]

    logit = (
        options["exog_care_single_mother_constant"]
        + options["exog_care_single_mother_age"] * mother_age
        + options["exog_care_single_mother_age_squared"] * (mother_age**2)
        + options["exog_care_single_mother_medium_health"]
        * (mother_health == MEDIUM_HEALTH)
        + options["exog_care_single_mother_bad_health"] * (mother_health == BAD_HEALTH)
    )
    return 1 / (1 + np.exp(-logit))


def _exog_care_demand_father(period, father_health, options):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    father_age = period + options["father_min_age"]

    logit = (
        options["exog_care_single_father_constant"]
        + options["exog_care_single_father_age"] * father_age
        + options["exog_care_single_father_age_squared"] * (father_age**2)
        + options["exog_care_single_father_medium_health"]
        * (father_health == MEDIUM_HEALTH)
        + options["exog_care_single_father_bad_health"] * (father_health == BAD_HEALTH)
    )
    return 1 / (1 + np.exp(-logit))


def _exog_care_demand_couple(period, mother_health, father_health, options):
    """Compute scalar care demand probability.

    Returns:
        float: Probability of needing care given health state.

    """
    mother_age = period + options["mother_min_age"]
    father_age = period + options["father_min_age"]

    logit = (
        options["exog_care_couple_constant"]
        + options["exog_care_couple_mother_age"] * mother_age
        + options["exog_care_couple_mother_age_squared"] * (mother_age**2)
        + options["exog_care_couple_mother_medium_health"]
        * (mother_health == MEDIUM_HEALTH)
        + options["exog_care_couple_mother_bad_health"] * (mother_health == BAD_HEALTH)
        + options["exog_care_couple_father_age"] * father_age
        + options["exog_care_couple_father_age_squared"] * (father_age**2)
        + options["exog_care_couple_father_medium_health"]
        * (father_health == MEDIUM_HEALTH)
        + options["exog_care_couple_father_bad_health"] * (father_health == BAD_HEALTH)
    )
    return 1 / (1 + np.exp(-logit))


# ==============================================================================
# State transition
# ==============================================================================


def update_endog_state(
    period,
    choice,
    options,
):
    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    # next_state["mother_age"] = options["mother_min_age"] + mother_age + 1
    # next_state["father_age"] = options["father_min_age"] + father_age + 1

    # alive based on exog state
    # health based on exog state

    return next_state


def get_state_specific_feasible_choice_set(
    part_time_offer, full_time_offer, mother_alive, father_alive, options,
):
    # formal_care = choice % 2 == 1  # uneven numbers mark formal care
    # light_informal_care = [2, 3, 8, 9, 14, 15]
    # intensive_informal_care =[4, 5, 10, 11, 16, 17]
    # NO_CARE = NO_CARE
    # CARE = [choice for choice in all_choices if choice not in NO_CARE]

    # state_vec including exog?
    feasible_choice_set = list(np.arange(options["n_choices"]))

    # care demand
    if mother_alive or father_alive:
        feasible_choice_set = [i for i in feasible_choice_set if i in CARE]
    else:
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_CARE]

    # job offer
    if (full_time_offer == True) | (part_time_offer == True):
        feasible_choice_set = [i for i in feasible_choice_set if i in WORK]
    elif (full_time_offer == False) & (part_time_offer == True):
        feasible_choice_set = [i for i in feasible_choice_set if i in PART_TIME]
    elif (full_time_offer == False) & (part_time_offer == False):
        feasible_choice_set = [i for i in feasible_choice_set if i in FULL_TIME]
    else:
        # (full_time_offer == False) & (part_time_offer == False)
        feasible_choice_set = [i for i in feasible_choice_set if i in NO_WORK]

    return np.array(feasible_choice_set)


# ==============================================================================
# Utility functions
# ==============================================================================


def utility_func(
    consumption: jnp.array, period, choice: int, options: dict, params: dict,
) -> jnp.array:
    """Computes the agent's current utility based on a CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        utility (jnp.array): Agent's utility . Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    rho = params["rho"]
    age = period + options["min_age"]

    informal_care = is_informal_care(choice)
    formal_care = is_formal_care(choice)
    part_time = is_part_time(choice)
    full_time = is_full_time(choice)

    working_hours_weekly = (
        part_time * WEEKLY_HOURS_PART_TIME + full_time * WEEKLY_HOURS_FULL_TIME
    )
    # From SOEP data we know that the 25% and 75% percentile in the care hours
    # distribution are 7 and 21 hours per week in a comparative sample.
    # We use these discrete mass-points as discrete choices of non-intensive and
    # intensive informal care.
    # In SHARE, respondents inform about the frequency with which they provide
    # informal care. We use this information to proxy the care provision in the data.
    caregiving_hours_weekly = informal_care * WEEKLY_INTENSIVE_INFORMAL_HOURS
    leisure_hours = (
        (TOTAL_WEEKLY_HOURS - working_hours_weekly - caregiving_hours_weekly)
        * 4.33  # month
        * 12  # year
    )

    utility_consumption = (consumption ** (1 - rho) - 1) / (1 - rho)

    # age is a proxy for health impacting the taste for free-time.
    utility_leisure = (
        params["utility_leisure_constant"]
        + params["utility_leisure_age"] * (age - MIN_AGE)
    ) * np.log(leisure_hours)

    return (
        utility_consumption
        + utility_leisure
        + params["disutility_part_time"] * part_time
        + params["disutility_full_time"] * full_time
        ## utility from caregiving
        + options["utility_informal_care"] * informal_care
        + options["utility_formal_care"] * formal_care
        + options["utility_informal_and_formal_care"] * (formal_care & informal_care)
    )


def marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])


def utility_final_consume_all(
    choice: int,
    period: int,
    resources: jnp.array,
    params: dict[str, float],
    options: dict[str, Any],
):
    age = period + options["min_age"]

    informal_care = is_informal_care(choice)
    formal_care = is_formal_care(choice)
    part_time = is_part_time(choice)
    full_time = is_full_time(choice)

    working_hours_weekly = (
        part_time * WEEKLY_HOURS_PART_TIME + full_time * WEEKLY_HOURS_FULL_TIME
    )
    # From SOEP data we know that the 25% and 75% percentile in the care hours
    # distribution are 7 and 21 hours per week in a comparative sample.
    # We use these discrete mass-points as discrete choices of non-intensive and
    # intensive informal care.
    # In SHARE, respondents inform about the frequency with which they provide
    # informal care. We use this information to proxy the care provision in the data.
    caregiving_hours_weekly = informal_care * WEEKLY_INTENSIVE_INFORMAL_HOURS
    leisure_hours = (
        (TOTAL_WEEKLY_HOURS - working_hours_weekly - caregiving_hours_weekly)
        * 4.33  # month
        * 12  # year
    )

    utility_consumption = (resources ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility_leisure = (
        params["utility_leisure_constant"]
        + params["utility_leisure_age"] * (age - options["min_age"])
    ) * np.log(leisure_hours)

    return (
        utility_consumption
        + utility_leisure
        + params["disutility_part_time"] * part_time
        + params["disutility_full_time"] * full_time
        ## utility from caregiving
        + options["utility_informal_care"] * informal_care
        + options["utility_formal_care"] * formal_care
        + options["utility_informal_and_formal_care"] * (formal_care & informal_care)
    )


def marginal_utility_final_consume_all(
    choice, resources: jnp.array, params: dict[str, float], options: dict[str, Any],
) -> jnp.array:
    """Computes marginal utility of CRRA utility function.

    Args:
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.

    Returns:
        marginal_utility (jnp.array): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    return resources ** (-params["rho"])


# ==============================================================================
# Budget constraint
# ==============================================================================


def budget_constraint(
    period: int,
    lagged_choice: int,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: dict[str, Any],
    params: dict[str, float],
) -> float:
    # already done in preprocessing
    # model_params = options["model_params"]

    # monthly
    working_hours = (
        is_part_time(lagged_choice) * 20 * 4.33 * 12  # week month year
        + is_full_time(lagged_choice) * 40 * 4.33 * 12  # week month year
    )

    wage_from_previous_period = calc_stochastic_wage(
        period=period,
        lagged_choice=lagged_choice,
        wage_shock=income_shock_previous_period,
        min_age=options["min_age"],
        constant=params["constant"],
    )

    wealth_beginning_of_period = (
        wage_from_previous_period * working_hours
        # + non_labor_income(age, high_educ, options)
        # + spousal_income(period, high_educ, options) * married
        + options["unemployment_benefits"] * is_not_working(lagged_choice)
        + options["informal_care_benefits"] * is_informal_care(lagged_choice)
        - options["formal_care_costs"] * is_formal_care(lagged_choice)
        + (1 + options["interest_rate"]) * savings_end_of_previous_period
    )

    # needed at all?
    wealth_beginning_of_period = jnp.maximum(
        wealth_beginning_of_period, options["consumption_floor"],
    )

    return wealth_beginning_of_period


# @jax.jit
def calc_stochastic_wage(
    period: int,
    lagged_choice: int,
    wage_shock: float,
    min_age: int,
    params: dict[str, float],
) -> float:
    """Computes the current level of deterministic and stochastic income.

    Note that income is paid at the end of the current period, i.e. after
    the (potential) labor supply choice has been made. This is equivalent to
    allowing income to be dependent on a lagged choice of labor supply.
    The agent starts working in period t = 0.
    Relevant for the wage equation (deterministic income) are age-dependent
    coefficients of work experience:
    labor_income = constant + alpha_1 * age + alpha_2 * age**2
    They include a constant as well as two coefficients on age and age squared,
    respectively. Note that the last one (alpha_2) typically has a negative sign.

    Args:
        state (jnp.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        wage_shock (float): Stochastic shock on labor income;
            may or may not be normally distributed. This float represents one
            particular realization of the income_shock_draws carried over from
            the previous period.
        params (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (float): The potential end of period income. It consists of a
            deterministic component, i.e. age-dependent labor income,
            and a stochastic shock.

    """
    # For simplicity, assume current_age - min_age = experience
    age = period + min_age

    # Determinisctic component of income depending on experience:
    # constant + alpha_1 * age + alpha_2 * age**2
    # exp_coeffs = jnp.array([constant, exp, exp_squared])
    # labor_income = exp_coeffs @ (age ** jnp.arange(len(exp_coeffs)))
    # working_income = jnp.exp(labor_income + wage_shock)

    log_wage = (
        params["wage_constant"]
        + params["wage_age"] * age
        + params["wage_age_squared"] * age**2
        # + params["wage_high_educ"] * high_educ
        + params["wage_part_time"] * is_part_time(lagged_choice)
        + params["wage_not_working"] * is_not_working(lagged_choice)
    )

    return jnp.exp(log_wage + wage_shock)


# ==============================================================================

# ==============================================================================
# Initial conditions
# ==============================================================================


def get_initial_share_three(initial_conditions, shares):
    return jnp.asarray(initial_conditions.loc[shares]).ravel()


def get_initial_share_two(initial_conditions, var):
    share_yes = initial_conditions.loc[var]
    return jnp.asarray([1 - share_yes, share_yes]).ravel()


def draw_random_sequence_from_array(seed, arr, n_agents):
    """Draw a random sequence from an array.

    rand = draw_random_sequence_from_array(     seed=2024,     n_agents=10_000,
    arr=jnp.array(initial_wealth), )

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, arr, shape=(n_agents,), replace=True)


def draw_random_array(seed, n_agents, values, probabilities):
    """Draw a random array with given probabilities.

    Usage:

    seed = 2024
    n_agents = 10_000

    # Parameters
    values = jnp.array([-1, 0, 1, 2])  # Values to choose from
    probabilities = jnp.array([0.3, 0.3, 0.2, 0.2])  # Corresponding probabilities

    table(pd.DataFrame(random_array)[0]) / 1000

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, values, shape=(n_agents,), p=probabilities)


def draw_parental_age(seed, n_agents, mean, std_dev):
    """Draw discrete parental age."""
    key = jax.random.PRNGKey(seed)

    sample_standard_normal = jax.random.normal(key, (n_agents,))

    # Scaling and shifting to get the desired mean and standard deviation, then rounding
    return jnp.round(mean + std_dev * sample_standard_normal).astype(jnp.int32)


# ==============================================================================
# Simulation
# ==============================================================================


def create_simulation_df(sim_dict, options, params):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    df = pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_agents)],
            names=["id"],
        ),
    )

    df["wealth"] = df["savings"] + df["consumption"]
    df["wage"] = calc_stochastic_wage(
        df["period"], df["lagged_choice"], df["wage_shock"], options["min_age"], params,
    )

    return df


def simulate_moments(df):
    """Df has multiindex ["period", "agent"] necessary?

    or "agent", "period" as columns. "age" is also a column

    .loc needed below?!

    """
    # share working by age
    share_not_working_by_age = get_share_by_age(df, lagged_choice=NO_WORK)  # 15
    share_working_part_time_by_age = get_share_by_age(df, lagged_choice=PART_TIME)  # 15
    share_working_full_time_by_age = get_share_by_age(df, lagged_choice=FULL_TIME)  # 15

    share_informal_care_by_age_bin = get_share_by_age_bin(
        df, lagged_choice=INFORMAL_CARE,
    )

    # yearly net income
    income_part_time_by_age_bin = get_income_by_age_bin(df, lagged_choice=PART_TIME)
    income_full_time_by_age_bin = get_income_by_age_bin(df, lagged_choice=FULL_TIME)

    # wealth
    wealth_by_age_bin = get_wealth_beginning_of_period_by_age_bin(df)

    # share working by caregiving type (and age bin) --> to be checked
    share_not_working_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            df, lagged_choice=NO_WORK, care_type=NO_INFORMAL_CARE,
        )
    )
    share_part_time_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            df, lagged_choice=PART_TIME, care_type=NO_INFORMAL_CARE,
        )
    )
    share_full_time_no_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            df, lagged_choice=FULL_TIME, care_type=NO_INFORMAL_CARE,
        )
    )

    share_not_working_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            df, lagged_choice=NO_WORK, care_type=INFORMAL_CARE,
        )
    )
    share_part_time_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            df, lagged_choice=PART_TIME, care_type=INFORMAL_CARE,
        )
    )
    share_full_time_informal_care_by_age_bin = (
        get_share_by_informal_care_type_by_age_bin(
            df, lagged_choice=FULL_TIME, care_type=INFORMAL_CARE,
        )
    )

    # parent child: mother
    informal_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        df, care_choice=INFORMAL_CARE, parent="mother", is_other_parent_alive=True,
    )
    formal_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        df, care_choice=FORMAL_CARE, parent="mother", is_other_parent_alive=True,
    )
    combination_care_by_mother_health_couple = get_caregiving_status_by_parental_health(
        df, care_choice=COMBINATION_CARE, parent="mother", is_other_parent_alive=True,
    )

    informal_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        df, care_choice=INFORMAL_CARE, parent="mother", is_other_parent_alive=False,
    )
    formal_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        df, care_choice=FORMAL_CARE, parent="mother", is_other_parent_alive=False,
    )

    combination_care_by_mother_health_single = get_caregiving_status_by_parental_health(
        df, care_choice=COMBINATION_CARE, parent="mother", is_other_parent_alive=False,
    )

    # parent child: father
    informal_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        df, care_choice=INFORMAL_CARE, parent="father", is_other_parent_alive=True,
    )
    formal_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        df, care_choice=FORMAL_CARE, parent="father", is_other_parent_alive=True,
    )
    combination_care_by_father_health_couple = get_caregiving_status_by_parental_health(
        df, care_choice=COMBINATION_CARE, parent="father", is_other_parent_alive=True,
    )

    informal_care_by_father_health_single = get_caregiving_status_by_parental_health(
        df, care_choice=INFORMAL_CARE, parent="father", is_other_parent_alive=False,
    )
    formal_care_by_father_health_single = get_caregiving_status_by_parental_health(
        df, care_choice=FORMAL_CARE, parent="father", is_other_parent_alive=False,
    )
    combination_care_by_father_health_single = get_caregiving_status_by_parental_health(
        df, care_choice=COMBINATION_CARE, parent="father", is_other_parent_alive=False,
    )

    # work transitions
    no_work_to_no_work = get_work_transition(df, NO_WORK, NO_WORK)
    no_work_to_part_time = get_work_transition(df, NO_WORK, PART_TIME)
    no_work_to_full_time = get_work_transition(df, NO_WORK, FULL_TIME)

    part_time_to_no_work = get_work_transition(df, PART_TIME, NO_WORK)
    part_time_to_part_time = get_work_transition(df, PART_TIME, PART_TIME)
    part_time_to_full_time = get_work_transition(df, PART_TIME, FULL_TIME)

    full_time_to_no_work = get_work_transition(df, FULL_TIME, NO_WORK)
    full_time_to_part_time = get_work_transition(df, FULL_TIME, PART_TIME)
    full_time_to_full_time = get_work_transition(df, FULL_TIME, FULL_TIME)

    # caregiving transitions
    no_informal_care_to_no_informal_care = get_work_transition(
        df, NO_INFORMAL_CARE, NO_INFORMAL_CARE,
    )
    no_informal_care_to_informal_care = get_work_transition(
        df, NO_INFORMAL_CARE, INFORMAL_CARE,
    )

    informal_care_to_no_informal_care = get_work_transition(
        df, INFORMAL_CARE, NO_INFORMAL_CARE,
    )
    informal_care_to_informal_care = get_work_transition(
        df, INFORMAL_CARE, INFORMAL_CARE,
    )

    no_informal_care_to_no_formal_care = get_work_transition(
        df, NO_INFORMAL_CARE, NO_FORMAL_CARE,
    )
    no_informal_care_to_formal_care = get_work_transition(
        df, NO_INFORMAL_CARE, FORMAL_CARE,
    )

    informal_care_to_no_formal_care = get_work_transition(
        df, INFORMAL_CARE, NO_FORMAL_CARE,
    )
    informal_care_to_formal_care = get_work_transition(df, INFORMAL_CARE, FORMAL_CARE)

    #
    no_formal_care_to_no_informal_care = get_work_transition(
        df, NO_FORMAL_CARE, NO_INFORMAL_CARE,
    )
    no_formal_care_to_informal_care = get_work_transition(
        df, NO_FORMAL_CARE, INFORMAL_CARE,
    )

    formal_care_to_no_informal_care = get_work_transition(
        df, FORMAL_CARE, NO_INFORMAL_CARE,
    )
    formal_care_to_informal_care = get_work_transition(df, FORMAL_CARE, INFORMAL_CARE)

    no_formal_care_to_no_formal_care = get_work_transition(
        df, NO_FORMAL_CARE, NO_FORMAL_CARE,
    )
    no_formal_care_to_formal_care = get_work_transition(df, NO_FORMAL_CARE, FORMAL_CARE)

    formal_care_to_no_formal_care = get_work_transition(df, FORMAL_CARE, NO_FORMAL_CARE)
    formal_care_to_formal_care = get_work_transition(df, FORMAL_CARE, FORMAL_CARE)

    return jnp.array(
        [
            share_not_working_by_age,
            share_working_part_time_by_age,
            share_working_full_time_by_age,
            #
            share_informal_care_by_age_bin,
            #
            income_part_time_by_age_bin,
            income_full_time_by_age_bin,
            wealth_by_age_bin,
            #
            share_not_working_no_informal_care_by_age_bin,
            share_part_time_no_informal_care_by_age_bin,
            share_full_time_no_informal_care_by_age_bin,
            share_not_working_informal_care_by_age_bin,
            share_part_time_informal_care_by_age_bin,
            share_full_time_informal_care_by_age_bin,
            #
            informal_care_by_mother_health_couple,
            formal_care_by_mother_health_couple,
            combination_care_by_mother_health_couple,
            informal_care_by_mother_health_single,
            formal_care_by_mother_health_single,
            combination_care_by_mother_health_single,
            #
            informal_care_by_father_health_couple,
            formal_care_by_father_health_couple,
            combination_care_by_father_health_couple,
            informal_care_by_father_health_single,
            formal_care_by_father_health_single,
            combination_care_by_father_health_single,
            # work transitions
            no_work_to_no_work,
            no_work_to_part_time,
            no_work_to_full_time,
            part_time_to_no_work,
            part_time_to_part_time,
            part_time_to_full_time,
            full_time_to_no_work,
            full_time_to_part_time,
            full_time_to_full_time,
            # caregiving transitions
            no_informal_care_to_no_informal_care,
            no_informal_care_to_informal_care,
            informal_care_to_no_informal_care,
            informal_care_to_informal_care,
            no_informal_care_to_no_formal_care,
            no_informal_care_to_formal_care,
            informal_care_to_no_formal_care,
            informal_care_to_formal_care,
            #
            no_formal_care_to_no_informal_care,
            no_formal_care_to_informal_care,
            formal_care_to_no_informal_care,
            formal_care_to_informal_care,
            no_formal_care_to_no_formal_care,
            no_formal_care_to_formal_care,
            formal_care_to_no_formal_care,
            formal_care_to_formal_care,
        ],
    )


# ==============================================================================


def get_share_by_age(df, lagged_choice):
    return [
        len(df[(df["age"] == age) & (df["lagged_choice"].isin(lagged_choice))])
        / len(df[(df["age"] == age)])
        for age in range(MIN_AGE, MAX_AGE + 1)
    ]


def get_share_by_age_bin(df, lagged_choice):
    return [
        len(
            df[
                (df["age"] > age_bin[0])
                & (df["age"] <= age_bin[1])
                & (df["lagged_choice"].isin(lagged_choice))
            ],
        )
        / len(df[(df["age"] > age_bin[0]) & (df["age"] <= age_bin[1])])
        for age_bin in AGE_BINS
    ]


def get_share_by_informal_care_type_by_age_bin(df, lagged_choice, care_type):
    """Really lagged choice or rather current (end of period) choice?"""
    return [
        len(
            df[
                (df["age"] > age_bin[0])
                & (df["age"] <= age_bin[1])
                & (df["lagged_choice"].isin(care_type))
                & (df["lagged_choice"].isin(lagged_choice))
            ],
        )
        / len(
            df[
                (df["age"] > age_bin[0])
                & (df["age"] <= age_bin[1])
                & (df["lagged_choice"].isin(care_type))
            ],
        )
        for age_bin in AGE_BINS
    ]


def _get_share_by_informal_care_type(df, lagged_choice, care_type):
    """Really lagged choice or rather current (end of period) choice?"""
    return [
        len(
            df[
                (df["lagged_choice"].isin(care_type))
                & (df["lagged_choice"].isin(lagged_choice))
            ],
        )
        / len(df[(df["lagged_choice"].isin(care_type))]),
    ]


def get_income_by_age_bin(df, lagged_choice):
    """Net income in absolute (non-log) terms.

    After taxes and transfers as reported in SHARE.

    """
    df["working_hours"] = (
        df[(df["lagged_choice"].isin(PART_TIME))] * PART_TIME_HOURS
        + df[(df["lagged_choice"].isin(FULL_TIME))] * FULL_TIME_HOURS
    )
    df["income"] = df["working_hours"] * df["wage"]

    return [
        df.loc[(df["age"] > age_bin[0]) & (df["age"] <= age_bin[1]), "income"].mean()
        for age_bin in AGE_BINS
    ]


def get_wealth_beginning_of_period_by_age_bin(df):
    """savings_current_period = resources_beginning_of_period - consumption

    beginning or end of period wealth?
    end of period wealth = savings

    matter of whether consumption is included or not (part of income that is consumed)

    """
    return [
        df.loc[(df["age"] > age_bin[0]) & (df["age"] <= age_bin[1]), "wealth"].mean()
        for age_bin in AGE_BINS
    ]


def _get_share_by_informal_care_type_by_age_bin(df, lagged_choice, is_informal_care):
    """Duplicate?

    What's the difference?

    """
    return [
        len(
            df[
                (df["lagged_choice"].isin(INFORMAL_CARE) * is_informal_care)
                & (df["age"] > age_bin[0])
                & (df["age"] <= age_bin[1])
                & (df["lagged_choice"].isin(lagged_choice))
            ],
        )
        / len(
            df[
                (df["lagged_choice"].isin(INFORMAL_CARE) * is_informal_care)
                & (df["age"] > age_bin[0])
                & (df["age"] <= age_bin[1])
            ],
        )
        for age_bin in AGE_BINS
    ]


def get_caregiving_status_by_parental_health(
    df, care_choice, parent, is_other_parent_alive,
):
    other_parent = ("father") * (parent == "mother") + ("mother") * (parent == "father")

    return [
        len(
            df[
                (df[f"{parent}_health"] == health)
                & (df[f"{other_parent}_alive"] == is_other_parent_alive)
                & (df["choice"].isin(care_choice))
            ],
        )
        for health in [GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH]
    ]


def get_work_transition(df, lagged_choice, current_choice):
    """df[(df["lagged_choice"].isin(NO_WORK)) & (df["choice"].isin(NO_WORK))]"""
    return len(
        df[
            (df["lagged_choice"].isin(lagged_choice))
            & (df["choice"].isin(current_choice))
        ],
    ) / len(df[(df["lagged_choice"].isin(lagged_choice))])


def get_care_transition(df, lagged_choice, current_choice):
    """"""
    return len(
        df[
            (df["lagged_choice"].isin(lagged_choice))
            & (df["choice"].isin(current_choice))
        ],
    ) / len(df[(df["lagged_choice"].isin(lagged_choice))])
