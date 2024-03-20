from typing import Any

import jax.numpy as jnp

from elder_care.model.shared import (
    is_bad_health,
    is_combination_care,
    is_formal_care,
    is_full_time,
    is_good_health,
    is_informal_care,
    is_medium_health,
    is_part_time,
)


def create_utility_functions():
    return {
        "utility": utility_func,
        "marginal_utility": marginal_utility,
        "inverse_marginal_utility": inverse_marginal_utility,
    }


def create_final_period_utility_functions():
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


# =====================================================================================
# Per-period utility
# =====================================================================================


def utility_func(
    consumption: jnp.array,
    choice: int,
    mother_alive: int,
    father_alive: int,
    mother_health: int,
    father_health: int,
    has_sibling: int,
    params: dict,
) -> jnp.array:
    """Computes the agent's current utility based on a CRRA utility function.

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

    # age is a proxy for health impacting the taste for free-time.
    utility_leisure = (
        (
            params["utility_leisure_constant"]
            + params["utility_leisure_age"] * (age - MIN_AGE)
        )
        * jnp.log(leisure_hours)
        / 4_000
    )

    Args:
        period (int): Current period.
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        mother_alive (int): Indicator for whether the mother is alive.
            0 = mother is not alive, 1 = mother is alive.
        father_alive (int): Indicator for whether the father is alive.
            0 = father is not alive, 1 = father is alive.
        mother_health (int): Health status of the mother. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        father_health (int): Health status of the father. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        has_sibling (int): Indicator for whether the agent has a sibling.
            0 = no sibling, 1 = has sibling.
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        utility (jnp.array): Agent's utility. Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    rho = params["rho"]

    mother_health = mother_health if mother_alive else -1
    father_health = father_health if father_alive else -1
    parental_health = jnp.maximum(mother_health, father_health)

    informal_care = is_informal_care(choice)
    formal_care = is_formal_care(choice)
    combination_care = is_combination_care(choice)
    part_time = is_part_time(choice)
    full_time = is_full_time(choice)

    utility_consumption = (consumption ** (1 - rho) - 1) / (1 - rho)

    disutility_working = (
        params["disutility_part_time"] * part_time
        + params["disutility_full_time"] * full_time
    )

    utility_caregiving = (
        # informal care by parental health status
        params["utility_informal_care_parent_good_health"]
        * informal_care
        * is_good_health(parental_health)
        + params["utility_informal_care_parent_medium_health"]
        * informal_care
        * is_medium_health(parental_health)
        + params["utility_informal_care_parent_bad_health"]
        * informal_care
        * is_bad_health(parental_health)
        # formal care by parental health status
        + params["utility_formal_care_parent_good_health"]
        * formal_care
        * is_good_health(parental_health)
        + params["utility_formal_care_parent_medium_health"]
        * formal_care
        * is_medium_health(parental_health)
        + params["utility_formal_care_parent_bad_health"]
        * formal_care
        * is_bad_health(parental_health)
        # combination care by parental health status
        + params["utility_combination_care_parent_good_health"]
        * combination_care
        * is_good_health(parental_health)
        + params["utility_combination_care_parent_medium_health"]
        * combination_care
        * is_medium_health(parental_health)
        + params["utility_combination_care_parent_bad_health"]
        * combination_care
        * is_bad_health(parental_health)
        # caregiving if sibling present
        + params["utility_informal_care_sibling"] * informal_care * has_sibling
        + params["utility_formal_care_sibling"] * formal_care * has_sibling
        + params["utility_combination_care_sibling"] * combination_care * has_sibling
    )

    return utility_consumption + disutility_working + utility_caregiving


def marginal_utility(consumption, params):
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    return marginal_utility ** (-1 / params["rho"])


# =====================================================================================
# Final period utility
# =====================================================================================


def utility_final_consume_all(
    resources: jnp.array,
    params: dict[str, float],
    options: dict[str, Any],
):
    rho = params["rho"]
    bequest_scale = options["bequest_scale"]

    return bequest_scale * (resources ** (1 - rho) - 1) / (1 - rho)


def marginal_utility_final_consume_all(
    resources: jnp.array,
    params: dict[str, float],
    options,
) -> jnp.array:
    """Computes marginal utility of CRRA utility function.

    Args:
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        resources (jnp.array): The agent's financial resources.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        marginal_utility (jnp.array): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    bequest_scale = options["bequest_scale"]
    return bequest_scale * (resources ** (-params["rho"]))
