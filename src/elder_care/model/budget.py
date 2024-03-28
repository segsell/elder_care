"""Budget functions for the Elder Care model."""

from typing import Any

import jax.numpy as jnp
import numpy as np

from elder_care.model.shared import is_full_time, is_not_working, is_part_time

# =====================================================================================
# Exogenous savings grid
# =====================================================================================


def create_savings_grid():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=10, step=0.5)  # 20
    section_2 = np.arange(start=10, stop=50, step=1)  # 40
    section_3 = np.arange(start=50, stop=100, step=5)  # 10
    section_4 = np.arange(start=100, stop=500, step=20)  # 20
    section_5 = np.arange(start=500, stop=1000, step=100)  # 5

    return (
        np.concatenate([section_1, section_2, section_3, section_4, section_5]) * 1_000
    )


# ==============================================================================
# Budget constraint
# ==============================================================================


def budget_constraint(
    period: int,
    lagged_choice: int,
    experience: int,
    high_educ: int,
    savings_end_of_previous_period: float,
    income_shock_previous_period: float,
    options: dict[str, Any],
) -> float:
    """Budget constraint.

    + non_labor_income(age, high_educ, options)

    + spousal_income(period, high_educ, options) * married

    return jnp.maximum(
        wealth_beginning_of_period,
        options["consumption_floor"],
    )


    + options["unemployment_benefits"] * is_not_working(lagged_choice) * 12
    + options["informal_care_benefits"] * is_informal_care(lagged_choice) * 12
    - options["formal_care_costs"] * is_formal_care(lagged_choice) * 12


    means_test = savings_end_of_previous_period < options["unemployment_wealth_thresh"]
    unemployment_benefits_yearly = means_test * options["unemployment_benefits"] * 12

    """
    working_hours_yearly = (
        is_part_time(lagged_choice) * 20 * 4.33 * 12  # week month year
        + is_full_time(lagged_choice) * 40 * 4.33 * 12  # week month year
    )

    wage_from_previous_period = get_exog_stochastic_wage(
        period=period,
        lagged_choice=lagged_choice,
        experience=experience,
        high_educ=high_educ,
        wage_shock=income_shock_previous_period,
        options=options,
    )

    wealth_beginning_of_period = (
        wage_from_previous_period * working_hours_yearly
        + options["unemployment_benefits"] * is_not_working(lagged_choice) * 12
        + (1 + options["interest_rate"]) * savings_end_of_previous_period
    )

    return jnp.maximum(
        wealth_beginning_of_period,
        options["consumption_floor"] * 12,
    )


def get_exog_stochastic_wage(
    period: int,
    lagged_choice: int,
    experience: int,
    high_educ: int,
    wage_shock: float,
    options: dict[str, float],
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


    Divide experience by 2 to get the number of years of experience.
    One year of part-time work experience is equivalent to
    0.5 years of full-time work experience. This is because the agent accumulates
    experience at a slower rate when working part-time.

    For computational reasons, we use the following transformation:
    exp = experience / 2
    exp_squared = (experience / 2) ** 2

    So one year of part-time experience is counted as 1 year of experience in the
    state space. But in the wage equation, it is counted as 0.5 years of experience.
    Analogously, one year of full-time experience is counted as 2 years of experience
    in the state space, but as 1 year of experience in the wage equation.


    Determinisctic component of income depending on experience:
    constant + alpha_1 * age + alpha_2 * age**2
    exp_coeffs = jnp.array([constant, exp, exp_squared])
    labor_income = exp_coeffs @ (age ** jnp.arange(len(exp_coeffs)))
    working_income = jnp.exp(labor_income + wage_shock)


    Args:
        period (int): Current period.
        state (jnp.ndarray): 1d array of shape (n_state_variables,) denoting
            the current child state.
        lagged_choice (int): The lagged choice of the agent.
        experience (int): Work experience. Full-time work experience adds 1, part-time
            experience adds 0.5, and no work experience adds 0 years of experience
            per period.
        wage_shock (float): Stochastic shock on labor income;
            may or may not be normally distributed. This float represents one
            particular realization of the income_shock_draws carried over from
            the previous period.
        high_educ (int): Indicator for whether the agent has a high education.
            0 = low education, 1 = high education. High education means that
            the agent has completed at least 15 years of education.
        params (dict): Dictionary containing model parameters.
            Relevant here are the coefficients of the wage equation.
        options (dict): Options dictionary.

    Returns:
        stochastic_income (float): The potential end of period income. It consists of a
            deterministic component, i.e. age-dependent labor income,
            and a stochastic shock.

    """
    age = period + options["start_age"]

    log_wage = (
        options["wage_constant"]
        + options["wage_age"] * age
        + options["wage_age_squared"] * age**2
        + options["wage_experience"] * (experience / 2)
        + options["wage_experience_squared"] * (experience / 2) ** 2
        + options["wage_high_education"] * high_educ
        + options["wage_part_time"] * is_part_time(lagged_choice)
    )

    return jnp.exp(log_wage + wage_shock)


def get_exog_spousal_income(period, options):
    """Income from the spouse."""
    age = period + options["start_age"]

    return (
        options["spousal_income_constant"]
        + options["spousal_income_age"] * age
        + options["spousal_income_age_squared"] * age**2
        + options["spousal_income_high_education"] * options["high_education"]
        + options["spousal_income_above_retirement_age"]
        * (age > options["retirement_age"])
    )
