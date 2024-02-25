import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytask
from dcegm.numerical_integration import quadrature_legendre
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_function

from elder_care.config import BLD
from elder_care.model import (
    budget_constraint,
    calc_stochastic_wage,
    get_state_specific_feasible_choice_set,
    inverse_marginal_utility,
    marginal_utility,
    marginal_utility_final_consume_all,
    prob_exog_care_demand_basic,
    prob_full_time_offer,
    prob_part_time_offer,
    prob_survival_father,
    prob_survival_mother,
    update_endog_state,
    utility_final_consume_all,
    utility_func,
)
from elder_care.simulate import (
    draw_initial_states,
    get_share_by_age,
    get_share_by_type,
    get_share_by_type_by_age_bin,
    get_transition,
    simulate_moments,
)

MIN_AGE = 51
MAX_AGE = 80

PARENT_MIN_AGE = 68
PARENT_MAX_AGE = 98

RETIREMENT_AGE = 62

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2

MIN_AGE = 51
MAX_AGE = 65

AGE_50 = 50 - MIN_AGE
AGE_53 = 53 - MIN_AGE
AGE_56 = 56 - MIN_AGE
AGE_59 = 59 - MIN_AGE
AGE_62 = 62 - MIN_AGE

AGE_55 = 55 - MIN_AGE
AGE_60 = 60 - MIN_AGE
AGE_65 = 65 - MIN_AGE


AGE_BINS = [(AGE_50, AGE_55), (AGE_55, AGE_60), (AGE_60, AGE_65)]

ALL = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

NO_WORK = jnp.array([0, 1, 2, 3])
PART_TIME = jnp.array([4, 5, 6, 7])
FULL_TIME = jnp.array([8, 9, 10, 11])
WORK = jnp.concatenate([PART_TIME, FULL_TIME])

NO_CARE = jnp.array([0, 4, 8])
FORMAL_CARE = jnp.array([1, 3, 5, 7, 9, 11])
INFORMAL_CARE = jnp.array([2, 3, 6, 7, 10, 11])
CARE = jnp.concatenate([FORMAL_CARE, INFORMAL_CARE])

COMBINATION_CARE = jnp.array([3, 7, 11])

# For NO_INFORMAL_CARE and NO_FORMAL_CARE, we need to perform set operations before converting to JAX arrays.
# This is because JAX doesn't support direct set operations.
# Convert the results of set operations to lists, then to JAX arrays.
NO_INFORMAL_CARE = jnp.array(list(set(ALL.tolist()) - set(INFORMAL_CARE.tolist())))
NO_FORMAL_CARE = jnp.array(list(set(ALL.tolist()) - set(FORMAL_CARE.tolist())))


TOTAL_WEEKLY_HOURS = 80
WEEKLY_HOURS_PART_TIME = 20
WEEKLY_HOURS_FULL_TIME = 40
WEEKLY_INTENSIVE_INFORMAL_HOURS = 14  # (21 + 7) / 2

N_MONTHS = 12
N_WEEKS = 4.33

PART_TIME_HOURS = 20 * N_WEEKS * N_MONTHS
FULL_TIME_HOURS = 40 * N_WEEKS * N_MONTHS


def create_exog_mapping(exog_state_space, exog_names):
    def exog_mapping(exog_proc_state):
        # Caution: JAX does not throw an error if the exog_proc_state is out of bounds
        # If the index is out of bounds, the last element of the array is returned.
        exog_state = jnp.take(exog_state_space, exog_proc_state, axis=0)
        exog_state_dict = {
            key: jnp.take(exog_state, i) for i, key in enumerate(exog_names)
        }
        return exog_state_dict

    return exog_mapping


def is_not_working(lagged_choice):
    return jnp.any(lagged_choice == NO_WORK)


def is_part_time(lagged_choice):
    return jnp.any(lagged_choice == PART_TIME)


def is_full_time(lagged_choice):
    return jnp.any(lagged_choice == FULL_TIME)


def is_informal_care(lagged_choice):
    # intensive only here
    return jnp.any(lagged_choice == INFORMAL_CARE)


def is_no_informal_care(lagged_choice):
    # intensive only here
    return jnp.all(lagged_choice != INFORMAL_CARE)


def is_formal_care(lagged_choice):
    return jnp.any(lagged_choice == FORMAL_CARE)


def is_no_formal_care(lagged_choice):
    return jnp.all(lagged_choice != FORMAL_CARE)


def create_simulation_array(sim_dict, options, params, n_agents):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    # Convert the dictionary to arrays
    agent = jnp.tile(jnp.arange(n_agents), n_periods)
    period = sim_dict["period"].ravel()
    savings = sim_dict["savings"].ravel()
    consumption = sim_dict["consumption"].ravel()
    lagged_choice = sim_dict["lagged_choice"].ravel()
    choice = sim_dict["choice"].ravel()
    income_shock = sim_dict["income_shock"].ravel()

    # Compute additional variables
    wealth = savings + consumption
    # savings_rate = savings / wealth
    savings_rate = jnp.where(wealth > 0, jnp.divide(savings, wealth), 0)
    period_indices = jnp.tile(jnp.arange(n_periods)[:, None], (1, n_agents)).ravel()
    age = period_indices + options["model_params"]["min_age"]

    # Adjusting the logic for PART_TIME and FULL_TIME checks
    is_part_time = jnp.isin(lagged_choice, PART_TIME)
    is_full_time = jnp.isin(lagged_choice, FULL_TIME)

    log_wage = (
        params["wage_constant"]
        + params["wage_age"] * age
        + params["wage_age_squared"] * age**2
        + params["wage_part_time"] * is_part_time
        + params["wage_not_working"] * is_full_time
    )

    wage = jnp.exp(log_wage + income_shock)

    # Adapt _assign_working_hours to work with vectorized operations in JAX
    # Example stub for _assign_working_hours function
    # Vectorize the _assign_working_hours function if it's not already suitable for vector operations
    working_hours = jax.vmap(_assign_working_hours_vectorized)(lagged_choice)

    income = working_hours * wage

    result = jnp.column_stack(
        (
            agent,
            period,
            lagged_choice,
            wealth,
            savings_rate,
            wage,
            working_hours,
            income,
            choice,
        ),
    )

    # Return the results as a dictionary or a structured array
    # result = sim_dict | {
    #    "wealth": wealth,
    #    "age": age,
    #    "log_wage": log_wage,
    #    "wage": wage,
    #    "working_hours": working_hours,
    #    "income": income
    # }
    return result


def _assign_working_hours_vectorized(choices):
    # Create boolean masks for each condition
    no_work_mask = jnp.isin(choices, NO_WORK)
    part_time_mask = jnp.isin(choices, PART_TIME)
    full_time_mask = jnp.isin(choices, FULL_TIME)

    # Use where to vectorize the conditional assignment of hours
    hours = jnp.where(
        no_work_mask,
        0,
        jnp.where(part_time_mask, 20, jnp.where(full_time_mask, 40, jnp.nan)),
    )

    return hours


def simulate_moments(arr, idx):
    """Df has multiindex ["period", "agent"] necessary?

    column_indices = {col: idx for idx, col in enumerate(sim.columns)} idx =
    column_indices.copy() arr = jnp.asarray(sim)

    """
    # share working by age
    share_not_working_by_age = get_share_by_age(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
    )  # 15
    share_working_part_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
    )  # 15
    share_working_full_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
    )  # 15
    # do timeit of jnp.array versus jnp.asarray?

    share_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        care_type=ALL,
        lagged_choice=INFORMAL_CARE,
    )

    # yearly net income
    # Caution: Some bug here!! Zero income, althouh working (full-time),
    # so people have to live off of their initial wealth
    income_part_time_by_age_bin = get_mean_by_age_bin_for_lagged_choice(
        arr,
        ind=idx,
        var="income",
        lagged_choice=PART_TIME,
    )
    income_full_time_by_age_bin = get_mean_by_age_bin_for_lagged_choice(
        arr,
        ind=idx,
        var="income",
        lagged_choice=FULL_TIME,
    )
    breakpoint()

    # savings rate
    # savings_rate_no_informal_care_by_age_bin = get_savings_rate_by_age_bin(
    #     arr,
    #     ind=idx,
    #     care_type=NO_INFORMAL_CARE,
    # )

    # savings_rate_informal_care_by_age_bin = get_savings_rate_by_age_bin(
    #     arr,
    #     ind=idx,
    #     care_type=INFORMAL_CARE,
    # )

    # share working by caregiving type (and age bin) --> to be checked

    #
    share_not_working_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        care_type=NO_INFORMAL_CARE,
    )
    share_part_time_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        care_type=NO_INFORMAL_CARE,
    )
    share_full_time_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        care_type=NO_INFORMAL_CARE,
    )

    share_not_working_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        care_type=INFORMAL_CARE,
    )
    share_part_time_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        care_type=INFORMAL_CARE,
    )
    share_full_time_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        care_type=INFORMAL_CARE,
    )

    # work transitions
    no_work_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        current_choice=NO_WORK,
    )
    no_work_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        current_choice=PART_TIME,
    )
    no_work_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        current_choice=FULL_TIME,
    )

    part_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=NO_WORK,
    )
    part_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=PART_TIME,
    )
    part_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=FULL_TIME,
    )

    full_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=NO_WORK,
    )
    full_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=PART_TIME,
    )
    full_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=FULL_TIME,
    )

    # caregiving transitions
    no_informal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    no_informal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    informal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    informal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    no_informal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    no_informal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    informal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    informal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    no_formal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    no_formal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    formal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    formal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    no_formal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    no_formal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    formal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    formal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    return jnp.asarray(
        income_part_time_by_age_bin
        + income_full_time_by_age_bin
        + share_not_working_by_age
        + share_working_part_time_by_age
        + share_working_full_time_by_age
        +
        #
        share_informal_care_by_age_bin
        #
        # income_part_time_by_age_bin
        # + income_full_time_by_age_bin
        # + savings_rate_no_informal_care_by_age_bin
        # + savings_rate_informal_care_by_age_bin
        #
        + [share_not_working_no_informal_care]
        + [share_part_time_no_informal_care]
        + [share_full_time_no_informal_care]
        + [share_not_working_informal_care]
        + [share_part_time_informal_care]
        + [share_full_time_informal_care]
        +
        #
        no_work_to_no_work
        + no_work_to_part_time
        + no_work_to_full_time
        + part_time_to_no_work
        + part_time_to_part_time
        + part_time_to_full_time
        + full_time_to_no_work
        + full_time_to_part_time
        + full_time_to_full_time
        +
        # caregiving transitions
        no_informal_care_to_no_informal_care
        + no_informal_care_to_informal_care
        + informal_care_to_no_informal_care
        + informal_care_to_informal_care
        + no_informal_care_to_no_formal_care
        + no_informal_care_to_formal_care
        + informal_care_to_no_formal_care
        + informal_care_to_formal_care
        +
        #
        no_formal_care_to_no_informal_care
        + no_formal_care_to_informal_care
        + formal_care_to_no_informal_care
        + formal_care_to_informal_care
        + no_formal_care_to_no_formal_care
        + no_formal_care_to_formal_care
        + formal_care_to_no_formal_care
        + formal_care_to_formal_care,
    )


model_params = {
    "quadrature_points_stochastic": 5,
    "n_choices": 12,
    "min_age": MIN_AGE,
    "max_age": MAX_AGE,
    "mother_min_age": PARENT_MIN_AGE,
    "father_min_age": PARENT_MIN_AGE,
    # annual
    "consumption_floor": 400 * 12,
    "unemployment_benefits": 500 * 12,
    "informal_care_benefits": 444.0466
    * 12,  # 0.4239 * 316 + 0.2793 * 545 + 728 *0.1405 + 901 * 0.0617
    "formal_care_costs": 118.10658099999999
    * 12,  # >>> 79.31 * 0.0944 + 0.4239 * 70.77 + 0.2793 * 176.16 + 224.26 *0.1401
    "interest_rate": 0.04,  # Adda et al (2017)
    # ===================
    # EXOGENOUS PROCESSES
    # ===================
    # survival probability
    "survival_probability_mother_constant": 17.01934835131644,
    "survival_probability_mother_age": -0.21245937682111807,
    "survival_probability_mother_age_squared": 0.00047537366767865137,
    "survival_probability_father_constant": 11.561515476144223,
    "survival_probability_father_age": -0.11058331994203506,
    "survival_probability_father_age_squared": -1.0998977981246952e-05,
    # health
    "mother_medium_health": {
        "medium_health_age": 0.0304,
        "medium_health_age_squared": -1.31e-05,
        "medium_health_lagged_good_health": -1.155,
        "medium_health_lagged_medium_health": 0.736,
        "medium_health_lagged_bad_health": 1.434,
        "medium_health_constant": -1.550,
    },
    "mother_bad_health": {
        "bad_health_age": 0.196,
        "bad_health_age_squared": -0.000885,
        "bad_health_lagged_good_health": -2.558,
        "bad_health_lagged_medium_health": -0.109,
        "bad_health_lagged_bad_health": 2.663,
        "bad_health_constant": -9.220,
    },
    "father_medium_health": {
        "medium_health_age": 0.176,
        "medium_health_age_squared": -0.000968,
        "medium_health_lagged_good_health": -1.047,
        "medium_health_lagged_medium_health": 1.016,
        "medium_health_lagged_bad_health": 1.743,
        "medium_health_constant": -7.374,
    },
    "father_bad_health": {
        "bad_health_age": 0.260,
        "bad_health_age_squared": -0.00134,
        "bad_health_lagged_good_health": -2.472,
        "bad_health_lagged_medium_health": 0.115,
        "bad_health_lagged_bad_health": 3.067,
        "bad_health_constant": -11.89,
    },
    # TODO: care demand
    # "exog_care_single_mother_constant": 27.894895,
    # "exog_care_single_mother_age": -0.815882,
    # "exog_care_single_mother_age_squared": 0.005773,
    # "exog_care_single_mother_medium_health": 0.652438,
    # "exog_care_single_mother_bad_health": 0.924265,
    #
    # "exog_care_single_father_constant": 17.833432,
    # "exog_care_single_father_age": -0.580729,
    # "exog_care_single_father_age_squared": 0.004380,
    # "exog_care_single_father_medium_health": 0.594160,
    # "exog_care_single_father_bad_health": 0.967142,
    #
    # "exog_care_couple_constant": 32.519891,
    # "exog_care_couple_mother_age": -0.916759,
    # "exog_care_couple_mother_age_squared": 0.006190,
    # "exog_care_couple_father_age": -0.046230,
    # "exog_care_couple_father_age_squared": 0.000583,
    # "exog_care_couple_mother_medium_health": 0.449386,
    # "exog_care_couple_mother_bad_health": 0.719621,
    # "exog_care_couple_father_medium_health": 0.360010,
    # "exog_care_couple_father_bad_health": 0.800824,
    #
    # TODO: care demand
    "exog_care_single_mother_constant": 22.322551,
    "exog_care_single_mother_age": -0.661611,
    "exog_care_single_mother_age_squared": 0.004840,
    #
    "exog_care_single_father_constant": 16.950484,
    "exog_care_single_father_age": -0.541042,
    "exog_care_single_father_age_squared": 0.004136,
    #
    "exog_care_couple_constant": 22.518664,
    "exog_care_couple_mother_age": -0.622648,
    "exog_care_couple_mother_age_squared": 0.004346,
    "exog_care_couple_father_age": -0.068347,
    "exog_care_couple_father_age_squared": 0.000769,
    #
}


options = {
    "state_space": {
        "n_periods": 20,
        "n_choices": 12,
        "choices": np.arange(12),
        "endogenous_states": {
            "married": np.arange(2),
            "has_sibling": np.arange(2),
        },
        "exogenous_processes": {
            "part_time_offer": {
                "states": np.arange(2),
                "transition": prob_part_time_offer,
            },
            "full_time_offer": {
                "states": np.arange(2),
                "transition": prob_full_time_offer,
            },
            "care_demand": {
                "states": np.arange(2),
                "transition": prob_exog_care_demand_basic,
            },
            "mother_alive": {
                "states": np.arange(2),
                "transition": prob_survival_mother,
            },
            "father_alive": {
                "states": np.arange(2),
                "transition": prob_survival_father,
            },
            # "mother_health": {
            #    "states": np.arange(3),
            #    "transition": exog_health_transition_mother,
            # },
            # "father_health": {
            #    "states": np.arange(3),
            #    "transition": exog_health_transition_father,
            # },
        },
    },
}

params_test = {
    # job offer
    "part_time_constant": -0.8,
    "part_time_not_working_last_period": -1.576,
    "part_time_working_full_time_last_period": 0.3,
    "part_time_above_retirement_age": 0.6,
    "full_time_constant": -0.3,
    "full_time_not_working_last_period": -2,
    "full_time_working_part_time_last_period": 0.5,
    "full_time_above_retirement_age": -1.75,
    # wage
    "wage_constant": 0.32,
    "wage_age": 0.05,
    "wage_age_squared": -0.0006,
    "wage_part_time": -0.1,
    "wage_not_working": -0.3,
    # utility
    "rho": 0.5,  # risk aversion
    "utility_leisure_constant": 3,
    "utility_leisure_age": 0.36,
    "disutility_part_time": -0.5,
    "disutility_full_time": -1,
    "utility_informal_care": 2,
    "utility_formal_care": 2,
    "utility_informal_and_formal_care": -1,
    ### fixed
    "beta": 0.95,  # 0.98
    "lambda": 1e-16,  # Taste shock scale/variance. Almost equal zero = no taste shocks
    "sigma": 1,  # Income shock scale/variance.
}


utility_functions = {
    "utility": utility_func,
    "marginal_utility": marginal_utility,
    "inverse_marginal_utility": inverse_marginal_utility,
}

utility_functions_final_period = {
    "utility": utility_final_consume_all,
    "marginal_utility": marginal_utility_final_consume_all,
}

state_space_functions = {
    # "update_endog_state_by_state_and_choice": update_endog_state,
    "get_next_period_state": update_endog_state,
    "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
}


def logspace(start, stop, n_points):
    start_lin = jnp.log(start)
    stop_lin = jnp.log(stop)
    return jnp.logspace(start_lin, stop_lin, n_points, base=2.718281828459045)


start_lin = 0
stop_lin = 1_000_000
n_points = 1_000
exog_savings_grid_one = jnp.arange(start=0, stop=100_000, step=200)
exog_savings_grid_two = jnp.arange(start=100_000, stop=1_000_000, step=10_000)
exog_savings_grid_three = jnp.arange(start=1_000_000, stop=11_000_000, step=1_000_000)

exog_savings_grid = jnp.concatenate(
    [exog_savings_grid_one, exog_savings_grid_two, exog_savings_grid_three],
)

pytask.mark.skip()


@pytask.mark.skip()
def task_debug():
    seed = 2024
    n_choices = 12
    n_agents = 10_000

    options["model_params"] = model_params

    path = BLD / "moments/initial_wealth_at_age_50.csv"
    initial_wealth_empirical = jnp.asarray(pd.read_csv(path)).ravel()

    path = BLD / "moments/initial_discrete_conditions_at_age_50.csv"
    initial_conditions = pd.read_csv(path, index_col=0)

    initial_resources, initial_states = draw_initial_states(
        initial_conditions,
        initial_wealth_empirical,
        n_agents,
        seed=seed,
    )
    for key, value in initial_states.items():
        initial_states[key] = value.astype(np.int32)

    (
        model_funcs,
        compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    solve_func = get_solve_function(
        options=options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
        state_space_functions=state_space_functions,
    )

    params = {
        "part_time_constant": 1.9039744394564848,
        "part_time_not_working_last_period": 0.8140032771945565,
        "part_time_working_full_time_last_period": 1.9986604717837122,
        "part_time_above_retirement_age": -0.10967500332770141,
        "full_time_constant": 2.424830416485194,
        "full_time_not_working_last_period": -2.549310086740609,
        "full_time_working_part_time_last_period": 3.6164929716663443,
        "full_time_above_retirement_age": -3.2105565169713763,
        "wage_constant": 2.6858232962606876,
        "wage_age": 1.074908693723371,
        "wage_age_squared": -3.3287973353563896,
        "wage_part_time": 0.4729857370437544,
        "wage_not_working": 1.272419411564533,
        "utility_leisure_constant": -4.435797729688349,
        "utility_leisure_age": -0.14202541204013341,
        "disutility_part_time": 0.0,
        "disutility_full_time": -2.576739660448464,
        "utility_informal_care": 1.2149265332064965,
        "utility_formal_care": 0.7843696831762427,
        "utility_informal_and_formal_care": -0.7145974649111337,
        "rho": 1.95,
        "beta": 0.95,
        "lambda": 1e-16,
        "sigma": 1.0,
        "interest_rate": 0.04,
    }

    value, policy_left, policy_right, endog_grid = solve_func(params)

    n_periods = options["state_space"]["n_periods"]

    # TODO: Make interface with several draw possibilities.
    # TODO: Some day make user supplied draw function.
    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"],
    )

    (
        model_funcs,
        compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_constraint,
    )

    (
        period_specific_state_objects,
        state_space,
        state_space_names,
        map_state_choice_to_index,
        exog_state_space,
        exog_state_names,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        get_next_period_state=update_endog_state_by_state_and_choice,
    )

    # exog_state_space = exog_state_space.astype(np.int16)
    exog_mapping = create_exog_mapping(exog_state_space, exog_state_names)

    result = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["state_space"]["n_periods"],
        params=params_test,
        #
        state_space_names=state_space_names,
        seed=seed,
        #
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        #
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        # choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int16),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int32),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_mapping,
        get_next_period_state=update_endog_state_by_state_and_choice,
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    np.save(BLD / "moments" / "result.npy", result)


# @pytask.mark.skip()
def task_debug_simulate():
    seed = 2024
    n_choices = 12
    n_agents = 10_000

    options["model_params"] = model_params

    idx = {
        "agent": 0,
        "period": 1,
        "lagged_choice": 2,
        "wealth": 3,
        "savings_rate": 4,
        "wage": 5,
        "working_hours": 6,
        "income": 7,
        "choice": 8,
    }

    result = np.load(BLD / "moments" / "result.npy", allow_pickle="TRUE").item()

    params = {
        "part_time_constant": 1.9039744394564848,
        "part_time_not_working_last_period": 0.8140032771945565,
        "part_time_working_full_time_last_period": 1.9986604717837122,
        "part_time_above_retirement_age": -0.10967500332770141,
        "full_time_constant": 2.424830416485194,
        "full_time_not_working_last_period": -2.549310086740609,
        "full_time_working_part_time_last_period": 3.6164929716663443,
        "full_time_above_retirement_age": -3.2105565169713763,
        "wage_constant": 2.6858232962606876,
        "wage_age": 1.074908693723371 + 0.6,
        "wage_age_squared": -3.3287973353563896 * 0.01,
        "wage_part_time": 0.4729857370437544,
        "wage_not_working": 1.272419411564533,
        "utility_leisure_constant": -4.435797729688349,
        "utility_leisure_age": -0.14202541204013341,
        "disutility_part_time": 0.0,
        "disutility_full_time": -2.576739660448464,
        "utility_informal_care": 1.2149265332064965,
        "utility_formal_care": 0.7843696831762427,
        "utility_informal_and_formal_care": -0.7145974649111337,
        "rho": 1.95,
        "beta": 0.95,
        "lambda": 1e-16,
        "sigma": 1.0,
        "interest_rate": 0.04,
    }

    wage = calc_stochastic_wage(
        period=0, lagged_choice=8, wage_shock=0, min_age=50, params=params,
    )
    breakpoint()

    arr = create_simulation_array(
        result,
        options=options,
        params=params,
        n_agents=n_agents,
    )

    out_arr = simulate_moments(arr, idx)
    breakpoint()
