"""Create exogenous transition from SOEP data."""

FEMALE = 2
MALE = 1

MIN_YEAR = 2004
MAX_YEAR = 2017
PARENT_MIN_AGE = 65


GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


RETIREMENT_AGE = 65


def task_create_exog_part_time_offer():
    """Create exogenous transition probabilities for part-time offer.

    If the individual was working part-time last period, the probability of receiving a
    part-time offer this period is assumed to be 1.

    """
    return {
        "part_time_constant": -2.568584,
        "part_time_not_working_last_period": 0.3201395,
        "part_time_high_education": 0.1691369,
        "part_time_above_retirement_age": -1.9976496,
    }


def task_create_exog_full_time_offer():
    """Create exogenous transition probabilities for full-time offer.

    If the individual was working full-time last period, the probability of receiving a
    full-time offer this period is assumed to be 1.

    """
    return {
        "full_time_constant": -2.445238,
        "full_time_not_working_last_period": -0.9964007,
        "full_time_high_education": 0.3019138,
        "full_time_above_retirement_age": -2.6571659,
    }


def task_create_exog_wage():
    """Create exogenous log wage offer."""
    return {
        "wage_constant": 1.997354,
        "wage_age": 0.0124007328,
        "wage_age_squared": -0.0001872683,
        "wage_experience": 0.0272860557,
        "wage_experience_squared": -0.0003839528,
        "wage_high_education": 0.4582799360,
        "wage_part_time": -0.0561647814,
    }


def task_create_non_labor_income():
    """Create exogenous non-labor income."""
    return {
        "non_labor_income_constant": 6.347577,
        "non_labor_income_age": -0.0544903182,
        "non_labor_income_age_squared": 0.0009502824,
        "non_labor_income_high_education": 0.8474811304,
        "non_labor_income_married": -0.0410315531,
        "non_labor_income_mabove_retirement_age": 0.0409211520,
    }


def task_create_spousal_income():
    """Create exogenous spousal income.

    Only available if the individual has a spouse or registered partner.

    """
    return {
        "spousal_income_constant": 9.036516,
        "spousal_income_age": 0.0385433771,
        "spousal_income_age_squared": -0.0003486319,
        "spousal_income_high_education": 0.2487519577,
        "spousal_income_above_retirement_age": -0.0157620788,
    }
