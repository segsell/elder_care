"""Create exogenous transition probabilities."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from elder_care.config import BLD
from elder_care.config import SRC
from statsmodels.stats.outliers_influence import variance_inflation_factor

FEMALE = 2
MALE = 1

MIN_YEAR = 2004
MAX_YEAR = 2017
PARENT_MIN_AGE = 65


RETIREMENT_AGE = 65


def task_create_exog_other_income(
    path_to_raw_data: Path = BLD / "data" / "estimation_data.csv",
) -> None:
    """Fit linear regression model to predict exogenous other income."""
    data = pd.read_csv(path_to_raw_data)

    data["age_squared"] = data["age"] ** 2
    data["age_65_and_older"] = np.where(data["age"] >= RETIREMENT_AGE, 1, 0)

    _dat = data[
        [
            "other_income",
            "age",
            "age_squared",
            "age_65_and_older",
            "married",
            "high_educ",
        ]
    ]
    dat = _dat.dropna()

    regressors = dat[["age", "age_65_and_older", "married", "high_educ"]]
    regressors = sm.add_constant(regressors)

    dat.loc[dat["other_income"] <= 0, "other_income"] = np.finfo(float).eps
    y_log = np.log(dat["other_income"])

    model = sm.OLS(y_log, regressors).fit()

    vif_data = pd.DataFrame()
    vif_data["feature"] = regressors.columns
    vif_data["VIF"] = [
        variance_inflation_factor(regressors.values, i)
        for i in range(regressors.shape[1])
    ]

    return model.params


def task_create_exog_care_demand(
    path_to_raw_data: Path = BLD / "data" / "parent_child_data.csv",
) -> None:
    """Create exogenous care demand probabilities."""
    data = pd.read_csv(path_to_raw_data)
    data = data.copy()

    data["age_squared"] = data["age"] ** 2
    x_with_nans = sm.add_constant(data[["age", "age_squared", "lagged_any_care"]])
    x = x_with_nans.dropna()
    data = data.dropna(subset=["age", "age_squared", "lagged_any_care"])

    x_single_male = x[(data["any_care"].notna()) & (data["gender"] == MALE)]
    y_single_male = data["any_care"][
        (data["any_care"].notna()) & (data["gender"] == MALE)
    ]
    x_single_male = x_single_male.reset_index(drop=True)
    y_single_male = y_single_male.reset_index(drop=True)

    x_single_female = x[(data["any_care"].notna()) & (data["gender"] == FEMALE)]
    y_single_female = data["any_care"][
        (data["any_care"].notna()) & (data["gender"] == FEMALE)
    ]
    x_single_female = x_single_female.reset_index(drop=True)
    y_single_female = y_single_female.reset_index(drop=True)

    x_couple = x[(data["any_care"].notna()) & (data["gender"].notna())]
    y_couple = data["any_care"][(data["any_care"].notna()) & (data["gender"].notna())]
    x_couple = x_couple.reset_index(drop=True)
    y_couple = y_couple.reset_index(drop=True)

    # regress dummy for any care on age and age squared
    # distance to parents?
    # any care in previous period

    logit_single_father = sm.Logit(y_single_male, x_single_male).fit()
    logit_single_mother = sm.Logit(y_single_female, x_single_female).fit()
    logit_couple = sm.Logit(y_couple, x_couple).fit()
    # care demand is zero if no parent is alive

    return logit_single_father.params, logit_single_mother.params, logit_couple.params


def task_create_survival_probabilities(
    path_to_raw_data: Path = SRC
    / "data"
    / "statistical_office"
    / "12621-0001_Sterbetafel_clean.csv",
) -> None:
    """Create exogenous survival probabilities for parents."""
    data = pd.read_csv(path_to_raw_data)

    # Filter data for the years 2004 to 2017 and age 65 and older
    data_filtered = data[
        (data["year"] >= MIN_YEAR)
        & (data["year"] <= MAX_YEAR)
        & (data["age"] >= PARENT_MIN_AGE)
    ]

    # Prepare the independent variables
    data_filtered = data_filtered.copy()
    data_filtered["age_squared"] = data_filtered["age"] ** 2
    x = sm.add_constant(data_filtered[["age", "age_squared"]])

    # Separate data for males and females within the specified years and age
    x_male = x[data_filtered["male_survival_probability"].notna()]
    y_male = data_filtered["male_survival_probability"][
        data_filtered["male_survival_probability"].notna()
    ]

    x_female = x[data_filtered["female_survival_probability"].notna()]
    y_female = data_filtered["female_survival_probability"][
        data_filtered["female_survival_probability"].notna()
    ]

    logit_male = sm.Logit(y_male, x_male).fit()
    logit_female = sm.Logit(y_female, x_female).fit()

    coefs_male = logit_male.params
    coefs_female = logit_female.params

    return coefs_male, coefs_female


def predict_other_income(age, married, high_educ, params):
    """Predict other income based on log-lin regression."""
    log_other_income = (
        params["other_income_const"]
        + params["other_income_age"] * age
        + params["other_income_age_65_and_older"] * (age >= RETIREMENT_AGE)
        + params["other_income_married"] * married
        + params["other_income_high_educ"] * high_educ
    )

    return np.exp(log_other_income)


def predict_care_demand(
    age,
    sex,
    lagged_care,
    coefs_single_father,
    coefs_single_mother,
    coefs_couple,
):
    """Predicts the survival probability based on logit parameters.

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
    if sex.lower() == "male":
        coefs = coefs_single_father
    elif sex.lower() == "female":
        coefs = coefs_single_mother
    else:
        coefs = coefs_couple

    # Logit prediction
    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2) + coefs[3] * lagged_care
    return 1 / (1 + np.exp(-logit))


def predict_survival_probability(age, sex):
    """Predicts the survival probability based on logit parameters.

    Parameters:
        age (int): The age of the individual. Age >= 65.
        sex (str): The gender of the individual ('male' or 'female').

    Returns:
        float: Predicted binary survival probability.

    """
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

    # Logit prediction
    logit = coefs[0] + coefs[1] * age + coefs[2] * (age**2)
    return 1 / (1 + np.exp(-logit))


def probability_full_time_offer(age, high_educ, lagged_choice, params):
    """Compute logit probability of full time offer."""
    logit = (
        params["full_time_constant"]
        + params["full_time_not_working_last_period"] * is_not_working(lagged_choice)
        + params["full_time_working_part_time_last_period"]
        * is_part_time(lagged_choice)
        + params["full_time_above_retirement_age"] * (age >= RETIREMENT_AGE)
        + params["full_time_high_education"] * high_educ
    )

    _prob = np.exp(logit) / (1 + np.exp(logit))
    prob = 1 / (1 + np.exp(-logit))

    return prob, _prob


def probability_part_time_offer(age, high_educ, lagged_choice, params):
    """Compute logit probability of part time offer."""
    logit = (
        params["part_time_constant"]
        + params["part_time_not_working_last_period"] * is_not_working(lagged_choice)
        + params["part_time_working_part_time_last_period"]
        * is_part_time(lagged_choice)
        + params["part_time_above_retirement_age"] * (age >= RETIREMENT_AGE)
        + params["part_time_high_education"] * high_educ
    )

    return 1 / (1 + np.exp(-logit))


def is_not_working(lagged_choice):
    return lagged_choice in (0, 1, 2, 3, 4, 5)


def is_part_time(lagged_choice):
    return lagged_choice in (6, 7, 8, 9, 10, 11)


def is_full_time(lagged_choice):
    return lagged_choice in (12, 13, 14, 15, 16, 17)


def is_formal_care(lagged_choice):
    return lagged_choice % 2 == 1


def is_informal_care(lagged_choice):
    return lagged_choice in (2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23)
