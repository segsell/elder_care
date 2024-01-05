"""Create the parent child data set of people older than 65."""
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from elder_care.config import BLD
from pytask import Product


WAVE_1 = 1
WAVE_2 = 2
WAVE_3 = 3
WAVE_4 = 4
WAVE_5 = 5
WAVE_6 = 6
WAVE_7 = 7
WAVE_8 = 8

MIN_AGE = 68
MAX_AGE = 105

HEALTH_EXCELLENT = 1
HEALTH_VERY_GOOD = 2
HEALTH_GOOD = 3
HEALTH_FAIR = 4
HEALTH_POOR = 5

RECEIVED_HELP_DAILY = 1


CHILD_ONE_GAVE_HELP = 10
STEP_CHILD_GAVE_HELP = 11
OTHER_CHILD_GAVE_HELP = 19

NO_HELP_FROM_OTHERS_OUTSIDE_HOUSEHOLD = 5
ANSWER_NO = 5
YES_TEMPORARILY = 1
YES_PERMANENTLY = 3


def table(df_col):
    """Return frequency table."""
    return pd.crosstab(df_col, columns="Count")["Count"]


def describe(df_col):
    """Return descriptive statistics."""
    return df_col.describe()


def count(df_col):
    """Count the number of non-missing observations."""
    return df_col.count()


def task_create_parent_child_data(
    path_to_raw_data: Path = BLD / "data" / "data_parent_child_merged.csv",
    path_to_main: Annotated[Path, Product] = BLD / "data" / "parent_child_data.csv",
    path_to_design_weight: Annotated[Path, Product] = BLD
    / "data"
    / "parent_child_data_design_weight.csv",
    path_to_hh_weight: Annotated[Path, Product] = BLD
    / "data"
    / "parent_child_data_hh_weight.csv",
    path_to_ind_weight: Annotated[Path, Product] = BLD
    / "data"
    / "parent_child_data_ind_weight.csv",
) -> None:
    """Create the estimation data set."""
    dat = pd.read_csv(path_to_raw_data)

    # Make prettier
    dat["age"] = dat.apply(
        lambda row: row["int_year"] - row["yrbirth"]
        if row["int_month"] >= row["mobirth"]
        else row["int_year"] - row["yrbirth"] - 1,
        axis=1,
    )

    # Keep only those aged 65 and older
    dat = dat[(dat["age"] > MIN_AGE) & (dat["age"] <= MAX_AGE)]

    dat = create_married_or_partner_alive(dat)

    dat = create_care_variables(dat)

    # dat = create_care_combinations(dat, informal_care_var="informal_care_general")
    dat = create_care_combinations(dat, informal_care_var="informal_care_child")

    dat = create_health_variables(dat)

    dat.reset_index(drop=True, inplace=True)

    dat_design_weight = multiply_rows_with_weight(dat, weight="design_weight")
    dat_hh_weight = multiply_rows_with_weight(dat, weight="hh_weight")
    dat_ind_weight = multiply_rows_with_weight(dat, weight="ind_weight")

    # Save
    dat.to_csv(path_to_main, index=False)
    dat_design_weight.to_csv(path_to_design_weight, index=False)
    dat_hh_weight.to_csv(path_to_hh_weight, index=False)
    dat_ind_weight.to_csv(path_to_ind_weight, index=False)


def multiply_rows_with_weight(dat, weight):
    # Create a DataFrame of weights with the same shape as dat
    weights = dat[weight].values.reshape(-1, 1)

    static_cols = [
        "mergeid",
        "gender",
        "int_year",
        "int_month",
        "age",
        "only_informal",
        "combination_care",
        "only_home_care",
        "informal_care_child",
        "informal_care_general",
        "home_care",
        "no_informal_care_child",
        "no_home_care",
        "lagged_home_care",
        "lagged_informal_care_general",
        "lagged_informal_care_child",
        "lagged_combination_care",
        "lagged_no_informal_care_child",
        "lagged_no_home_care",
        "health",
        "married",
        "wave",
        weight,
    ]
    data_columns = dat.drop(columns=static_cols).values

    result = data_columns * weights

    dat_weighted = pd.DataFrame(
        result,
        columns=[col for col in dat.columns if col not in static_cols],
    )
    dat_weighted.insert(0, "mergeid", dat["mergeid"])
    dat_weighted.insert(1, "int_year", dat["int_year"])
    dat_weighted.insert(2, "int_month", dat["int_month"])
    dat_weighted.insert(3, "age", dat["age"])
    dat_weighted.insert(4, weight, dat[weight])
    dat_weighted.insert(5, "only_informal", dat["only_informal"])
    dat_weighted.insert(6, "combination_care", dat["combination_care"])
    dat_weighted.insert(7, "only_home_care", dat["only_home_care"])
    dat_weighted.insert(8, "informal_care_child", dat["informal_care_child"])
    dat_weighted.insert(9, "informal_care_general", dat["informal_care_general"])
    dat_weighted.insert(10, "home_care", dat["home_care"])
    dat_weighted.insert(11, "health", dat["health"])
    dat_weighted.insert(12, "married", dat["married"])
    dat_weighted.insert(13, "wave", dat["wave"])
    dat_weighted.insert(14, "lagged_home_care", dat["lagged_home_care"])
    dat_weighted.insert(
        15,
        "lagged_informal_care_general",
        dat["lagged_informal_care_general"],
    )
    dat_weighted.insert(
        16,
        "lagged_informal_care_child",
        dat["lagged_informal_care_child"],
    )
    dat_weighted.insert(
        17,
        "lagged_combination_care",
        dat["lagged_combination_care"],
    )
    dat_weighted.insert(18, "gender", dat["gender"])
    dat_weighted.insert(
        19, "lagged_no_informal_care_child", dat["lagged_no_informal_care_child"]
    )
    dat_weighted.insert(20, "lagged_no_home_care", dat["lagged_no_home_care"])
    dat_weighted.insert(21, "no_informal_care_child", dat["no_informal_care_child"])
    dat_weighted.insert(22, "no_home_care", dat["no_home_care"])

    dat_weighted[f"{weight}_avg"] = dat_weighted.groupby("mergeid")[weight].transform(
        "mean",
    )

    return dat_weighted


def create_health_variables(dat):
    """Create dummy for health status."""
    dat = replace_negative_values_with_nan(dat, "ph003_")

    _cond = [
        (dat["ph003_"] == HEALTH_EXCELLENT) | (dat["ph003_"] == HEALTH_VERY_GOOD),
        (dat["ph003_"] == HEALTH_GOOD) | (dat["ph003_"] == HEALTH_FAIR),
        (dat["ph003_"] == HEALTH_POOR),
    ]
    _val = [0, 1, 2]

    dat["health"] = np.select(_cond, _val, default=np.nan)

    return dat


def create_care_variables(dat):
    """Create a dummy for formal care."""
    dat = replace_negative_values_with_nan(dat, "hc029_")  # was in nursing home
    dat = replace_negative_values_with_nan(dat, "hc031_")  # Weeks in nursing home
    dat = replace_negative_values_with_nan(dat, "hc032d1")
    dat = replace_negative_values_with_nan(dat, "hc032d2")
    dat = replace_negative_values_with_nan(dat, "hc032d3")
    dat = replace_negative_values_with_nan(dat, "hc032dno")
    dat = replace_negative_values_with_nan(dat, "hc033_")
    dat = replace_negative_values_with_nan(dat, "hc034_")
    dat = replace_negative_values_with_nan(dat, "hc035_")
    dat = replace_negative_values_with_nan(dat, "hc036_")
    dat = replace_negative_values_with_nan(dat, "hc127d1")
    dat = replace_negative_values_with_nan(dat, "hc127d2")
    dat = replace_negative_values_with_nan(dat, "hc127d3")
    dat = replace_negative_values_with_nan(dat, "hc127d4")
    dat = replace_negative_values_with_nan(dat, "hc127dno")

    # nursing home
    _cond = [
        (dat["hc029_"].isin([YES_TEMPORARILY, YES_PERMANENTLY]) | (dat["hc031_"] > 0)),
        dat["hc029_"] == ANSWER_NO,
    ]
    _val = [1, 0]
    dat["nursing_home"] = np.select(_cond, _val, default=np.nan)

    # RENAME WAVES 1, 2 (3, 4 missing): hc127d1

    # formal home care by professional nursing service
    _cond = [
        (dat["hc032d1"] == 1)
        | (dat["hc032d2"] == 1)
        | (dat["hc032d3"] == 1)
        | (dat["hc032dno"] == 0)
        | (dat["hc127d1"] == 1)
        | (dat["hc127d2"] == 1)
        | (dat["hc127d3"] == 1)
        | (dat["hc127d4"] == 1),
        # | (dat["hc127dno"] == 0),
        (
            dat["hc032d1"].isna()
            & dat["hc032d2"].isna()
            & dat["hc032d3"].isna()
            & dat["hc032dno"].isna()
            & dat["hc127d1"].isna()
            & dat["hc127d2"].isna()
            & dat["hc127d3"].isna()
            & dat["hc127d4"].isna()
        ),
        # & (dat["hc127dno"] == 1),
    ]
    _val = [1, np.nan]
    dat["home_care"] = np.select(_cond, _val, default=0)

    dat = replace_negative_values_with_nan(dat, "sp020_")
    dat = replace_negative_values_with_nan(dat, "sp021d10")
    dat = replace_negative_values_with_nan(dat, "sp021d11")
    dat = replace_negative_values_with_nan(dat, "sp021d20")
    dat = replace_negative_values_with_nan(dat, "sp021d21")
    dat = replace_negative_values_with_nan(dat, "sp002_")
    dat = replace_negative_values_with_nan(dat, "sp003_1")
    dat = replace_negative_values_with_nan(dat, "sp003_2")
    dat = replace_negative_values_with_nan(dat, "sp003_3")

    # informal care by own children
    _cond = [
        dat["sp021d10"] == 1,  # help within household from own children
        # help outside the household from own children
        (
            dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
            & (
                (
                    dat["sp003_1"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
                    # & (dat["sp005_1"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_2"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
                    # & (dat["sp005_2"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_3"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
                    # & (dat["sp005_1"] == RECEIVED_HELP_DAILY)
                )
            )
        )
        | (
            (dat["wave"].isin([WAVE_6, WAVE_7, WAVE_8]))
            & (dat["sp002_"] == NO_HELP_FROM_OTHERS_OUTSIDE_HOUSEHOLD)
            & (
                (
                    dat["sp003_1"]
                    == CHILD_ONE_GAVE_HELP
                    # & (dat["sp005_1"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_2"]
                    == CHILD_ONE_GAVE_HELP
                    # & (dat["sp005_2"] == RECEIVED_HELP_DAILY)
                )
                | (
                    dat["sp003_3"]
                    == CHILD_ONE_GAVE_HELP
                    # & (dat["sp005_3"] == RECEIVED_HELP_DAILY)
                )
            )
        ),
        ((dat["sp021d10"] == 0) | (dat["sp021d10"].isna()))
        & (
            dat["wave"].isin([WAVE_1, WAVE_2, WAVE_5])
            & (dat["sp002_"] == NO_HELP_FROM_OTHERS_OUTSIDE_HOUSEHOLD)
            #     & ~(
            #         dat["sp003_1"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
            #         | dat["sp003_2"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
            #         | dat["sp003_3"].between(CHILD_ONE_GAVE_HELP, OTHER_CHILD_GAVE_HELP)
            #     )
            # )
            | (
                (dat["wave"].isin([WAVE_6, WAVE_7, WAVE_8]))
                & (dat["sp002_"] == NO_HELP_FROM_OTHERS_OUTSIDE_HOUSEHOLD)
                #     & ~(
                #         (dat["sp003_1"] == CHILD_ONE_GAVE_HELP)
                #         | (dat["sp003_2"] == CHILD_ONE_GAVE_HELP)
                #         | (dat["sp003_3"] == CHILD_ONE_GAVE_HELP)
            )
        ),
    ]
    _val = [1, 1, 0]
    dat["informal_care_child"] = np.select(_cond, _val, default=np.nan)

    # informal care general
    _cond = [
        (dat["sp021d20"] == 1) | (dat["sp002_"] == 1),
        (dat["sp021d20"].isna()) & (dat["sp002_"].isna()),
    ]
    _val = [1, np.nan]
    dat["informal_care_general"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat["informal_care_general"] == 1),
        (dat["home_care"].isna()) & (dat["informal_care_general"].isna()),
    ]
    _val = [1, np.nan]
    dat["combination_care"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) | (dat["informal_care_general"] == 1),
        (dat["home_care"] == 0) & (dat["informal_care_general"] == 0),
    ]
    _val = [1, 0]
    dat["any_care"] = np.select(_cond, _val, default=np.nan)

    # lagged care
    dat = dat.sort_values(by=["mergeid", "int_year"], ascending=[True, True])

    _cond = [dat["informal_care_child"] == 1, dat["informal_care_child"] == 0]
    _val = [0, 1]
    dat["no_informal_care_child"] = np.select(_cond, _val, default=np.nan)

    _cond = [dat["home_care"] == 1, dat["home_care"] == 0]
    _val = [0, 1]
    dat["no_home_care"] = np.select(_cond, _val, default=np.nan)

    dat = _create_lagged_var(dat, "home_care")
    dat = _create_lagged_var(dat, "informal_care_general")
    dat = _create_lagged_var(dat, "informal_care_child")
    dat = _create_lagged_var(dat, "combination_care")
    dat = _create_lagged_var(dat, "any_care")
    dat = _create_lagged_var(dat, "no_informal_care_child")
    dat = _create_lagged_var(dat, "no_home_care")

    return dat


def create_care_combinations(dat, informal_care_var):
    _cond = [
        (dat["home_care"] == 0) & (dat[informal_care_var] == 1),
        (dat["home_care"].isna()) & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["only_informal"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["home_care"] == 1) & (dat[informal_care_var] == 0),
        (dat["home_care"].isna()) & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["only_home_care"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["nursing_home"] == 1)
        & (dat["home_care"] == 0)
        & (dat[informal_care_var] == 0),
        (dat["nursing_home"].isna())
        & (dat["home_care"].isna())
        & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["only_nursing_home"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["nursing_home"] == 1)
        | (dat["home_care"] == 1) & (dat[informal_care_var] == 1),
        (dat["nursing_home"].isna())
        & (dat["home_care"].isna())
        & (dat[informal_care_var].isna()),
    ]
    _val = [1, np.nan]
    dat["only_formal"] = np.select(_cond, _val, default=0)

    return dat


def replace_negative_values_with_nan(dat, col):
    """Replace negative values with NaN."""
    dat[col] = np.where(dat[col] < 0, np.nan, dat[col])
    return dat


def create_means(dat):
    mean_home_care = dat.loc[dat["any_care"] == 1, "only_home_care"].mean()
    mean_combination_care = dat.loc[dat["any_care"] == 1, "combination_care"].mean()
    mean_informal_care = dat.loc[dat["any_care"] == 1, "only_informal"].mean()
    mean_formal_care = dat.loc[dat["any_care"] == 1, "only_formal"].mean()
    mean_nursing_home = dat.loc[dat["any_care"] == 1, "only_nursing_home"].mean()

    return (
        mean_home_care,
        mean_combination_care,
        mean_informal_care,
        mean_formal_care,
        mean_nursing_home,
    )


def create_married_or_partner_alive(dat):
    """Create married variable."""
    # We use marriage information in SHARE to construct an indicator on the
    # existence of a partner living in the same household.
    # We do not distinguish between marriage and registered partnership.
    # dn014_
    # Widowed

    conditions_married_or_partner = [
        dat["mstat"].isin([1, 2]),
        dat["mstat"].isin([3, 4, 5, 6]),
    ]
    values_married_or_partner = [1, 0]
    # replace with zeros or nans
    dat["married"] = np.select(
        conditions_married_or_partner,
        values_married_or_partner,
        np.nan,
    )

    return dat


def _create_lagged_var(dat, var):
    """Create lagged variable by mergeid."""
    dat[f"lagged_{var}"] = dat.groupby("mergeid")[var].shift(1)
    return dat
