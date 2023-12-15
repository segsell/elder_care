"""Create the parent child data set of females between 50 and 68."""
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from elder_care.config import BLD
from pytask import Product

MIN_AGE = 65
MAX_AGE = 105

HEALTH_EXCELLENT = 1
HEALTH_VERY_GOOD = 2
HEALTH_GOOD = 3
HEALTH_FAIR = 4
HEALTH_POOR = 5


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
    path: Annotated[Path, Product] = BLD / "data" / "parent_child_data.csv",
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
    dat = dat[(dat["age"] >= MIN_AGE) & (dat["age"] <= MAX_AGE)]

    dat = create_care_variables(dat)

    dat = create_health_variables(dat)
    # breakpoint()

    dat.to_csv(path, index=False)


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
    # (Pdb++) table(dat["hc029_"])
    # hc029_
    # -2.0        4
    # -1.0        1
    # 1.0       66
    # 3.0       21
    # 5.0    12254
    # Name: Count, dtype: int64
    # (Pdb++)
    # hc029_
    # -2.0        4
    # -1.0        1
    # 1.0       66
    # 3.0       21
    # 5.0    12254
    # Name: Count, dtype: int64
    # (Pdb++) 66 + 21
    # 87
    # (Pdb++) (66 + 21) / 12254
    # 0.007099722539578913
    # (Pdb++) 0.04 * 0.15
    # 0.006

    # Define age brackets
    # age_brackets = {
    #     "[65, 70)": (65, 70),
    #     "[70, 75)": (70, 75),
    #     "[75, 80)": (75, 80),
    #     "[80, 85)": (80, 85),
    #     "[85, 90)": (85, 90),
    #     "[90 and older]": (90, float("inf")),
    # }

    # # Create an empty DataFrame to store results
    # result_df = pd.DataFrame(columns=["age_bracket", "hc029_"])

    # # Iterate through age brackets and filter data
    # for label, (start_age, end_age) in age_brackets.items():
    #     filtered_data = dat[(dat["age"] >= start_age) & (dat["age"] < end_age)]
    #     filtered_data["age_bracket"] = label
    #     result_df = pd.concat([result_df, filtered_data])

    # # Reset index of the result DataFrame
    # result_df.reset_index(drop=True, inplace=True)

    # Display the result
    # print(result_df[["age_bracket", "hc029_"]])

    # Iterate through columns and set values < 0 to NA
    # for job in job_start:

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
        (dat["hc029_"].isin([1, 3]) | (dat["hc031_"] > 0)),
        dat["hc029_"] == 5,
    ]
    _val = [1, 0]
    dat["nursing_home"] = np.select(_cond, _val, default=np.nan)

    # TODO: RENAME WAVES 1, 2 (3, 4 missing): hc127d1

    # formal home care by professional nursing service
    # _cond = (dat["hc035_"] > 0) | (dat["hc036_"] > 0)
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
        dat["sp021d10"] == 1,
        dat["sp021d10"] == 0,
        (
            dat["wave"].isin([1, 2, 5])
            & (
                dat["sp003_1"].between(10, 19)
                | dat["sp003_2"].between(10, 19)
                | dat["sp003_3"].between(10, 19)
            )
        )
        | (
            (dat["wave"].isin([6, 7, 8]))
            & (dat["sp002_"] == 5)
            & ((dat["sp003_1"] == 10) | (dat["sp003_2"] == 10) | (dat["sp003_3"] == 10))
        ),
        (
            dat["wave"].isin([1, 2, 5])
            & (dat["sp002_"] == 5)
            & ~(
                dat["sp003_1"].between(10, 19)
                | dat["sp003_2"].between(10, 19)
                | dat["sp003_3"].between(10, 19)
            )
        )
        | (
            (dat["wave"].isin([6, 7, 8]))
            & (dat["sp002_"] == 5)
            & ~(
                (dat["sp003_1"] == 10) | (dat["sp003_2"] == 10) | (dat["sp003_3"] == 10)
            )
        ),
    ]
    _val = [1, 0, 1, 0]
    dat["informal_care"] = np.select(_cond, _val, default=np.nan)

    # informal care general
    _cond = [
        (dat["sp021d20"] == 1) | (dat["sp002_"] == 1),
        (dat["sp021d20"].isna()) & (dat["sp002_"].isna()),
    ]
    _val = [1, np.nan]
    dat["informal_care_general"] = np.select(_cond, _val, default=0)

    _cond = [
        # (dat["nursing_home"] == 1)
        (dat["home_care"] == 1) | (dat["informal_care_general"] == 1),
        # (dat["nursing_home"] == 0)
        (dat["home_care"] == 0) & (dat["informal_care_general"] == 0),
    ]
    _val = [1, 0]
    dat["any_care"] = np.select(_cond, _val, default=np.nan)

    _cond = [
        # (dat["nursing_home"] == 0)
        (dat["home_care"] == 1) & (dat["informal_care_general"] == 1),
        # (dat["nursing_home"].isna())
        (dat["home_care"].isna()) & (dat["informal_care_general"].isna()),
    ]
    _val = [1, np.nan]
    dat["combination_care"] = np.select(_cond, _val, default=0)

    _cond = [
        # (dat["nursing_home"] == 0)
        (dat["home_care"] == 0) & (dat["informal_care_general"] == 1),
        (dat["home_care"].isna()) & (dat["informal_care_general"].isna()),
    ]
    _val = [1, np.nan]
    dat["only_informal"] = np.select(_cond, _val, default=0)

    _cond = [
        # ((dat["nursing_home"] == 0) &
        (dat["home_care"] == 1) & (dat["informal_care_general"] == 0),
        # (dat["nursing_home"].isna())
        (dat["home_care"].isna()) & (dat["informal_care_general"].isna()),
    ]
    _val = [1, np.nan]
    dat["only_home_care"] = np.select(_cond, _val, default=0)

    # _cond = [
    #     (dat["nursing_home"] == 1)
    #     & (dat["home_care"] == 0)
    #     & (dat["informal_care_general"] == 0),
    #     (dat["nursing_home"].isna())
    #     & (dat["home_care"].isna())
    #     & (dat["informal_care_general"].isna()),
    # ]
    # _val = [1, np.nan]
    # dat["only_nursing_home"] = np.select(_cond, _val, default=0)

    # _cond = [
    #     (dat["nursing_home"] == 1)
    #     | (dat["home_care"] == 1) & (dat["informal_care_general"] == 1),
    #     (dat["nursing_home"].isna())
    #     & (dat["home_care"].isna())
    #     & (dat["informal_care_general"].isna()),
    # ]
    # _val = [1, np.nan]
    # dat["only_formal"] = np.select(_cond, _val, default=0)

    mean_home_care = dat.loc[dat["any_care"] == 1, "only_home_care"].mean()
    mean_combination_care = dat.loc[dat["any_care"] == 1, "combination_care"].mean()
    mean_informal_care = dat.loc[dat["any_care"] == 1, "only_informal"].mean()
    # mean_formal_care = dat.loc[dat["any_care"] == 1, "only_formal"].mean()
    # mean_nursing_home = dat.loc[dat["any_care"] == 1, "only_nursing_home"].mean()

    # dat["formal_care"] = np.where(dat["formal_care"] > 0, 1, 0)

    return dat


def replace_negative_values_with_nan(dat, col):
    """Replace negative values with NaN."""
    dat[col] = np.where(dat[col] < 0, np.nan, dat[col])
    return dat
