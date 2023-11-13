"""Create the estimation data set of females between 50 and 68."""
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from elder_care.config import BLD
from pytask import Product

FEMALE = 2

MIN_AGE = 55
MAX_AGE = 68

MIN_YEARS_SCHOOLING = 0
MAX_YEARS_SCHOOLING = 25
HIGH_EDUC_YEARS_SCHOOLING = 15
HIGH_EDUC_INDICATOR_WAVE_FOUR = 3
HIGH_EDUC_INDICATOR_WAVE_FIVE = 10

FURTHER_EDUC = [
    "dn012d1",
    "dn012d2",
    "dn012d3",
    "dn012d4",
    "dn012d5",
    "dn012d6",
    "dn012d7",
    "dn012d8",
    "dn012d9",
    "dn012d10",
    "dn012d11",
    "dn012d12",
    "dn012d13",
    "dn012d14",
    "dn012d15",
    "dn012d16",
    "dn012d17",
    "dn012d18",
    "dn012d19",
    "dn012d20",
    #'dn012d95' # currently in education --> not needed
]

HOCHSCHUL_DEGREE = 5

GIVEN_HELP = 1
GIVEN_NO_HELP = 5
MOTHER = 2
FATHER = 3
GIVEN_HELP_LESS_THAN_DAILY = 2
GIVEN_HELP_DAILY = 1


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_create_estimation_data(
    path: Annotated[Path, Product] = BLD / "data" / "estimation_data.csv",
) -> None:
    """Create the estimation data set."""
    # nurs#Out-of-pocket payment for nursing home / home care

    # Load the data
    dat = pd.read_csv(BLD / "data" / "data_merged.csv")

    # Filter for females
    dat = dat[dat["gender"] == FEMALE]

    # number of siblings alive

    # Set negative values to missing
    dat["dn036_"] = np.where(dat["dn036_"] < 0, np.nan, dat["dn036_"])
    dat["dn037_"] = np.where(dat["dn037_"] < 0, np.nan, dat["dn037_"])

    dat["siblings"] = np.select(
        [
            (~dat["dn036_"].isna())
            & (~dat["dn037_"].isna()),  # Both columns are not NaN
            (~dat["dn036_"].isna()) & dat["dn037_"].isna(),  # Only dn036_ is not NaN
            dat["dn036_"].isna() & (~dat["dn037_"].isna()),  # Only dn037_ is not NaN
            (dat["dn036_"].isna()) & (dat["dn037_"].isna()),  # Both columns are NaN
        ],
        [
            dat["dn036_"] + dat["dn037_"],  # Addition when both columns are not NaN
            dat["dn036_"],  # Value from dn036_ when only dn036_ is not NaN
            dat["dn037_"],  # Value from dn037_ when only dn037_ is not NaN
            np.nan,  # Result is NaN when both columns are NaN
        ],
        default=np.nan,
    )

    # Make prettier
    dat["age"] = dat.apply(
        lambda row: row["int_year"] - row["yrbirth"]
        if row["int_month"] >= row["mobirth"]
        else row["int_year"] - row["yrbirth"] - 1,
        axis=1,
    )

    # Keep only those aged 55 to 68
    dat = dat[(dat["age"] >= MIN_AGE) & (dat["age"] <= MAX_AGE)]

    # !!! Still not 0.35 share high educ... rahter 0.25
    dat = create_high_educ(dat)

    # number of children
    dat = create_number_of_children(dat)

    # current job situation

    # retired

    # married
    dat = create_married(dat)

    dat = create_caregving(dat)

    # !!! Replace mother_alive = 1 if health status >= 0
    dat.to_csv(path, index=False)


# =====================================================================================


def create_caregving(dat):
    """Create caregiving variables and care experience."""
    cols = ["sp008_", "sp009_1", "sp009_2", "sp009_3"]
    dat[cols] = np.where(dat[cols] >= 0, dat[cols], np.nan)

    conditions_care = [
        (dat["sp008_"] == GIVEN_HELP) | (dat["sp018_"] == GIVEN_HELP),
        ((dat["sp008_"] == GIVEN_NO_HELP) & (dat["sp018_"] == GIVEN_NO_HELP))
        | ((dat["sp008_"] == GIVEN_NO_HELP) & dat["sp018_"].isna())
        | (dat["sp008_"].isna() & (dat["sp018_"] == GIVEN_NO_HELP)),
    ]
    choices_care = [1, 0]

    dat["care"] = np.select(conditions_care, choices_care, default=np.nan)

    conditions_parents_outside = [
        (dat["sp008_"] == 1)
        & (
            (dat["sp009_1"].isin([2, 3]))  # to whom
            | (dat["sp009_2"].isin([2, 3]))  # to whom
            | (dat["sp009_3"].isin([2, 3]))  # to whom
        ),
        dat["sp008_"].isna(),
    ]
    choices_parents_outside = [1, np.nan]

    dat["care_parents_outside"] = np.select(
        conditions_parents_outside,
        choices_parents_outside,
        default=0,
    )

    conditions_parents_within = [
        (dat["sp018_"] == 1) & ((dat["sp019d2"] == 1) | (dat["sp019d3"] == 1)),
        dat["sp018_"].isna(),
    ]
    choices_parents_within = [1, np.nan]

    dat["care_parents_within"] = np.select(
        conditions_parents_within,
        choices_parents_within,
        default=0,
    )

    # Create the 'ever_cared_parents' column
    conditions_parents = [
        (dat["care_parents_outside"] == 1) | (dat["care_parents_within"] == 1),
        (dat["care_parents_within"].isna()) & (dat["care_parents_outside"].isna()),
    ]
    choices_parents = [1, np.nan]

    dat["care_parents"] = np.select(conditions_parents, choices_parents, default=0)

    conditions = [
        # personal care in hh
        (dat["sp018_"] == 1) & ((dat["sp019d2"] == 1) | (dat["sp019d3"] == 1)),
        # care outside hh to mother
        (dat["sp008_"] == 1)
        & (
            (dat["sp009_1"] == MOTHER)
            | (dat["sp009_2"] == MOTHER)
            | (dat["sp009_3"] == MOTHER)
        ),
        # care outside hh to father
        (dat["sp008_"] == 1)
        & (
            (dat["sp009_1"] == FATHER)
            | (dat["sp009_2"] == FATHER)
            | (dat["sp009_3"] == FATHER)
        ),
    ]

    choices = [1, 1, 1]  # Assign 1 if the conditions are met

    # Use np.select to create the 'care_in_year' column
    dat["care_in_year"] = np.select(conditions, choices, default=0)

    # care_parents and care_in_year identical except more zeros in care_in_year
    # because default is 0, not nan
    # --> take care parents?

    # sp011_1: how often inside
    condition = [
        (
            (
                (dat["sp011_1"] >= GIVEN_HELP_LESS_THAN_DAILY)
                & (dat["sp009_1"].isin([2, 3]))
            )
            | (
                (dat["sp011_2"] >= GIVEN_HELP_LESS_THAN_DAILY)
                & (dat["sp009_2"].isin([2, 3]))
            )
            | (
                (dat["sp011_3"] >= GIVEN_HELP_LESS_THAN_DAILY)
                & (dat["sp009_3"].isin([2, 3]))
            )
        )
        & (dat["sp018_"] != 1)
        & ((dat["sp019d2"] != 1) & (dat["sp019d3"] != 1)),  # to whom in hh
        # no personal care in hh (e.g. to partner, want those excluded)
    ]
    choice = [1]  # Assign 1 if the conditions are met

    dat["light_care"] = np.select(condition, choice, default=0)

    conditions = [
        (
            (
                (dat["sp011_1"] == GIVEN_HELP_DAILY)
                & (dat["sp009_1"].isin([MOTHER, FATHER]))
            )
            | (
                (dat["sp011_2"] == GIVEN_HELP_DAILY)
                & (dat["sp009_2"].isin([MOTHER, FATHER]))
            )
            | (
                (dat["sp011_3"] == GIVEN_HELP_DAILY)
                & (dat["sp009_3"].isin([MOTHER, FATHER]))
            )
        )
        | (
            (dat["sp018_"] == 1)  # or personal care in hh
            & ((dat["sp019d2"] == 1) | (dat["sp019d3"] == 1))
        ),  # include mother and father in law?
        # & (dat["sp018_"].isna()),
    ]
    choices = [1]
    dat["intensive_care"] = np.select(conditions, choices, default=0)

    dat["light_care"] = np.where(
        (dat["intensive_care"] == 1) & (dat["light_care"] == 1),
        0,
        dat["light_care"],
    )

    dat["care"] = np.where(
        (dat["intensive_care"] == 1) | (dat["light_care"] == 1),
        1,
        0,
    )

    dat["care"] = np.where(
        (dat["care_parents"] == 1)
        & (dat["intensive_care"] == 0)
        & (dat["light_care"] == 0),
        np.nan,
        dat["care"],
    )

    # care experience
    # Calculate cumulative sum for 'care_in_year' within each 'mergeid' group
    dat = dat.sort_values(by=["mergeid", "int_year"], ascending=[True, True])
    # dat["care_experience"] = (
    #     .cumsum()
    #     .where(dat["care_parents"] >= 0, np.nan)
    dat["care_experience"] = (
        dat.groupby(["mergeid", "int_year"])["care"]
        .cumsum()
        .where(dat["care"] >= 0, np.nan)
    )

    # dat["care_experience"] = (
    #     .apply(lambda group: group.interpolate(method="linear",
    # limit_direction="both"))
    #     .fillna(0)
    #     .astype(int)

    return dat


def create_married(dat):
    """Create married variable."""
    # Partner We use marriage information in SHARE to construct an indicator on the
    # existence of a partner living in the same household.
    # We do not distinguish between marriage and registered partnership.

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


def create_number_of_children(dat):
    """Create number of children variable."""
    dat = dat.rename(columns={"ch001_": "nchild"})
    dat["nchild"] = np.where(dat["nchild"] >= 0, dat["nchild"], np.nan)
    return dat


def create_retired(dat: pd.DataFrame) -> pd.DataFrame:
    """Define retirement status."""
    # Current job situation
    # -2 Refusal
    # -1 Don't know
    # 1 Retired
    # 2 Employed or self-employed (including working for family business)
    # 3 Unemployed
    # 4 Permanently sick or disabled
    # 5 Homemaker
    # 97 Other

    return dat


def create_high_educ(dat: pd.DataFrame) -> pd.DataFrame:
    dat["years_educ"] = dat["yedu"].copy()

    conditions = [
        (dat["years_educ"] < MIN_YEARS_SCHOOLING),
        (dat["years_educ"] > MAX_YEARS_SCHOOLING),
    ]
    values = [np.nan, np.nan]

    # Use numpy.select to set values in the 'years_educ' column based on conditions
    dat["years_educ"] = np.select(conditions, values, dat["years_educ"])

    # Create 'high_educ' column, setting NaN when 'years_educ' is NaN
    dat["high_educ"] = np.where(
        dat["years_educ"].isna(),
        np.nan,
        (dat["years_educ"] >= HIGH_EDUC_YEARS_SCHOOLING).astype(int),
    )

    for educ in FURTHER_EDUC:
        number = int(re.search(r"\d+", educ).group())
        conditions = [dat[educ] < 0, dat[educ] == number]
        values = [np.nan, 1]

        dat[educ] = np.select(conditions, values, dat[educ])

    dat["dn012dno"] = np.where(dat["dn012dno"] < 0, np.nan, dat["dn012dno"])
    dat["dn012dot"] = np.where(dat["dn012dot"] < 0, np.nan, dat["dn012dot"])
    dat["dn012dno"] = np.where(dat["dn012dno"] == 1, 0, dat["dn012dno"])
    dat["further_educ_max"] = dat.apply(find_max_suffix, axis=1)

    dat["high_educ_012"] = (
        (
            dat["wave"].isin([1, 2, 4])
            & (dat["further_educ_max"] >= HIGH_EDUC_INDICATOR_WAVE_FOUR)
        )
        | (
            dat["wave"].between(5, 8)
            & (dat["further_educ_max"] >= HIGH_EDUC_INDICATOR_WAVE_FIVE)
        )
    ).astype(int)

    # Stufe 0: Kindergarten
    # Stufe 1: Grundschule
    # Stufe 2A: Realschule, Mittlere Reife und Polytechnische Oberschule (DDR)
    # Stufe 2B: Volks- und Hauptschule und Anlernausbildung
    # Stufe 3A: Fachhochschulreife und Abitur
    # Stufe 3B: Berufsausbildung im dualen System, mittlere Verwaltungsausbildung,
    # Berufsfach-/Kollegschulabschluss und einjährige Schulen des Gesundheitswesens
    # Stufe 4A: Abschluss von 3A UND 3B
    # Stufe 5A: Fachhochschulabschluss
    # Stufe 5A: Universitätsabschluss
    # Stufe 5B: Meister/Techniker, 2- bis 3-jährige Schule des Gesundheitswesens,
    # Fach-/Berufsakademie, Fachschulabschluss (DDR) und Verwaltungsfachschule
    # Stufe 6: Promotion

    dat["high_educ_comb"] = (
        (dat["high_educ"] == 1)
        # | (dat["high_educ_012"] == 1)
        | (dat["isced"] >= HOCHSCHUL_DEGREE)
    ).astype(int)

    conditions = [(dat["isced"] >= HOCHSCHUL_DEGREE), (dat["isced"] < HOCHSCHUL_DEGREE)]
    values = [1, 0]
    dat["high_isced"] = np.select(conditions, values, default=np.nan)

    return dat


def find_max_suffix(row):
    active_cols = [
        int(col.split("dn012d")[-1]) for col in FURTHER_EDUC if row[col] == 1
    ]
    max_suffix = max(active_cols) if active_cols else 0
    return max_suffix if max_suffix >= 0 else np.nan


def calculate_retired(row):
    """Define whether the person is retired."""
    if row["ep005_"] == 1 or (not pd.isna(row["ep329_"])):
        out = 1
    elif pd.isna(row["ep005_"]) and pd.isna(row["ep329_"]):
        out = np.nan
    else:
        out = 0

    return out


def calculate_years_since_retirement(row):
    """Calculate the years since retirement."""
    if row["retired"] == 1 or (not pd.isna(row["ep329_"])):
        out = row["int_year"] - row["ep329_"]
    elif row["retired"] == 0:
        out = 0
    else:
        out = np.nan

    return out


def _find_max_suffix(row):
    max_suffix = 0
    for col in FURTHER_EDUC:
        if row[col] == 1:
            suffix = int(col.split("dn012d")[-1])
            max_suffix = max(max_suffix, suffix)

    return max_suffix if max_suffix >= 0 else np.nan
