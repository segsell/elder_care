"""Create the estimation data set of females between 50 and 68."""
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from elder_care.config import BLD
from pytask import Product


FEMALE = 2

REFUSAL = -2
DONT_KNOW = -1
ANSWER_YES = 1
ANSWER_NO = 5

MIN_AGE = 55
MAX_AGE = 68

MIN_YEARS_SCHOOLING = 0
MAX_YEARS_SCHOOLING = 25
HIGH_EDUC_YEARS_SCHOOLING = 15
HIGH_EDUC_INDICATOR_WAVE_FOUR = 3
HIGH_EDUC_INDICATOR_WAVE_FIVE = 10
HOCHSCHUL_DEGREE = 5

MOTHER = 2
FATHER = 3
GIVEN_HELP_LESS_THAN_DAILY = 2
GIVEN_HELP_DAILY = 1

HEALTH_EXCELLENT = 1
HEALTH_VERY_GOOD = 2
HEALTH_GOOD = 3
HEALTH_FAIR = 4
HEALTH_POOR = 5

WORKING_PART_TIME_THRESH = 10
WORKING_FULL_TIME_THRESH = 32
EMPLOYED_OR_SELF_EMPLOYED = 2

# Distance to parents
SAME_HOUSEHOLD = 1
SAME_BUILDING = 2
LESS_THAN_1_KM_AWAY = 3
BETWEEN_1_AND_5_KM_AWAY = 4
BETWEEN_5_AND_25_KM_AWAY = 5
BETWEEN_25_AND_100_KM_AWAY = 6
BETWEEN_100_AND_500_KM_AWAY = 7
MORE_THAN_500_KM_AWAY = 8
MORE_THAN_500_KM_AWAY_IN_ANOTHER_COUNTRY = 9

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

    dat = create_married(dat)

    dat = create_caregving(dat)

    dat = create_parental_health_status(dat, parent="mother")
    dat = create_parental_health_status(dat, parent="father")

    dat = create_age_parent_and_parent_alive(dat, parent="mother")
    dat = create_age_parent_and_parent_alive(dat, parent="father")
    # !!! Replace mother_alive = 1 if health status >= 0

    dat = create_parents_live_close(dat)

    dat = create_working(dat)

    dat = create_retired(dat)

    dat = create_years_since_retirement(dat)

    dat.to_csv(path, index=False)


# =====================================================================================


def create_years_since_retirement(dat):
    dat["_years_since_retirement"] = dat["int_year"] - dat["ep329_"]

    return dat


def create_retired(dat: pd.DataFrame) -> pd.DataFrame:
    _cond = [
        (dat["ep005_"] == 1)
        | (~dat["ep329_"].isna() & (dat["ep329_"] <= dat["int_year"]))
        | (dat["ypen1"] > 0),  # or >= 200??
        dat["ep005_"].isna() & dat["ep329_"].isna(),
    ]
    _val = [1, np.nan]

    dat["retired"] = np.select(_cond, _val, 0)

    return dat


def create_parents_live_close(dat):
    dat["dist_mother"] = np.where(dat["dn030_1"] >= 0, dat["dn030_1"], np.nan)
    dat["dist_father"] = np.where(dat["dn030_2"] >= 0, dat["dn030_2"], np.nan)

    conditions_distance = [
        (dat["dist_father"] <= BETWEEN_5_AND_25_KM_AWAY)
        | (dat["dist_mother"] <= BETWEEN_5_AND_25_KM_AWAY),
        (dat["dist_father"] > BETWEEN_5_AND_25_KM_AWAY)
        | (dat["dist_mother"] > BETWEEN_5_AND_25_KM_AWAY),
    ]
    choices_distance = [1, 0]

    dat["parents_live_close"] = np.select(
        conditions_distance,
        choices_distance,
        default=np.nan,
    )

    return dat


def create_parental_health_status(dat, parent):
    """Aggregate health status of parents from 5 into 3 levels."""
    # this is just a dummy comment
    if parent == "mother":
        parent_indicator = 1
    elif parent == "father":
        parent_indicator = 2
    _cond = [
        (dat[f"dn033_{parent_indicator}"] == HEALTH_EXCELLENT)
        | (dat[f"dn033_{parent_indicator}"] == HEALTH_VERY_GOOD),
        (dat[f"dn033_{parent_indicator}"] == HEALTH_GOOD)
        | (dat[f"dn033_{parent_indicator}"] == HEALTH_FAIR),
        (dat[f"dn033_{parent_indicator}"] == HEALTH_POOR),
    ]
    _val = [0, 1, 2]

    dat[f"{parent}_health"] = np.select(_cond, _val, default=np.nan)

    return dat


def create_age_parent_and_parent_alive(dat, parent):
    """Create age and alive variables for parents."""
    if parent == "mother":
        parent_indicator = 1
    elif parent == "father":
        parent_indicator = 2

    dat = dat.sort_values(by=["mergeid", "int_year"])
    dat[f"{parent}_age"] = dat[f"dn028_{parent_indicator}"].copy()

    dat[f"lagged_{parent}_age"] = dat.groupby("mergeid")[f"{parent}_age"].shift(1)
    # Get the first non-NaN value of '{parent}_age'
    dat[f"{parent}_age_first"] = dat.groupby("mergeid")[f"{parent}_age"].transform(
        "first",
    )
    dat[f"{parent}_first_int_year"] = dat.groupby("mergeid")["int_year"].transform(
        "first",
    )

    _cond = [
        (dat[f"dn026_{parent_indicator}"] == ANSWER_YES),
        (dat[f"dn026_{parent_indicator}"] == ANSWER_NO),
    ]
    _val = [1, 0]
    dat[f"{parent}_alive"] = np.select(_cond, _val, default=np.nan)

    dat[f"{parent}_dead"] = np.where(
        dat[f"{parent}_age"].isna() & (dat[f"lagged_{parent}_age"] > 0),
        1,
        np.nan,
    )

    dat[f"lagged_{parent}_alive"] = dat.groupby("mergeid")[f"{parent}_alive"].shift(1)

    _cond = [(dat[f"lagged_{parent}_alive"] == 0), (dat[f"lagged_{parent}_alive"] == 1)]
    _val = [1, 0]
    dat[f"{parent}_dead_since_last"] = np.select(_cond, _val, np.nan)

    # dn027_{parent_indicator}: age of death of {parent}
    # Determine the most common non-empty value in 'dn027_1' for each 'mergeid'
    _grouped = dat.groupby("mergeid")
    dat[f"dn027_{parent_indicator}"] = np.where(
        dat[f"dn027_{parent_indicator}"] < 0,
        np.nan,
        dat[f"dn027_{parent_indicator}"],
    )
    _age_at_death = _grouped[f"dn027_{parent_indicator}"].apply(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan,
    )
    dat[f"{parent}_age_at_death"] = dat["mergeid"].map(_age_at_death)

    # Fill any remaining NaN values with np.nan
    dat[f"{parent}_age_at_death"] = np.where(
        dat[f"{parent}_age_at_death"] < 0,
        np.nan,
        dat[f"{parent}_age_at_death"],
    )

    dat = _impute_missing_values_parent_alive(dat, parent)
    dat = _impute_missing_values_parent_age(dat, parent)

    # If health status is not missing, impute missing values for age and alive
    dat[f"{parent}_alive"] = np.where(
        dat[f"{parent}_health"].notna() & dat[f"{parent}_alive"].isna(),
        1,
        dat[f"{parent}_alive"],
    )

    _cond = [
        (dat[f"{parent}_health"].notna() & dat[f"{parent}_age"].isna()),
        (dat[f"{parent}_health"].notna() & dat[f"{parent}_age"].isna()),
        (dat[f"{parent}_health"].notna() & dat[f"{parent}_age"].isna()),
    ]
    _val = [
        dat[f"{parent}_age_imputed_raw"],
        dat[f"{parent}_age_imputed"],
        dat[f"{parent}_age_imputed_two"],
    ]
    dat[f"{parent}_age"] = np.select(_cond, _val, default=dat[f"{parent}_age"])

    return dat


def _impute_missing_values_parent_age(dat, parent):
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Identify the rows where '{parent}_alive' switches from 1 to 0
    # within each 'mergeid'
    parent_passes_away = (dat[f"{parent}_alive"] == 1) & (
        dat[f"{parent}_alive"].shift(-1) == 0
    )

    # Calculate '{parent}_birth_year' based on 'int_year' and f"{parent}_age_at_death"
    # for the switching year
    dat.loc[parent_passes_away, f"{parent}_birth_year"] = (
        dat["int_year"] - dat[f"{parent}_age_at_death"]
    )
    dat[f"{parent}_birth_year"] = (
        dat.groupby("mergeid")[f"{parent}_birth_year"].ffill().bfill()
    )

    dat[f"{parent}_age_imputed"] = np.where(
        dat[f"{parent}_alive"] == 1,
        dat["int_year"] - dat[f"{parent}_birth_year"],
        np.nan,
    )
    dat[f"{parent}_age"] = dat[f"{parent}_age"].fillna(dat[f"{parent}_age_imputed"])

    # Alternative
    dat[f"{parent}_birth_year_two"] = np.where(
        dat[f"{parent}_alive"] == 1,
        dat[f"{parent}_first_int_year"] - dat[f"{parent}_age_first"],
        np.nan,
    )

    dat[f"{parent}_age_imputed_two"] = dat["int_year"] - dat[f"{parent}_birth_year_two"]

    dat[f"{parent}_age"] = dat[f"{parent}_age"].fillna(dat[f"{parent}_age_imputed_two"])
    # does not change anything

    # # Careful: Parent might be dead!
    # dat[f"{parent}_age"] = np.where(

    return dat


def _impute_missing_values_parent_alive(dat, parent):
    dat = dat.sort_values(by=["mergeid", "int_year"])

    dat[f"{parent}_age_imputed_raw"] = (
        dat["int_year"]
        - dat[f"{parent}_first_int_year"]
        + dat.groupby("mergeid")[f"{parent}_age_first"].transform("first")
    )

    # Identify the first and next observation in the panel for each 'mergeid'
    _first_observation = (
        dat.groupby("mergeid")["int_year"].transform("first") == dat["int_year"]
    )
    _next_observation = (
        dat.groupby("mergeid")["int_year"].transform("first") == dat["int_year"] + 1
    )

    # Filter for rows where '{parent}_alive' is NaN and the conditions are met
    _nan_parent_dead_before = (
        dat[f"{parent}_alive"].isna() & _first_observation & _next_observation
    )

    # Replace '{parent}_alive' with 1 for the specified rows
    dat.loc[
        _nan_parent_dead_before & (dat[f"{parent}_alive"].shift(1) == 1),
        f"{parent}_alive",
    ] = 1
    dat.loc[
        _nan_parent_dead_before & (dat[f"{parent}_alive"].shift(1) == 0),
        f"{parent}_alive",
    ] = 0

    _parent_dead = (
        (
            (
                dat[f"{parent}_age_imputed_raw"].notna()
                & dat[f"{parent}_age_at_death"].notna()
            )
            & (dat[f"{parent}_age_imputed_raw"] > dat[f"{parent}_age_at_death"])
        )
        .groupby(dat["mergeid"])
        .idxmax()
    )
    _parent_dead_next = (dat[f"{parent}_alive"] != 1).groupby(dat["mergeid"]).shift(-1)

    dat.loc[
        (dat[f"{parent}_alive"].isna())
        & (dat.index.isin(_parent_dead))
        & _parent_dead_next,
        f"{parent}_alive",
    ] = 0

    _nan_but_parent_dead_before = dat.groupby("mergeid").apply(
        lambda group: group[f"{parent}_alive"].isna()
        & (group[f"{parent}_alive"].shift(1) == 0),
    )
    dat.loc[_nan_but_parent_dead_before.to_numpy(), f"{parent}_alive"] = 0

    _parent_dead_next = dat.groupby("mergeid")[f"{parent}_alive"].shift(-1) == 1
    dat.loc[
        (dat[f"{parent}_alive"].isna()) & _parent_dead_next,
        f"{parent}_alive",
    ] = 1

    return dat


def create_caregving(dat):
    """Create caregiving variables and care experience."""
    cols = ["sp008_", "sp009_1", "sp009_2", "sp009_3"]
    dat[cols] = np.where(dat[cols] >= 0, dat[cols], np.nan)

    conditions_care = [
        (dat["sp008_"] == ANSWER_YES) | (dat["sp018_"] == ANSWER_YES),
        ((dat["sp008_"] == ANSWER_NO) & (dat["sp018_"] == ANSWER_NO))
        | ((dat["sp008_"] == ANSWER_NO) & dat["sp018_"].isna())
        | (dat["sp008_"].isna() & (dat["sp018_"] == ANSWER_NO)),
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


def create_working(dat):
    """Create full and part time work."""
    # Current job situation
    # -2 Refusal
    # -1 Don't know
    # 1 Retired
    # 2 Employed or self-employed (including working for family business)
    # 3 Unemployed
    # 4 Permanently sick or disabled
    # 5 Homemaker
    # 97 Other

    _cond = [
        (dat["cjs"] == EMPLOYED_OR_SELF_EMPLOYED),
        (dat["cjs"] < 0),
        (dat["cjs"] < 0) & (dat["empstat"] >= 1),
    ]
    _val = [1, np.nan, 1]
    dat["working"] = np.select(_cond, _val, default=0)

    _cond = [
        (dat["cjs"] == EMPLOYED_OR_SELF_EMPLOYED),
        (dat["pwork"] == 1),
        (dat["cjs"] < 0) & (dat["pwork"] != 1),
    ]
    _val = [1, 1, np.nan]
    dat["working_new"] = np.select(_cond, _val, default=0)

    dat["_full_time"] = np.where(dat["ep013_"] > WORKING_FULL_TIME_THRESH, 1, 0)
    dat["full_time"] = np.where(
        (dat["working"] == 1) & (dat["ep013_"] > WORKING_FULL_TIME_THRESH),
        1,
        0,
    )
    dat["full_time_new"] = np.where(
        (dat["working_new"] == 1) & (dat["ep013_"] > WORKING_FULL_TIME_THRESH),
        1,
        0,
    )

    dat["_part_time"] = np.where(
        (dat["ep013_"] >= WORKING_PART_TIME_THRESH)
        & (dat["ep013_"] <= WORKING_FULL_TIME_THRESH),
        1,
        0,
    )
    dat["part_time"] = np.where(
        (dat["working"] == 1)
        & (dat["ep013_"] >= WORKING_PART_TIME_THRESH)
        & (dat["ep013_"] <= WORKING_FULL_TIME_THRESH),
        1,
        0,
    )
    dat["part_time_new"] = np.where(
        (dat["working_new"] == 1)
        & (dat["ep013_"] >= WORKING_PART_TIME_THRESH)
        & (dat["ep013_"] <= WORKING_FULL_TIME_THRESH),
        1,
        0,
    )

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
