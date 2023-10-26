"""Create the estimation data set of females between 50 and 78."""
import re

import numpy as np
import pandas as pd

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


def task_create_estimation_data(data):
    # only females
    dat = data.copy()

    # Filter for females
    dat = dat[dat["gender"] == 2]

    dat["age"] = dat.apply(
        lambda row: row["int_year"] - row["yrbirth"]
        if row["int_month"] >= row["mobirth"]
        else row["int_year"] - row["yrbirth"] - 1,
        axis=1,
    )

    # Keep only those aged 55 to 68
    dat = dat[(dat["age"] >= 55) & (dat["age"] <= 68)]

    # Rename 'dn041_' to 'years_educ'
    dat = dat.rename(columns={"dn041_": "years_educ"})

    # Filter rows where 'years_educ' is less than or equal to 25 or is NaN
    dat = dat[(dat["years_educ"] <= 25) | dat["years_educ"].isna()]

    # Replace negative 'years_educ' values with NaN
    dat["years_educ"] = dat["years_educ"].apply(lambda x: np.nan if x < 0 else x)

    # Create 'high_educ' column, setting NaN when 'years_educ' is NaN
    dat["high_educ"] = np.where(
        dat["years_educ"].isna(),
        np.nan,
        (dat["years_educ"] >= 15).astype(int),
    )

    #     "dn012d1",
    #     "dn012d2",
    #     "dn012d3",
    #     "dn012d4",
    #     "dn012d5",
    #     "dn012d6",
    #     "dn012d7",
    #     "dn012d8",
    #     "dn012d9",
    #     "dn012d10",
    #     "dn012d11",
    #     "dn012d12",
    #     "dn012d13",
    #     "dn012d14",
    #     "dn012d15",
    #     "dn012d16",
    #     "dn012d17",
    #     "dn012d18",
    #     "dn012d19",
    #     "dn012d20",
    #     #'dn012d95' # currently in education --> not needed

    # Process each 'further_educ' column
    for educ in FURTHER_EDUC:
        dat[educ] = np.where(dat[educ] < 0, np.nan, dat[educ])
        number = int(re.search(r"\d+", educ).group())
        dat[educ] = np.where(dat[educ] == number, 1, dat[educ])

    # np.select?
    dat["dn012dno"] = np.where(dat["dn012dno"] < 0, np.nan, dat["dn012dno"])
    dat["dn012dot"] = np.where(dat["dn012dot"] < 0, np.nan, dat["dn012dot"])
    dat["dn012dno"] = np.where(dat["dn012dno"] == 1, 0, dat["dn012dno"])

    # Calculate the max for columns starting with 'dn012' for each row
    dat["dn012_max"] = dat.loc[:, dat.columns.str.startswith("dn012")].max(axis=1)

    dat["further_educ_max"] = dat.apply(find_max_suffix, axis=1)

    # NEEDED?!!
    # Find columns that start with 'dn012'
    dn012_columns = [col for col in dat.columns if col.startswith("dn012")]

    # Add a new column 'dn012_max' with the maximum value across 'dn012' columns
    dat["dn012_max"] = dat[dn012_columns].max(axis=1)

    # Replace NaN values with 0
    dat["dn012_max"] = dat["dn012_max"].fillna(0)

    dat["high_educ_012"] = (
        (dat["wave"].isin([1, 2, 4]) & (dat["further_educ_max"] >= 3))
        | (dat["wave"].between(5, 7) & (dat["further_educ_max"] >= 10))
    ).astype(int)

    dat.loc[dat["further_educ_max"].isna(), "high_educ_012"] = None

    # Create a new column "high_educ_comb" based on conditions
    dat["high_educ_comb"] = (
        (dat["high_educ"] == 1) | (dat["high_educ_012"] == 1)
    ).astype(int)

    dat = dat.rename(columns={"ch001_": "nchild"})
    dat["nchild"] = dat["nchild"].apply(lambda x: x if x >= 0 else np.nan)

    dat["ep005_"] = np.where(dat["ep005_"] >= 0, dat["ep005_"], np.nan)

    ### retired

    dat["retired"] = dat.apply(calculate_retired, axis=1)

    dat["ep329_"] = np.where(dat["ep329_"] >= 0, dat["ep329_"], np.nan)
    dat["ep328_"] = np.where(dat["ep328_"] >= 0, dat["ep328_"], np.nan)

    dat["years_since_retirement"] = dat.apply(calculate_years_since_retirement, axis=1)

    dat["married"] = dat["dn014_"].apply(
        lambda x: 1 if x in (1, 3) else (0 if x in (2, 4, 5, 6) else np.nan),
    )

    dat["in_partnership"] = dat["dn014_"].apply(
        lambda x: 1 if x in (1, 2) else (0 if x in (3, 4, 5, 6) else np.nan),
    )

    conditions = [
        (dat["married"] == 1) | (dat["in_partnership"] == 1),
        (dat["married"].isna()) & (dat["in_partnership"].isna()),
    ]

    choices = [1, np.nan]

    dat["has_partner"] = np.select(conditions, choices, default=0)

    ### new

    # Update 'sp008_' to handle negative values
    dat["sp008_"] = dat["sp008_"].apply(lambda x: x if x >= 0 else np.nan)

    # Update 'sp009_1', 'sp009_2', and 'sp009_3' to handle negative values
    columns_to_update = ["sp009_1", "sp009_2", "sp009_3"]
    for col in columns_to_update:
        dat[col] = dat[col].apply(lambda x: x if x >= 0 else np.nan)

    # Create the 'ever_cared' column
    dat["ever_cared"] = np.where(
        (dat["sp008_"] == 1) | (dat["sp018_"] == 1),
        1,
        np.where(
            ((dat["sp008_"] == 5) & (dat["sp018_"] == 5))
            | ((dat["sp008_"] == 5) & dat["sp018_"].isna())
            | (dat["sp008_"].isna() & (dat["sp018_"] == 5)),
            0,
            np.nan,
        ),
    )

    # Create the 'ever_cared' column
    conditions_ever_cared = [
        (dat["sp008_"] == 1) | (dat["sp018_"] == 1),
        ((dat["sp008_"] == 5) & (dat["sp018_"] == 5))
        | ((dat["sp008_"] == 5) & dat["sp018_"].isna())
        | (dat["sp008_"].isna() & (dat["sp018_"] == 5)),
    ]

    choices_ever_cared = [1, 0]

    dat["ever_cared"] = np.select(
        conditions_ever_cared,
        choices_ever_cared,
        default=np.nan,
    )

    # Create the 'ever_cared_parents_outside' column
    conditions_parents_outside = [
        (dat["sp008_"] == 1)
        & (
            (dat["sp009_1"].isin([2, 3]))
            | (dat["sp009_2"].isin([2, 3]))
            | (dat["sp009_3"].isin([2, 3]))
        ),
        dat["sp008_"].isna(),
    ]

    choices_parents_outside = [1, np.nan]

    dat["ever_cared_parents_outside"] = np.select(
        conditions_parents_outside,
        choices_parents_outside,
        default=0,
    )

    # Create the 'ever_cared_parents_within' column
    conditions_parents_within = [
        (dat["sp018_"] == 1) & ((dat["sp019d2"] == 1) | (dat["sp019d3"] == 1)),
        dat["sp018_"].isna(),
    ]

    choices_parents_within = [1, np.nan]

    dat["ever_cared_parents_within"] = np.select(
        conditions_parents_within,
        choices_parents_within,
        default=0,
    )

    # Create the 'ever_cared_parents' column
    conditions_parents = [
        (dat["ever_cared_parents_outside"] == 1)
        | (dat["ever_cared_parents_within"] == 1),
        (dat["ever_cared_parents_within"].isna())
        & (dat["ever_cared_parents_outside"].isna()),
    ]

    choices_parents = [1, np.nan]

    dat["ever_cared_parents"] = np.select(
        conditions_parents,
        choices_parents,
        default=0,
    )

    # Define conditions and choices for np.select
    conditions = [
        (dat["sp018_"] == 1) & ((dat["sp019d2"] == 1) | (dat["sp019d3"] == 1)),
        (dat["sp008_"] == 1)
        & ((dat["sp009_1"] == 2) | (dat["sp009_2"] == 2) | (dat["sp009_3"] == 2)),
        (dat["sp008_"] == 1)
        & ((dat["sp009_1"] == 3) | (dat["sp009_2"] == 3) | (dat["sp009_3"] == 3)),
    ]

    choices = [1, 1, 1]  # Assign 1 if the conditions are met

    # Use np.select to create the 'care_in_year' column
    dat["care_in_year"] = np.select(conditions, choices, default=0)

    # still care stuff
    dat = dat.sort_values(by=["mergeid", "int_year"], ascending=[True, True])

    # Calculate cumulative sum for 'care_in_year' within each 'mergeid' group
    dat["care_experience"] = (
        dat.groupby("mergeid")["care_in_year"]
        .cumsum()
        .where(dat["care_in_year"] >= 0, np.nan)
    )

    # outside the household
    #     | ((dat["sp009_2"] == 1) & (dat["sp010d1_2"] == 1))
    #     | ((dat["sp009_3"] == 1) & (dat["sp010d1_3"] == 1))

    # # need to drop personal (intensive care) INSIDE the houeshold to any other than parent
    # # need to add variables sp019d1
    # # rename in waves 4, 5 --> sp/sn above
    # # variables to add

    ### parents health and alive

    # Define conditions and choices for np.select
    conditions_dn026 = [(dat["dn026_1"] == 1), (dat["dn026_1"] == 5)]

    choices_dn026 = [1, 0]

    # Create 'mother_alive' based on 'dn026_1' using np.select
    dat["mother_alive"] = np.select(conditions_dn026, choices_dn026, default=np.nan)

    # Rename 'dn028_1' to 'age_mother'
    dat = dat.rename(columns={"dn028_1": "age_mother"})

    # Handle negative values in 'dn033_1' and convert to 0 for Excellent, 1 for Very good, and 2 for the rest
    conditions_dn033 = [
        (dat["dn033_1"] == 1) | (dat["dn033_1"] == 2),
        (dat["dn033_1"] == 3) | (dat["dn033_1"] == 4),
        (dat["dn033_1"] == 5),
    ]

    choices_dn033 = [0, 1, 2]

    # Create 'health_mother' based on 'dn033_1' using np.select
    dat["health_mother"] = np.select(conditions_dn033, choices_dn033, default=np.nan)

    # Rename 'health_mother_3' to 'health_mother'
    dat = dat.rename(columns={"health_mother_3": "health_mother"})

    # Re-map values to 0=good, 1=medium, 2=bad

    # Handle negative values in 'dn026_2' and create 'father_alive'
    conditions_dn026_2 = [(dat["dn026_2"] == 1), (dat["dn026_2"] == 5)]

    choices_dn026_2 = [1, 0]

    dat["father_alive"] = np.select(conditions_dn026_2, choices_dn026_2, default=np.nan)

    # Rename 'dn028_2' to 'age_father'
    dat = dat.rename(columns={"dn028_2": "age_father"})

    # Handle negative values in 'dn033_2' and create 'health_father_3'
    conditions_dn033_2 = [
        (dat["dn033_2"] == 1) | (dat["dn033_2"] == 2),
        (dat["dn033_2"] == 3) | (dat["dn033_2"] == 4),
        (dat["dn033_2"] == 5),
    ]

    choices_dn033_2 = [0, 1, 2]

    dat["health_father_3"] = np.select(
        conditions_dn033_2,
        choices_dn033_2,
        default=np.nan,
    )

    ### dist to parents
    # Handle negative values in 'dn030_1' and 'dn030_2', and create 'dist_father' and 'dist_mother'
    dat["dist_father"] = dat["dn030_2"].apply(lambda x: x if x >= 0 else np.nan)
    dat["dist_mother"] = dat["dn030_1"].apply(lambda x: x if x >= 0 else np.nan)

    # Create 'parents_live_close' based on distance criteria using np.select
    conditions_distance = [(dat["dist_father"] <= 4) | (dat["dist_mother"] <= 4)]

    choices_distance = [1]

    dat["parents_live_close"] = np.select(
        conditions_distance,
        choices_distance,
        default=0,
    )

    dat["freq_visits_mother"] = dat["dn032_1"]
    dat["freq_visits_father"] = dat["dn032_2"]

    ### missing age mother

    # low share of parent alive in Fischer
    # age == nan --> parent dead?
    # or can the come "back alive" if nan means simply just missing

    # dn127_1 (mother) dn127_2 (father)
    # only since wave 6

    # could use age / health of mothers to check this
    # if data about age / health of mother in period before and now not, assume
    # that mother died
    # same for father

    # Group the data by 'age' and count missing values in 'age_mother'
    dat[dat["age_mother"].isna()].groupby("age")["age"].count()

    mask = (dat["age_mother"].isna()) & (dat["mother_alive"] == 1)
    dat[mask]
    #   1259

    # Create 'mother_alive_2' based on 'mother_alive'
    dat["mother_alive_2"] = np.where(dat["mother_alive"] == 1, 1, np.nan)

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Create 'lagged_age_mother' using 'shift' to represent the previous period's values
    dat["lagged_age_mother"] = dat.groupby("mergeid")["age_mother"].shift(1)

    # Create 'mother_dead' based on the specified conditions
    dat["mother_dead"] = np.where(
        dat["age_mother"].isna() & (dat["lagged_age_mother"] > 0),
        1,
        np.nan,
    )

    dat["lagged_mother_alive"] = dat.groupby("mergeid")["mother_alive"].shift(1)

    # Create 'mother_dead' based on conditions using np.select
    conditions = [(dat["lagged_mother_alive"] == 0), (dat["lagged_mother_alive"] == 1)]

    choices = [1, 0]  # 1 for True, 0 for False

    dat["mother_dead_since_last"] = np.select(conditions, choices, np.nan)

    ## more age info?
    # Create 'mother_alive' based on 'dn026_1' using np.select
    dat["mother_alive"] = np.select(conditions_dn026, choices_dn026, default=np.nan)

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Create 'lagged_age_mother' using 'shift' to represent the previous period's values
    dat["lagged_age_mother"] = dat.groupby("mergeid")["age_mother"].shift(1)

    # Create 'mother_dead' based on the specified conditions
    dat["mother_dead"] = np.where(
        dat["age_mother"].isna() & (dat["lagged_age_mother"] > 0),
        1,
        np.nan,
    )

    dat["lagged_mother_alive"] = dat.groupby("mergeid")["mother_alive"].shift(1)

    # Create 'mother_dead' based on conditions using np.select
    conditions = [(dat["lagged_mother_alive"] == 0), (dat["lagged_mother_alive"] == 1)]

    choices = [1, 0]  # 1 for True, 0 for False

    dat["mother_dead_since_last"] = np.select(conditions, choices, np.nan)

    ## more age info
    # Create 'mother_alive' based on 'dn026_1' using np.select
    dat["mother_alive"] = np.select(conditions_dn026, choices_dn026, default=np.nan)

    # Sort the data by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Group the data by 'mergeid' and transform to get the first non-NaN value of 'age_mother'
    dat["age_mother_first"] = dat.groupby("mergeid")["age_mother"].transform("first")
    dat["int_year_mother_first"] = dat.groupby("mergeid")["int_year"].transform("first")

    # Calculate the first non-NaN value in 'age_mother_first' within each group
    first_age_mother = dat.groupby("mergeid")["age_mother_first"].transform("first")

    # Create 'birth_year_mother' based on the calculation
    dat["age_year_mother_new"] = (
        dat["int_year"] - dat["int_year_mother_first"] + first_age_mother
    )

    # Group the data by 'mergeid'
    grouped = dat.groupby("mergeid")

    # Determine the most common non-empty value in 'dn027_1' for each 'mergeid'
    most_common_value = grouped["dn027_1"].apply(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan,
    )

    # Assign the most common value to all rows within the 'mergeid' group
    dat["age_mother_death"] = dat["mergeid"].map(most_common_value)

    # Fill any remaining NaN values with np.nan
    dat["age_mother_death"] = dat["age_mother_death"].fillna(np.nan)

    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Initialize an auxiliary variable 'death_transition' to track the transition from 1 to 0
    dat["death_transition"] = (dat["mother_alive"] == 0) & (
        dat.groupby("mergeid")["mother_alive"].shift(1) == 1
    )

    # Calculate 'year_mother_death' based on the first transition from 1 to 0 within each 'mergeid'
    dat["year_mother_death"] = dat.groupby("mergeid")["int_year"].transform(
        lambda x: x.where(dat["death_transition"]).min(),
    )

    # Fill remaining NaN values in 'year_mother_death' with np.nan

    # Identify the first observation in the panel for each 'mergeid'
    #
    ## Further filter for rows where 'mother_alive' is 0
    #
    ## Replace values in 'year_mother_death' with 'int_year - 1' for the first observations
    # dat.loc[first_observation_mother_alive_zero_mask, "year_mother_death"] = (

    # Assuming 'dat' is a pandas DataFrame

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Identify the first observation in the panel for each 'mergeid'
    first_observation_mask = (
        dat.groupby("mergeid")["int_year"].transform("first") == dat["int_year"]
    )

    # Identify the next observation in the panel for each 'mergeid'
    next_observation_mask = (
        dat.groupby("mergeid")["int_year"].transform("first") == dat["int_year"] + 1
    )

    # Filter for rows where 'mother_alive' is NaN and the conditions are met
    nan_mother_alive_mask = (
        dat["mother_alive"].isna() & first_observation_mask & next_observation_mask
    )

    # Replace 'mother_alive' with 1 for the specified rows
    dat.loc[
        nan_mother_alive_mask & (dat["mother_alive"].shift(1) == 1),
        "mother_alive",
    ] = 1

    # Replace 'mother_alive' with 0 for the specified rows
    dat.loc[
        nan_mother_alive_mask & (dat["mother_alive"].shift(1) == 0),
        "mother_alive",
    ] = 0

    first_occurrence_condition = (
        (
            (dat["age_year_mother_new"].notna() & dat["age_mother_death"].notna())
            & (dat["age_year_mother_new"] > dat["age_mother_death"])
        )
        .groupby(dat["mergeid"])
        .idxmax()
    )

    # Identify the next occurrence of "mother_alive == 1" per "mergeid"
    next_occurrence_condition = (
        (dat["mother_alive"] != 1).groupby(dat["mergeid"]).shift(-1)
    )
    # Replace 'mother_alive' with 0 for rows where it is NaN and the conditions are met
    dat.loc[
        (dat["mother_alive"].isna())
        & (dat.index.isin(first_occurrence_condition))
        & (next_occurrence_condition),
        "mother_alive",
    ] = 0

    # # Sort the DataFrame by 'mergeid' and 'int_year'
    #
    #
    # # Create a mask for rows where 'mother_alive' is NaN and the preceding row's 'mother_alive' is 0
    #
    # # Set 'mother_alive' to 0 for the identified rows

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Define a custom function to handle grouping within 'mergeid'
    def custom_condition(group):
        return group["mother_alive"].isna() & (group["mother_alive"].shift(1) == 0)

    # Apply the custom function within each 'mergeid'
    nan_mother_alive_mask = dat.groupby("mergeid").apply(custom_condition)

    # Flatten the result to a boolean array
    nan_mother_alive_mask = nan_mother_alive_mask.values

    # Set 'mother_alive' to 0 for the identified rows
    dat.loc[nan_mother_alive_mask, "mother_alive"] = 0

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Identify the next occurrence of "mother_alive == 1" per "mergeid"
    next_occurrence_condition = dat.groupby("mergeid")["mother_alive"].shift(-1) == 1

    # Replace 'mother_alive' with 1 for rows where it is NaN and the next occurrence condition is met
    dat.loc[
        (dat["mother_alive"].isna()) & next_occurrence_condition,
        "mother_alive",
    ] = 1

    # dat["birth_year_mother"] = (
    #     .apply(lambda group: group["year_mother_death"] - group["age_mother_death"])
    #     .reset_index(drop=True)

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Identify the rows where 'mother_alive' switches from 1 to 0 within each 'mergeid'
    switch_condition = (dat["mother_alive"] == 1) & (dat["mother_alive"].shift(-1) == 0)

    # Calculate 'birth_year_mother' based on 'int_year' and 'age_mother_death' for the switching year
    dat.loc[switch_condition, "birth_year_mother"] = (
        dat["int_year"] - dat["age_mother_death"]
    )

    # Forward-fill the values within each 'mergeid' group
    dat["birth_year_mother"] = dat.groupby("mergeid")["birth_year_mother"].ffill()

    dat["birth_year_mother"] = dat.groupby("mergeid")["birth_year_mother"].bfill()

    # dat["age_year_mother_new"] = dat["age_year_mother_new"].fillna(

    dat["age_year_mother_new"] = dat.apply(
        lambda row: row["int_year"] - row["birth_year_mother"]
        if row["mother_alive"] == 1
        else np.nan,
        axis=1,
    )

    dat["age_mother"] = dat["age_mother"].fillna(dat["age_year_mother_new"])

    # SAME FOR FATHER @

    ### Missing age info

    ###### JOB STUFF @@@@@

    dat["ep002_"] = dat["ep002_"].apply(lambda x: x if x >= 0 else np.nan)

    dat["worked_last_period"] = np.where(
        (dat["ep005_"] == 2) | (dat["ep002_"] == 1),
        1,
        0,
    )

    # EP141_ChangeInJob
    # EP125_ContWork
    # EP006_EverWorked

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Shift the 'ep005_' and 'ep002_' variables by one period
    dat["lagged_ep005_"] = dat.groupby("mergeid")["ep005_"].shift(1)
    dat["lagged_ep002_"] = dat.groupby("mergeid")["ep002_"].shift(1)

    # Create 'worked_last_period' based on the lagged values
    dat["worked_last_period"] = np.where(
        (dat["lagged_ep005_"] == 2) | (dat["ep002_"] == 1),
        1,
        0,
    )

    # Drop the lagged columns if not needed
    dat = dat.drop(["lagged_ep005_", "lagged_ep002_"], axis=1)

    # **Retirement** Individuals are considered retired if they respond to be retired in the question on their
    # current job situation. In addition, individuals are considered retired if they respond not to be working
    # and respond to be receiving old age pension benefits.

    # **Working**  Individuals are considered part-time employed if they respond to be working and provide
    # a number of working hours within the 5th to 50th percentile of the distribution of working hours. This
    # corresponds to 10 to 32 hours per week. Individuals are considered full-time employed if they work
    # more than the median of hours in the distribution of working hours (more than 32 hours per week).
    # In the model we consider the mass-points of the distribution at the 25th percentile (20hours per week)
    # and 75th percentile of the distribution (40 hours per week) for working women as part- and full- time
    # work.

    dat["full_time"] = np.where((dat["working"] == 1) & (dat["ep013_"] > 32), 1, 0)

    dat["part_time"] = np.where(
        (dat["working"] == 1) & (dat["ep013_"] >= 10) & (dat["ep013_"] <= 32),
        1,
        0,
    )

    conditions = [
        (dat["working"] == 1) & (dat["ep013_"] > 32),
        (dat["working"] == 1) & (dat["ep013_"] >= 10) & (dat["ep013_"] <= 32),
    ]

    values = [1, 0]

    # Use numpy.select to create the 'full_time' variable
    dat["full_time_nan"] = np.select(conditions, values, np.nan)

    # Identify columns that start with "sl_re011"
    job_start = [col for col in dat.columns if col.startswith("sl_re011")]

    # Iterate through columns and set values < 0 to NA
    for job in job_start:
        dat[job] = np.where(dat[job] < 0, np.nan, dat[job])

    # Identify columns that start with "sl_re026"
    job_end = [col for col in dat.columns if col.startswith("sl_re026")]

    # Iterate through columns and set values < 0 to NA, and values == 9997 to int_year
    for job in job_end:
        dat[job] = np.where(
            dat[job] < 0,
            np.nan,
            np.where(dat[job] == 9997, dat["int_year"], dat[job]),
        )

    # Create a list of column names that start with 'sl_re011_'
    sl_re011_columns = [f"sl_re011_{i}" for i in range(1, 21)]

    # Function to find the most recent job started
    def most_recent_job(row):
        for col in reversed(sl_re011_columns):
            if not pd.isna(row[col]):
                return row[col]
        return np.nan

    # Create the 'most_recent_job_started' variable
    dat["most_recent_job_started"] = dat.apply(most_recent_job, axis=1)

    # Create a list of column names that start with 'sl_re026_'
    sl_re026_columns = [f"sl_re026_{i}" for i in range(1, 21)]

    # Function to find the most recent job ended
    def most_recent_job_ended(row):
        for col in reversed(sl_re026_columns):
            if not pd.isna(row[col]):
                return row[col]
        return np.nan

    # Create the 'most_recent_job_ended' variable
    dat["most_recent_job_ended"] = dat.apply(most_recent_job_ended, axis=1)

    dat["most_recent_job_ended"] = dat.groupby("mergeid")[
        "most_recent_job_ended"
    ].transform(lambda x: x.ffill().bfill())

    dat["most_recent_job_started"] = dat.groupby("mergeid")[
        "most_recent_job_started"
    ].transform(lambda x: x.ffill().bfill())

    conditions = [(dat["sl_re011_1"].notna() & (dat["wave"] == 3))]
    values = [1]

    # Use numpy.select to create the 'wave_3_response' variable

    # # Define the conditions and corresponding values for wave 7 response

    # # Use numpy.select to create the 'wave_7_response' variable

    # dat["both_wave_3_and_7"] = (
    # ).astype(int)

    prefixes = ["sl_re011_", "sl_re026_"]

    # Iterate over the prefixes and apply forward and backward fill
    for prefix in prefixes:
        relevant_cols = [col for col in dat.columns if col.startswith(prefix)]
        dat[relevant_cols] = dat.groupby("mergeid")[relevant_cols].transform(
            lambda x: x.ffill().bfill(),
        )

    dat["job_just_started"] = 0

    dat.loc[
        dat["most_recent_job_started"].notna() & dat["most_recent_job_ended"].isna(),
        "job_just_started",
    ] = 1

    dat["job_just_ended"] = 0

    dat.loc[
        dat["most_recent_job_ended"].notna() & dat["most_recent_job_started"].isna(),
        "job_just_ended",
    ] = 1

    conditions = [(dat["full_time"] == 1), (dat["part_time"] == 1)]

    choices = [1, 0.5]

    dat["exp_weight"] = np.select(conditions, choices, default=0)

    dat["lagged_exp_weight"] = dat.groupby("mergeid")["exp_weight"].shift(1)

    # List of columns starting with "sl_re026_" or "sl_re011_"
    columns_to_check = [
        col for col in dat.columns if col.startswith(("sl_re026_", "sl_re011_"))
    ]

    # Use map with a lambda function to replace both negative and values greater than or equal to 9997 with np.nan
    dat[columns_to_check] = dat[columns_to_check].apply(
        lambda x: x.map(lambda val: np.nan if val < 0 else val),
    )

    ### Compute exper weights (normal exper not needed)
    suffixes = range(1, 21)

    weight_columns = []

    for suffix in suffixes:
        sl_re_column = f"sl_re016_{suffix}"
        weight_exper_column = f"weight_exper_{suffix}"

        weight_values = dat[sl_re_column].apply(
            lambda x: 1 if x == 1 else (0.5 if x == 2 else 0),
        )

        weight_columns.append(pd.Series(weight_values, name=weight_exper_column))

    dat = pd.concat([dat, *weight_columns], axis=1)

    # Job was full-time or part-time: sl_re016_{suffix}

    # start spell: sl_re011_{suffix}
    # stop spell: sl_re026_{suffix}

    # year switch to part-time: sl_re018_{suffix}
    # year switch to full-time: sl_re020_{suffix}

    #     dat["job_ended"] = np.where(

    suffixes = range(1, 17)

    for suffix in suffixes:
        dat[f"weight_exper_{suffix}"] = np.nan

        job_ended = np.where(
            dat[f"sl_re026_{suffix}"] >= dat["first_int_year"],
            dat["first_int_year"],
            dat[f"sl_re026_{suffix}"],
        )

        always_full_time = dat[f"sl_re016_{suffix}"] == 1.0
        dat.loc[always_full_time, f"weight_exper_{suffix}"] = 1.0 * np.abs(
            job_ended - dat[f"sl_re011_{suffix}"],
        )

        always_part_time = dat[f"sl_re016_{suffix}"] == 2.0
        dat.loc[always_part_time, f"weight_exper_{suffix}"] = 0.5 * np.abs(
            job_ended - dat[f"sl_re011_{suffix}"],
        )

        switched_from_full_to_part_time = dat[f"sl_re016_{suffix}"] == 3.0
        dat.loc[switched_from_full_to_part_time, f"weight_exper_{suffix}"] = 1 * np.abs(
            dat[f"sl_re018_{suffix}"] - dat[f"sl_re011_{suffix}"],
        ) + 0.5 * np.abs(job_ended - dat[f"sl_re018_{suffix}"])

        switched_from_part_to_full_time = dat[f"sl_re016_{suffix}"] == 4.0
        dat.loc[
            switched_from_part_to_full_time,
            f"weight_exper_{suffix}",
        ] = 0.5 * np.abs(
            dat[f"sl_re018_{suffix}"] - dat[f"sl_re011_{suffix}"],
        ) + 1 * np.abs(
            job_ended - dat[f"sl_re020_{suffix}"],
        )

    # work experience
    suffixes = range(1, 17)

    # Create a list of column names for 'weight_exper_' columns
    weight_columns = [f"weight_exper_{i}" for i in suffixes]

    # Calculate work_experience row-wise and store the result in a new column
    dat["_retro_work_exp"] = dat[weight_columns].sum(axis=1)

    # Group by 'mergeid' and transform to propagate work_experience value

    # Calculate the maximum work_experience value within each 'mergeid' group
    dat["retro_work_exp"] = dat.groupby("mergeid")["_retro_work_exp"].transform("max")

    # Create a copy of the DataFrame to de-fragment it
    dat = dat.copy()

    ###

    dat["lagged_int_year"] = dat.groupby("mergeid")["int_year"].shift(1)
    dat["lagged_working"] = dat.groupby("mergeid")["working"].shift(1)

    # Sort the DataFrame by 'mergeid' and 'int_year'
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Sort the DataFrame by mergeid and int_year
    dat = dat.sort_values(by=["mergeid", "int_year"])

    # Initialize the work_exp_cum column with zeros
    dat["work_exp_cum"] = np.nan

    # Define the conditions and values for np.select
    conditions = dat["lagged_working"] == 1
    values = (dat["int_year"] - dat["lagged_int_year"]) * dat["lagged_exp_weight"]

    # Use np.select to update work_exp_cum

    dat["work_exp_cum"] = np.where(
        dat["lagged_working"] == 1,
        (dat["int_year"] - dat["lagged_int_year"]) * dat["lagged_exp_weight"],
        0,
    )

    # Calculate the cumulative sum of work_exp_cum by mergeid
    dat["work_exp_cum"] = dat.groupby("mergeid")["work_exp_cum"].cumsum()

    ### Putting it together ###
    dat["work_exp"] = dat["retro_work_exp"] + dat["work_exp_cum"]

    # try


# =====================================================================================


def calculate_retired(row):
    if row["ep005_"] == 1 or (not pd.isna(row["ep329_"])):
        out = 1
    elif pd.isna(row["ep005_"]) and pd.isna(row["ep329_"]):
        out = np.nan
    else:
        out = 0

    return out


def calculate_years_since_retirement(row):
    if row["retired"] == 1 or (not pd.isna(row["ep329_"])):
        out = row["int_year"] - row["ep329_"]
    elif row["retired"] == 0:
        out = 0
    else:
        out = np.nan

    return out


def find_max_suffix(row):
    max_suffix = 0
    for col in FURTHER_EDUC:
        if row[col] == 1:
            suffix = int(col.split("dn012d")[-1])
            max_suffix = max(max_suffix, suffix)

    return max_suffix if max_suffix >= 0 else np.nan
