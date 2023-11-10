"""Create the estimation data set of females between 50 and 68."""
import numpy as np
import pandas as pd

FEMALE = 2

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


# def task_create_estimation_data(data):
def create_estimation_data(data):
    """Create the estimation data set."""
    # only females
    dat = data.copy()

    # Filter for females
    dat = dat[dat["gender"] == FEMALE]


# =====================================================================================


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


def find_max_suffix(row):
    max_suffix = 0
    for col in FURTHER_EDUC:
        if row[col] == 1:
            suffix = int(col.split("dn012d")[-1])
            max_suffix = max(max_suffix, suffix)

    return max_suffix if max_suffix >= 0 else np.nan
