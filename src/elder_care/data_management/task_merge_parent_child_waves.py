"""Merge parent information."""
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from elder_care.config import BLD
from elder_care.config import SRC
from pytask import Product

GERMANY = 12
MISSING_VALUE = -9
WAVE_1 = 1
WAVE_2 = 2
WAVE_3 = 3
WAVE_4 = 4
WAVE_5 = 5
WAVE_6 = 6
WAVE_7 = 7

CV_R = [
    "int_year",
    "int_month",
    "gender",
    "mobirth",
    "yrbirth",
    "age_int",
    "hhsize",
]

CHILDREN = [
    "ch005_",  # gender
    "ch006_",  # year of birth
    "ch007_",  # where does child live
    "ch012_",  # marital status
    "ch014_",  # frequency contact with child
    "ch016_",  # employment status
    "ch017_",  # highest education
]

PHYSICAL_HEALTH = ["ph003_"]

HEALTH_CARE = [
    "hc029_",  # in nursing home during last 12 months
    # 1 = yes, temporarily
    # 3 = yes, permanently
    # 5 = no
    "hc031_",  # Weeks stayed in a nursing home or residential care facility
    "hc032d1",  # nursing or personal care
    "hc032d2",  # domestic tasks
    "hc032d3",  # meals on wheels
    "hc032dno",  # none of these
    "hc033_",  # How many weeks did you receive paid help for personal care
    "hc034_",  # How many hours per week did you receive such professional help?
    "hc035_",  # How many weeks did you receive professional help for domestic tasks
    "hc036_",  # How many hours per week did you receive such professional help?
    "hc037_",  # How many weeks did you receive meals-on-wheel
    # since wave 5
    "hc696_",  # Paid anything yourself stay in nursing home
    "hc127d1",  # help with personal care
    "hc127d2",  # help with domestic tasks in own home
    "hc127d3",  # meals on wheels
    "hc127d4",  # helpt with other activities
    "hc127dno",  # none of these
]


SOCIAL_SUPPORT = [
    "sp020_",  # someone in this household helped you regularly with personal care
    "sp021d10",  # from own child
    "sp021d11",  # from stepchild
    "sp021d20",  # from son-in-law
    "sp021d21",  # from daughter-in-law
    # only from wave 7 on :9
    # "sp033_1",  # from which child in household
    # "sp033_2",  # from which child in household
    # "sp033_3",  # from which child in household
    # "sp033_4",  # from which child in household
    # "sp033_5",  # from which child in household
    # "sp033_6",  # from which child in household
    # "sp033_7",  # from which child in household
    "sp002_",  # help from outsided the household
    "sp003_1",  # from whom, person 1: 10, 11, 20 ,21
    "sp003_2",  # from whom, person 2: 10, 11, 20 ,21
    "sp003_3",  # from whom, person 3: 10, 11, 20 ,21
    # only from wave 7 on :9
    # "sp027_1",  # from which child
    # "sp027_2",  # from which child
    # "sp027_3",  # from which child
    "sp004d1_1",  # help with personal care
    "sp004d1_2",  # help with personal care
    "sp004d1_3",  # help with personal care
    "sp004d2_1",  # practical household help
    "sp004d2_2",  # practical household help
    "sp004d2_3",  # practical household help
    "sp005_1",  # how often per week from this person
    "sp005_2",  # how often per week from this person
    "sp005_3",  # how often per week from this person
]

GV_CHILDREN = [
    "ch_gender_",
    "ch_yrbirth_",
    "ch_yrbirth_youngest_child_",
    "ch_occupation_",
    "ch_proximity_",
    "ch_marital_status_",
    "ch_school_education_",
    "ch_outhh_receive_care_",
    "ch_hh_receive_care_",
]


# =============================================================================
def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_merge_parent_child_waves_and_modules(
    path: Annotated[Path, Product] = BLD / "data" / "data_parent_child_merged.csv",
) -> None:
    """Merge raw parent information."""
    child_suffixes = [str(i) for i in range(1, 21)]

    vars_children = [var + child for var in CHILDREN for child in child_suffixes]

    data_modules = {
        "cv_r": CV_R,
        "ph": PHYSICAL_HEALTH,
        "ch": vars_children,
        "sp": SOCIAL_SUPPORT,
        "hc": HEALTH_CARE,
    }

    wave1 = process_wave(wave=1, data_modules=data_modules)
    wave2 = process_wave(wave=2, data_modules=data_modules)
    wave4 = process_wave(wave=4, data_modules=data_modules)
    wave5 = process_wave(wave=5, data_modules=data_modules)
    wave6 = process_wave(wave=6, data_modules=data_modules)
    wave7 = process_wave(wave=7, data_modules=data_modules)
    wave8 = process_wave(wave=8, data_modules=data_modules)

    waves_list = [wave1, wave2, wave4, wave5, wave6, wave7, wave8]

    data = merge_wave_datasets(waves_list)

    vars_gv_children = [var + child for var in GV_CHILDREN for child in child_suffixes]
    gv_6 = process_gv_children(wave=6, args=vars_gv_children)
    gv_7 = process_gv_children(wave=7, args=vars_gv_children)
    gv_8 = process_gv_children(wave=8, args=vars_gv_children)

    gv_datasets = [gv_6, gv_7, gv_8]
    gv_data = pd.concat(gv_datasets, axis=0, ignore_index=True)
    gv_data = gv_data.sort_values(by=["mergeid", "wave"])

    data_merged = data.merge(gv_data, on=["mergeid", "wave"], how="left")
    # save data
    data_merged.to_csv(path, index=False)

    # create moments of formal care by informal care from children
    # (parent age brackets), (no informal care, light, intensive),
    # (child close, child far away)
    # and also age of caregiving child? (maybe later, too cumbersome now)
    # later maybe also education type of children


def process_wave(wave, data_modules):
    """Process wave of the standard SHARE modules."""
    wave_data = {}

    for module in data_modules:
        _wave_module = process_module(module, wave, data_modules[module])
        wave_data[module] = _wave_module

    merged_data = wave_data["cv_r"]

    for module_key in ("ph", "ch", "sp", "hc"):
        merged_data = merged_data.merge(
            wave_data[module_key],
            on="mergeid",
            how="outer",
        )

    merged_data = merged_data.copy()
    merged_data["wave"] = wave

    return merged_data


def process_module(module, wave, args):
    """Process a single wave module."""
    module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel8-0-0_{module}.dta"
    data = pd.read_stata(module_file, convert_categoricals=False)

    data.columns = [col[:-2] if col.endswith("sp") else col for col in data.columns]

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    selected_columns = ["mergeid"] + [col for col in args if col in data.columns]
    columns = ["mergeid", *args]

    if wave == WAVE_7:
        not_included = [
            "ch007_1",
            "ch007_2",
            "ch007_3",
            "ch007_4",
            "ch014_1",
            "ch014_2",
            "ch014_3",
            "ch014_4",
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave in (WAVE_1, WAVE_2):
        not_included = [
            "hc696_",  # Paid anything yourself stay in nursing home
            "hc127d1",  # help with personal care
            "hc127d2",  # help with domestic tasks in own home
            "hc127d3",  # meals on wheels
            "hc127d4",  # helpt with other activities
            "hc127dno",
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave == WAVE_4:
        not_included = [
            "sp021d10",
            "sp021d11",
            "sp004d1_1",
            "sp004d1_2",
            "sp004d1_3",
            "sp004d2_1",
            "sp004d2_2",
            "sp004d2_3",
            # "hc032_",
            # "hc033_",
            # "hc034_",
            "hc035_",  # How many weeks did you receive paid help for domestic tasks
            "hc036_",  # How many hours per week did you receive such professional help?
            "hc037_",  # How many weeks did you receive meals-on-wheel
            "hc696_",  # Paid anything yourself stay in nursing home
            "hc696_",  # Paid anything yourself stay in nursing home
            "hc127d1",  # help with personal care
            "hc127d2",  # help with domestic tasks in own home
            "hc127d3",  # meals on wheels
            "hc127d4",  # helpt with other activities
            "hc127dno",
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave == WAVE_5:
        not_included = [
            "sp004d1_1",
            "sp004d1_2",
            "sp004d1_3",
            "sp004d2_1",
            "sp004d2_2",
            "sp004d2_3",
            "hc035_",  # How many weeks did you receive paid help for domestic tasks
            "hc036_",  # How many hours per week did you receive such professional help?
            "hc037_",  # How many weeks did you receive meals-on-wheel
            "hc696_",  # Paid anything yourself stay in nursing home
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave == WAVE_6:
        not_included = [
            "hc035_",  # How many weeks did you receive paid help for domestic tasks
            "hc036_",  # How many hours per week did you receive such professional help?
            "hc037_",  # How many weeks did you receive meals-on-wheel
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    else:
        columns = ["mergeid", *args]

    # Replace negative values with NaN using NumPy
    # Create missing columns and fill with NaN
    data = data.copy()
    for col in args:
        if col not in selected_columns:
            data[col] = np.nan

    return data[columns]


def process_gv_children(wave, args):
    """Process single wave of the gv_children module."""
    module = "gv_children"
    module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel8-0-0_{module}.dta"
    data = pd.read_stata(module_file, convert_categoricals=False)

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    # Select columns 'mergeid' and the specified args (create missing columns with NaN)
    selected_columns = ["mergeid"] + [col for col in args if col in data.columns]
    columns = ["mergeid", *args]

    # Replace negative values with NaN using NumPy
    # Create missing columns and fill with NaN
    data = data.copy()
    for col in args:
        if col not in selected_columns:
            data[col] = np.nan

    data = data[columns]
    data["wave"] = wave

    return data


def merge_wave_datasets(wave_datasets):
    """Combine data frames in the wave_datasets into one data frame."""
    combined_data = pd.concat(wave_datasets, axis=0, ignore_index=True)

    # Filter out rows where the 'int_year' column is not equal to -9
    combined_data = combined_data[combined_data["int_year"] != MISSING_VALUE]

    return combined_data.sort_values(by=["mergeid", "int_year"])
