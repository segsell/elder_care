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
WAVE_4 = 4
WAVE_5 = 5
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

HEALTH = [
    "hc029_",  # in nursing home during last 12 months
    # 1 = yes, temporarily
    # 3 = yes, permanently
    # 5 = no
    "hc031_",  # Weeks stayed in a nursing home or residential care facility
    "hc035_",  # How many weeks did you receive professional help for domestic tasks
    "hc036_",  # How many hours per week did you receive such professional help?
    "hc037_",  # How many weeks did you receive meals-on-wheel
    # since wave 5
    "hc696_",  # Payed anything yourself stay in nursing home
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


def task_merge_parent_child_waves_and_modules(
    path: Annotated[Path, Product] = BLD / "data" / "data_parent_child_merged.csv",
) -> None:
    child_suffixes = [str(i) for i in range(1, 5)]

    vars_children = [var + child for var in CHILDREN for child in child_suffixes]
    children_1 = process_module(module="ch", wave=1, args=vars_children)
    children_2 = process_module(module="ch", wave=2, args=vars_children)
    # children_3 = process_module(module="ch", wave=3, args=vars_children)
    children_4 = process_module(module="ch", wave=4, args=vars_children)
    children_5 = process_module(module="ch", wave=5, args=vars_children)
    children_6 = process_module(module="ch", wave=6, args=vars_children)
    children_7 = process_module(module="ch", wave=7, args=vars_children)
    children_8 = process_module(module="ch", wave=8, args=vars_children)

    children_datasets = [
        children_1,
        children_2,
        children_4,
        children_5,
        children_6,
        children_7,
        children_8,
    ]
    # children_data = merge_wave_datasets(children_datasets)

    social_support_1 = process_module(module="sp", wave=1, args=SOCIAL_SUPPORT)
    social_support_2 = process_module(module="sp", wave=2, args=SOCIAL_SUPPORT)
    social_support_4 = process_module(module="sp", wave=4, args=SOCIAL_SUPPORT)
    social_support_5 = process_module(module="sp", wave=5, args=SOCIAL_SUPPORT)
    social_support_6 = process_module(module="sp", wave=6, args=SOCIAL_SUPPORT)
    social_support_7 = process_module(module="sp", wave=7, args=SOCIAL_SUPPORT)
    social_support_8 = process_module(module="sp", wave=8, args=SOCIAL_SUPPORT)

    social_support_datasets = [
        social_support_1,
        social_support_2,
        social_support_4,
        social_support_5,
        social_support_6,
        social_support_7,
        social_support_8,
    ]
    social_support_data = pd.concat(social_support_datasets, axis=0, ignore_index=True)

    health_1 = process_module(module="hc", wave=1, args=HEALTH)
    health_2 = process_module(module="hc", wave=2, args=HEALTH)
    health_3 = process_module(module="hc", wave=3, args=HEALTH)
    health_4 = process_module(module="hc", wave=4, args=HEALTH)
    health_5 = process_module(module="hc", wave=5, args=HEALTH)
    health_6 = process_module(module="hc", wave=6, args=HEALTH)
    health_7 = process_module(module="hc", wave=7, args=HEALTH)
    health_8 = process_module(module="hc", wave=8, args=HEALTH)

    health_datasets = [
        health_1,
        health_2,
        health_3,
        health_4,
        health_5,
        health_6,
        health_7,
        health_8,
    ]
    health_data = pd.concat(health_datasets, axis=0, ignore_index=True)

    vars_gv_children = [var + child for var in GV_CHILDREN for child in child_suffixes]
    gv_6 = process_gv_children(wave=6, args=vars_gv_children)
    gv_7 = process_gv_children(wave=7, args=vars_gv_children)
    gv_8 = process_gv_children(wave=8, args=vars_gv_children)

    gv_datasets = [gv_6, gv_7, gv_8]
    gv_data = pd.concat(gv_datasets, axis=0, ignore_index=True)

    combined_children_health_data = pd.concat([children_data, health_data], axis=1)

    data_merged = pd.merge(
        combined_children_health_data, gv_data, on=["mergeid", "wave"], how="left",
    )

    breakpoint()

    # save data
    data_merged.to_csv(path, index=False)

    # create moments of formal care by informal care from children
    # (parent age brackets), (no informal care, light, intensive),
    # (child close, child far away)
    # and also age of caregiving child? (maybe later, too cumbersome now)
    # later maybe also education type of children


def process_module(module, wave, args):
    module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel8-0-0_{module}.dta"
    data = pd.read_stata(module_file, convert_categoricals=False)

    data.columns = [col[:-2] if col.endswith("sp") else col for col in data.columns]

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    # Select columns 'mergeid' and the specified args (create missing columns with NaN)
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
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    else:
        columns = ["mergeid", *args]

    # Create missing columns and fill with NaN
    # for col in args:
    #     if col not in selected_columns:
    #         data[col] = np.nan

    data = data[columns]

    # Replace negative values with NaN using NumPy

    data["wave"] = wave

    return data


def process_gv_children(wave, args):
    module = "gv_children"
    module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel8-0-0_{module}.dta"
    data = pd.read_stata(module_file, convert_categoricals=False)

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    # Select columns 'mergeid' and the specified args (create missing columns with NaN)
    selected_columns = ["mergeid"] + [col for col in args if col in data.columns]
    columns = ["mergeid", *args]

    # Create missing columns and fill with NaN
    for col in args:
        if col not in selected_columns:
            data[col] = np.nan

    data = data[columns]

    # Replace negative values with NaN using NumPy

    data["wave"] = wave

    return data


def merge_wave_datasets(wave_datasets):
    # Combine the data frames in wave_datasets into one data frame
    combined_data = pd.concat(wave_datasets, axis=0, ignore_index=True)

    # Filter out rows where the 'int_year' column is not equal to -9
    combined_data = combined_data[combined_data["int_year"] != MISSING_VALUE]

    # Sort the data frame by 'mergeid' and 'int_year'
    return combined_data.sort_values(by=["mergeid", "int_year"])
