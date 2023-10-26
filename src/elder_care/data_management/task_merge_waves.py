"""Merge all SHARE waves and modules."""
import pandas as pd
from elder_care.config import SRC

ALL_VARIABLES = {
    "cv_r": [
        "int_year",
        "int_month",
        "gender",
        "mobirth",
        "yrbirth",
        "age_int",
        "hhsize",
    ],
    "dn": [
        "dn002_",
        "dn003_",
        "dn010_",
        "dn041_",
        "dn009_",
        "dn014_",
        "dn015_",
        "dn016_",
        "dn026_1",
        "dn026_2",
        "dn033_1",
        "dn033_2",
        "dn027_1",
        "dn027_2",
        "dn028_1",
        "dn028_2",
        "dn030_1",
        "dn030_2",
        "dn127_1",
        "dn127_2",
        "dn032_1",
        "dn032_2",
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
        # "dn012d95",
        "dn012dno",
        "dn012dot",
    ],
    "ep": [
        "ep005_",
        "ep013_",
        "ep002_",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
        "ep328_",
        "ep329_",
        "ep213_1",
        # "ep213_2",
        # "ep213_3",
        # "ep213_4",
        # "ep213_5",
        # "ep213_6",
        # "ep213_7",
        # "ep213_8",
        # "ep213_9",
        # "ep213_10",
        # "ep213_11",
        # "ep213_12",
        # "ep213_13",
        # "ep213_14",
        # "ep213_15",
        # "ep213_16",
    ],
    "sp": [
        "sp008_",
        "sp018_",
        "sp009_1",
        "sp009_2",
        "sp009_3",
        "sp010d1_1",
        "sp010d1_2",
        "sp010d1_3",
        "sp011_1",
        "sp011_2",
        "sp011_3",
        # "sp019d1",
        "sp019d2",
        "sp019d3",
        "sp019d4",
        "sp019d5",
        "sp019d6",
        "sp019d7",
        # "sp019d8",
        # "sp019d9",
        # "sp019d10",
        # "sp019d11",
        # "sp019d12",
        # "sp019d13",
        # "sp019d14",
        # "sp019d15",
        # "sp019d16",
        # "sp019d17",
        # "sp019d18",
        # "sp019d19",
        # "sp019d20",
    ],
    "gv_isced": ["isced1997_r"],
    # "gv_imputations": [
    #    "hnetw"
    # ],  # household net worth = total gross financial assets + total real assets - total libailities
    "ch": ["ch001_"],
}

KEYS_TO_REMOVE_WAVE1 = {
    "dn": [
        "dn041_",
        "dn127_1",
        "dn127_2",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
    ],
    "ep": [
        "ep013_",
        "ep328_",
        "ep329_",
        "ep213_12",
        "ep213_13",
        "ep213_14",
        "ep213_15",
        "ep213_16",
    ],
}

KEYS_TO_REMOVE_WAVE2 = {
    "dn": [
        "dn127_1",
        "dn127_2",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
    ],
}


KEYS_TO_REMOVE_WAVE4 = {
    "dn": [
        "dn127_1",
        "dn127_2",
        "dn012d14",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
    ],
    "sp": [
        "sp010d1_1",
        "sp010d1_2",
        "sp010d1_3",
    ],
}


KEYS_TO_REMOVE_WAVE5 = {
    "dn": [
        "dn127_1",
        "dn127_2",
        "dn012d20",
        "dn012dno",
    ],
    "sp": [
        "sp010d1_1",
        "sp010d1_2",
        "sp010d1_3",
    ],
}

KEYS_TO_REMOVE_WAVE6 = {
    "dn": [
        "dn012dno",
    ],
    "ep": [
        "ep213_14",
        "ep213_15",
        "ep213_16",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
    ],
}

KEYS_TO_REMOVE_WAVE7 = {
    "dn": [
        "dn012dno",
    ],
    "ep": [
        "ep213_14",
        "ep213_15",
        "ep213_16",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
    ],
    "re": [
        "sl_re018_1",
        "sl_re018_2",
        "sl_re018_3",
        "sl_re018_4",
        "sl_re018_5",
        "sl_re018_6",
        "sl_re018_7",
        "sl_re018_8",
        "sl_re018_9",
        "sl_re018_10",
        "sl_re018_11",
        "sl_re018_12",
        "sl_re018_13",
        "sl_re018_14",
        "sl_re018_15",
        "sl_re018_16",
        "sl_re018_17",
        "sl_re020_1",
        "sl_re020_2",
        "sl_re020_3",
        "sl_re020_4",
        "sl_re020_5",
        "sl_re020_6",
        "sl_re020_7",
        "sl_re020_8",
        "sl_re020_9",
        "sl_re020_10",
        "sl_re020_11",
        "sl_re020_12",
        "sl_re020_13",
        "sl_re020_14",
        "sl_re020_15",
        "sl_re020_16",
        "sl_re020_17",
    ],
}

KEYS_TO_REMOVE_WAVE8 = {
    "dn": [
        "dn012dno",
    ],
    "ep": [
        "ep213_14",
        "ep213_15",
        "ep213_16",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
    ],
}


def task_merge_waves():
    # Retrospective waves
    re_vars = (
        [f"sl_re011_{i}" for i in range(1, 21)]
        + [f"sl_re016_{i}" for i in range(1, 21)]
        + [f"sl_re026_{i}" for i in range(1, 21)]
        + [f"sl_re018_{i}" for i in range(1, 17)]
        + [f"sl_re020_{i}" for i in range(1, 17)]
    )

    # Data modules for wave 3
    variables_wave3 = {
        "cv_r": [
            "int_year",
            "int_month",
            "gender",
            "mobirth",
            "yrbirth",
            "age_int",
            "hhsize",
        ],
        "re": re_vars,
    }

    # Separate modules for partly retrospective wave 7
    variables_wave7 = filter_nested_dict(
        ALL_VARIABLES | {"re": re_vars},
        KEYS_TO_REMOVE_WAVE7,
    )

    variables_wave1 = filter_nested_dict(ALL_VARIABLES, KEYS_TO_REMOVE_WAVE1)
    variables_wave2 = filter_nested_dict(ALL_VARIABLES, KEYS_TO_REMOVE_WAVE2)
    variables_wave4 = filter_nested_dict(ALL_VARIABLES, KEYS_TO_REMOVE_WAVE4)
    variables_wave5 = filter_nested_dict(ALL_VARIABLES, KEYS_TO_REMOVE_WAVE5)
    variables_wave6 = filter_nested_dict(ALL_VARIABLES, KEYS_TO_REMOVE_WAVE6)
    variables_wave8 = filter_nested_dict(ALL_VARIABLES, KEYS_TO_REMOVE_WAVE8)

    wave1 = process_wave(wave_number=1, data_modules=variables_wave1)
    wave2 = process_wave(wave_number=2, data_modules=variables_wave2)
    wave3 = process_wave(wave_number=3, data_modules=variables_wave3)
    wave4 = process_wave(wave_number=4, data_modules=variables_wave4)
    wave5 = process_wave(wave_number=5, data_modules=variables_wave5)
    wave6 = process_wave(wave_number=6, data_modules=variables_wave6)
    wave7 = process_wave(wave_number=7, data_modules=variables_wave7)
    wave8 = process_wave(wave_number=8, data_modules=variables_wave8)

    waves_list = [wave1, wave2, wave3, wave4, wave5, wave6, wave7, wave8]

    # Drop all nan rows
    for i, df in enumerate(waves_list):
        waves_list[i] = df.dropna(how="all", axis=0, inplace=False)

    merge_wave_datasets(waves_list)


def merge_wave_datasets(wave_datasets):
    # Combine the data frames in wave_datasets into one data frame
    combined_data = pd.concat(wave_datasets, axis=0, ignore_index=True)

    # Filter out rows where the 'int_year' column is not equal to -9
    combined_data = combined_data[combined_data["int_year"] != -9]

    # Sort the data frame by 'mergeid' and 'int_year'
    return combined_data.sort_values(by=["mergeid", "int_year"])


def process_wave(wave_number, data_modules):
    wave_data = {}

    for module in data_modules:
        module_file = (
            SRC / f"data/sharew{wave_number}/sharew{wave_number}_rel8-0-0_{module}.dta"
        )

        wave_module = pd.read_stata(module_file, convert_categoricals=False)
        wave_module = wave_module[wave_module["country"] == 12]

        # Read and filter
        if module == "re" and wave_number == 7:
            #    for var in data_modules["re"]
            #    if var.startswith("sl")
            lookup = {
                f"{var[3:]}": f"{var}"
                for var in data_modules["re"]
                if var.startswith("sl")
            }
        else:
            lookup = {
                "sp009_1sp": "sp009_1",
                "sp009_2sp": "sp009_2",
                "sp009_3sp": "sp009_3",
                "sp019d2sp": "sp019d2",
                "sp019d3sp": "sp019d3",
                "sp019d4sp": "sp019d4",
                "sp019d5sp": "sp019d5",
                "sp019d6sp": "sp019d6",
                "sp019d7sp": "sp019d7",
            }

        # Rename columns using the dictionary
        wave_module = wave_module.rename(columns=lookup)

        module_vars = ["mergeid"] + data_modules[module]

        # Select columns
        wave_module = wave_module[module_vars]

        wave_data[module] = wave_module

    add_wealth_data = "gv_imputations" in data_modules
    merged_data = wave_data["cv_r"]

    data_modules.pop("cv_r")
    data_modules.pop("gv_imputations", None)

    for module_key in data_modules:
        merged_data = merged_data.merge(
            wave_data[module_key],
            on="mergeid",
            how="outer",
        )

    if add_wealth_data:
        merged_data = merged_data.merge(
            wave_data["gv_imputations"],
            on="mergeid",
            how="left",
        )

    merged_data["wave"] = wave_number

    return merged_data


def filter_nested_dict(original_dict, keys_to_remove):
    return {
        key: [value for value in values if value not in keys_to_remove.get(key, [])]
        if key in keys_to_remove
        else values
        for key, values in original_dict.items()
    }
