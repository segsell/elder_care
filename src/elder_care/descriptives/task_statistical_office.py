"""Descriptives from the German statistical office."""
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from elder_care.config import BLD
from elder_care.config import SRC
from pytask import Product


def task_create_type_of_care_by_age_group(
    path_to_data: Path = SRC / "data/statistical_office" / "22421-0001_$F_modified.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "descriptives"
    / "type_of_care_by_age_group.csv",
    path_to_save_fig: Annotated[Path, Product] = BLD
    / "descriptives"
    / "all_care_by_age_group.png",
    path_to_save_fig_by_type_of_care: Annotated[Path, Product] = BLD
    / "descriptives"
    / "type_of_care_by_age_group.png",
):
    """Function to process and modify the dataset as per the specified requirements and
    extend it with additional rows.

    Args:
    path (str): Path to the dataset file.

    Returns:
    DataFrame: The processed and extended dataset.

    """
    # Load the dataset with appropriate encoding
    df_wide = pd.read_csv(path_to_data, encoding="utf-8")

    # Transforming the dataframe from wide format to long format
    df = df_wide.melt(
        id_vars=["Versorgungsart", "sex", "age_group"],
        var_name="year",
        value_name="value",
    )

    # Renaming 'Versorgungsart' to 'type_of_care'
    df = df.rename(columns={"Versorgungsart": "type_of_care"})

    # Removing the "year_" prefix from the 'year' column and converting to int
    df["year"] = df["year"].str.replace("year_", "").astype(int)

    # Updating the 'sex' column
    df["sex"] = (
        df["sex"]
        .replace({"männlich": 1, "weiblich": 2, "Insgesamt": 0, "insgesamt": 0})
        .astype(int)
    )

    # Updating the 'age_group' column with specific replacements
    age_group_replacements = {
        "unter 5 Jahre": 0,
        "5 bis unter 10 Jahre": 1,
        "10 bis unter 15 Jahre": 2,
        "15 bis unter 20 Jahre": 3,
        "20 bis unter 25 Jahre": 4,
        "25 bis unter 30 Jahre": 5,
        "30 bis unter 35 Jahre": 6,
        "35 bis unter 40 Jahre": 7,
        "40 bis unter 45 Jahre": 8,
        "45 bis unter 50 Jahre": 9,
        "50 bis unter 55 Jahre": 10,
        "55 bis unter 60 Jahre": 11,
        "60 bis unter 65 Jahre": 12,
        "65 bis unter 70 Jahre": 13,
        "70 bis unter 75 Jahre": 14,
        "75 bis unter 80 Jahre": 15,
        "80 bis unter 85 Jahre": 16,
        "85 bis unter 90 Jahre": 17,
        "90 bis unter 95 Jahre": 18,
        "95 Jahre und mehr": 19,
        "Insgesamt": 99,
    }
    df["age_group"] = df["age_group"].replace(age_group_replacements)

    # Updating 'type_of_care' column values
    type_of_care_mapping = {
        "Versorgung zu Hause allein durch Angehörige": "only_informal",
        "Versorgung zu Hause mit/durch ambul. Pflegedienste": "informal_and_home_care",
        "Vollstationäre Pflege": "nursing_home",
        "Pflegegrad 1 und teilstationäre Pflege": "care_degree_one_a",
        "Pflegegrad 1 u. nur landesrechtl. bzw. ohne Leist.": "care_degree_one_b",
        "Insgesamt": "total",
    }
    df["type_of_care"] = df["type_of_care"].map(type_of_care_mapping)

    # Renaming 'value' column to 'number'
    df = df.rename(columns={"value": "number"})

    # Adding more lines for 'total_without_care_degree_one' for the years 2019 and 2021
    # Filtering the dataset for the years 2019 and 2021
    total_df_filtered = df[df["year"].isin([2019, 2021])]
    care_degree_one_a_filtered = df[df["type_of_care"] == "care_degree_one_a"]
    care_degree_one_b_filtered = df[df["type_of_care"] == "care_degree_one_b"]

    # Merging and subtracting the values
    # total_without_care_degree_one_df = total_df_filtered.copy()
    # total_without_care_degree_one_df = total_without_care_degree_one_df.set_index(
    #     ["sex", "age_group", "year"]
    # )
    # total_without_care_degree_one_df["number"] -= care_degree_one_a_filtered.set_index(
    #     ["sex", "age_group", "year"]
    # )["number"].fillna(0)
    # total_without_care_degree_one_df["number"] -= care_degree_one_b_filtered.set_index(
    #     ["sex", "age_group", "year"]
    # )["number"].fillna(0)
    # total_without_care_degree_one_df.reset_index(inplace=True)
    # total_without_care_degree_one_df["type_of_care"] = "total_without_care_degree_one"

    # Append these new rows to the original dataset
    # df_extended = df.append(total_without_care_degree_one_df, ignore_index=True)

    # save
    df.to_csv(path_to_save, index=False)

    plot_numbers_by_age_group_all(df, 0, 2021, path_to_save_fig)
    plot_numbers_by_age_group(df, 0, 2021, path_to_save_fig_by_type_of_care)


# ==============================================================================
def plot_share_by_age_group_width(dat, sex, year, save_path):
    """Function to plot the percentage share in the 'number' column by age group for a
    given sex and year. The width of each bin differs by the number of people in each
    respective age group.

    Args:
    dat (DataFrame): The dataset containing the data.
    sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
    year (int): The year for which to plot the data.

    Returns:
    None (Displays and saves the plot).

    """
    age_bins = [
        "<5",
        "5-10",
        "10-15",
        "15-20",
        "20-25",
        "25-30",
        "30-35",
        "35-40",
        "40-45",
        "45-50",
        "50-55",
        "55-60",
        "60-65",
        "65-70",
        "70-75",
        "75-80",
        "80-85",
        "85-90",
        "90-95",
        "95+",
    ]
    df = dat[(dat["sex"] == sex) & (dat["year"] == year)]
    df = df[
        ~df["type_of_care"].isin(["total", "care_degree_one_a", "care_degree_one_b"])
    ]
    df["number"] = df["number"].astype(int)
    # Calculate total for each age group
    total_by_age_group = df.groupby("age_group")["number"].sum()
    # Calculate the widths for the bars
    total_people = total_by_age_group.sum()
    widths = [count / total_people * len(age_bins) for count in total_by_age_group]

    # Calculate the starting x position for each bar
    x_positions = np.cumsum([0] + widths[:-1])

    # Filter the DataFrame based on the provided sex and year

    df["age_group"] = pd.Categorical(df["age_group"], ordered=True, categories=age_bins)
    df = df[df["age_group"] != 99]  # Filter out the "all" age group
    df = df.sort_values("age_group")

    # Create subsets
    only_informal = df[df["type_of_care"] == "only_informal"]
    informal_and_home_care = df[df["type_of_care"] == "informal_and_home_care"]
    nursing_home = df[df["type_of_care"] == "nursing_home"]

    share_combination_care_in_home_care = [
        0.4026772102524974,
        0.4990651870170441,
        0.5113503611840557,
        0.5425971988562102,
        0.5952632679491388,
        0.44028522574716733,
    ]

    # Extract numbers
    _only_informal = only_informal["number"].tolist()
    _informal_and_home_care = informal_and_home_care["number"].tolist()
    _nursing_home = nursing_home["number"].tolist()

    _combination_care = 0.4 * np.array(_informal_and_home_care)
    _only_home_care = 0.6 * np.array(_informal_and_home_care)

    for val in [14, 15, 16, 17, 18, 19]:
        i = val - 14
        _combination_care[val] = (
            share_combination_care_in_home_care[i] * _informal_and_home_care[val]
        )
        _only_home_care[val] = (
            1 - share_combination_care_in_home_care[i]
        ) * _informal_and_home_care[val]

    _only_home_care = _only_home_care.tolist()
    _combination_care = _combination_care.tolist()

    # Calculate percentage share for each category
    def calculate_percentage_share(numbers, total):
        return [
            n / total[i] * 100 if total[i] > 0 else 0
            for i, n in enumerate(numbers)
            if i < 20
        ]

    _only_informal = calculate_percentage_share(_only_informal, total_by_age_group)
    _combination_care = calculate_percentage_share(
        _combination_care, total_by_age_group,
    )
    _only_home_care = calculate_percentage_share(_only_home_care, total_by_age_group)
    _nursing_home = calculate_percentage_share(_nursing_home, total_by_age_group)

    # Plotting
    # Plotting
    plt.figure(figsize=(12, 8))

    # Fixed space between bars
    fixed_space = 0.05

    # Initial position for the first bar
    current_position = 0

    # Adjust bar plotting to include fixed space between bars
    for i, age_bin in enumerate(age_bins):
        plt.bar(
            current_position,
            _only_informal[i],
            width=widths[i],
            label="Pure Informal Care" if i == 0 else "",
            color="lightgreen",
            align="edge",
        )
        plt.bar(
            current_position,
            _combination_care[i],
            bottom=_only_informal[i],
            width=widths[i],
            label="Combination Care" if i == 0 else "",
            color="khaki",
            align="edge",
        )
        plt.bar(
            current_position,
            _only_home_care[i],
            bottom=[i + j for i, j in zip(_only_informal, _combination_care, strict=False)][i],
            width=widths[i],
            label="Formal Home Care" if i == 0 else "",
            color="lightcoral",
            align="edge",
        )
        plt.bar(
            current_position,
            _nursing_home[i],
            bottom=[
                i + j + k
                for i, j, k in zip(_only_informal, _combination_care, _only_home_care, strict=False)
            ][i],
            width=widths[i],
            label="Nursing Home" if i == 0 else "",
            color="steelblue",
            align="edge",
        )

        # Update position for the next bar
        current_position += widths[i] + fixed_space

    # Set x-ticks to the center of each group of bars
    plt.xticks(
        [pos + widths[i] / 2 for i, pos in enumerate(np.cumsum(widths[:-1]))],
        age_bins,
        rotation=45,
    )

    # Rest of the formatting

    plt.xlabel("Age Bin")
    plt.ylabel("Percentage of Care-Dependent People (%)")
    # plt.xticks(
    #     x_positions[:-1] + np.array(widths[:-1]) / 2, age_bins, rotation=90
    # )  # Adjust x-ticks
    plt.tight_layout()  # Adjust layout for better display
    plt.grid(axis="y")
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter())

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [3, 2, 1, 0]

    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Save plot
    plt.savefig(save_path)


def _plot_share_by_age_group(dat, sex, year, save_path):
    """Function to plot the percentage share in the 'number' column by age group for a
    given sex and year.

    Args:
    df (DataFrame): The dataset containing the data.
    sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
    year (int): The year for which to plot the data.

    Returns:
    None (Displays and saves the plot).

    """
    age_bins = [
        "<5",
        "5-10",
        "10-15",
        "15-20",
        "20-25",
        "25-30",
        "30-35",
        "35-40",
        "40-45",
        "45-50",
        "50-55",
        "55-60",
        "60-65",
        "65-70",
        "70-75",
        "75-80",
        "80-85",
        "85-90",
        "90-95",
        "95+",
    ]

    # Filter the DataFrame based on the provided sex and year
    df = dat[(dat["sex"] == sex) & (dat["year"] == year)]
    df = df[
        ~df["type_of_care"].isin(["total", "care_degree_one_a", "care_degree_one_b"])
    ]
    df["number"] = df["number"].astype(int)

    df["age_group"] = pd.Categorical(df["age_group"], ordered=True)
    df = df[df["age_group"] != 99]  # Filter out the "all" age group
    df = df.sort_values("age_group")

    # Create subsets
    only_informal = df[(df["type_of_care"] == "only_informal")]
    informal_and_home_care = df[(df["type_of_care"] == "informal_and_home_care")]
    nursing_home = df[(df["type_of_care"] == "nursing_home")]

    share_combination_care_in_home_care = [
        0.4026772102524974,
        0.4990651870170441,
        0.5113503611840557,
        0.5425971988562102,
        0.5952632679491388,
        0.44028522574716733,
    ]

    _only_informal = only_informal["number"].tolist()
    _informal_and_home_care = informal_and_home_care["number"].tolist()
    _nursing_home = nursing_home["number"].tolist()

    _combination_care = 0.4 * np.array(_informal_and_home_care)
    _only_home_care = 0.6 * np.array(_informal_and_home_care)

    for val in [14, 15, 16, 17, 18, 19]:
        i = val - 14
        _combination_care[val] = (
            share_combination_care_in_home_care[i] * _informal_and_home_care[val]
        )
        _only_home_care[val] = (
            1 - share_combination_care_in_home_care[i]
        ) * _informal_and_home_care[val]

    _only_home_care = _only_home_care.tolist()
    _combination_care = _combination_care.tolist()

    # Calculate total for each age group
    total_by_age_group = df.groupby("age_group")["number"].sum()

    # Calculate percentage share for each category
    def calculate_percentage_share(numbers, total):
        return [
            n / total[i] * 100 if total[i] > 0 else 0 for i, n in enumerate(numbers)
        ]

    _only_informal = calculate_percentage_share(_only_informal, total_by_age_group)
    _combination_care = calculate_percentage_share(
        _combination_care,
        total_by_age_group,
    )
    _only_home_care = calculate_percentage_share(_only_home_care, total_by_age_group)
    _nursing_home = calculate_percentage_share(_nursing_home, total_by_age_group)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(
        age_bins,
        _only_informal,
        label="Pure Informal Care",
        color="lightgreen",
    )
    plt.bar(
        age_bins,
        _combination_care,
        bottom=_only_informal,
        label="Combination Care",
        color="khaki",
    )
    plt.bar(
        age_bins,
        _only_home_care,
        bottom=[i + j for i, j in zip(_only_informal, _combination_care, strict=False)],
        label="Formal Home Care",
        color="lightcoral",
    )
    plt.bar(
        age_bins,
        _nursing_home,
        bottom=[
            i + j + k
            for i, j, k in zip(
                _only_informal,
                _combination_care,
                _only_home_care,
                strict=False,
            )
        ],
        label="Nursing Home",
        color="steelblue",
    )

    plt.xlabel("Age Bin")
    plt.ylabel("Percentage of Care-Dependent People (%)")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout for better display
    plt.grid(axis="y")
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter())

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [3, 2, 1, 0]

    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Save plot
    plt.savefig(save_path)


def plot_numbers_by_age_group_all(dat, sex, year, save_path):
    """Function to plot the numbers in the 'number' column by age group for a given sex
    and year.

    Args:
    df (DataFrame): The dataset containing the data.
    sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
    year (int): The year for which to plot the data.

    Returns:
    None (Displays the plot).

    """
    age_bins = [
        "<5",
        "5-10",
        "10-15",
        "15-20",
        "20-25",
        "25-30",
        "30-35",
        "35-40",
        "40-45",
        "45-50",
        "50-55",
        "55-60",
        "60-65",
        "65-70",
        "70-75",
        "75-80",
        "80-85",
        "85-90",
        "90-95",
        "95+",
    ]

    # Filter the DataFrame based on the provided sex and year
    df = dat[(dat["sex"] == sex) & (dat["year"] == year)]
    df = df[
        ~df["type_of_care"].isin(["total", "care_degree_one_a", "care_degree_one_b"])
    ]
    df["number"] = df["number"].astype(int)

    df["age_group"] = pd.Categorical(df["age_group"], ordered=True)
    # Filter out the "all" age group
    df = df[df["age_group"] != 99]

    # Sort the DataFrame based on age_group
    df = df.sort_values("age_group")

    # Create subsets
    only_informal = df[(df["type_of_care"] == "only_informal")]
    informal_and_home_care = df[(df["type_of_care"] == "informal_and_home_care")]
    nursing_home = df[(df["type_of_care"] == "nursing_home")]

    share_combination_care_in_home_care = [
        0.4026772102524974,
        0.4990651870170441,
        0.5113503611840557,
        0.5425971988562102,
        0.5952632679491388,
        0.44028522574716733,
    ]

    _only_informal = only_informal["number"].tolist()
    _informal_and_home_care = informal_and_home_care["number"].tolist()
    _nursing_home = nursing_home["number"].tolist()

    _combination_care = 0.4 * np.array(_informal_and_home_care)
    _only_home_care = 0.6 * np.array(_informal_and_home_care)

    for val in [14, 15, 16, 17, 18, 19]:
        i = val - 14
        _combination_care[val] = (
            share_combination_care_in_home_care[i] * _informal_and_home_care[val]
        )
        _only_home_care[val] = (
            1 - share_combination_care_in_home_care[i]
        ) * _informal_and_home_care[val]

    _only_home_care = _only_home_care.tolist()
    _combination_care = _combination_care.tolist()

    color = "steelblue"
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(
        age_bins,
        _only_informal,
        label="Pure Informal Care",
        # color="lightgreen",
        color=color,
    )
    plt.bar(
        age_bins,
        _combination_care,
        bottom=_only_informal,
        label="Combination Care",
        # color="green",
        # color="moccasin",
        color=color,
    )
    plt.bar(
        age_bins,
        _only_home_care,
        bottom=[i + j for i, j in zip(_only_informal, _combination_care, strict=False)],
        label="Formal Home Care",
        color=color,
    )
    plt.bar(
        age_bins,
        _nursing_home,
        bottom=[
            i + j + k
            for i, j, k in zip(
                _only_informal,
                _combination_care,
                _only_home_care,
                strict=False,
            )
        ],
        label="Nursing Home",
        color=color,
    )

    plt.xlabel("Age Bin")
    plt.ylabel("Number of Care-Dependent People")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout for better display
    # plt.ylim(0, 0.13)  # Set y-axis range from 0 to 15%
    plt.grid(axis="y")
    # Add thousand separators (commas) to y-axis label ticks
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [3, 2, 1, 0]

    # add legend to plot
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Save plot
    plt.savefig(save_path)


def plot_numbers_by_age_group(dat, sex, year, save_path):
    """Function to plot the numbers in the 'number' column by age group for a given sex
    and year.

    Args:
    df (DataFrame): The dataset containing the data.
    sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
    year (int): The year for which to plot the data.

    Returns:
    None (Displays the plot).

    """
    age_bins = [
        "<5",
        "5-10",
        "10-15",
        "15-20",
        "20-25",
        "25-30",
        "30-35",
        "35-40",
        "40-45",
        "45-50",
        "50-55",
        "55-60",
        "60-65",
        "65-70",
        "70-75",
        "75-80",
        "80-85",
        "85-90",
        "90-95",
        "95+",
    ]

    # Filter the DataFrame based on the provided sex and year
    df = dat[(dat["sex"] == sex) & (dat["year"] == year)]
    df = df[
        ~df["type_of_care"].isin(["total", "care_degree_one_a", "care_degree_one_b"])
    ]
    df["number"] = df["number"].astype(int)

    df["age_group"] = pd.Categorical(df["age_group"], ordered=True)
    # Filter out the "all" age group
    df = df[df["age_group"] != 99]

    # Sort the DataFrame based on age_group
    df = df.sort_values("age_group")

    # Create subsets
    only_informal = df[(df["type_of_care"] == "only_informal")]
    informal_and_home_care = df[(df["type_of_care"] == "informal_and_home_care")]
    nursing_home = df[(df["type_of_care"] == "nursing_home")]

    share_combination_care_in_home_care = [
        0.4026772102524974,
        0.4990651870170441,
        0.5113503611840557,
        0.5425971988562102,
        0.5952632679491388,
        0.44028522574716733,
    ]

    _only_informal = only_informal["number"].tolist()
    _informal_and_home_care = informal_and_home_care["number"].tolist()
    _nursing_home = nursing_home["number"].tolist()

    _combination_care = 0.4 * np.array(_informal_and_home_care)
    _only_home_care = 0.6 * np.array(_informal_and_home_care)

    for val in [14, 15, 16, 17, 18, 19]:
        i = val - 14
        _combination_care[val] = (
            share_combination_care_in_home_care[i] * _informal_and_home_care[val]
        )
        _only_home_care[val] = (
            1 - share_combination_care_in_home_care[i]
        ) * _informal_and_home_care[val]

    _only_home_care = _only_home_care.tolist()
    _combination_care = _combination_care.tolist()

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(
        age_bins,
        _only_informal,
        label="Pure Informal Care",
        color="lightgreen",
        # color="sandybrown",
        # color="moccasin",
    )
    plt.bar(
        age_bins,
        _combination_care,
        bottom=_only_informal,
        label="Combination Care",
        # color="green",
        # color="moccasin",
        color="khaki",
    )
    plt.bar(
        age_bins,
        _only_home_care,
        bottom=[i + j for i, j in zip(_only_informal, _combination_care, strict=False)],
        label="Formal Home Care",
        color="lightcoral",
    )
    plt.bar(
        age_bins,
        _nursing_home,
        bottom=[
            i + j + k
            for i, j, k in zip(
                _only_informal,
                _combination_care,
                _only_home_care,
                strict=False,
            )
        ],
        label="Nursing Home",
        color="steelblue",
    )

    plt.xlabel("Age Bin")
    plt.ylabel("Number of Care-Dependent People")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout for better display
    # plt.ylim(0, 0.13)  # Set y-axis range from 0 to 15%
    plt.grid(axis="y")
    # Add thousand separators (commas) to y-axis label ticks
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [3, 2, 1, 0]

    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Save plot
    plt.savefig(save_path)


def check_age_group_sums(df, year, sex, type_of_care):
    """Function to check if the numbers of different age groups (excluding 'total') sum
    to the 'total' age group for a given year, sex, and 'type_of_care' category.

    Args:
    df (DataFrame): The dataset containing the data.
    year (int): The year for which to perform the check.
    sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
    type_of_care (str): The 'type_of_care' category to check.

    Returns:
    bool: True if the sums match, False otherwise.

    """
    # Filter the DataFrame based on the provided year, sex, and type_of_care
    filtered_df = df[
        (df["year"] == year) & (df["sex"] == sex) & (df["type_of_care"] == type_of_care)
    ]

    # Filter for age groups that are not 'total'
    age_groups = filtered_df[filtered_df["age_group"] != "total"]

    # Calculate the sum of numbers for age groups (excluding 'total')
    age_group_sum = age_groups["number"].sum()

    # Filter for 'total' age group
    total_age_group = filtered_df[filtered_df["age_group"] == "total"]

    # Get the number for the 'total' age group
    total_number = total_age_group["number"].values[0]

    # Check if the sums match
    return age_group_sum == total_number
