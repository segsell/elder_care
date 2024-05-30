"""Functions to evaluate the fit of the simulated moments to the data moments."""

import pytask
import numpy as np
import jax.numpy as jnp
import pandas as pd

import matplotlib.pyplot as plt

from elder_care.config import BLD

SIM_MOMENTS = np.array(
    [
        3.97190000e-01,
        3.43500000e-01,
        4.43190000e-01,
        3.57900000e-01,
        4.04360000e-01,
        3.50200000e-01,
        3.52920000e-01,
        3.24730000e-01,
        3.27700000e-01,
        3.21960000e-01,
        3.30890000e-01,
        3.42060000e-01,
        3.75110000e-01,
        4.26360000e-01,
        4.76910000e-01,
        5.29050000e-01,
        5.75760000e-01,
        6.41770000e-01,
        7.20600000e-01,
        7.97070000e-01,
        8.66850000e-01,
        9.15300000e-01,
        9.57800000e-01,
        9.88870000e-01,
        9.94840000e-01,
        9.97420000e-01,
        9.99170000e-01,
        9.99750000e-01,
        9.99910000e-01,
        9.99970000e-01,
        3.48790000e-01,
        2.83600000e-01,
        1.47780000e-01,
        2.04130000e-01,
        1.78730000e-01,
        2.39440000e-01,
        2.58270000e-01,
        3.22970000e-01,
        3.50650000e-01,
        3.75720000e-01,
        3.67790000e-01,
        3.64090000e-01,
        3.47470000e-01,
        3.35570000e-01,
        3.16690000e-01,
        2.92040000e-01,
        2.66870000e-01,
        2.22990000e-01,
        1.60160000e-01,
        1.02670000e-01,
        5.40200000e-02,
        2.68100000e-02,
        4.08000000e-03,
        3.40000000e-04,
        1.20000000e-04,
        3.00000000e-05,
        1.00000000e-05,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.54020000e-01,
        3.72900000e-01,
        4.09030000e-01,
        4.37970000e-01,
        4.16910000e-01,
        4.10360000e-01,
        3.88810000e-01,
        3.52300000e-01,
        3.21650000e-01,
        3.02320000e-01,
        3.01320000e-01,
        2.93850000e-01,
        2.77420000e-01,
        2.38070000e-01,
        2.06400000e-01,
        1.78910000e-01,
        1.57370000e-01,
        1.35240000e-01,
        1.19240000e-01,
        1.00260000e-01,
        7.91300000e-02,
        5.78900000e-02,
        3.81200000e-02,
        1.07900000e-02,
        5.04000000e-03,
        2.55000000e-03,
        8.20000000e-04,
        2.50000000e-04,
        9.00000000e-05,
        3.00000000e-05,
        -2.54494592e00,
        1.15201183e-01,
        -1.00526114e-03,
        1.17432400e-01,
        5.41931660e-02,
        6.46730981e-02,
        4.43863500e-01,
        3.55593712e-01,
        4.99436647e-01,
        8.48281648e-01,
        4.93794898e-01,
        4.89096023e-02,
        3.70494067e-01,
        3.52409871e-01,
        3.49844331e-01,
        3.53738916e-01,
        1.28076821e-01,
        2.18638972e-04,
        3.69276746e-01,
        3.56876224e-01,
        1.56642350e-01,
        8.13673665e-03,
        0.00000000e00,
        0.00000000e00,
        2.24545995e-01,
        2.96016463e-01,
        2.30109585e-01,
        2.19936597e-01,
        2.37119830e-01,
        2.19185570e-02,
        3.70494067e-01,
        3.52409871e-01,
        3.49844331e-01,
        3.53738916e-01,
        1.28076821e-01,
        2.18638972e-04,
        4.04959938e-01,
        3.51573666e-01,
        4.20046084e-01,
        4.26324487e-01,
        2.97697548e-01,
        2.04427439e-02,
        2.07662217e-04,
        6.83500718e-01,
        4.57518800e-02,
        2.70539740e-01,
        8.72692509e-01,
        7.22552390e-02,
        5.50522523e-02,
        3.72650469e-01,
        3.40938009e-01,
        2.86411522e-01,
        3.41230153e-01,
        2.38036367e-01,
        4.20733480e-01,
        7.49798549e-01,
        2.50201451e-01,
        3.73215871e-01,
        6.26784129e-01,
        9.10612799e-01,
        8.93872010e-02,
        7.63638570e-01,
        2.36361430e-01,
        6.42070696e-01,
        3.57929304e-01,
        3.71293813e-01,
        6.28706187e-01,
        8.70325833e-01,
        1.29674167e-01,
        7.51462716e-01,
        2.48537284e-01,
    ],
)

_SIM_MOMENTS = np.array(
    [
        4.03420000e-01,
        3.45380000e-01,
        4.41540000e-01,
        3.61050000e-01,
        4.01410000e-01,
        3.52200000e-01,
        3.53010000e-01,
        3.26690000e-01,
        3.23900000e-01,
        3.09630000e-01,
        3.09830000e-01,
        3.15190000e-01,
        3.40780000e-01,
        3.92640000e-01,
        4.45390000e-01,
        4.88950000e-01,
        5.21590000e-01,
        5.56790000e-01,
        6.02770000e-01,
        6.73000000e-01,
        7.71960000e-01,
        8.48610000e-01,
        9.27740000e-01,
        9.79110000e-01,
        9.90640000e-01,
        9.95780000e-01,
        9.98400000e-01,
        9.99560000e-01,
        9.99840000e-01,
        9.99970000e-01,
        3.42510000e-01,
        2.80120000e-01,
        1.45840000e-01,
        1.97840000e-01,
        1.74560000e-01,
        2.26690000e-01,
        2.48140000e-01,
        3.15510000e-01,
        3.57690000e-01,
        3.86950000e-01,
        3.84780000e-01,
        3.84580000e-01,
        3.72190000e-01,
        3.63650000e-01,
        3.49270000e-01,
        3.31030000e-01,
        3.20510000e-01,
        3.03300000e-01,
        2.67150000e-01,
        2.08300000e-01,
        1.20770000e-01,
        6.07600000e-02,
        8.19000000e-03,
        7.40000000e-04,
        3.60000000e-04,
        3.00000000e-05,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.54070000e-01,
        3.74500000e-01,
        4.12620000e-01,
        4.41110000e-01,
        4.24030000e-01,
        4.21110000e-01,
        3.98850000e-01,
        3.57800000e-01,
        3.18410000e-01,
        3.03420000e-01,
        3.05390000e-01,
        3.00230000e-01,
        2.87030000e-01,
        2.43710000e-01,
        2.05340000e-01,
        1.80020000e-01,
        1.57900000e-01,
        1.39910000e-01,
        1.30080000e-01,
        1.18700000e-01,
        1.07270000e-01,
        9.06300000e-02,
        6.40700000e-02,
        2.01500000e-02,
        9.00000000e-03,
        4.19000000e-03,
        1.60000000e-03,
        4.40000000e-04,
        1.60000000e-04,
        3.00000000e-05,
        -2.71071480e00,
        1.20976132e-01,
        -1.04914479e-03,
        1.07026740e-01,
        3.09536591e-02,
        4.83311217e-02,
        4.45148084e-01,
        3.44254292e-01,
        4.60257357e-01,
        7.60210342e-01,
        4.55801030e-01,
        3.51300913e-02,
        3.90908460e-01,
        3.65366962e-01,
        3.68445127e-01,
        3.80453622e-01,
        1.68027716e-01,
        9.54562810e-05,
        3.83678507e-01,
        3.79986122e-01,
        1.67012539e-01,
        4.47745130e-03,
        0.00000000e00,
        0.00000000e00,
        2.34713152e-01,
        3.12216000e-01,
        2.28071076e-01,
        2.12230072e-01,
        1.96819556e-01,
        1.84230622e-02,
        3.90908460e-01,
        3.65366962e-01,
        3.68445127e-01,
        3.80453622e-01,
        1.68027716e-01,
        9.54562810e-05,
        3.74378388e-01,
        3.22417038e-01,
        4.03483797e-01,
        4.07316306e-01,
        3.13224234e-01,
        3.06414662e-02,
        2.79214493e-04,
        9.32254517e-01,
        1.41887295e-02,
        5.32775392e-02,
        8.53062172e-01,
        8.83842805e-02,
        5.85535471e-02,
        4.00197443e-01,
        3.27189457e-01,
        2.72613100e-01,
        3.39431065e-01,
        2.42087625e-01,
        4.18481311e-01,
        7.28489232e-01,
        2.71510768e-01,
        3.79333008e-01,
        6.20666992e-01,
        9.50870245e-01,
        4.91297552e-02,
        8.21523466e-01,
        1.78476534e-01,
        6.20851890e-01,
        3.79148110e-01,
        2.64530428e-01,
        7.35469572e-01,
        9.17568668e-01,
        8.24313318e-02,
        7.11274245e-01,
        2.88725755e-01,
    ],
)


@pytask.mark.skip()
def task_model_fit(path_to_empirical_moments=BLD / "moments" / "empirical_moments.csv"):
    """Evaluate the fit of the simulated moments to the data moments."""

    empirical_moments = pd.read_csv(path_to_empirical_moments, index_col=0).iloc[:, 0]

    simulated_moments = add_index_to_simulated_moments(
        empirical_moments=empirical_moments, simulated_moments=SIM_MOMENTS
    )

    comparison = pd.concat([empirical_moments, simulated_moments], axis=1)
    comparison.columns = ["empirical", "simulated"]

    plot_labor_shares(comparison, "part_time")
    plot_labor_shares(comparison, "full_time")
    plot_labor_shares(comparison, "not_working")
    breakpoint()


def add_index_to_simulated_moments(
    empirical_moments: pd.DataFrame, simulated_moments: np.ndarray
) -> pd.DataFrame:
    """Convert simulated moments to pandas DataFrame.

    Args:
        empirical_moments (pd.DataFrame): Empirical moments.
        simulated_moments (np.array): Simulated moments.

    Returns:
        pd.DataFrame: Simulated moments as pandas DataFrame.
    """
    idx = dict(enumerate(empirical_moments.index, start=0))

    df_sim = pd.Series(simulated_moments, index=idx.values(), name="moment")

    return df_sim


def plot_labor_shares(df, working_status):
    """
    Plots the labor shares of empirical and simulated moments against age for a specified working status.

    Parameters:
        df (pandas.DataFrame): DataFrame with the index as "working_status_age_x" and two columns 'empirical' and 'simulated'.
        working_status (str): Working status to filter the DataFrame on.
    """
    # Prepare the data: filter rows for the specified working status and extract age
    pattern = f"{working_status}_age_"
    filtered_df = df[df.index.str.startswith(pattern)]
    # ages = filtered_df.index.str.replace(pattern, "").astype(
    #     int
    # )  # Extract and convert age part of the index to integer
    ages = np.array(
        [
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
        ]
    )

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(ages, filtered_df["empirical"][:30], label="Empirical Moments", marker="o")
    plt.plot(
        ages,
        filtered_df["simulated"][:30],
        label="Simulated Moments",
        linestyle="--",
        marker="x",
    )

    plt.title(f"Labor Shares - {working_status.capitalize()}")
    plt.xlabel("Age")
    plt.ylabel("Labor Shares")
    plt.xticks(
        ages, rotation=45
    )  # Set x-ticks to be the ages, rotate for better visibility
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ====================================================================================


def plot_working_status(empirical, simulated, working_status):
    # Extract ages and corresponding shares from empirical
    ages_empirical = [
        int(idx.split("_")[-1])
        for idx in empirical.index
        if idx.startswith(f"{working_status}_age_")
    ]
    shares_empirical = empirical.loc[
        empirical.index.str.startswith(f"{working_status}_age_")
    ].values.ravel()

    # Extract ages and corresponding shares from simulated
    ages_simulated = [
        int(idx.split("_")[-1])
        for idx in simulated.index
        if idx.startswith(f"{working_status}_age_")
    ]
    shares_simulated = simulated.loc[
        simulated.index.str.startswith(f"{working_status}_age_")
    ].values.ravel()

    # Create DataFrames for easier plotting
    df_empirical = pd.DataFrame(
        {
            "Age": ages_empirical,
            f"Share of {working_status.capitalize()} (Empirical)": shares_empirical,
        }
    )
    df_simulated = pd.DataFrame(
        {
            "Age": ages_simulated,
            f"Share of {working_status.capitalize()} (Simulated)": shares_simulated,
        }
    )

    # Sort the DataFrames by Age for better visualization
    df_empirical = df_empirical.sort_values(by="Age")
    df_simulated = df_simulated.sort_values(by="Age")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_empirical["Age"],
        df_empirical[f"Share of {working_status.capitalize()} (Empirical)"],
        marker="o",
        linestyle="-",
        label="Empirical",
    )
    plt.plot(
        df_simulated["Age"],
        df_simulated[f"Share of {working_status.capitalize()} (Simulated)"],
        marker="s",
        linestyle="--",
        label="Simulated",
    )
    plt.title(f"Share of {working_status.capitalize()} People by Age")
    plt.xlabel("Age")
    plt.ylabel(f"Share of {working_status.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.show()
