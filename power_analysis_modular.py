# %%
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import genfromtxt
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr, ttest_1samp, ttest_ind

warnings.filterwarnings("ignore")

CONFIG = {
    "data_path": Path(
        "/Users/karahanyilmazer/Coding/Python/Neuro/data/charite-neurofeedback/ID285/"
    ),
    "days": ["day1", "day2", "day3", "day4", "day6", "day7", "day8"],
    "num_trials": 5,
    "power_start": 200,
    "power_end": 4000,
    "channel": "Cz",
    "sampling_rate": 500,
    "results_dir": Path("results/analysis_plots"),
    "colormap": "viridis",
}


# %%
def load_power_file(fname):
    try:
        if fname.suffix == ".txt":
            power = genfromtxt(fname, delimiter="\n", skip_header=1, dtype=str)
            power_arr = [float(re.split("b |,", row)[1]) for row in power]
            return np.array(power_arr)
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        return None


def collect_power_data(config):
    records = []
    for d_idx, day in enumerate(config["days"]):
        day_path = config["data_path"] / day
        if not day_path.exists():
            print(f"Warning: {day_path} does not exist.")
            continue
        files = os.listdir(day_path)
        files_arr = [file.split("_")[0] for file in files]
        files_idx = np.argsort(files_arr)
        for i, f_idx in enumerate(files_idx):
            if i >= config["num_trials"]:
                print(f"Skipping {files[f_idx]}")
                break
            fname_power = day_path / files[f_idx]
            power_arr = load_power_file(fname_power)
            if power_arr is None:
                continue
            trimmed = power_arr[config["power_start"] : config["power_end"]]
            records.append({"day": day, "trial": i + 1, "power": trimmed})
    df = pd.DataFrame(records)
    return df


def plot_trial_timeseries(df, config):
    n_days = len(config["days"])
    n_trials = config["num_trials"]
    fig, axes = plt.subplots(
        n_days, n_trials, figsize=(18, 18), sharex="col", sharey="all"
    )
    fig.suptitle(
        f"Alpha Power Time Series for Channel {config['channel']}", fontsize=16
    )

    # Store regression parameters for each trial
    regression_params = []

    # Plot data and fit regression manually
    for idx, row in df.iterrows():
        d_idx = config["days"].index(row["day"])
        t_idx = row["trial"] - 1
        ax = axes[d_idx, t_idx]
        x = np.arange(len(row["power"]))
        y = row["power"]
        # Scatter plot
        ax.scatter(x, y, color="black", s=10)
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)  # degree 1 polynomial (linear)
        slope, intercept = coeffs
        # Regression line
        y_pred = slope * x + intercept
        ax.plot(x, y_pred, color="red")
        # Store parameters
        regression_params.append(
            {
                "day": row["day"],
                "trial": row["trial"],
                "slope": slope,
                "intercept": intercept,
            }
        )

    # Titles for top row
    for t in range(n_trials):
        axes[0, t].set_title(f"Trial {t+1}")

    # Ylabels for rightmost column
    for d in range(n_days):
        axes[d, 0].set_ylabel("Alpha Power")
        axes[d, n_trials - 1].yaxis.set_label_position("right")
        axes[d, n_trials - 1].set_ylabel(f"Day {d+1}", rotation=270, labelpad=20)

    # Only bottom row gets xlabel
    for t in range(n_trials):
        axes[n_days - 1, t].set_xlabel("Time")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    plt.savefig(config["results_dir"] / "trial_timeseries.png")
    plt.show()

    return regression_params

    return regression_params


def perform_statistical_tests(regression_params, config):
    """
    Perform comprehensive statistical tests on slope and intercept data
    """
    # Convert to DataFrame for easier analysis
    params_df = pd.DataFrame(regression_params)

    print("=" * 80)
    print("STATISTICAL ANALYSIS OF REGRESSION PARAMETERS")
    print("=" * 80)
    print(f"Total number of trials analyzed: {len(params_df)}")
    print(f"Number of days: {len(config['days'])}")
    print(f"Trials per day: {config['num_trials']}")
    print("\n")

    # Basic descriptive statistics
    print("DESCRIPTIVE STATISTICS")
    print("-" * 40)
    print("Slopes:")
    print(f"  Mean: {params_df['slope'].mean():.6f}")
    print(f"  Std:  {params_df['slope'].std():.6f}")
    print(f"  Min:  {params_df['slope'].min():.6f}")
    print(f"  Max:  {params_df['slope'].max():.6f}")
    print(f"  Median: {params_df['slope'].median():.6f}")

    print("\nIntercepts:")
    print(f"  Mean: {params_df['intercept'].mean():.6f}")
    print(f"  Std:  {params_df['intercept'].std():.6f}")
    print(f"  Min:  {params_df['intercept'].min():.6f}")
    print(f"  Max:  {params_df['intercept'].max():.6f}")
    print(f"  Median: {params_df['intercept'].median():.6f}")

    # Test 1: One-sample t-test (slopes significantly different from 0?)
    print("\n" + "=" * 80)
    print("TEST 1: ONE-SAMPLE T-TEST (Slopes vs 0)")
    print("-" * 40)
    slope_ttest = ttest_1samp(params_df["slope"], 0)
    print(f"H0: Mean slope = 0 (no trend in alpha power over time)")
    print(f"H1: Mean slope ≠ 0 (significant trend exists)")
    print(f"t-statistic: {slope_ttest.statistic:.4f}")
    print(f"p-value: {slope_ttest.pvalue:.6f}")
    print(
        f"Result: {'SIGNIFICANT' if slope_ttest.pvalue < 0.05 else 'NOT SIGNIFICANT'} at α = 0.05"
    )

    # Test 2: ANOVA - Compare slopes across days
    print("\n" + "=" * 80)
    print("TEST 2: ONE-WAY ANOVA (Slopes across days)")
    print("-" * 40)
    slope_groups = [
        params_df[params_df["day"] == day]["slope"].values for day in config["days"]
    ]
    slope_groups = [
        group for group in slope_groups if len(group) > 0
    ]  # Remove empty groups

    if len(slope_groups) > 1:
        f_stat, p_value = f_oneway(*slope_groups)
        print(f"H0: Mean slopes are equal across all days")
        print(f"H1: At least one day has different mean slope")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(
            f"Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α = 0.05"
        )
    else:
        print("Not enough groups for ANOVA")

    # Test 3: ANOVA - Compare intercepts across days
    print("\n" + "=" * 80)
    print("TEST 3: ONE-WAY ANOVA (Intercepts across days)")
    print("-" * 40)
    intercept_groups = [
        params_df[params_df["day"] == day]["intercept"].values for day in config["days"]
    ]
    intercept_groups = [
        group for group in intercept_groups if len(group) > 0
    ]  # Remove empty groups

    if len(intercept_groups) > 1:
        f_stat, p_value = f_oneway(*intercept_groups)
        print(f"H0: Mean intercepts are equal across all days")
        print(f"H1: At least one day has different mean intercept")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(
            f"Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α = 0.05"
        )
    else:
        print("Not enough groups for ANOVA")

    # Test 4: Correlation between slopes and intercepts
    print("\n" + "=" * 80)
    print("TEST 4: CORRELATION ANALYSIS")
    print("-" * 40)
    pearson_corr, pearson_p = pearsonr(params_df["slope"], params_df["intercept"])
    spearman_corr, spearman_p = spearmanr(params_df["slope"], params_df["intercept"])

    print(f"Correlation between slopes and intercepts:")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p = {pearson_p:.6f})")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p = {spearman_p:.6f})")
    print(
        f"  Pearson result: {'SIGNIFICANT' if pearson_p < 0.05 else 'NOT SIGNIFICANT'} at α = 0.05"
    )
    print(
        f"  Spearman result: {'SIGNIFICANT' if spearman_p < 0.05 else 'NOT SIGNIFICANT'} at α = 0.05"
    )

    # Test 5: Day progression analysis (linear trend across days)
    print("\n" + "=" * 80)
    print("TEST 5: DAY PROGRESSION ANALYSIS")
    print("-" * 40)

    # Calculate mean slope and intercept per day
    daily_stats = (
        params_df.groupby("day")
        .agg({"slope": ["mean", "std", "count"], "intercept": ["mean", "std", "count"]})
        .round(6)
    )

    print("Daily statistics:")
    print(daily_stats)

    # Test for linear trend in daily mean slopes
    day_nums = list(range(1, len(config["days"]) + 1))
    daily_mean_slopes = [
        params_df[params_df["day"] == day]["slope"].mean() for day in config["days"]
    ]
    daily_mean_intercepts = [
        params_df[params_df["day"] == day]["intercept"].mean() for day in config["days"]
    ]

    # Remove NaN values
    valid_indices = [
        i
        for i, (s, ic) in enumerate(zip(daily_mean_slopes, daily_mean_intercepts))
        if not (np.isnan(s) or np.isnan(ic))
    ]

    if len(valid_indices) > 2:
        clean_day_nums = [day_nums[i] for i in valid_indices]
        clean_slopes = [daily_mean_slopes[i] for i in valid_indices]
        clean_intercepts = [daily_mean_intercepts[i] for i in valid_indices]

        slope_trend_corr, slope_trend_p = pearsonr(clean_day_nums, clean_slopes)
        intercept_trend_corr, intercept_trend_p = pearsonr(
            clean_day_nums, clean_intercepts
        )

        print(f"\nTrend analysis across days:")
        print(f"  Slope trend: r = {slope_trend_corr:.4f}, p = {slope_trend_p:.6f}")
        print(
            f"  Intercept trend: r = {intercept_trend_corr:.4f}, p = {intercept_trend_p:.6f}"
        )
        print(
            f"  Slope trend result: {'SIGNIFICANT' if slope_trend_p < 0.05 else 'NOT SIGNIFICANT'}"
        )
        print(
            f"  Intercept trend result: {'SIGNIFICANT' if intercept_trend_p < 0.05 else 'NOT SIGNIFICANT'}"
        )

    return params_df


def plot_statistical_results(params_df, config):
    """
    Create visualizations for the statistical analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Statistical Analysis of Regression Parameters", fontsize=16)

    # Plot 1: Slope distribution
    axes[0, 0].hist(
        params_df["slope"], bins=20, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[0, 0].axvline(
        params_df["slope"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {params_df["slope"].mean():.6f}',
    )
    axes[0, 0].axvline(0, color="orange", linestyle="-", label="Zero line")
    axes[0, 0].set_xlabel("Slope")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Slopes")
    axes[0, 0].legend()

    # Plot 2: Intercept distribution
    axes[0, 1].hist(
        params_df["intercept"],
        bins=20,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
    )
    axes[0, 1].axvline(
        params_df["intercept"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {params_df["intercept"].mean():.2f}',
    )
    axes[0, 1].set_xlabel("Intercept")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Intercepts")
    axes[0, 1].legend()

    # Plot 3: Slope vs Intercept scatter
    axes[0, 2].scatter(params_df["slope"], params_df["intercept"], alpha=0.6)
    axes[0, 2].set_xlabel("Slope")
    axes[0, 2].set_ylabel("Intercept")
    axes[0, 2].set_title("Slope vs Intercept")

    # Add correlation line
    z = np.polyfit(params_df["slope"], params_df["intercept"], 1)
    p = np.poly1d(z)
    axes[0, 2].plot(params_df["slope"], p(params_df["slope"]), "r--", alpha=0.8)

    # Plot 4: Slopes by day (boxplot)
    params_df.boxplot(column="slope", by="day", ax=axes[1, 0])
    axes[1, 0].set_title("Slopes by Day")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Slope")

    # Plot 5: Intercepts by day (boxplot)
    params_df.boxplot(column="intercept", by="day", ax=axes[1, 1])
    axes[1, 1].set_title("Intercepts by Day")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_ylabel("Intercept")

    # Plot 6: Daily means trend
    daily_means = params_df.groupby("day").agg({"slope": "mean", "intercept": "mean"})
    day_nums = list(range(1, len(daily_means) + 1))

    ax2 = axes[1, 2]
    ax2.plot(day_nums, daily_means["slope"], "bo-", label="Slope", markersize=8)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Mean Slope", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    ax3 = ax2.twinx()
    ax3.plot(day_nums, daily_means["intercept"], "ro-", label="Intercept", markersize=8)
    ax3.set_ylabel("Mean Intercept", color="r")
    ax3.tick_params(axis="y", labelcolor="r")

    axes[1, 2].set_title("Daily Mean Trends")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        config["results_dir"] / "statistical_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def pairwise_day_comparisons(params_df, config):
    """
    Perform pairwise t-tests between days for both slopes and intercepts
    """
    from itertools import combinations

    from scipy.stats import ttest_ind

    print("\n" + "=" * 80)
    print("PAIRWISE DAY COMPARISONS")
    print("=" * 80)

    days_with_data = params_df["day"].unique()

    print("SLOPE COMPARISONS:")
    print("-" * 40)

    slope_results = []
    for day1, day2 in combinations(days_with_data, 2):
        slopes1 = params_df[params_df["day"] == day1]["slope"]
        slopes2 = params_df[params_df["day"] == day2]["slope"]

        if len(slopes1) > 0 and len(slopes2) > 0:
            t_stat, p_val = ttest_ind(slopes1, slopes2)
            slope_results.append(
                {
                    "comparison": f"{day1} vs {day2}",
                    "mean1": slopes1.mean(),
                    "mean2": slopes2.mean(),
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                }
            )

            print(f"{day1} vs {day2}:")
            print(f"  Mean slopes: {slopes1.mean():.6f} vs {slopes2.mean():.6f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")
            print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")
            print()

    print("INTERCEPT COMPARISONS:")
    print("-" * 40)

    intercept_results = []
    for day1, day2 in combinations(days_with_data, 2):
        intercepts1 = params_df[params_df["day"] == day1]["intercept"]
        intercepts2 = params_df[params_df["day"] == day2]["intercept"]

        if len(intercepts1) > 0 and len(intercepts2) > 0:
            t_stat, p_val = ttest_ind(intercepts1, intercepts2)
            intercept_results.append(
                {
                    "comparison": f"{day1} vs {day2}",
                    "mean1": intercepts1.mean(),
                    "mean2": intercepts2.mean(),
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                }
            )

            print(f"{day1} vs {day2}:")
            print(
                f"  Mean intercepts: {intercepts1.mean():.4f} vs {intercepts2.mean():.4f}"
            )
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")
            print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")
            print()

    # Save results
    if slope_results:
        slope_df = pd.DataFrame(slope_results)
        slope_df.to_csv(
            config["results_dir"] / "slope_pairwise_comparisons.csv", index=False
        )

    if intercept_results:
        intercept_df = pd.DataFrame(intercept_results)
        intercept_df.to_csv(
            config["results_dir"] / "intercept_pairwise_comparisons.csv", index=False
        )

    return slope_results, intercept_results


# Execute the analysis
df = collect_power_data(CONFIG)
regression_params = plot_trial_timeseries(df, CONFIG)

# Perform statistical tests
params_df = perform_statistical_tests(regression_params, CONFIG)

# Create statistical visualizations
plot_statistical_results(params_df, CONFIG)

# Perform pairwise comparisons
slope_comparisons, intercept_comparisons = pairwise_day_comparisons(params_df, CONFIG)

# Save regression parameters to CSV for further analysis
params_df.to_csv(CONFIG["results_dir"] / "regression_parameters.csv", index=False)
print(
    f"\nRegression parameters saved to: {CONFIG['results_dir'] / 'regression_parameters.csv'}"
)
# %%

# %%


def plot_day_aggregate(df, config):
    fig, axes = plt.subplots(len(config["days"]), 1, figsize=(8, 20))
    for d_idx, day in enumerate(config["days"]):
        day_trials = df[df["day"] == day]
        if day_trials.empty:
            continue
        agg_power = np.concatenate(day_trials["power"].values)
        x = np.arange(len(agg_power))
        data = pd.DataFrame({"time": x, "alpha power": agg_power})
        sns.regplot(
            x="time",
            y="alpha power",
            data=data,
            ax=axes[d_idx],
            scatter_kws={"color": "black"},
            line_kws={"color": "blue"},
        )
        axes[d_idx].set_title(f"{day} Aggregate")
    plt.tight_layout()
    plt.savefig(config["results_dir"] / "day_aggregate.png")
    plt.show()


print(f"Loaded {len(df)} trials.")

plot_trial_timeseries(df, CONFIG)
# plot_day_aggregate(df, CONFIG)

# %%
