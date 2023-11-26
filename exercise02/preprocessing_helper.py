import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


########################################################################################################################
# Preprocessing Helper Function Definitions
########################################################################################################################
# Checks if there are missing values in a row
def has_missing_value(row):
    acc_data = row.iloc[1:]
    nona_count = acc_data.dropna().count()
    first_nona_count_values = acc_data[:nona_count]
    # If a row has 100 values that are not na, there are no missing values if the first 100 values have 0 not na values
    return first_nona_count_values.dropna().count() != nona_count


# Used for resizing the acceleration data using interpolation
def resize_acc_data(data, target_length):
    data_nona = data.dropna()
    x_data = np.arange(0, data_nona.shape[0])
    interpolate = interp1d(x_data, data_nona, kind='linear')
    # I had the problem, that the interpolation data had less than target_length values, if the original dataset was
    # shorter than the target length and fixed it by this workaround
    n = 0
    step_size = data_nona.shape[0] / target_length
    while len(np.arange(0, data_nona.shape[0] - 1, step_size)) < target_length:
        n = n + 1
        step_size = data_nona.shape[0] / (target_length + n)
    return interpolate(
        np.arange(0, data_nona.shape[0] - 1, step_size)
    )


# Used to resize the whole dataframe
def resize_df(df, target_length):
    resized_df = pd.DataFrame(
        df.apply(lambda row: pd.Series(resize_acc_data(row.iloc[1:], target_length)), axis=1),
    )
    resized_df["gesture"] = df.iloc[:, 0]
    return resized_df


# Plots each sample for a given gesture
def plot_gesture_samples(df, gesture, axs=plt):
    gestures_df = df[df["gesture"] == gesture]
    # we can not use 0:50, because we can have more than 50 values...
    acc_data = gestures_df.loc[:, df.columns != 'gesture'].T
    axs.plot(acc_data)


def plot_interpolation_comparison(original_df, resized_df_200, resized_df_100, resized_df_50):
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        gestures_figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        gestures_figure.suptitle(f"{gesture} gesture interpolation and resizing comparison")
        gestures_figure.tight_layout(pad=2)

        ax1.set_title("Original Length")
        ax2.set_title("Resized to 200 values")
        ax3.set_title("Resized to 100 values")
        ax4.set_title("Resized to 50 values")

        plot_gesture_samples(original_df, gesture, ax1)
        plot_gesture_samples(resized_df_200, gesture, ax2)
        plot_gesture_samples(resized_df_100, gesture, ax3)
        plot_gesture_samples(resized_df_50, gesture, ax4)
        gestures_figure.savefig(f"plots/interpolation_{gesture}.svg")


def plot_density_comparison(unscaled_df, scaled_df):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.set_title("Acceleration data before scaling")
    ax2.set_title("Acceleration data after scaling")

    unscaled_df.plot.density(ax=ax1, legend=False)
    scaled_df.plot.density(ax=ax2, legend=False)
    plt.savefig("plots/density.svg")


def plot_filter_comparison(unfiltered_df, std_filtered_df, mean_filtered_df, sv_filtered_df):
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        gestures_figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        gestures_figure.suptitle(f"{gesture} gesture Filter comparison")
        gestures_figure.tight_layout(pad=2)

        ax1.set_title("Data without Filter")
        ax2.set_title("STD Filter window size 8")
        ax3.set_title("Mean Filter window size 8")
        ax4.set_title("Savgol Filter window size 8")

        plot_gesture_samples(unfiltered_df, gesture, ax1)
        plot_gesture_samples(std_filtered_df, gesture, ax2)
        plot_gesture_samples(mean_filtered_df, gesture, ax3)
        plot_gesture_samples(sv_filtered_df, gesture, ax4)
        gestures_figure.savefig(f"plots/filter_{gesture}.svg")


def plot_outliers(original_data, outliers, gesture):
    fig = plt.figure()
    plt.title(f"{gesture} Outliers")
    plt.plot(original_data[original_data[0] == gesture].iloc[:, 3:].T, color="gray", linestyle='-', alpha=0.5)
    plt.plot(outliers.iloc[:, 3:].T, color="red")
    plt.ylabel("X acceleration value")
    plt.xlabel("time")
    fig.savefig(f"plots/{gesture}_outliers.svg")

