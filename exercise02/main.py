import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")


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
    return interpolate(
        np.arange(0, data_nona.shape[0] - 1, data_nona.shape[0] / target_length)
    )


# Used to resize the whole dataframe
def resize_df(df, target_length):
    resized_df = pd.DataFrame(
        df.apply(lambda row: pd.Series(resize_acc_data(row.iloc[1:], target_length)), axis=1),
    )
    # Restore the labels
    resized_df[0] = df.iloc[:, 0]
    return resized_df


# Plots all gestures in the dataset
def plot_by_gestures(df, pre_title=""):
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        plt.figure()
        plot_gesture_samples(df, gesture, plt)
        plt.title(f"{pre_title} {gesture}")
        plt.ylabel("x acceleration value")
        plt.xlabel("time")
        plt.savefig(f"plots/{pre_title.replace(' ', '_')}_{gesture}.svg")


# Plots each sample for a given gesture
def plot_gesture_samples(df, gesture, axs=plt):
    gestures_df = df[df[0] == gesture]
    acc_data = gestures_df.iloc[:, 1:].T
    axs.plot(acc_data)


df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

# The sampleNr and participantNr are not more necessary for this assignment
df.drop(columns=[1, 2], inplace=True)

# Outliers first try
mean_data = pd.DataFrame()
std_data = pd.DataFrame()

for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
    box_data_gesture = df[df[0] == gesture].iloc[:, 1:].T
    mean_data[gesture] = box_data_gesture.mean().values
    std_data[gesture] = box_data_gesture.std().values

mean_data.plot.box()
plt.title("Mean distribution per gesture")
plt.ylabel("mean x axis")
plt.savefig(f"plots/mean_distribution_per_gesture.svg")

std_data.plot.box()
plt.title("Standard Deviation distribution per gesture")
plt.ylabel("std x axis")
plt.savefig(f"plots/std_distribution_per_gesture.svg")
plt.show()

#
# plot_by_gestures(df, "original data")
# plt.show()

# box_data.plot.box()
# plt.show()
# plt.savefig(f"plots/boxplot_lengths.svg")


# Missing Values
# missing_values = df.apply(has_missing_value, axis=1)
# print(f"The dataset has {missing_values.sum()} samples with missing values!")


# resized_df_200 = resize_df(df, 200)
# resized_df_100 = resize_df(df, 100)
# resized_df_50 = resize_df(df, 50)

# for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
#     gestures_figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#     gestures_figure.suptitle(f"{gesture} gesture interpolation and resizing comparison")
#     gestures_figure.tight_layout(pad=2)
#
#     ax1.set_title("Original Data")
#     ax2.set_title("Resized to 200 values")
#     ax3.set_title("Resized to 100 values")
#     ax4.set_title("Resized to 50 values")
#
#     plot_gesture_samples(df, gesture, ax1)
#     plot_gesture_samples(resized_df_200, gesture, ax2)
#     plot_gesture_samples(resized_df_100, gesture, ax3)
#     plot_gesture_samples(resized_df_50, gesture, ax4)
#     gestures_figure.savefig(f"plots/interpolation_{gesture}.svg")
#
# plt.show()

# resized_df = resize_df(df, 50)
# plot_by_gestures(resized_df, pre_title="Resized data for gesture: ")
# plt.show()


# scaled_df = apply_to_df(df, lambda row: pd.Series(preprocessing.scale(row.iloc[3:])))
# std_filtered_df = apply_to_df(scaled_df, lambda row: pd.Series(row.iloc[3:]).rolling(10).mean())


# plt.show()

# print(std_filtered_df)
# Normalizing

# Filtering


# Features

# Gesture Length in Acc Values
# Number of Minimums
# Number of Maximums
# 1st Derivate
# 2nd Derivate
# Frequency Signal
#
