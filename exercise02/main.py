import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn import preprocessing
from scipy import signal
from scipy.signal import savgol_filter

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


# Plots all gestures in the dataset
def plot_by_gestures(df, pre_title=""):
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        plt.figure()
        plot_gesture_samples(df, gesture, plt)
        plt.title(f"{pre_title} {gesture}")
        plt.ylabel("x acceleration value")
        plt.xlabel("time")
        plt.savefig(f"plots/{pre_title.replace(' ', '_')}_{gesture}.svg")


def boxplot_by_gestures(df, y_label="", title=""):
    df_with_labels = df.copy(deep=True)
    df_with_labels["gesture"] = original_df.iloc[:, 0]
    boxplot_data = pd.DataFrame()
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        gesture_data = df_with_labels[df_with_labels["gesture"] == gesture]
        boxplot_data[gesture] = gesture_data[0].T.values
    boxplot_data.plot.box()
    plt.title(title)
    plt.ylabel(y_label)
    plt.savefig(f"plots/{title.replace(' ', '_')}.svg")


# Plots each sample for a given gesture
def plot_gesture_samples(df, gesture, axs=plt):
    gestures_df = df[df["gesture"] == gesture]
    # we can not use 0:50, because we can have more than 50 values...
    acc_data = gestures_df.loc[:, df.columns != 'gesture'].T
    axs.plot(acc_data)


# Setup
original_df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

# The sampleNr and participantNr are not more necessary for this assignment
original_df.drop(columns=[1, 2], inplace=True)

# Preprocessing - Missing Values
missing_values = original_df.apply(has_missing_value, axis=1)
print(f"The dataset has {missing_values.sum()} samples with missing values!")

# Preprocessing - Feature Reduction
resized_df_200 = resize_df(original_df, 200)
resized_df_100 = resize_df(original_df, 100)
resized_df_50 = resize_df(original_df, 50)

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

# Preprocessing - Normalizing, scaling
scaled_df = resized_df_50.apply(
    lambda row: pd.Series(preprocessing.scale(row.iloc[0:50])), axis=1
)

print(scaled_df)

fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_title("Acceleration data before scaling")
ax2.set_title("Acceleration data after scaling")

resized_df_50.plot.density(ax=ax1, legend=False)
scaled_df.plot.density(ax=ax2, legend=False)
plt.savefig("plots/density.svg")
plt.show()

# Preprocessing - Outliers
# left_data = resized_df_50[resized_df_50[0] == "left"]
# left_data.iloc[:, 0:50].T.plot()


# Preprocessing - Filtering
std_filtered_df = scaled_df.apply(lambda row: pd.Series(row).rolling(8).mean())
mean_filtered_df = scaled_df.apply(lambda row: pd.Series(row).rolling(8).std())
sv_filtered_df = scaled_df.apply(lambda row: savgol_filter(row, 8, 2))

print(std_filtered_df)
print(mean_filtered_df)
print(sv_filtered_df)

# Add labels for the plots...
scaled_df["gesture"] = original_df.iloc[:, 0]
std_filtered_df["gesture"] = original_df.iloc[:, 0]
mean_filtered_df["gesture"] = original_df.iloc[:, 0]
sv_filtered_df["gesture"] = original_df.iloc[:, 0]

for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
    gestures_figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    gestures_figure.suptitle(f"{gesture} gesture Filter comparison")
    gestures_figure.tight_layout(pad=2)

    ax1.set_title("Data without Filter")
    ax2.set_title("STD Filter window size 8")
    ax3.set_title("Mean Filter window size 8")
    ax4.set_title("Savgol Filter window size 8")

    plot_gesture_samples(scaled_df, gesture, ax1)
    plot_gesture_samples(std_filtered_df, gesture, ax2)
    plot_gesture_samples(mean_filtered_df, gesture, ax3)
    plot_gesture_samples(sv_filtered_df, gesture, ax4)
    gestures_figure.savefig(f"plots/filter_{gesture}.svg")

plt.show()
#
# # left_data = sv_filtered_df[sv_filtered_df[0] == "left"]
# # first_col = left_data.iloc[:, 5]
# # first_col.plot.box()
# # plt.show()
#
# # print(left_data_first_col)
#
# # scaled_df = apply_to_df(df, lambda row: pd.Series(preprocessing.scale(row.iloc[3:])))
# # std_filtered_df = apply_to_df(scaled_df, lambda row: pd.Series(row.iloc[3:]).rolling(10).mean())
#
# # Preprocessing - Feature Addition
# gesture_lengths = original_df.iloc[:, 1:].T.count()
#
# # plt.show()
#
# # print(std_filtered_df)
#
# # Filtering
#
# # Features
#
# # Feature Extraction
#
# # Min
# min_values = sv_filtered_df.apply(lambda row: pd.Series(row.iloc[0:50].min()), axis=1)
# boxplot_by_gestures(min_values, y_label="minimum value", title="min values per gesture")
#
# # Max
# max_values = sv_filtered_df.apply(lambda row: pd.Series(row.iloc[0:50].max()), axis=1)
# boxplot_by_gestures(max_values, y_label="maximum value", title="max values per gesture")
#
# # Mean
# mean_values = original_df.apply(lambda row: pd.Series(row.iloc[3:].mean()), axis=1)
# boxplot_by_gestures(mean_values, y_label="mean", title="mean values per gesture")
#
# # Median
# median_values = sv_filtered_df.apply(lambda row: pd.Series(row.iloc[0:50].median()), axis=1)
# boxplot_by_gestures(median_values, y_label="median", title="median values per gesture")
#
# # STD
# std_values = original_df.apply(lambda row: pd.Series(row.iloc[3:].std()), axis=1)
# boxplot_by_gestures(std_values, y_label="std", title="std values per gesture")
#
#
# # Zero Crossing Rate
# def count_zero_crossings(row):
#     # https://stackoverflow.com/questions/44319374/how-to-calculate-zero-crossing-rate-with-pyaudio-stream-data
#     crossings = np.nonzero(np.diff(row.iloc[0:50].dropna() > 0))[0]
#     return pd.Series(len(crossings))
#
#
# zero_crossings = sv_filtered_df.apply(count_zero_crossings, axis=1)
# boxplot_by_gestures(zero_crossings, y_label="number of zero crossings", title="zero crossings per gesture")
#
#
# # Median Crossing Rate
# def count_median_crossings(row):
#     median = row.iloc[0:50].median()
#     crossings = np.nonzero(np.diff(row.iloc[0:50].dropna() > median))[0]
#     return pd.Series(len(crossings))
#
#
# median_crossings = sv_filtered_df.apply(count_median_crossings, axis=1)
# boxplot_by_gestures(median_crossings, y_label="number of median crossings", title="median crossings per gesture")
# plt.show()
#
# # Number of Minimas
# number_of_minimas = sv_filtered_df.apply(
#     lambda row: pd.Series(len(signal.argrelmin(row.iloc[0:50].dropna().values)[0])), axis=1
# )
# boxplot_by_gestures(number_of_minimas, y_label="number of minimas", title="minimas per gesture")
#
# # Number of Maximas
# number_of_maximas = sv_filtered_df.apply(
#     lambda row: pd.Series(len(signal.argrelmax(row.iloc[0:50].dropna().values)[0])), axis=1
# )
# boxplot_by_gestures(number_of_maximas, y_label="number of maximas", title="maximas per gesture")
#
# # plt.show()
#
# # 1st Derivative
# derivatives_1st = sv_filtered_df.apply(lambda row: pd.Series(np.diff(row.iloc[0:50])), axis=1)
#
# # 2nd Derivative
# derivatives_2nd = derivatives_1st.apply(lambda row: pd.Series(np.diff(row.dropna())), axis=1)
#
#
# def calc_power(row):
#     fft_calc = np.fft.fft(row.iloc[0:50].dropna())
#     return pd.Series(np.abs(fft_calc) ** 2)
#
#
# def calc_phase(row):
#     fft_calc = np.fft.fft(row.iloc[0:50].dropna())
#     return pd.Series(np.angle(fft_calc))
#
#
# # FFT power and phase
# power_spectrum = scaled_df.apply(calc_power, axis=1)
# phase_spectrum = scaled_df.apply(calc_phase, axis=1)
#
# # plot for power and phase
# # bar_x_data = list(range(len(fft_calc)))
# # plt.plot(first_left)
# # plt.bar(bar_x_data, phase_spectrum)
# # plt.bar(bar_x_data, power_spectrum)
# #
# # plt.legend(['original scaled', 'FFT phase', 'FFT power'])
# # plt.show()
#
# # AKF
#
# # Wavelets
# # beaver_temp_scaled = preprocessing.scale(beaver.iloc[:,3])
# # plt.plot(beaver_temp_scaled)
# # plot the wavelet
# # [phi, psi, x] = pywt.Wavelet('db3').wavefun(level=1)
# # plt.plot(x, psi)
