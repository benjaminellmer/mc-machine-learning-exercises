from matplotlib import gridspec
from statsmodels.graphics import tsaplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


########################################################################################################################
# Feature Extraction Helper Function Definitions
########################################################################################################################
def get_short_label(label):
    if label == "circleCw":
        return "cCw"
    elif label == "circleCcw":
        return "cCcw"
    else:
        return label[0]


def sub_boxplot_by_gestures(df, labels, title="", ax=plt):
    df_with_labels = df.copy(deep=True)
    short_labels = labels.apply(get_short_label)
    df_with_labels["gesture"] = short_labels
    df_with_labels.boxplot(ax=ax, by="gesture")
    ax.set_xlabel("")
    ax.set_title(title)


def plot_raw_min_max_mean(raw_values, min_values, max_values, mean_values, labels):
    figure, ((raw_ax, min_ax), (max_ax, mean_ax)) = plt.subplots(2, 2)
    figure.tight_layout(pad=2)

    raw_ax.plot(raw_values)
    raw_ax.set_title("Raw Data")
    sub_boxplot_by_gestures(min_values, labels, title="min values per gesture", ax=min_ax)
    sub_boxplot_by_gestures(max_values, labels, title="max values per gesture", ax=max_ax)
    sub_boxplot_by_gestures(mean_values, labels, title="mean values per gesture", ax=mean_ax)


def plot_median_std_iqr_mad(median_values, std_values, iqr_values, mad_values, labels):
    figure, ((median_ax, std_ax), (iqr_ax, mad_ax)) = plt.subplots(2, 2)
    figure.tight_layout(pad=2)

    sub_boxplot_by_gestures(median_values, labels, title="median values per gesture", ax=median_ax)
    sub_boxplot_by_gestures(std_values, labels, title="std values per gesture", ax=std_ax)
    sub_boxplot_by_gestures(iqr_values, labels, title="iqr values per gesture", ax=iqr_ax)
    sub_boxplot_by_gestures(mad_values, labels, title="mad values per gesture", ax=mad_ax)


def plot_minimas_maximas_zcr_mcr(number_of_minimas, number_of_maximas, zero_crossings, median_crossings, labels):
    figure, ((minimas_ax, maximas_ax), (zero_crossings_ax, median_crossings_ax)) = plt.subplots(2, 2)
    figure.tight_layout(pad=2)

    sub_boxplot_by_gestures(number_of_minimas, labels, title="number of minimas per gesture", ax=minimas_ax)
    sub_boxplot_by_gestures(number_of_maximas, labels, title="number of maximas per gesture", ax=maximas_ax)
    sub_boxplot_by_gestures(zero_crossings, labels, title="zero crossing rate per gesture", ax=zero_crossings_ax)
    sub_boxplot_by_gestures(median_crossings, labels, title="median crossing rate per gesture", ax=median_crossings_ax)


def plot_power_phase_acf(power_spectrum, phase_spectrum, sv_filtered_data):
    figure = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
    (power_ax, phase_ax, acf_ax) = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, :])]

    figure.tight_layout(pad=2)
    bar_x_data = list(range(50))

    power_ax.set_title("FFT Power Spectrum")
    for index, row in power_spectrum.iterrows():
        power_ax.bar(bar_x_data, row)

    phase_ax.set_title("FFT Phase Spectrum")
    for index, row in phase_spectrum.iterrows():
        phase_ax.bar(bar_x_data, row)

    acf_ax.set_title("Autocorrelation")
    for index, row in sv_filtered_data.iterrows():
        tsaplots.plot_acf(row.iloc[0:50], lags=len(row.iloc[0:50]) - 1, ax=acf_ax)


def plot_wavelet(approx_coeff, detail1_coeff, detail2_coeff, detail3_coeff, detail4_coeff, title=""):
    figure = plt.figure(figsize=(8, 6))
    figure.tight_layout()
    figure.suptitle(title)
    gs = gridspec.GridSpec(3, 2)
    (ap_ax, d1_ax, d2_ax, d3_ax, d4_ax) = [
        plt.subplot(gs[0, 0]),
        plt.subplot(gs[0, 1]),
        plt.subplot(gs[1, 0]),
        plt.subplot(gs[1, 1]),
        plt.subplot(gs[2, 0:]),
    ]

    ap_ax.set_title("Approximation coefficients")
    d1_ax.set_title("Detail 1 coefficients")
    d2_ax.set_title("Detail 2 coefficients")
    d3_ax.set_title("Detail 3 coefficients")
    d4_ax.set_title("Detail 4 coefficients")

    approx_coeff.T.plot(legend=False, ax=ap_ax)
    detail1_coeff.T.plot(legend=False, ax=d1_ax)
    detail2_coeff.T.plot(legend=False, ax=d2_ax)
    detail3_coeff.T.plot(legend=False, ax=d3_ax)
    detail4_coeff.T.plot(legend=False, ax=d4_ax)


# Zero Crossing Rate
def count_zero_crossings(row):
    # https://stackoverflow.com/questions/44319374/how-to-calculate-zero-crossing-rate-with-pyaudio-stream-data
    crossings = np.nonzero(np.diff(row.iloc[0:50].dropna() > 0))[0]
    return pd.Series(len(crossings))


# Median Crossing Rate
def count_median_crossings(row):
    median = row.iloc[0:50].median()
    crossings = np.nonzero(np.diff(row.iloc[0:50].dropna() > median))[0]
    return pd.Series(len(crossings))


def calc_power(row):
    fft_calc = np.fft.fft(row.iloc[0:50].dropna())
    return pd.Series(np.abs(fft_calc) ** 2)


def calc_phase(row):
    fft_calc = np.fft.fft(row.iloc[0:50].dropna())
    return pd.Series(np.angle(fft_calc))
