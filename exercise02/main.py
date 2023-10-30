import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def resize(data, target_length):
    data_nona = data.dropna()
    x_data = np.arange(0, data_nona.shape[0])
    interpolate = interp1d(x_data, data_nona, kind='linear')
    return interpolate(
        np.arange(0, data_nona.shape[0] - 1, data_nona.shape[0] / target_length)
    )


def plot_interpolated_gesture(df, gesture, plt_axis, gesture_length=100, participant_nr=None):
    gestures_df = df[df[0] == gesture]
    if participant_nr is not None:
        gestures_df = gestures_df[gestures_df[1] == participant_nr]
    resized_df = pd.DataFrame(
        gestures_df.apply(lambda row: pd.Series(resize(row.iloc[3:], gesture_length)), axis=1)
    )
    plt_axis.plot(resized_df.std(), label=gesture)


df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
    plot_interpolated_gesture(df, gesture, plt, 150)

plt.title("Median Curve for all participants and all gestures")
plt.xlabel("time")
plt.ylabel("x accelerometer value")
plt.savefig("plots/median_curve_all_participants.svg")
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 3)
fig.tight_layout(pad=1.5)

for participant_nr in range(9):
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        plt_axis = axs[int(participant_nr / 3), participant_nr % 3]
        plot_interpolated_gesture(df, gesture, plt_axis, 150, participant_nr=participant_nr)

    plt_axis.set_title(f"Participant {participant_nr}")
    plt_axis.set_xlabel("time")
    plt_axis.set_ylabel("x acc")

# plt.savefig("plots/median_curve_participant_0.svg")
fig.legend(labels=["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"], loc="upper right")
plt.show()
