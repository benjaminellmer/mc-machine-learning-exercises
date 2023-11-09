import pandas as pd

df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

print("---------- Part 1 ----------")

# 1.1
print(df.describe())

# 1.2
print(df.dtypes)

# 1.3
print(df[0].value_counts())

# %%
import pandas as pd
import matplotlib.pyplot as plt

print("---------- Part 2 ----------")

df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)


def plot_gesture_per_participant(gesture, sample_nr=14):
    gesture_frame = df[df[0] == gesture]
    for i, participant_sample in gesture_frame[gesture_frame[2] == sample_nr].iterrows():
        plt.plot(participant_sample[3:].T, label=f"Participant {participant_sample[1]}")

    plt.legend()
    plt.title(f"Sample number {sample_nr} of {gesture} gesture per participant")
    plt.xlabel("time")
    plt.ylabel("x accelerometer value")
    plt.savefig(f"plots/per_participant_{gesture}.svg")
    plt.show()


def plot_gesture_for_participant(gesture, participant_nr):
    gesture_frame = df[df[0] == gesture]
    gesture_frame_for_participant = gesture_frame[gesture_frame[1] == participant_nr]
    gesture_frame_for_participant.iloc[:, 3:].T.plot(legend=False)

    plt.title(f"All samples of gesture {gesture} for participant {participant_nr}")
    plt.xlabel("time")
    plt.ylabel("x accelerometer value")
    plt.savefig(f"plots/{gesture}_participant_{participant_nr}.svg")
    plt.show()


# 2.1.
plot_gesture_per_participant("right")

# 2.2.
plot_gesture_for_participant("right", 0)

# %%
import pandas as pd
import matplotlib.pyplot as plt

print("---------- Part 3 ----------")
df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)


def get_gesture_length_data(gesture, participant_nr=None):
    if participant_nr is None:
        participant_frame = df
    else:
        participant_frame = df[df[1] == participant_nr]

    gesture_frame = participant_frame[participant_frame[0] == gesture]
    return gesture_frame.iloc[:, 3:].T.count()


def calculate_gesture_length_stats(participant_nr=None):
    gesture_length_stats = pd.DataFrame(index=[
        'Mean Length', 'Median Length',
        'MAD',
        'STD',
        '1st Quartile',
        '3rd Quartile',
        'Inner Quartile Range'
    ])

    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        gesture_acc_lengths = get_gesture_length_data(gesture, participant_nr)

        gesture_length_stats[gesture] = [
            gesture_acc_lengths.mean(),
            gesture_acc_lengths.median(),
            (gesture_acc_lengths - gesture_acc_lengths.mean()).abs().mean(),
            gesture_acc_lengths.std(),
            gesture_acc_lengths.quantile(0.25),
            gesture_acc_lengths.quantile(0.75),
            gesture_acc_lengths.quantile(0.75) - gesture_acc_lengths.quantile(0.25),
        ]
    return gesture_length_stats


print("\n----- Calculations for all Participants -----\n")
print(calculate_gesture_length_stats())

for participant_nr in range(9):
    print(f"\n----- Calculations for Participant {participant_nr} -----\n")
    print(calculate_gesture_length_stats(participant_nr=participant_nr))


# %%
def get_gesture_length_data(gesture):
    participant_frame = df[df[1] == participant_nr]

    gesture_frame = participant_frame[participant_frame[0] == gesture]
    return gesture_frame.iloc[:, 3:].T.count()


import pandas as pd
import matplotlib.pyplot as plt

print("---------- Part 4 ----------")

df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

gesture_plot_data = pd.DataFrame()

for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
    gesture_plot_data[gesture] = get_gesture_length_data(gesture).values

gesture_plot_data.plot.box()
plt.show()
plt.savefig(f"plots/boxplot_lengths.svg")

# %%
import pandas as pd
import matplotlib.pyplot as plt

print("---------- Part 5 ----------")

df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

counts_df = pd.DataFrame({
    'label': df[0],
    'participant_nr': df[1],
    'acc_count': df.iloc[:, 3:].T.count(),
})

# How many gestures have a length of more than 100 acceleration values in total ?
print(len(counts_df.query("acc_count > 100")))

# How many gestures have a length of more than 100 acceleration values in per participant ?
print(counts_df.drop("label", axis=1).query("acc_count > 100").groupby("participant_nr").count())

# How many gestures have a length below 100 acceleration values in total ?
print(len(counts_df.query("acc_count < 100")))

# How many gestures have a length below 100 acceleration values in per participant ?
print(counts_df.drop("label", axis=1).query("acc_count < 100").groupby("participant_nr").count())

# How many participants performed at least half of their left gestures in less than 70 acceleration values
print(len(counts_df.query("acc_count < 70 and label == 'left'").groupby("participant_nr").count().query(
    "acc_count > 15")))

# How many participants needed more than 200 accleration values to perform at least of their square gestures
print(len(counts_df.query("acc_count > 200 and label == 'square'").groupby("participant_nr").count().query(
    "acc_count > 15")))

print("---------- Part 2 after we got assignment 2 ----------")

# %%

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
plt.legend()
plt.savefig("plots/median_curve_all_participants.svg")
plt.show()

for participant_nr in range(9):
    for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
        plot_interpolated_gesture(df, gesture, plt, 150, participant_nr=participant_nr)

    plt.title(f"Median Curve for participant {participant_nr} and all gestures")
    plt.xlabel("time")
    plt.ylabel("x accelerometer value")
    plt.legend()
    plt.savefig(f"plots/median_curve_participant_{participant_nr}.svg")
    plt.show()

# Solution Using Subplot, sadly I did not get the subplot in a good format for the protocol

# fig, axs = plt.subplots(3, 3)
# fig.tight_layout(pad=2.5)
#
# for participant_nr in range(9):
#     for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
#         plt_axis = axs[int(participant_nr / 3), participant_nr % 3]
#         plot_interpolated_gesture(df, gesture, plt_axis, 150, participant_nr=participant_nr)
#
#     plt_axis.set_title(f"Participant {participant_nr}")
#     plt_axis.set_xlabel("time")
#     plt_axis.set_ylabel("x acc")
#
# fig.legend(labels=["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"],
#            loc="right",
#            )
# fig.subplots_adjust(right=0.75)
#
# plt.savefig("plots/median_curve_per_participant.svg")
# plt.show()
