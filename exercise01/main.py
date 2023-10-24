import pandas as pd
import matplotlib.pyplot as plt

# %%

print("Part 1")

df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

# 1.
# 1.1 The dataset has 2160 samples with 423 features (without counting the label, participantNr and sampleNr
print(df.describe())

# 1.2 The features have the datatype float64
print(df.dtypes)

# 1.3 Yes, the gestures are balanced we have 270 samples for each of the 9 classes
print(df[0].value_counts())

# %%
print("Part 2")


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


df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

# 2.
# 2.1.
plot_gesture_per_participant("right")
# Yes we can see the trend that the gesture looks very similar for each participant.
# The gesture consists of one huge peak that has a different breadth depending on the speed in which the gesture was
# executed e.g. Participant 4 executed the gesture very fast and the gesture of participant 8 was the slowest one.

# Yes we can see that each participant has a different delay, speed and strength while the pattern looks similar
# for all of them.

# 2.2.
plot_gesture_for_participant("up", 0)
plot_gesture_for_participant("up", 1)
# Yes we can see the trend, that each sample of the participant has a very similar pattern.
# Especially we can see that each try of the participant had a very similar delay, speed and strength.

# Yes we can see, that the previous assumption does not work for both participants.
# We can see that participant 1 had one (outlier) sample where the delay was very high, which could be the result
# of a distraction during the recording.


# %%

df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

gesture_length_stats = pd.DataFrame(index=[
    'Mean Length',
    'Median Length',
    'MAD',
    'STD',
    '1st Quantile',
    '3rd Quantile',
    'Inner Quantile Range'
])

for gesture in ["left", "right", "up", "down", "square", "triangle", "circleCw", "circleCcw"]:
    participant_frame = df[df[1] == participant_nr]
    gesture_frame = participant_frame[participant_frame[0] == gesture]
    gesture_acc_lengths = gesture_frame.iloc[:, 3:].T.count()

    gesture_length_stats[gesture] = [
        gesture_acc_lengths.mean(),
        gesture_acc_lengths.median(),
        (gesture_acc_lengths - gesture_acc_lengths.mean()).abs().mean(),
        gesture_acc_lengths.std(),
        gesture_acc_lengths.quantile(0.25),
        gesture_acc_lengths.quantile(0.75),
        gesture_acc_lengths.quantile(0.75) - gesture_acc_lengths.quantile(0.25),
    ]

print(gesture_length_stats)

# Concatenate the individual DataFrames into one DataFrame
# print(result_df)

# print(result_df.reset_index(drop=True))

# Rename the columns to match the gestures
# result_df.columns = gestures

# print(gesture_length_stats)
#
# print("Left Mean Length:")
# print(left_frame_lengths.mean())
#
# print("Left Median Length:")
# print(left_frame_lengths.median())
#
# print("Left MAD:")
# print((left_frame_lengths - left_frame_lengths.mean()).abs().mean())
#
# print("Left Standard Deviation:")
# print(left_frame_lengths.std())
#
# print("1st Quantile:")
# print(left_frame_lengths.quantile(0.25))
#
# print("3rd Quantile:")
# print(left_frame_lengths.quantile(0.75))
#
# print("Inner Quartile Range:")
# print(left_frame_lengths.quantile(0.75) - left_frame_lengths.quantile(0.25))

# 3.1 Calculate and state the mean and median length (number of acceleration values) of each gesture type of each
# participants

# As well as per single participants together with its standard deviation median absolute deviation 1st
# and 3rd quartile and inner quartile range

# %%
