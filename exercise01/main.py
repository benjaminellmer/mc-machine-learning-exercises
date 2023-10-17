import pandas as pd
import matplotlib.pyplot as plt


def plot_gesture_per_participant(gesture, sample_nr=14):
    gesture_frame = df[df[0] == gesture]
    for i, participant_sample in gesture_frame[gesture_frame[2] == sample_nr].iterrows():
        plt.plot(participant_sample[3:].T, label=f"Participant {participant_sample[1]}")

    plt.legend()
    plt.title(f"Sample number {sample_nr} of {gesture} gesture per participant")
    plt.xlabel("time")
    plt.ylabel("x accelerometer value")
    plt.show()


def plot_gesture_for_participant(gesture, participant_nr):
    gesture_frame = df[df[0] == gesture]
    gesture_frame_for_participant = gesture_frame[gesture_frame[1] == participant_nr]
    gesture_frame_for_participant.iloc[:, 3:].T.plot(legend=False)

    plt.title(f"All samples of gesture {gesture} for participant {participant_nr}")
    plt.xlabel("time")
    plt.ylabel("x accelerometer value")
    plt.show()


# df stands for dataframe obviously, first I named it gesture_data, but writing df saves a lot of time
df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

# 1.
# 1.1 The dataset has 2160 samples with 423 features (without counting the label, participantNr and sampleNr
print(df.describe())

# 1.2 The features have the datatype float64
print(df.dtypes)

# 1.3 Yes, the gestures are balanced we have 270 samples for each of the 9 classes
print(df[0].value_counts())

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

