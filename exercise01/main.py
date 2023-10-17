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
plot_gesture_per_participant("right")
plot_gesture_for_participant("right", 0)
