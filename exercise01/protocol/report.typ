#align(center)[
    #v(2cm)
    #text(22pt)[
        #smallcaps[Machine Learning]
    ] 
    #v(1.5cm)
    #text(28pt)[
        *Submission Assignment 01*
    ]
    #v(1.5cm)
    #text(16pt)[
        von \
        Benjamin Ellmer \
        (S2210455012)
    ]
    #v(0.5cm)
    #image("images/logo.svg", height: 30%)
    #v(0.5cm)
    #text(16pt)[
        Mobile Computing Master \
        FH Hagenberg
    ]
    #v(2cm)
    #text(16pt)[
        #datetime.today().display("[month repr:long] [day], [year]")
    ]
]
#pagebreak()

#set text(
  size: 14pt
)
#set page(
    numbering: "1 / 1",
)



= Task 1: Overview of the Dataset
*How many samples and features are in the dataset ?* \
Samples: 2160 \
Features: 423 (Without counting participantNr and sampleNr as features) \

*What are the features datatypes?* \
float64 

*Are the gestures balanced ?* \
Yes, the gestures are balanced the dataset contains 270 samples for each of the 9 classes

= Task 2: Visualizations 
I did this exercise, before we had the infos for our second assignment, therefore I tried to plot at least something. \

#figure(
    image("images/per_participant_right.svg")
)

*Is there a trend visible ?* \
Yes we can see the trend that the gesture looks very similar for each participant.
The gesture consists of one huge peak that has a different breadth depending on the speed in which the gesture was executed 
e.g. Participant 4 executed the gesture very fast and the gesture of participant 8 was the slowest one.

*Are there differences between the participants visible ?* \
Yes we can see that each participant has a different delay, speed and strength while the pattern looks similar.

#figure(
    image("images/right_participant_0.svg")
)

*Is there a trend visible ?* \
Yes we can see that the participant executed all his samples almost identical.
There are very less differences between the samples comparing to the figure of all participants.

#pagebreak()



= Task 3: Calculations
#figure(
    image("images/all_participants.png")
)
#figure(
    image("images/participant_0.png")
)
#figure(
    image("images/participant_1.png")
)
#figure(
    image("images/participant_2.png")
)
#figure(
    image("images/participant_3.png")
)
#figure(
    image("images/participant_4.png")
)
#figure(
    image("images/participant_5.png")
)
#figure(
    image("images/participant_6.png")
)
#figure(
    image("images/participant_7.png")
)
#figure(
    image("images/participant_8.png")
)

#pagebreak()

= Task 4: Correlation between gesture lengths
I did this exercise, before we had the infos for our second assignment, therefore I used a boxplot, which shows the correlation very good.

#figure(
    image("images/boxplot_lengths.svg")
)

*Is there a trend visible?* \
Yes we can see a trend between the gesture lengths and the gesture types.
The very primitive gestures left, right, up and down very usually very short, still we can always see some outliers,
which can e.g. be caused by a delayed start, as we already saw in the plots of assignment 2

 The "longest gesture" is of course the square, which makes sense, because the square gestures consists of 4 movements
 right, down left, and up, the second longest gesture is the triangle, which also makes sense, because it consists of
 3 movements.

#pagebreak()

= Task 5: Vectorized commands
*How many gestures have a length of more than 100 acceleration values?* \
In total 1114 \
#table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [*P0*], [*P1*], [*P2*], [*P3*], [*P4*], [*P5*], [*P6*], [*P7*], [*P8*],
    [116], [118], [110], [122], [111], [131], [132], [136], [138]
)
*How many gestures have a length below 100?* \
In total 1035
#table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [*P0*], [*P1*], [*P2*], [*P3*], [*P4*], [*P5*], [*P6*], [*P7*], [*P8*],
    [124], [120], [126], [117], [129], [108], [107], [103], [101]
)

*How many participants performed at least half of their left gestures in less than 70 acceleration values?* \
2 Participants

*How many participants needed more than 200 acceleration values to perform at least half of their square gestures?* \
2 Participants

= Task 2: Visualizations (after we got assignment 2)
I tried to create some plots with the code we got for assignment 2.
I interpolated the data to get samples with equal lenghts.
Then I used the median of each datapoint for each gesture. 
I'm not sure if using the median is that clever, but I did not find a better way.
On the internet I read that DTW makes sense for such tasks, but I didn't get it working.

#figure(
    image("images/median_curve_all_participants.svg")
)

#figure(
    image("images/median_curve_participant_0.svg")
)
#figure(
    image("images/median_curve_participant_1.svg")
)
#figure(
    image("images/median_curve_participant_2.svg")
)
#figure(
    image("images/median_curve_participant_3.svg")
)
#figure(
    image("images/median_curve_participant_4.svg")
)
#figure(
    image("images/median_curve_participant_5.svg")
)
#figure(
    image("images/median_curve_participant_6.svg")
)
#figure(
    image("images/median_curve_participant_7.svg")
)
#figure(
    image("images/median_curve_participant_8.svg")
)
