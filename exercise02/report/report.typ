#align(center)[
    #v(2cm)
    #text(22pt)[
        #smallcaps[Machine Learning]
    ] 
    #v(1.5cm)
    #text(28pt)[
        *Submission Assignment 02*
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

= Data Preprocessing

== Preprocessing - Missing values
*Are there any missing values which need to be taken care of?* \
No, I did not find a sample that has missing values. This was done by counting the number of values that are not nan for each sample.
Then I counted the number of nan values for the first values.

This means if the sample has 100 not nan values I took the first 100 values and counted the number of nan values.
If the number of nan values in the first 100 values is 0, then there is no missing data in the sample.

== Preprocessing - Outliers
My first idea to find outliers was visualizing the data using a boxplot. 
Therefore, I visualized the standard deviation in @boxstd and the median in @boxmean for each gesture as boxplot:

#figure(
    caption: "Boxplot Standard Deviation",
    image("images/std_distribution_per_gesture.svg")
)<boxstd>

#figure(
    caption: "Boxplot mean",
    image("images/mean_distribution_per_gesture.svg")
)<boxmean>

As we can see in the boxplots, each gesture shows some ouliers for both std and mean.
On the std diagram, at least the simpler gestures (left, right, up, down, circles) do not seem to have very much outliers.
But, especially the outliers of the triangle samples shows, that this analysis probably is not optimal.
Therefore, I decided to visualize all samples for each gesture and analyze this diagrams.

#pagebreak()

In @left we can see that the blue line at the end is a clear outlier and should be removed.
The brown line at the end is also an outlier but I decided not to remove it because it is only a little delayed.
#figure(
    caption: "left",
    image("images/original_data_left.svg", height: 35%)
)<left>

@right (right) shows one clear outlier, the blue line, which will be removed.
#figure(
    caption: "right",
    image("images/original_data_right.svg", height: 35%)
)<right>

#pagebreak()

In @up we can see the light green sample at the end, which is an oultier, which will be removed.
Additionally I removed the orange sample, which looks like a problem with the recording.
#figure(
    caption: "up",
    image("images/original_data_up.svg", height: 35%)
)<up>

In @down we can see one clear outlier the brown line which has to be removed.
#figure(
    caption: "down",
    image("images/original_data_down.svg", height: 35%)
)<down>

#pagebreak()

@square shows some samples that seem to have much smaller values, but I do not think they should be removed.
#figure(
    caption: "square",
    image("images/original_data_square.svg", height: 35%)
) <square>

@triangle shows two samples, where the gesture looks like it is very delayed, therefore I removed the green and the blue sample.
#figure(
    caption: "triangle",
    image("images/original_data_triangle.svg", height: 35%)
)<triangle>

#pagebreak()

@circleCw shows two samples the red and the grey one which seem to be outliers, therefore I removed both.
#figure(
    caption: "circleCw",
    image("images/original_data_circleCw.svg", height: 35%)
)<circleCw>

In @circleCcw I can not recognize any outliers.
#figure(
    image("images/original_data_circleCcw.svg", height: 35%)
)<circleCcw>

#pagebreak()
== Preprocessing - Normalization
I decided to normalize the data using the scale function. 
Normalizing does not affect the trend of the data, I just centers the data, therefore I did not create any plots.

== Preprocessing - Filtering
I decided to use filter to reduce the noise in the data, as suggested in the exercise hints.
I tried the suggested filters (running mean, running median and SV).

=== Running Mean
=== Running Median
=== SV Filter

== Preprocessing - Feature Reduction
To get an equal amount of acceleration values I interpolated the gesture data and took an equal amount of values for each sample.
According to the description the sensor was recording with 100Hz and the max frequency that makes sense is 20Hz, this means 84.6 (423/5) values should be sufficient.
Therefore, I visualized the comparison of the interpolated data for each gesture and using 50, 100 and 200 values compared to the original data.
As a result I decided to continue to work with only 50 values, because I think it still shows the mandatory information.
The comparisons can be seen on the next pages.

#pagebreak()

#image("images/interpolation_left.svg", height: 45%)
#image("images/interpolation_right.svg", height: 45%)
#image("images/interpolation_up.svg", height: 45%)
#image("images/interpolation_down.svg", height: 45%)
#image("images/interpolation_square.svg", height: 45%)
#image("images/interpolation_triangle.svg", height: 45%)
#image("images/interpolation_circleCw.svg", height: 45%)
#image("images/interpolation_circleCcw.svg", height: 45%)

== Preprocessing - Feature Addition

/*
#figure(
image("images/all_participants.png")
)
*/

See appendix for the comparison plots of all gestures.


= Feature Extraction

= Plots Appendix

