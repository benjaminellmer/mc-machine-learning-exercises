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
#set heading(numbering: "1.")

= Data Preprocessing

== Preprocessing - Missing values
*Are there any missing values which need to be taken care of?* \
No, I did not find a sample that has missing values. This was done by counting the number of values that are not nan for each sample.
Then I counted the number of nan values for the first values.

This means if the sample has 100 not nan values I took the first 100 values and counted the number of nan values.
If the number of nan values in the first 100 values is 0, then there is no missing data in the sample.

== Preprocessing - Feature Reduction
To get an equal amount of acceleration values I interpolated the gesture data and took an equal amount of values for each sample.
According to the description the sensor was recording with 100Hz and the max frequency that makes sense is 20Hz, this means 84.6 (423/5) values should be sufficient.
Therefore, I visualized the comparison of the interpolated data for each gesture and using 50, 100 and 200 values compared to the original data.
As a result I decided to continue to work with only 50 values, because I think it still shows the mandatory information.
The comparisons can be seen in @ainterpolations[Appendix Interpolations].

== Preprocessing - Normalization
I decided to normalize the data using scaling.
@density shows the distribution of the data before and after normalization.
The normalization was done, because it helps the models to work with the data.

#figure(
    image("images/density.svg", height: 40%)
) <density>

== Preprocessing - Filtering
I decided to use filter to reduce the noise in the data, as suggested in the exercise hints.
I tried the suggested filters (running mean, running median and savgol filter).

In my opinion the savgol filter preservs the trends of the gestures best, therefore I continued with the data that was filtered using the savgol filter.
The comparisons can be seen in @afiltering[Appendix Filters].
Regarding the windowsize I tried some sizes and ended up with 8, but I think including theses plots here would be too much.

/*
== Preprocessing - Outliers
Before preprocesssing, the data, it could see a trend in all samples, but they were very different from each others.
This means, I could find many outliers looking on column values, their stds or their means, therefore I decided to first preprocess the data and search for outliers afterwards.
*/

== Preprocessing - Feature Addition
In the last exercise we already saw, that there is a correlation between the length of a sample and the gesture type.
But, by processing the samples to get samples with equal lenghts, we lost this information.
Therefore, I added it manually as an extra feature, describing the original length of the recording.

#pagebreak()

= Feature Extraction
Yes I think it makes sense to derive more features besides the acceleration values, or at least try and look if there might be ones that make sense.
I chose to extract the following features.

*Based on the x-axis data (preprocessed):*
- raw values
- min
- max
- mean
- median
- standard deviation
- innerquartile range
- median absolute deviation
- number of maximas
- number of minimas
- zero crossing rate
- median crossing rate
- frequency power
- frequency angle
- autocorrelation
*Based on the 1st derivative of the x-axis data:*
- raw values
- min
- max
- mean
- median
- standard deviation
- innerquartile range
- median absolute deviation
- autocorrelation
#pagebreak()
*Based on the 2nd derivative of the x-axis data:*
- raw values
- min
- max
- mean
- median
- standard deviation
- innerquartile range
- median absolute deviation
- autocorrelation
*Based on the wavelet transformation of the x-axis data:*
- raw values
- min
- max
- mean
- median
- standard deviation
- innerquartile range
- median absolute deviation
- autocorrelation

Afterwards I evaluated the features and chose the most promising ones, as described in ... .
This will be described in depth in the next assignment.

#pagebreak()

== Raw x-axis data
== 1st derivative of x-axis data
== 2nd derivative of x-axis data
== Wavelet transformation of x-axis data

/* 
#pagebreak()

== Feature Extraction - Mean
By normalizing the data during the preprocessing, the mean was "destroyed", therefore I had to use the mean of the original data.
In @mean we can sense, that this value was calculated before preprocessing the data, especially looking on the amount of outliers.
#figure(
    image("../plots/mean_values_per_gesture.svg", height: 35%)
) <mean>

== Feature Extraction - Median
The median could be taken from the preprocessed data in @median it looks like there is at least some correlation with the gesture types.
#figure(
    image("images/median_values_per_gesture.svg", height: 35%)
) <median>

== Feature Extraction - Standard Deviation
Same as with the mean, the std had to be taken from the original data.
In @std we can see that the std is more robust compared to the mean in @mean, still there is a great amount of outliers.
#figure(
    image("images/std_values_per_gesture.svg", height: 35%)
) <std>

== Feature Extraction - Min value
@min shows the correlation between the min value and the gesture types.
It seems like there is at least some correlation, espeically the gestures left and right and the circles have very high minimum values.
#figure(
    image("images/min_values_per_gesture.svg", height: 35%)
) <min>

== Feature Extraction - Max value
@max shows that the max value does not correlate very good with the gesture type.
#figure(
    image("images/max_values_per_gesture.svg", height: 35%)
) <max>

== Feature Extraction - Innerquartile Range

== Feature Extraction - Median Absolute deviation (MAD)

== Feature Extraction - Zero Crossing Rate
@zerocrossings shows the number of zero crossings per gesture, which looks like a potentially good feature.
#figure(
    image("images/zero_crossings_per_gesture.svg", height: 35%)
) <zerocrossings>

== Feature Extraction - Median Crossing Rate
@mediancrossings shows the number of median crossings per gesture, which does not look as promising compared to the zero crossing rate in @zerocrossings.
#figure(
    image("images/median_crossings_per_gesture.svg", height: 35%)
) <mediancrossings>

== Feature Extraction - number of maximas 
I calculated the number of maximas per gesture, because I thought this information would be a very good feature.
But looking at @maximas we can see that this is not the case.
Still I kept it and waited for the PCA, to tell us how good this feature is.

#figure(
    image("images/maximas_per_gesture.svg", height: 35%)
) <maximas>

== Feature Extraction - number of minimas 
Same as with the maximas, we can see in @minimas that the number of minimas probably not a good feature.
#figure(
    image("images/minimas_per_gesture.svg", height: 35%)
) <minimas>



== Feature Extraction - 1st. derivative
plot by gesture?

== Feature Extraction - 2nd. derivative
plot by gesture? 

== Feature Extraction - frequency transformation
dont know how to plot

== Feature Extraction - autocorrelation function
dont know how to calculate

== Feature Extraction - wavelets
dont understand
*/

#pagebreak()

= Appendix

== Appendix Interpolations <ainterpolations>
#image("images/interpolation_left.svg", height: 40%)
#image("images/interpolation_right.svg", height: 40%)
#image("images/interpolation_up.svg", height: 45%)
#image("images/interpolation_down.svg", height: 45%)
#image("images/interpolation_square.svg", height: 45%)
#image("images/interpolation_triangle.svg", height: 45%)
#image("images/interpolation_circleCw.svg", height: 45%)
#image("images/interpolation_circleCcw.svg", height: 45%)

#pagebreak()

== Appendix Filtering <afiltering>
#image("images/filter_left.svg", height: 40%)
#image("images/filter_right.svg", height: 40%)
#image("images/filter_up.svg", height: 45%)
#image("images/filter_down.svg", height: 45%)
#image("images/filter_square.svg", height: 45%)
#image("images/filter_triangle.svg", height: 45%)
#image("images/filter_circleCw.svg", height: 45%)
#image("images/filter_circleCcw.svg", height: 45%)

