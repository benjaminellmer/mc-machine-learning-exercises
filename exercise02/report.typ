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

== Preprocessing - Outliers
Looking at the mean, or the std it is hard to determine which samples are real outliers.
In @x-median-std-iqr-mad and @x-raw-min-max-mean we can see that there is a large amount of outliers std and mean.
Since the data will also be filtered I decided to remove only the outliers that are visually recognizeable, especially by having an irregular length.
Those outliers are visualized in the figures @left-outliers, @right-outliers @up-outliers, @down-outliers, @triangle-outliers, @circleCw-outliers as the red samples. 

#figure(
    image("plots/left_outliers.svg", height: 35%)
) <left-outliers>
#figure(
    image("plots/right_outliers.svg", height: 35%)
) <right-outliers>
#figure(
    image("plots/up_outliers.svg", height: 35%)
) <up-outliers>
#figure(
    image("plots/down_outliers.svg", height: 35%)
) <down-outliers>
#figure(
    image("plots/triangle_outliers.svg", height: 35%)
) <triangle-outliers> 
#figure(
    image("plots/circleCw_outliers.svg", height: 35%)
) <circleCw-outliers>

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
    image("plots/density.svg", height: 40%)
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
The proof that it did make sense to derive more features is discussed in @feature-selection.

*Based on the x-axis data (preprocessed):*
- raw values
- min
- max
- mean 
- median
- standard deviation
- innerquartile range
- median absolute deviation
- number of minimas
- number of maximas
- zero crossing rate
- median crossing rate
- frequency power
- frequency angle
- autocorrelation
- wavelet components
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
- wavelet components
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
- wavelet components

Afterwards I evaluated the features and chose the most promising ones, as described in @feature-selection.

#pagebreak()

== Feature Plots based on raw x-axis data
*Note: * The mean and the std are calculated based on the original data, because the samples were scaled during preprocessing.
#figure(
    image("plots/x_data_raw_min_max_mean.svg", height: 40%)
) <x-raw-min-max-mean>
#figure(
    image("plots/x_data_median_std_iqr_mad.svg", height: 40%)
) <x-median-std-iqr-mad>
#figure(
    image("plots/x_data_minimas_maximas_zcr_mcr.svg", height: 40%)
) 

*FFT power, FFT phase, ACF:* \
I created a plot that includes FFT power, phase and the ACF plots, but the created svg had almost 80MB, therefore I did not include it in the protocol.

#figure(
    image("plots/x_data_wavelet.svg", height: 40%)
) 

#pagebreak()

== Feature plots based on 1st derivative 
#figure(
    image("plots/d1_raw_min_max_mean.svg", height: 40%)
) 
#figure(
    image("plots/d1_median_std_iqr_mad.svg", height: 40%)
) 
#figure(
    image("plots/d1_wavelet.svg", height: 50%)
) 

#pagebreak()

== Feature plots based on 2nd derivative of x-axis data
#figure(
    image("plots/d2_raw_min_max_mean.svg", height: 40%)
) 
#figure(
    image("plots/d2_median_std_iqr_mad.svg", height: 40%)
) 
#figure(
    image("plots/d2_wavelet.svg", height: 50%)
) 

#pagebreak()

= Feature Selection <feature-selection>
First of all, I plotted the correlation matrix to get a picture of the correlations beyond the features.
Because of the huge amount of features, it is hard to get any useful information out of the plot.
I removed the ticks and labels, because the texts were overlapping.

#figure(
    image("plots/feature_correlations.svg", height: 40%)
) <correlations>

Still, we can see in @correlations, that many features have a strong correlation and that we might be able to drop some of them.
This is very interesting in the next assignment I will check which accuracy I can reach with only 10 features.


#pagebreak()

Then I chose PCA to look at the variance of the features.
In @cumsum we can see that we should have about 99% variance with only 10 features.

#figure(
    image("plots/pca_cumsum.svg", height: 40%)
) <cumsum>

#pagebreak()

Before looking at the cross validation score of all extracted features I wanted to check the cross validation scores of only the acceleration data.
If there would not be much difference, all the extracted features would be useless.
@cv-acc shows the results with only the acceleration values.

#figure(
    image("plots/cross_validation_acc.svg", height: 35%)
) <cv-acc>

@cv-all shows that it was worth extracting the features reaching almost 55%, which is about 25% better than using only the acceleration data.

#figure(
    image("plots/cross_validation_all.svg", height: 35%)
) <cv-all>


#pagebreak()



= Appendix

== Appendix Interpolations <ainterpolations>
#image("plots/interpolation_left.svg", height: 40%)
#image("plots/interpolation_right.svg", height: 40%)
#image("plots/interpolation_up.svg", height: 45%)
#image("plots/interpolation_down.svg", height: 45%)
#image("plots/interpolation_square.svg", height: 45%)
#image("plots/interpolation_triangle.svg", height: 45%)
#image("plots/interpolation_circleCw.svg", height: 45%)
#image("plots/interpolation_circleCcw.svg", height: 45%)

#pagebreak()

== Appendix Filtering <afiltering>
#image("plots/filter_left.svg", height: 40%)
#image("plots/filter_right.svg", height: 40%)
#image("plots/filter_up.svg", height: 45%)
#image("plots/filter_down.svg", height: 45%)
#image("plots/filter_square.svg", height: 45%)
#image("plots/filter_triangle.svg", height: 45%)
#image("plots/filter_circleCw.svg", height: 45%)
#image("plots/filter_circleCcw.svg", height: 45%)

