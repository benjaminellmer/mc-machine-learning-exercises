import warnings

import pywt
import statsmodels.api as sm
from scipy import signal
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.feature_selection import RFECV  # recursive feature eliminiation with CV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

from feature_extraction_helper import *
from preprocessing_helper import *
from selection_helper import *

warnings.filterwarnings("ignore")

########################################################################################################################
# Preprocessing
########################################################################################################################

original_df = pd.read_csv("raw_gesture_data_x_axis.csv", header=None)

# The sampleNr and participantNr are not more necessary for this assignment
original_df.drop(columns=[1, 2], inplace=True)

# Preprocessing - Missing Values
missing_values = original_df.apply(has_missing_value, axis=1)
print(f"The dataset has {missing_values.sum()} samples with missing values!")

# Preprocessing - Outliers
left_df = original_df[original_df[0] == "left"]
right_df = original_df[original_df[0] == "right"]
up_df = original_df[original_df[0] == "up"]
down_df = original_df[original_df[0] == "down"]
triangle_df = original_df[original_df[0] == "triangle"]
circleCw_df = original_df[original_df[0] == "circleCw"]

left_outliers = left_df[left_df.notna().sum(axis=1) > 150]
right_outliers = right_df[right_df.notna().sum(axis=1) > 200]
up_outliers = up_df[up_df.notna().sum(axis=1) > 150]
down_outliers = down_df[down_df.notna().sum(axis=1) > 200]
triangle_outliers = triangle_df[triangle_df.notna().sum(axis=1) > 350]
circleCw_outliers = circleCw_df[circleCw_df.notna().sum(axis=1) > 350]

original_df.drop(left_outliers.index, inplace=True)
original_df.drop(right_outliers.index, inplace=True)
original_df.drop(up_outliers.index, inplace=True)
original_df.drop(down_outliers.index, inplace=True)
original_df.drop(triangle_outliers.index, inplace=True)
original_df.drop(circleCw_outliers.index, inplace=True)

plot_outliers(original_df, left_outliers, "left")
plot_outliers(original_df, right_outliers, "right")
plot_outliers(original_df, up_outliers, "up")
plot_outliers(original_df, down_outliers, "down")
plot_outliers(original_df, triangle_outliers, "triangle")
plot_outliers(original_df, circleCw_outliers, "circleCw")

# Preprocessing - Feature Reduction
resized_df_200 = resize_df(original_df, 200)
resized_df_100 = resize_df(original_df, 100)
resized_df_50 = resize_df(original_df, 50)

# Makes it simpler to plot using functions
original_df.rename(columns={0: 'gesture'}, inplace=True)

plot_interpolation_comparison(original_df, resized_df_200, resized_df_100, resized_df_50)

# Preprocessing - Normalizing, scaling
scaled_df = resized_df_50.apply(lambda row: pd.Series(preprocessing.scale(row.iloc[0:50])), axis=1)

plot_density_comparison(resized_df_50, scaled_df)

# Preprocessing - Filtering
std_filtered_df = scaled_df.apply(lambda row: pd.Series(row).rolling(8).mean())
mean_filtered_df = scaled_df.apply(lambda row: pd.Series(row).rolling(8).std())
sv_filtered_df = scaled_df.apply(lambda row: savgol_filter(row, 8, 2))

# Add labels for the plots...
scaled_df["gesture"] = original_df.iloc[:, 0]
std_filtered_df["gesture"] = original_df.iloc[:, 0]
mean_filtered_df["gesture"] = original_df.iloc[:, 0]
sv_filtered_df["gesture"] = original_df.iloc[:, 0]

plot_filter_comparison(scaled_df, std_filtered_df, mean_filtered_df, sv_filtered_df)

# Preprocessing - Feature Addition
# Restore the lengths from the original acceleration samples
gesture_lengths = original_df.iloc[:, 1:].T.count()

# plt.show()

########################################################################################################################
# Feature Extraction
########################################################################################################################
labels = original_df.iloc[:, 0]

# Min
x_data_min_values = sv_filtered_df.apply(lambda row: pd.Series(row.iloc[0:50].min()), axis=1)

# Max
x_data_max_values = sv_filtered_df.apply(lambda row: pd.Series(row.iloc[0:50].max()), axis=1)

# Mean
x_data_mean_values = original_df.apply(lambda row: pd.Series(row.iloc[3:].mean()), axis=1)

# Median
x_data_median_values = sv_filtered_df.apply(lambda row: pd.Series(row.iloc[0:50].median()), axis=1)

# STD
x_data_std_values = original_df.apply(lambda row: pd.Series(row.iloc[3:].std()), axis=1)

# Inner Quartile Range
x_data_iqr_values = sv_filtered_df.apply(
    lambda row: pd.Series(row.iloc[0:50].quantile(0.75) - row.iloc[0:50].quantile(0.25)), axis=1
)

# MAD
x_data_mad_values = sv_filtered_df.apply(lambda row: pd.Series((row.iloc[0:50] - row.iloc[0:50].mean()).abs().mean()),
                                         axis=1)

# Zero Crossing Rate
x_data_zero_crossings = sv_filtered_df.apply(count_zero_crossings, axis=1)

# Median Crossing Rate
x_data_median_crossings = sv_filtered_df.apply(count_median_crossings, axis=1)

# Number of Minimas
x_data_number_of_minimas = sv_filtered_df.apply(
    lambda row: pd.Series(len(signal.argrelmin(row.iloc[0:50].dropna().values)[0])), axis=1
)

# Number of Maximas
x_data_number_of_maximas = sv_filtered_df.apply(
    lambda row: pd.Series(len(signal.argrelmax(row.iloc[0:50].dropna().values)[0])), axis=1
)

# FFT power and phase
x_data_power_spectrum = scaled_df.apply(calc_power, axis=1)
x_data_phase_spectrum = scaled_df.apply(calc_phase, axis=1)

# ACF
x_data_acf = sv_filtered_df.apply(
    lambda row: pd.Series(sm.tsa.acf(row.iloc[0:50], nlags=len(row.iloc[0:50]) - 1)), axis=1
)
x_data_acf.drop(columns=[0], inplace=True)  # is always 1

# Wavelets
# I chose level 4, same as in the lecturer code examples, I guess we could get more out of it by tuning the parameters
x_data_coeff = sv_filtered_df.apply(lambda row: pywt.wavedec(row.iloc[0:50], 'db3', level=4), axis=1)
x_data_approx_coeff = x_data_coeff.apply(lambda row: pd.Series(row[0]))
x_data_detail1_coeff = x_data_coeff.apply(lambda row: pd.Series(row[1]))
x_data_detail2_coeff = x_data_coeff.apply(lambda row: pd.Series(row[2]))
x_data_detail3_coeff = x_data_coeff.apply(lambda row: pd.Series(row[3]))
x_data_detail4_coeff = x_data_coeff.apply(lambda row: pd.Series(row[4]))

plot_raw_min_max_mean(sv_filtered_df.iloc[:, 0:50].T, x_data_min_values, x_data_max_values, x_data_mean_values, labels)
plt.savefig("plots/x_data_raw_min_max_mean.svg")

plot_median_std_iqr_mad(x_data_median_values, x_data_std_values, x_data_iqr_values, x_data_mad_values, labels)
plt.savefig("plots/x_data_median_std_iqr_mad.svg")

plot_minimas_maximas_zcr_mcr(
    x_data_number_of_minimas, x_data_number_of_maximas, x_data_zero_crossings, x_data_median_crossings, labels
)
plt.savefig("plots/x_data_minimas_maximas_zcr_mcr.svg")

# Caution this takes decades, because I plotted each row with its own command :/
plot_power_phase_acf(x_data_power_spectrum, x_data_phase_spectrum, sv_filtered_df)
plt.savefig("plots/x_data_power_phase_acf.svg")

plot_wavelet(
    x_data_approx_coeff,
    x_data_detail1_coeff,
    x_data_detail2_coeff,
    x_data_detail3_coeff,
    x_data_detail4_coeff,
    title="x-data wavelet transformation"
)
plt.savefig("plots/x_data_wavelet.svg")

# 1st Derivative
derivatives_1st = sv_filtered_df.apply(lambda row: pd.Series(np.diff(row.iloc[0:50])), axis=1)

# Min 1st Derivative
d1_min_values = derivatives_1st.apply(lambda row: pd.Series(row.iloc[0:49].min()), axis=1)

# Max 1st Derivative
d1_max_values = derivatives_1st.apply(lambda row: pd.Series(row.iloc[0:49].max()), axis=1)

# Mean 1st Derivative
d1_mean_values = derivatives_1st.apply(lambda row: pd.Series(row.iloc[0:49].mean()), axis=1)

# Median 1st Derivative
d1_median_values = derivatives_1st.apply(lambda row: pd.Series(row.iloc[0:49].median()), axis=1)

# STD 1st Derivative
d1_std_values = derivatives_1st.apply(lambda row: pd.Series(row.iloc[0:49].std()), axis=1)

# Inner Quartile Range 1st Derivative
d1_iqr_values = derivatives_1st.apply(
    lambda row: pd.Series(row.iloc[0:49].quantile(0.75) - row.iloc[0:49].quantile(0.25)), axis=1
)

# MAD 1st Derivative
d1_mad_values = derivatives_1st.apply(lambda row: pd.Series((row.iloc[0:49] - row.iloc[0:49].mean()).abs().mean()),
                                      axis=1)

plot_raw_min_max_mean(derivatives_1st.iloc[:, 0:49].T, d1_min_values, d1_max_values, d1_mean_values, labels)
plt.savefig("plots/d1_raw_min_max_mean.svg")

plot_median_std_iqr_mad(d1_median_values, d1_std_values, d1_iqr_values, d1_mad_values, labels)
plt.savefig("plots/d1_median_std_iqr_mad.svg")

# Wavelet 1st Derivative
# I chose level 4, same as in the lecturer code examples, I guess we could get more out of it by tuning the parameters
d1_coeff = derivatives_1st.apply(lambda row: pywt.wavedec(row.iloc[0:49], 'db3', level=4), axis=1)
d1_approx_coeff = d1_coeff.apply(lambda row: pd.Series(row[0]))
d1_detail1_coeff = d1_coeff.apply(lambda row: pd.Series(row[1]))
d1_detail2_coeff = d1_coeff.apply(lambda row: pd.Series(row[2]))
d1_detail3_coeff = d1_coeff.apply(lambda row: pd.Series(row[3]))
d1_detail4_coeff = d1_coeff.apply(lambda row: pd.Series(row[4]))

plot_wavelet(
    d1_approx_coeff,
    d1_detail1_coeff,
    d1_detail2_coeff,
    d1_detail3_coeff,
    d1_detail4_coeff,
    title="1st derivative wavelet transformation"
)
plt.savefig("plots/d1_wavelet.svg")

# 2nd Derivative
derivatives_2nd = derivatives_1st.apply(lambda row: pd.Series(np.diff(row.dropna())), axis=1)

# Min 2nd Derivative
d2_min_values = derivatives_2nd.apply(lambda row: pd.Series(row.iloc[0:49].min()), axis=1)

# Max 2nd Derivative
d2_max_values = derivatives_2nd.apply(lambda row: pd.Series(row.iloc[0:49].max()), axis=1)

# Mean 2nd Derivative
d2_mean_values = derivatives_2nd.apply(lambda row: pd.Series(row.iloc[0:49].mean()), axis=1)

# Median 2nd Derivative
d2_median_values = derivatives_2nd.apply(lambda row: pd.Series(row.iloc[0:49].median()), axis=1)

# STD 2nd Derivative
d2_std_values = derivatives_2nd.apply(lambda row: pd.Series(row.iloc[0:49].std()), axis=1)

# Inner Quartile Range 2nd Derivative
d2_iqr_values = derivatives_2nd.apply(
    lambda row: pd.Series(row.iloc[0:49].quantile(0.75) - row.iloc[0:49].quantile(0.25)), axis=1
)

# MAD 2nd Derivative
d2_mad_values = derivatives_2nd.apply(lambda row: pd.Series((row.iloc[0:49] - row.iloc[0:49].mean()).abs().mean()),
                                      axis=1)

d2_coeff = derivatives_2nd.apply(lambda row: pywt.wavedec(row.iloc[0:49], 'db3', level=4), axis=1)
d2_approx_coeff = d2_coeff.apply(lambda row: pd.Series(row[0]))
d2_detail1_coeff = d2_coeff.apply(lambda row: pd.Series(row[1]))
d2_detail2_coeff = d2_coeff.apply(lambda row: pd.Series(row[2]))
d2_detail3_coeff = d2_coeff.apply(lambda row: pd.Series(row[3]))
d2_detail4_coeff = d2_coeff.apply(lambda row: pd.Series(row[4]))

plot_wavelet(
    d2_approx_coeff,
    d2_detail1_coeff,
    d2_detail2_coeff,
    d2_detail3_coeff,
    d2_detail4_coeff,
    title="2nd derivative wavelet transformation"
)
plt.savefig("plots/d2_wavelet.svg")

plot_raw_min_max_mean(derivatives_2nd.iloc[:, 0:49].T, d2_min_values, d2_max_values, d2_mean_values, labels)
plt.savefig("plots/d2_raw_min_max_mean.svg")

plot_median_std_iqr_mad(d2_median_values, d2_std_values, d2_iqr_values, d2_mad_values, labels)
plt.savefig("plots/d2_median_std_iqr_mad.svg")

# plt.show()

########################################################################################################################
# Selection
########################################################################################################################
# Concat the final feature dataframe without labels/targets
sv_filtered_df.drop(columns=["gesture"], inplace=True)
features_df = pd.concat([
    gesture_lengths,
    sv_filtered_df, x_data_min_values, x_data_max_values, x_data_mean_values,
    x_data_median_values, x_data_std_values, x_data_iqr_values, x_data_mad_values,
    x_data_number_of_minimas, x_data_number_of_maximas, x_data_zero_crossings, x_data_median_crossings,
    x_data_power_spectrum, x_data_phase_spectrum,
    x_data_acf,
    x_data_approx_coeff, x_data_detail1_coeff, x_data_detail2_coeff, x_data_detail3_coeff, x_data_detail4_coeff,
    derivatives_1st, d1_min_values, d1_max_values, d1_mean_values,
    d1_median_values, d1_std_values, d1_iqr_values, d1_mad_values,
    d1_approx_coeff, d1_detail1_coeff, d1_detail2_coeff, d1_detail3_coeff, d1_detail4_coeff,
    derivatives_2nd, d2_min_values, d2_max_values, d2_mean_values,
    d2_median_values, d2_std_values, d2_iqr_values, d2_mad_values,
    d2_approx_coeff, d2_detail1_coeff, d2_detail2_coeff, d2_detail3_coeff, d2_detail4_coeff
], axis=1)

plot_correlation_matrix(features_df)

features_df.columns = features_df.columns.astype(str)

# PCA, Feature variance
plot_cumsum(features_df)
plt.savefig("plots/pca_cumsum.svg")

# Select K Best Features
targets = original_df.iloc[:, 0]

rfecv = RFECV(
    estimator=SGDClassifier(max_iter=1000, tol=1e-3),
    cv=KFold(3),
    scoring='accuracy'
)

# with only acceleration data
data_filtered_only_acc = SelectKBest(mutual_info_classif, k=50).fit_transform(sv_filtered_df, targets)
rfecv.fit(data_filtered_only_acc, targets)
plot_cross_validation_(rfecv, "Cross validation using SelectKBest with k=50 and acceleration data")
plt.savefig("plots/cross_validation_acc.svg")

# with all extracted features
data_filtered_all_features = SelectKBest(mutual_info_classif, k=100).fit_transform(features_df, targets)
rfecv.fit(data_filtered_all_features, targets)
plot_cross_validation_(rfecv, "Cross validation using SelectKBest with k=100 and all extracted features")
plt.savefig("plots/cross_validation_all.svg")

plt.show()
