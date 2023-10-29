# -*- coding: utf-8 -*-
"""
Created on Thu 19 Oct 2023

@author: Kefer.Kathrin
"""
#%%
########################################################
# DATA NORMALIZATION
########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

iris = pd.read_csv('iris.csv')
iris.boxplot()

#scale features
iris_scaled = preprocessing.scale(iris.iloc[:,:-1])
fig = plt.figure()
fig.suptitle('Scaled data')
ax = fig.add_subplot(111)
plt.boxplot(iris_scaled)
ax.set_xticklabels(iris.iloc[:,:-1].columns)
plt.show()

# mean and sd are 0 and 1 for each feature accross all samples
print(iris_scaled.mean(axis=0))
print(iris_scaled.std(axis=0))


#%%
########################################################
# DATA PREPROCESSING: EXAMPLES
########################################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

iris = pd.read_csv('iris.csv')

# feature correlation as plot
correlations = iris.corr(numeric_only=True)
print(correlations)
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(iris.iloc[:,:-1].columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(iris.iloc[:,:-1].columns)
ax.set_yticklabels(iris.iloc[:,:-1].columns)
plt.show()


#PCA-only example
pca = decomposition.PCA() # NOTE: if you put n_components=3 as parameter to the pca, the features are already stripped to 3! If you use a number >0 and <1, then it is the percentage of features that should be kept 
pca.fit(iris.iloc[:,:-1])

#the variance of the pca components in the same order as the original features
print(pca.explained_variance_)

pca_data = pca.transform(iris.iloc[:,:-1])
iris_targets = iris.iloc[:,-1]

ax = plt.figure().add_subplot(projection='3d')
colors = ('r', 'g', 'b') # Plot scatterplot data (150 samples --> 50 points per colour) on the x and z axes
c_list = []
for c in colors:
    c_list.extend([c] * 50)

    
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=c_list)
for name in pd.unique(iris_targets):
    ax.text(pca_data[iris_targets == name, 0].mean(),
            pca_data[iris_targets == name, 1].mean() + 1.5,
            pca_data[iris_targets == name, 2].mean(), name,
            horizontalalignment='center',color='k')
ax.legend()
ax.view_init(elev=48, azim=134)
plt.show()


#%%
########################################################
# FEATURE SELECTION
########################################################
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV # recursive feature eliminiation with CV
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

mtcars = pd.read_csv('mtcars.csv') #cyl should be the class = idx2
data=mtcars.drop(['cyl'], axis=1).drop(['model'], axis=1)
targets=mtcars.iloc[:,2]

# feature correlation as plot
correlations = mtcars.corr(numeric_only=True)
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()


# minimal example: feature filter using univariate filters
print("Original Data:" + str(data.shape))
data_filtered = SelectKBest(chi2, k=2).fit_transform(data, targets)
print("Filtered Data:" + str(data_filtered.shape))

# minimal example: feature wrapper using recursive feature elimination
sgdclass = SGDClassifier(max_iter=1000, tol=1e-3)# Create the RFE object and compute a cross-validated score.
rfecv = RFECV(estimator=sgdclass, cv=KFold(10), scoring='accuracy')# The "accuracy" scoring is proportional to the number of correct classifications
rfecv.fit(data, targets)

print("Optimal number of features: " + str(rfecv.n_features_))
print("Feature Ranking:" + str(rfecv.ranking_)) # the relative ranking of the features

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()


#%%
########################################################
# TIME SERIES FEATURE EXTRACTION
########################################################

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from statsmodels.graphics import tsaplots
import pywt # in anaconda environment it is called "pywavelets"
from dtw import dtw #needs to be installed via: python -m pip install dtw, https://github.com/pierre-rouanet/dtw
from scipy.spatial.distance import euclidean

beaver = pd.read_csv("beaver.csv") # body tempearture data of a beaver over time --> detect whether it is active or not!

#%%
# derivative (simplified...)
beaver_temp_scaled = preprocessing.scale(beaver.iloc[:,3])
fig=plt.figure()
plt.plot(beaver_temp_scaled) 

derivatives1st = np.diff(beaver_temp_scaled) # derivative = difference over values
plt.plot(derivatives1st) #1st derivative

derivatives2nd = np.diff(derivatives1st)
plt.plot(derivatives2nd)
plt.legend(['beaver_temp_scaled', '1st derivative', '2nd derivative'])
plt.show()

# filtering, e.g. with a runnig median, sg-filter, etc
d2 = medfilt(beaver.iloc[:,3], 5)
d3 = medfilt(beaver.iloc[:,3], 15)
d4 = medfilt(beaver.iloc[:,3], 31)
fig=plt.figure()
plt.plot(beaver.iloc[:,3])
plt.plot(d2)
plt.plot(d3)
plt.plot(d4)
plt.legend(['original', 'median filter window=5', 'median filter window=15', 'median filter window=31'])
plt.show()

d21=savgol_filter(beaver.iloc[:,3], 9, 1)
d22=savgol_filter(beaver.iloc[:,3], 9, 2)
d23=savgol_filter(beaver.iloc[:,3], 9, 3)
d24=savgol_filter(beaver.iloc[:,3], 9, 4)

fig=plt.figure()
plt.plot(beaver.iloc[:,3])
plt.plot(d21)
plt.plot(d22)
plt.plot(d23)
plt.plot(d24)
plt.legend(['original', 'savgol filter order=1', 'savgol filter order=2', 'savgol filter order=3', 'savgol filter order=4'])
plt.show()
# how much filtering? use domain specific knowledge ("do I need this spike or not?")

#%%
# interpolation
fig=plt.figure()
plt.plot(beaver.iloc[:,3], 'o') 
interpolate_linear = interp1d(np.arange(0,beaver.shape[0]), beaver.iloc[:,3], kind='linear') #this generates the interpolation function that still needs to be applied/carried out with the new step size!
beaver_lin1=interpolate_linear(np.arange(0,beaver.shape[0]-1,0.1))# generate more samples
plt.plot(np.arange(0,beaver.shape[0]-1,0.1),beaver_lin1, 'x')
interpolate_cubic = interp1d(np.arange(0,beaver.shape[0]), beaver.iloc[:,3], kind='cubic') 
beaver_cub=interpolate_cubic(np.arange(0,beaver.shape[0]-1,0.1))# generate more samples
plt.plot(np.arange(0,beaver.shape[0]-1,0.1),beaver_cub, '--')
beaver_lin2=interpolate_linear(np.arange(0,beaver.shape[0],2))
plt.plot(np.arange(0,beaver.shape[0]-1,2),beaver_lin2, 'o-')# generate less samples
plt.legend(['original', 'linear', 'cubic', 'linear - less samples'])
plt.show()

#%%
# approx can be used to easily normalize lengths of time series to a defined amount of data points
beaver_temp1=beaver.iloc[:55,3] #55 values
beaver_temp2=beaver.iloc[20:40,3] # 20 values
fig=plt.figure()
plt.plot(np.arange(0,beaver_temp1.shape[0]), beaver_temp1)
plt.plot(np.arange(0,beaver_temp2.shape[0]), beaver_temp2)
plt.show()
stepwidth = 0.01 
interpolate_linear_temp1=interp1d(np.arange(0,beaver_temp1.shape[0]), beaver_temp1, kind='linear')
interpolate_linear_temp2=interp1d(np.arange(0,beaver_temp2.shape[0]), beaver_temp2, kind='linear')
beaver_lin3=interpolate_linear_temp1(np.arange(0,beaver_temp1.shape[0]-1,beaver_temp1.shape[0]/100))   # we want 100 values
beaver_lin4=interpolate_linear_temp2(np.arange(0,beaver_temp2.shape[0]-1,beaver_temp2.shape[0]/100))              # we want 100 values
fig=plt.figure()
plt.plot(beaver_lin3)
plt.plot(beaver_lin4)
plt.show()

#%%
# very simple sliding window example
beaver_temp_scaled = preprocessing.scale(beaver.iloc[:,3])
fig=plt.figure()
plt.plot(beaver_temp_scaled,'o-')
sliding_win_1=pd.DataFrame(beaver_temp_scaled).rolling(3).std()
plt.plot(sliding_win_1, '--')
sliding_win_2=pd.DataFrame(beaver_temp_scaled).rolling(11).std()
plt.plot(sliding_win_2, 'x-')
plt.legend(['original_scaled', 'sliding window size=3', 'sliding window size=11'])
plt.show()
#the rolling can be applied accross multiple columns in one data frame at once (careful: applying it over columns is default and might not be what you want)

#%%
# autocorrelation ACF
beaver_temp_scaled = preprocessing.scale(beaver.iloc[:,3])
fig=plt.figure()
plt.plot(beaver_temp_scaled,'o-')
# Display the autocorrelation plot of your time series
fig = tsaplots.plot_acf(beaver_temp_scaled, lags=len(beaver_temp_scaled)-1)



#%%
# frequency transformation FFT
beaver_temp_scaled = preprocessing.scale(beaver.iloc[:,3])
fig=plt.figure()
plt.plot(beaver_temp_scaled,'o-')
fft_calc =  np.fft.fft(beaver_temp_scaled)
phase_spectrum = np.angle(fft_calc) # phase
plt.bar(list(range(len(fft_calc))),phase_spectrum)
power_spectrum=np.abs(fft_calc)**2 # power
plt.bar(list(range(len(fft_calc))), power_spectrum)
plt.legend(['original_scaled', 'FFT phase', 'FFT power'])
plt.show()


#%%
# wavelets
beaver_temp_scaled = preprocessing.scale(beaver.iloc[:,3])
plt.plot(beaver_temp_scaled)
#plot the wavelet
[phi, psi, x] = pywt.Wavelet('db3').wavefun(level=1)
plt.plot(x, psi)

 # DWT: Multilevel 1D Discrete Wavelet Transform of data.
coeffs = pywt.wavedec(beaver_temp_scaled, 'db3', level = 4) # db=Daubechie Wavelets, 3 = the specific wavelet number
#split up the coefficients to the details (one per level) plus the approximation of the coefficients
(approx_coeff, detail_coeff1, detail_coeff2, detail_coeff3,detail_coeff4) = coeffs 
#detail data for each level
plt.stem(detail_coeff1); 
plt.stem(detail_coeff2); 
plt.stem(detail_coeff3); 
plt.stem(detail_coeff4); 

plt.stem(approx_coeff); # ...and these are its obtained features for different levels (=scalings of wavelet)
# wavelets and multi resolution analysis are a very powerful concept: 
#   * they capture time and "frequency" information at once
#   * remember them if you need to go deep into time series analysis!
plt.legend(['Lvl 1 detail coefficients', 'Lvl 2 detail coefficients', 'Lvl 3 detail coefficients', 'Lvl 4 detail coefficients', 'Approximation Coefficients'])
plt.show()

#%%
# DTW
plt.plot(beaver.iloc[:,3])

beaver1 = beaver.iloc[:int(len(beaver)/2),3].values.reshape(-1, 1) #type must be numpy.ndarray!
beaver2 = beaver.iloc[int(len(beaver)/2):,3].values.reshape(-1, 1)

distance, cost_matrix, acc_cost_matrix, path = dtw(beaver1, beaver2, dist=euclidean)
print(distance)

fig=plt.figure()
plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()
