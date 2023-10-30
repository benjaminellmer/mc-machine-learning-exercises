# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:48:43 2019

@author: Kefer.Kathrin
"""
#%%
########################################################
# TIME SERIES: EXAMPLES
########################################################
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from statsmodels.graphics import tsaplots
import pywt
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
plt.plot(beaver_temp_scaled,'o-')
fft_calc =  np.fft.fft(beaver_temp_scaled)
phase_spectrum = np.angle(fft_calc) # phase
plt.bar(list(range(len(fft_calc))),phase_spectrum)
power_spectrum=np.abs(fft_calc)**2 # power
plt.bar(list(range(len(fft_calc))), power_spectrum)


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
plt.stem(detail_coeff1); plt.legend(['Lvl 1 detail coefficients'])
plt.stem(detail_coeff2); plt.legend(['Lvl 2 detail coefficients'])
plt.stem(detail_coeff3); plt.legend(['Lvl 3 detail coefficients'])
plt.stem(detail_coeff4); plt.legend(['Lvl 4 detail coefficients'])

plt.stem(approx_coeff); plt.legend(['Approximation Coefficients']) # ...and these are its obtained features for different levels (=scalings of wavelet)
# wavelets and multi resolution analysis are a very powerful concept: 
#   * they capture time and "frequency" information at once
#   * remember them if you need to go deep into time series analysis!

#%%
# DTW
plt.plot(beaver.iloc[:,3])

beaver1 = beaver.iloc[:int(len(beaver)/2),3].values #type must be numpy.ndarray!
beaver2 = beaver.iloc[int(len(beaver)/2):,3].values

distance, cost_matrix, acc_cost_matrix, path = dtw(beaver1, beaver2, dist=euclidean)
print(distance)

fig=plt.figure()
plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()
