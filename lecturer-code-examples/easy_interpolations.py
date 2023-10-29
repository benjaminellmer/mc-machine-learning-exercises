# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:48:43 2019

@author: Kefer.Kathrin
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.interpolate import interp1d

beaver = pd.read_csv(
    "beaver.csv")  # body tempearture data of a beaver over time --> detect whether it is active or not!

# interpolation
fig = plt.figure()

temperature_data = beaver.iloc[:, 3]
x_data_1_steps = np.arange(0, beaver.shape[0])
x_data_01_steps = np.arange(0, beaver.shape[0] - 1, 0.1)

plt.plot(temperature_data, 'o')

# this generates the interpolation function that still needs to be applied/carried out with the new step size!
interpolate_linear = interp1d(x_data_1_steps, temperature_data, kind='linear')
interpolate_cubic = interp1d(x_data_1_steps, temperature_data, kind='cubic')

# plt.plot(x_data_01_steps, interpolate_linear(x_data_01_steps), '-')
# plt.plot(x_data_1_steps, interpolate_linear(x_data_1_steps), '-')
plt.plot(x_data_01_steps, interpolate_cubic(x_data_01_steps), 'x')

# beaver_cub = interpolate_cubic(np.arange(0, beaver.shape[0] - 1, 0.1))  # generate more samples

# plt.plot(np.arange(0, beaver.shape[0] - 1, 0.1), beaver_cub, '--')
# beaver_lin2 = interpolate_linear(np.arange(0, beaver.shape[0], 2))

# plt.plot(np.arange(0, beaver.shape[0] - 1, 2), beaver_lin2, 'o-')  # generate less samples
plt.show()

# %%
