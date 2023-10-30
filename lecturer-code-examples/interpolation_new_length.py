import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

beaver = pd.read_csv(
    "beaver.csv")  # body tempearture data of a beaver over time --> detect whether it is active or not!

# %%
# approx can be used to easily normalize lengths of time series to a defined amount of data points

# This part here is just to "fake" that we have uneqal data lenghts
temperature_data_55 = beaver.iloc[:55, 3]  # 55 values
temperature_data_20 = beaver.iloc[20:40, 3]  # 20 values

print(temperature_data_20)

temperature_data_20_x1 = np.arange(0, temperature_data_20.shape[0])
temperature_data_55_x1 = np.arange(0, temperature_data_55.shape[0])

fig = plt.figure()
plt.plot(temperature_data_20_x1, temperature_data_20)
plt.plot(temperature_data_55_x1, temperature_data_55)
plt.show()

targetSize = 100
interpolate_linear_55 = interp1d(temperature_data_55_x1, temperature_data_55, kind='linear')
interpolate_linear_20 = interp1d(temperature_data_20_x1, temperature_data_20, kind='linear')

lin55 = interpolate_linear_55(
    np.arange(0, temperature_data_55.shape[0] - 1, temperature_data_55.shape[0] / targetSize)
)

lin20 = interpolate_linear_20(
    np.arange(0, temperature_data_20.shape[0] - 1, temperature_data_20.shape[0] / targetSize)
)

fig = plt.figure()
plt.plot(lin55)
plt.plot(lin20)
plt.show()
