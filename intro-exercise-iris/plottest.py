import pandas as pd
import matplotlib.pyplot as plt

df_iris =  pd.read_csv('iris.csv')

# line plots
df_iris.plot()

plt.show()
