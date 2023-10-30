import pandas as pd

df_iris =  pd.read_csv('iris.csv')

print(df_iris)

# STATISTICS & INFO ABOUT DATA - a few examples
df_iris.dtypes


df_iris.describe() # statistics

# statistics about the discrete feature = the class
print(df_iris['Name'].value_counts())

df_iris_class = df_iris['Name']
print(df_iris_class)
df_iris_class.describe()
df_iris_data = df_iris.iloc[:,:-1]
df_iris_data.describe()
print(df_iris_data)


#%%

print("------- MEAN + MAD -----------")
# measures of "average" (average != mean)
# print(df_iris.mean()) #--> warning here due to non-numeric values in the dataset (= the "Name" column)
print(df_iris_data.mean()) # Option 1: select only numeric colums by hand
print(df_iris.mean(numeric_only=True)) #Option 2: add "numeric_only" parameter

print(df_iris_data.mean(axis=1)) # not meaningful on the iris data!
print(df_iris_data.median()) # don't know what the median is? google it!

#%%

print("------- Quartiles + IRQ -----------")
# Q1 = 25%
print(df_iris_data.quantile(0.25))
# median = Q2 = 50%
print(df_iris_data.quantile(0.5))
# Q3 = 75%
print(df_iris_data.quantile(0.75))
# innerquartile range = Q3 - Q1
print(df_iris_data.quantile(0.75)-df_iris_data.quantile(0.25))

#%%
# measures of spread: standard deviation (sd or std), mad
print("---------- Standard Deviation + MAD ------------")
print (df_iris_data.std())
# print(df_iris_data.mad()) # will be deprecated soon --> (df - df.mean()).abs().mean()

#%% PLOTTING WITH PANDAS
import pandas as pd

df_iris =  pd.read_csv('iris.csv')

# line plots
df_iris.plot()
df_iris.plot(linestyle=':', linewidth=2) # color='red', ...

# scatterplot matrix, pairplot
pd.plotting.scatter_matrix(df_iris)
pd.plotting.scatter_matrix(df_iris, alpha=0.2)
pd.plotting.scatter_matrix(df_iris, alpha=0.2, diagonal='kde')

# with selected features
df_iris.plot.scatter(x=0, y=1, c='red')
# "." as a marker can speed up plotting significantly
# you can also use column names for x and y
df_iris.plot.scatter(x="SepalLength", y="SepalWidth", marker='.', c='blue')

# boxplot, histograms, density plots, etc...
df_iris.plot.box()
df_iris.boxplot(by='Name')
df_iris.hist()
df_iris.plot.hist(alpha=0.5)
df_iris.plot.hist(alpha=0.5, stacked=True)
df_iris.plot.density()


#%% PLOTTING WITH MATPLOTLIB
import matplotlib.pyplot as plt
import pandas as pd

df_iris =  pd.read_csv('iris.csv')

fig = plt.figure()
plt.plot(df_iris['SepalLength'])
plt.plot(df_iris['SepalWidth'], color='red', linestyle=':')
plt.title('title!')
plt.xlabel('x axis!')
plt.ylabel('y axis!')
plt.legend(['SepalLength', 'SepalWidth']) #adding legend manually
# if we have the labels specified in the plot like that:
# plt.plot(data, label=['col1','col2','col3'])
# we could also use that command to plot the legend plt.legend()

# save plot info file
plt.savefig('iris_boxplot.png')
plt.savefig('iris_boxplot.svg')
plt.savefig('iris_boxplot.pdf')
# in the interactive console you need to execute the savefig in the same call (e.g. running all code with F9 at once)
#   --> line by line execution in the interactive console causes the plots to be empty
df_iris.plot.box()
plt.savefig('iris_boxplot.png')
plt.savefig('iris_boxplot.svg')
plt.savefig('iris_boxplot.pdf')

#%% VECTORIZED COMMANDS
import pandas as pd

df_iris =  pd.read_csv('iris.csv')

len(df_iris.query("SepalLength > 5.5"))
len(df_iris.query("SepalWidth < 3.5 and Name == 'Iris-setosa'"))
