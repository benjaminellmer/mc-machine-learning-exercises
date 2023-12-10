# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:35:12 2019

@author: Kefer.Kathrin
"""

#%%
########################################################
# DATA PARTITIONING AND REGRESSION MODEL TRAINING
########################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
iris = pd.read_csv('iris.csv')

data_train, data_test, target_train, target_test = train_test_split(iris.iloc[:,1:4], iris.iloc[:,0], train_size=0.75, random_state=123456)

parameters_to_tune = {'n_neighbors': [n for n in range(1,10)]} 
# predict first numeric feature (target variable)
knnRegression = GridSearchCV(KNeighborsRegressor(), parameters_to_tune, cv=10, scoring='neg_mean_squared_error')
knnRegression.fit(data_train, target_train)  # ensure that the second parameter is a numeric for regression and a factor for classification

#
# A) Internally, train does random splits into CV partions, so results will be different each run
# To make train results reproducible use the "random_state" parameter of the estimator (in this case the KNeighborsRegressor) if available!
#   this can be used to do gallery independent partitioning with CV
#   (needs to contain n+1 seeds for n CV partitions)
# B) If you repeat your training and CV performance changes largely, you probably need more repeats
#

# computed model -- provides further details 
print(knnRegression)
#Best parameters set found
print (knnRegression.best_params_)
#Best score found
print(knnRegression.best_score_)

# regression error metrics 
# training data performance 
training_predicted = knnRegression.predict(data_train) # let model predict output variable for all training set samples
fig, ax=plt.subplots()
ax.plot(training_predicted,target_train, "o") # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c=".3") #draws the diagonal line that represents the ideal fit: with an error free model, all predictions would be on this line
plt.show()
print(np.sqrt(np.mean((training_predicted - target_train)**2))) # RMSE on training data

# test data performance
test_predicted = knnRegression.predict(data_test)
fig, ax=plt.subplots()
ax.plot(test_predicted,target_test, "o") # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c=".3") #draws the diagonal line that represents the ideal fit: with an error free model, all predictions would be on this line
plt.show()
print(np.sqrt(np.mean((test_predicted - target_test)**2))) # RMSE on test data

# try different models + parameter grids!

#%%
########################################################
# DATA PARTITIONING AND CLASSIFICATION MODEL TRAINING
########################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
iris = pd.read_csv('iris.csv')

#train/test split
data_train, data_test, target_train, target_test = train_test_split(iris.iloc[:,0:4], iris.iloc[:,4], train_size=0.75, random_state=123456)

parameters_to_tune = {'n_neighbors': [n for n in range(1,10)]} 

knnClassification = GridSearchCV(KNeighborsClassifier(), parameters_to_tune, cv=10, scoring='accuracy')
knnClassification.fit(data_train, target_train)  

# computed model -- provides further details 
print(knnClassification)

train_predicted=knnClassification.predict(data_train);# let model predict output variable for all training set samples
test_predicted = knnClassification.predict(data_test) # let model predict output variable for all test set samples

# Confusion matrices
# Training confusion Matrix - this is based on the apparent error!
cm = confusion_matrix(target_train, train_predicted) # create confusion matrix over all involved classes
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    # ... and label them with the respective list entries
    xticklabels=pd.unique(iris.iloc[:,4]), yticklabels=pd.unique(iris.iloc[:,4]),
    title='Confusion Matrix Iris Dataset',
    ylabel='True label',
    xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

plt.show()

# Test confusion matrix: realistic error estimate from held back test set, usually higher than apparent error
# This is the one you usually want to use during development
cm = confusion_matrix(target_test, test_predicted) # create confusion matrix over all involved classes
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    # ... and label them with the respective list entries
    xticklabels=pd.unique(iris.iloc[:,4]), yticklabels=pd.unique(iris.iloc[:,4]),
    title='Confusion Matrix Iris Dataset',
    ylabel='True label',
    xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

plt.show()

#%%
########################################################
# MODEL SELECTION
########################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib 
iris = pd.read_csv('iris.csv')

data_train, data_test, target_train, target_test = train_test_split(iris.iloc[:,0:4], iris.iloc[:,4], train_size=0.75, random_state=123456)

names = [
        "NearestNeighbors", "SVC", "DecisionTree", "RandomForest"]

classifiers = {
    'NearestNeighbors': KNeighborsClassifier(), 
    'SVC': SVC(), 
    'DecisionTree': DecisionTreeClassifier(), 
    'RandomForest': RandomForestClassifier()
}

params = {
    'NearestNeighbors': {'n_neighbors': [1, 3, 5, 7, 10, 15]},
    'SVC': [
        {'kernel': ['linear'], 'C': [3**n for n in range(-5,5)]},
        {'kernel': ['rbf'], 'C': [3**n for n in range(-5,5)], 'gamma':[3**n for n in range(-5,5)]}
    ],
    'DecisionTree': {'max_depth':[5, 10, 20]},
    'RandomForest':{'max_depth':[5, 10, 20], 'n_estimators':[1, 3, 5, 10]}
}

kfold = KFold(n_splits=10)
grid_searches={}

#fit for all defined models and parameter configurations
for name in names:
    print("Running GridSearchCV for %s." % name)
    model = classifiers[name]
    parameterset = params[name]
    gs = GridSearchCV(model, parameterset, cv=kfold, scoring='accuracy', verbose=1)
    gs.fit(data_train,target_train)
    grid_searches[name] = gs    
 
# we can easily save and load all models for later usage/analysis
joblib.dump(grid_searches, 'trained_models.sav')

#load models again
loaded_models = joblib.load('trained_models.sav')

#extract the results and the hyperparameter configuration as labels
results_test_acc = []
results_test_std=[]
results_train_acc=[]
results_train_std=[]
labels = []
for model_name in loaded_models:
    results_test_acc.extend(loaded_models[model_name].cv_results_['mean_test_score'])
    results_test_std.extend(loaded_models[model_name].cv_results_['std_test_score'])
    for params in loaded_models[model_name].cv_results_['params']:
        label = model_name
        for param in params:
                label = label + '_' +param+':'+str(params[param])
        labels.append(label)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison - Test Data Accuracy with Standard Deviation')
plt.bar(np.arange(len(results_test_acc)), results_test_acc, yerr=results_test_std)
plt.xticks(np.arange(len(results_test_acc)), labels, rotation='vertical')
plt.ylabel('Accuracy')
plt.show()


#%%
########################################################
# Specialties: Gallery Independent CV partitioning
########################################################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

# example: predict iris sepal length from sepal width and petal length. this should also work 
#   on new species of flowers = be gallery independent accross flower spcies.
# therefore need to split data in gallery independent way
iris = pd.read_csv('iris.csv')
sns.pairplot(iris, hue='Name')

# a) NOT gallery independent (across iris classes)
# shuffle data + create train/test partitions
data_train, data_test, target_train, target_test = train_test_split(iris.iloc[:,1:3], iris.iloc[:,0], train_size=0.5, random_state=123456)
sns.pairplot(pd.concat([data_train,target_train],axis=1))

# train
linearRegression = LinearRegression()
linearRegression.fit(data_train, target_train)

# computed model -- provides further details 
print(linearRegression)

# train error
train_predicted = linearRegression.predict(data_train)
fig, ax=plt.subplots()
ax.plot(train_predicted,target_train, "o") # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c=".3") #draws the diagonal line that represents the ideal fit: with an error free model, all predictions would be on this line
plt.show()
print("RMSE on training data: " + str(np.sqrt(np.mean((train_predicted - target_train)**2)))) # RMSE on training data

# test data performance
test_predicted = linearRegression.predict(data_test)
fig, ax=plt.subplots()
ax.plot(test_predicted,target_test, "o") # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c=".3") #draws the diagonal line that represents the ideal fit: with an error free model, all predictions would be on this line
plt.show()
print(" RMSE on test data:" + str(np.sqrt(np.mean((test_predicted - target_test)**2))))

# all looks good so far - but there is a hidden problem...

# b) gallery independent (across iris classes)
# as we only have 3 classes here we don't keep a separate test partition now
# with real-life scenarios always keep a gallery idependent test partition as well (data from people not used in training at all)
# create the cv-folds per hand = indexes of iris sample split by their classes

for i in pd.unique(iris.iloc[:,4]): # iterate over datasets
    # divide the data by classes
    train_data=iris[iris['Name']!=i].iloc[:,1:3] 
    train_target=iris[iris['Name']!=i].iloc[:,0]
    test_data=iris[iris['Name']==i].iloc[:,1:3]
    test_target=iris[iris['Name']==i].iloc[:,0]
    sns.pairplot(iris[iris['Name']!=i], hue='Name')
    
    linearRegression = LinearRegression()
    linearRegression.fit(train_data, train_target) 
    #   prediciton works well for some but not all subjects (in our case: classes of flowers)
    #   when training from versicolor and virginica, predicting the target variable for the new subject setosa is difficult
    #   this has been hidden with gallery dependent data splitting (using sample of all subjects in training)
    #   review why: comes from different relation between features and target variable over subjects
    #   clearly visible in our data (often difficult to find out with real data!)
    
    predicted_train=linearRegression.predict(train_data)
    #print("RMSE for predicting training data: "+ str(np.sqrt(np.mean((predicted_train-train_target)**2))))
    predicted = linearRegression.predict(test_data)#predict the remaining target
    print("RMSE for predicting " + i + ":" + str(np.sqrt(np.mean((predicted - test_target)**2))))