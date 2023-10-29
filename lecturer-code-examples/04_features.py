# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:56:52 2019

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

iris = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv') # 'iris.csv')
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

iris = pd.read_csv('iris.csv')

data_train, data_test, target_train, target_test = train_test_split(iris.iloc[:,:-1], iris.iloc[:,4], train_size=0.75, random_state=123456)

# feature correlation as plot
correlations = iris.corr()
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

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
for name in pd.unique(iris_targets):
    ax.text3D(pca_data[iris_targets == name, 0].mean(),
              pca_data[iris_targets == name, 1].mean() + 1.5,
              pca_data[iris_targets == name, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

#PCA in preprocessing pipeline - example
tuning_params_svm = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}
param_grid = {}

for key,value in tuning_params_svm.items():
    hyperparam_key = "classify__" + key
    param_grid[hyperparam_key] = value

print(param_grid)

pipe = Pipeline([
        ('normalization', StandardScaler()), #= center + scale
        ('pca', decomposition.PCA(n_components=0.85)), #n_components=0.85 equals to 85% of the features that should be kept. if this is set to an integer >=1, then this is the exact number of features that should be kept!
        ('classify', SVC(kernel='rbf'))
])

gs = GridSearchCV(pipe, param_grid=param_grid, cv=10)
gs.fit(data_train, target_train)

#Best estimator hyperparameters:
print(gs.best_params_)

#Best estimator scaling for each feature:
print(gs.best_estimator_.steps[0][1].scale_)
print(gs.best_estimator_.steps[0][1].var_)
print(gs.best_estimator_.steps[0][1].mean_)

#Best estimator selected PCA components' variance
print(gs.best_estimator_.steps[1][1].explained_variance_)

#Apply the best estimator to the test data. THIS ESTIMATOR ALREADY CONTAINS THE NORMALIZATION + FINAL PCA WHICH IS ALSO APPLIED TO THIS TEST DATA
predicted_classes = gs.best_estimator_.predict(data_test)
predicted_accuracy = accuracy_score(target_test, predicted_classes)
cm = confusion_matrix(target_test, predicted_classes) # create confusion matrix over all involved classes
print(cm)
print(predicted_accuracy)

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
correlations = mtcars.corr()
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
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#%%
########################################################
# FEATURE IMPORTANCE FROM MODEL
########################################################
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


mtcars = pd.read_csv('mtcars.csv') #cyl should be the class = idx2
data=mtcars.drop(['cyl'], axis=1).drop(['model'], axis=1)
targets=mtcars.iloc[:,2]

sgdclass = SGDClassifier(max_iter=1000, tol=1e-3)
sgdclass.fit(data, targets)
print(sgdclass.coef_) # Weights assigned to the features.

#Other models offer different ways of extracting the feature importance, e.g.:
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(data, targets)
print(rf.feature_importances_)