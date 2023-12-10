import warnings

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

from selection_helper import *
import pandas as pd

features_df = pd.read_csv("features.csv")
targets = features_df["target"]
features_df.drop(columns=["target"], inplace=True)

run = 2

# PCA
# plot_cumsum(features_df)
# plt.savefig("plots/pca_cumsum.svg")

# Get the 10 principal components with the highest variance 10 components = 99% variance in this case
pca = decomposition.PCA(n_components=10)
pca_result = pca.fit_transform(features_df)
pca_df = pd.DataFrame(pca_result, columns=[f"PC_{i}" for i in range(pca.n_components)])

data_train, data_test, target_train, target_test = train_test_split(
    pca_df, targets, train_size=0.75, random_state=432000
)

# KNN
# knn_neighbours = 20
# knn_cv = 15
#
# parameters_to_tune = {'n_neighbors': [n for n in range(1, knn_neighbours)]}
# knnClassification = GridSearchCV(KNeighborsClassifier(), parameters_to_tune, cv=knn_cv, scoring='accuracy')
# knnClassification.fit(data_train, target_train)
#
# train_predicted = knnClassification.predict(data_train)
# test_predicted = knnClassification.predict(data_test)
#
# train_accuracy = accuracy_score(target_train, train_predicted)
# test_accuracy = accuracy_score(target_test, test_predicted)

# train_confusion_matrix = confusion_matrix(target_train, train_predicted)
# plot_confusion_matrix(
#     train_confusion_matrix,
#     targets,
#     f"KNN Train Confusion Matrix (k=1-{knn_neighbours}) (cv={knn_cv})\nAccuracy={train_accuracy:.3f}"
# )
# plt.savefig("plots/knn_train_cm.svg")

# test_confusion_matrix = confusion_matrix(target_test, test_predicted)
# plot_confusion_matrix(
#     test_confusion_matrix,
#     targets,
#     f"KNN Test Confusion Matrix (k=1-{knn_neighbours}) (cv={knn_cv})\nAccuracy={test_accuracy:.3f}"
# )
# plt.savefig("plots/knn_test_cm.svg")

# Real Model Selection

names = ["NearestNeighbors", "SVC", "DecisionTree", "RandomForest"]
classifiers = {
    'NearestNeighbors': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier()
}

params = {
    'NearestNeighbors': {'n_neighbors': [7, 8, 9]},
    'SVC': [
        {'kernel': ['linear'], 'C': [3 ** n for n in range(-5, 5, 2)]},
    ],
    'DecisionTree': {'max_depth': range(10, 20)},
    'RandomForest': {'max_depth': [5, 10, 20], 'n_estimators': [10, 20, 30]}
}

search_algorithms = {
    'NearestNeighbors': SearchAlgorithm.GridSearchCV,
    'SVC': SearchAlgorithm.RandomizedSearchCV,
    'DecisionTree': SearchAlgorithm.GridSearchCV,
    'RandomForest': SearchAlgorithm.GridSearchCV,
}

kfold = KFold(n_splits=5)

for name in names:
    if os.path.exists(f"trained_{name}_{run}.sav"):
        print(f"Loaded model for {name}. If you changed the parameters, delete trained_{name}.sav and rerun!")
    else:
        print(f"Running {search_algorithms[name].name} for {name}.")
        model = classifiers[name]
        parameter_set = params[name]
        gs = instantiate_search(search_algorithms[name], model, parameter_set, kfold)
        gs.fit(data_train, target_train)
        joblib.dump(gs, f"trained_{name}_{run}.sav")

loaded_models = {}
for name in names:
    loaded_models[name] = joblib.load(f"trained_{name}_{run}.sav")

results_test_acc = []
results_test_std = []
results_train_acc = []
results_train_std = []

labels = []
for model_name in loaded_models:
    results_test_acc.extend(loaded_models[model_name].cv_results_['mean_test_score'])
    results_test_std.extend(loaded_models[model_name].cv_results_['std_test_score'])
    for params in loaded_models[model_name].cv_results_['params']:
        labels.append(get_label(model_name, params))

# boxplot algorithm comparison
fig = plt.figure(figsize=(8, 6))
fig.suptitle(f"Algorithm Comparison - Test Data Accuracy with Standard Deviation and cv={kfold.n_splits}")
plt.subplots_adjust(bottom=0.25)
plt.bar(np.arange(len(results_test_acc)), results_test_acc, yerr=results_test_std)
plt.xticks(np.arange(len(results_test_acc)), labels, rotation='vertical')
plt.ylabel("Accuracy")
plt.savefig(f"plots/comparison_run_{run}.svg")
plt.show()
