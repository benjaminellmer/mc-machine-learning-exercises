########################################################################################################################
# Feature Selection Helper Function Definitions
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from enum import Enum


def plot_correlation_matrix(features_df):
    correlations = features_df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.suptitle("Feature Correlation Matrix")
    fig.colorbar(cax)
    fig.savefig("plots/feature_correlations.svg")
    # ticks = np.arange(0, len(corr_df.columns) + 1, 10)
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(["0", "9", "18", "27", "36", "gesture"])
    # ax.set_yticklabels(["0", "10", "20", "30", "40", "gesture"])


def plot_cumsum(features_df):
    pca_all = decomposition.PCA()
    pca_all.fit(features_df)

    pca_20 = decomposition.PCA(n_components=20)
    pca_20.fit(features_df)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.tight_layout(pad=3)

    fig.suptitle("Cumulative Variance vs. Number of Features")
    fig.supxlabel('Number of Features')
    fig.supylabel('Cumulative Explained Variance')

    ax1.plot(np.cumsum(pca_all.explained_variance_ratio_))
    ax2.plot(np.cumsum(pca_20.explained_variance_ratio_))
    return fig


def plot_cross_validation(rfecv, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of selected features")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])


def get_label(name, parameters):
    shortened_label = ""
    if name == "NearestNeighbors":
        param_nn = parameters["n_neighbors"]
        shortened_label += f"KNN n:{param_nn}"
    elif name == "DecisionTree":
        param_max_depth = parameters["max_depth"]
        shortened_label += f"DT d:{param_max_depth}"
    elif name == "RandomForest":
        param_max_depth = parameters["max_depth"]
        shortened_label += f"RF d:{param_max_depth}"
        param_n_estimators = parameters.get("n_estimators", None)
        if param_n_estimators is not None:
            shortened_label += f" n_est:{param_n_estimators}"
    elif name == "SVC":
        kernel = parameters["kernel"]
        param_c = parameters["C"]
        shortened_label += f"SVC_{kernel[:3]} C:{param_c:.1f}"
        param_gamma = parameters.get("gamma", None)
        if param_gamma is not None:
            shortened_label += f" G:{param_gamma:.1f}"
    else:
        shortened_label += name
    return shortened_label


def plot_confusion_matrix(confusion_matrix, targets, title=""):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        title=title,
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=pd.unique(targets),
        yticklabels=pd.unique(targets),
        ylabel="True label",
        xlabel="Predicted label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


class SearchAlgorithm(Enum):
    GridSearchCV = 1
    RandomizedSearchCV = 2


def instantiate_search(search_algorithm, model, parameter_set, cv, scoring='accuracy'):
    # n_jobs = -1 means, all CPUs are used
    # https://stackoverflow.com/questions/50183080/how-to-find-an-optimum-number-of-processes-in-gridsearchcv-n-jobs
    if search_algorithm == SearchAlgorithm.RandomizedSearchCV:
        return RandomizedSearchCV(model, parameter_set, cv=cv, scoring=scoring, verbose=1, n_jobs=-1)
    elif search_algorithm == SearchAlgorithm.GridSearchCV:
        return GridSearchCV(model, parameter_set, cv=cv, scoring=scoring, verbose=1, n_jobs=-1)
