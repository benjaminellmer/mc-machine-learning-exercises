########################################################################################################################
# Feature Selection Helper Function Definitions
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition


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


def plot_cross_validation_(rfecv, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Number of selected features")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
