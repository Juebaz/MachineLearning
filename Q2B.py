from sklearn import datasets
from matplotlib import pyplot
import time
from scipy.integrate._ivp.common import norm
import collections
from sklearn import datasets, svm, model_selection, metrics
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
import time
import pandas
from IPython import display


TMAX_Q2B = 1.5
ERRMAX_Q2B = 0.22


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, classificator, xx, yy, **params):
    Z = classificator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plotClassicationGraphForFeatures(f1, f2, data, titles, models):

    X = data.data[:, [f1, f2]]
    y = data.target

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                                   tight_layout=True)

    fig.suptitle(data.feature_names[f1]+' vs ' +
                 data.feature_names[f2], fontsize=12)

    erreurs = []

    for classificator, subfig, title in zip(models, subfigs.reshape(-1), titles):
        clf_name = title

        trainModels = classificator.fit(X, y)

        err = 1 - metrics.accuracy_score(classificator.predict(X), y)
        erreurs.append(err)

        plot_contours(subfig, classificator, xx, yy,
                      cmap=pyplot.cm.coolwarm, alpha=0.8)

        subfig.scatter(X0, X1, c=y, cmap=pyplot.cm.coolwarm,
                       s=20, edgecolors='k')
        subfig.set_xlim(xx.min(), xx.max())
        subfig.set_ylim(yy.min(), yy.max())
        subfig.set_xlabel(data.feature_names[f1])
        subfig.set_ylabel(data.feature_names[f2])
        subfig.set_xticks(())
        subfig.set_yticks(())
        subfig.set_title(title)

    return fig, erreurs


def main():
    iris = datasets.load_iris()

    pairsOfFeatures = [(i, j) for i in range(4) for j in range(i+1, 4)]

    regularizeParam = 1.0  # SVM regularization parameter

    classificators = (QuadraticDiscriminantAnalysis(),
                      LinearDiscriminantAnalysis(),
                      GaussianNB(),
                      NearestCentroid())

    titles = ('QuadraticDiscriminantAnalysis',
              'LinearDiscriminantAnalysis',
              'GaussianNB',
              'NearestCentroid')

    errors = {}

    for (f1, f2) in pairsOfFeatures:
        f1_name = iris.feature_names[f1]
        f2_name = iris.feature_names[f2]
        regularizeParam = 1.0  # SVM regularization parameter

        fig, errorForFeatures = plotClassicationGraphForFeatures(
            f1, f2, iris, titles, classificators)

        errors[f'{f1_name} {f2_name}'] = errorForFeatures

    pyplot.show()

    df = pandas.DataFrame(errors, index=['QuadraticDiscriminantAnalysis',
                                         'LinearDiscriminantAnalysis',
                                         'GaussianNB',
                                         'NearestCentroid'])
    display.display(df)


main()
