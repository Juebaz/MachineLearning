from sklearn import datasets
from matplotlib import pyplot
import time
from scipy.integrate._ivp.common import norm
import collections
from sklearn import svm, datasets
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
import time
# import pandas
# from IPython import display


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


def plotClassicationGraphForFeatures(f1, f2, data):
    regularizeParam = 1.0  # SVM regularization parameter

    models = (svm.SVC(kernel='linear', C=regularizeParam),
              svm.LinearSVC(C=regularizeParam, max_iter=10000),
              svm.SVC(kernel='rbf', gamma=0.7, C=regularizeParam),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=regularizeParam))

    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    X = data.data[:, [f1, f2]]
    y = data.target

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                                   tight_layout=True)

    for classificator, subfig, title in zip(models, subfigs.reshape(-1), titles):
        trainModels = classificator.fit(X, y)

        plot_contours(subfig, classificator, xx, yy,
                      cmap=pyplot.cm.coolwarm, alpha=0.8)
        subfig.scatter(X0, X1, c=y, cmap=pyplot.cm.coolwarm,
                       s=20, edgecolors='k')
        subfig.set_xlim(xx.min(), xx.max())
        subfig.set_ylim(yy.min(), yy.max())
        subfig.set_xlabel('Sepal length')
        subfig.set_ylabel('Sepal width')
        subfig.set_xticks(())
        subfig.set_yticks(())
        subfig.set_title(title)

        return fig


iris = datasets.load_iris()

pairsOfFeatures = [(i, j) for i in range(4) for j in range(i+1, 4)]

regularizeParam = 1.0  # SVM regularization parameter


# Dictionnaire pour enregistrer les erreurs selon les classifieurs
# Dictionary to record errors according to classifiers
erreurs = collections.defaultdict(list)
erreurs['Classifieurs'] = []

mainFigure, subfigs = pyplot.subplots(2, 3, sharex='all', sharey='all',
                                      tight_layout=True)

# Traiter par paires de mesures avec réentraînement
# Process with pairs of measures with retraining
for (f1, f2), subfig in zip(pairsOfFeatures, subfigs.reshape(-1)):
    f1_name = iris.feature_names[f1]
    f2_name = iris.feature_names[f2]
    regularizeParam = 1.0  # SVM regularization parameter

    models = (svm.SVC(kernel='linear', C=regularizeParam),
              svm.LinearSVC(C=regularizeParam, max_iter=10000),
              svm.SVC(kernel='rbf', gamma=0.7, C=regularizeParam),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=regularizeParam))

    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    mainTitle = f1_name + 'vs' + f2_name

    X = iris.data[:, [f1, f2]]
    y = iris.target

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    classFigure, subClassFigure = pyplot.subplots(2, 2, sharex='all', sharey='all',
                                                  tight_layout=True)
    classFigure.set_title(mainTitle)

    for classificator, subfig, title in zip(models, subClassFigure.reshape(-1), titles):
        trainModels = classificator.fit(X, y)

        plot_contours(subfig, classificator, xx, yy,
                      cmap=pyplot.cm.coolwarm, alpha=0.8)
        subfig.scatter(X0, X1, c=y, cmap=pyplot.cm.coolwarm,
                       s=20, edgecolors='k')
        subfig.set_xlim(xx.min(), xx.max())
        subfig.set_ylim(yy.min(), yy.max())
        subfig.set_xlabel(f1_name)
        subfig.set_ylabel(f2_name)
        subfig.set_xticks(())
        subfig.set_yticks(())
        subfig.set_title(title)


pyplot.show()

# Affichage des erreurs / display errors
# clfs = erreurs.pop('Classifieurs')
# df = pandas.DataFrame(erreurs, index=clfs)
# display.display(df)
