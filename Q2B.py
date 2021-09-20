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

# Durée d'exécution maximale / maximum execution time
TMAX_Q2B = 1.5

# Erreur maximale attendue / maximum error expected
ERRMAX_Q2B = 0.22

iris = datasets.load_iris()

pairsOfFeatures = [(i, j) for i in range(4) for j in range(i+1, 4)]

# Dictionnaire pour enregistrer les erreurs selon les classifieurs
# Dictionary to record errors according to classifiers
erreurs = collections.defaultdict(list)
erreurs['Classifieurs'] = []

regularizeParam = 1.0  # SVM regularization parameter

models = (svm.SVC(kernel='linear', C=regularizeParam),
          svm.LinearSVC(C=regularizeParam, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=regularizeParam),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=regularizeParam))

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Traiter par paires de mesures avec réentraînement
# Process with pairs of measures with retraining
for (f1, f2) in pairsOfFeatures:
    f1_name = iris.feature_names[f1]
    f2_name = iris.feature_names[f2]
    
     # *** TODO Q2B ***
    # Créez un jeu de données contenant seulement f1 et f2
    # Create a dataset with only f1 and f2
    X = iris.data[:, [f1,f2]]
    y = iris.target
    
   
    # Initialisez différents classifieurs dans la liste nommée 'classifieurs'
    # Initialize various classifiers in the list named 'classifiers'


    trainModels = (classificator.fit(X, y) for classificator in models)
    

    fig, sub = pyplot.subplots(2, 2)
    pyplot.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for classificator, title, ax in zip(trainModels, titles, sub.flatten()):
        plot_contours(ax, classificator, xx, yy,
                    cmap=pyplot.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=pyplot.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    pyplot.show()




def plotClassicationGraphForFeatures(f1,f2,data): 
    regularizeParam = 1.0  # SVM regularization parameter

    models = (svm.SVC(kernel='linear', C=regularizeParam),
          svm.LinearSVC(C=regularizeParam, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=regularizeParam),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=regularizeParam))

    titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

    X = data.data[:, [f1,f2]]
    y = data.target

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                                  tight_layout=True)

    for classificator,subfig,title in zip(models, subfigs.reshape(-1),titles):
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
