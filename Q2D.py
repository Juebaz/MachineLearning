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
from sklearn.model_selection import RepeatedKFold
from IPython import display
from sklearn.datasets import make_circles

# Durée d'exécution maximale / maximum execution time
TMAX_Q2D = 1.0

# Dictionnaire pour enregistrer les erreurs selon les classifieurs
# Dictionary to record error according to the classifiers
erreurs = collections.OrderedDict()

X, Y = make_circles(factor=0.3)

# *** TODO Q2D ***
# Initialisez les différents classifieurs dans une liste nommée 'classifieurs'
# Initialize the various classifiers in a list named 'classifieurs'

classificators = [QuadraticDiscriminantAnalysis(
), LinearDiscriminantAnalysis(), GaussianNB(), NearestCentroid()]


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


# Créer une figure à plusieurs sous-graphes
# Create a figure with several subplots
fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                               tight_layout=True)
t1 = time.time()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, train_size=0.50, test_size=0.50)

for classificator, subfig in zip(classificators, subfigs.reshape(-1)):
    clf_name = classificator.__class__.__name__

    classificator.fit(X_train, Y_train)

    err = 1 - metrics.accuracy_score(classificator.predict(X_test), Y_test)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(subfig, classificator, xx, yy,
                  cmap=pyplot.cm.coolwarm, alpha=0.8)

    subfig.scatter(X0, X1, c=Y, cmap=pyplot.cm.coolwarm,
                   s=20, edgecolors='k')

    erreurs[clf_name] = err
    subfig.set_title(clf_name)

pyplot.show()

### Ne pas modifier ###
t2 = time.time()
duration = t2 - t1
if duration > TMAX_Q2D:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2D " +
          f"met trop de temps à s'exécuter! Le temps maximum " +
          f"permis est de {TMAX_Q2D:.4f} secondes, mais votre " +
          f"code a requis {duration:.4f} secondes! Assurez-vous " +
          f"que vous ne faites pas d'appels bloquants (par " +
          f"exemple à show()) dans cette boucle!\x1b[0m")
df = pandas.DataFrame(erreurs, index=['Erreurs'])
display.display(df)

# ***********************REPONSE QUESTION 3D ******************************
# Les classificateurs gaussian et quadratique semblent les plus approprié.
#  Ils ont tout deux des erreurs de 0. Il reussi a toute classer sans erreurs les doonées de tests.
