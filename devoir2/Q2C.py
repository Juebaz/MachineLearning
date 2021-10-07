
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
from scipy.stats import norm
from matplotlib import pyplot
import matplotlib
from IPython import display
import collections
import time
import numpy
import pandas
from Q2B import DiscriminantLineaire

_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")


# Durée d'exécution maximale / maximum execution time
TMAX_Q2C = 10.0
_times.append(time.time())


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, classificator, xx, yy, **params):
    Z = classificator.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# Durée d'exécution maximale / maximum execution time

# Problème à 2 classes / 2-class problem
X, y = make_classification(n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           n_clusters_per_class=1)

discriminant = DiscriminantLineaire()

discriminant.fit(X, y)
ypred = discriminant.predict(X)

# Testez la performance du discriminant linéaire pour le problème
# Test the performance of the linear discriminant for the problem

score = discriminant.score(X, y)

# Création de la figure à afficher
# Create the figure to display
fig = pyplot.figure()
ax = fig.add_subplot(111)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)


# À modifier / to be modified
ax.set_title("Classification with Linear Discriminant and Gradient Descent")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")

plot_contours(ax, discriminant, xx, yy, cmap=pyplot.cm.coolwarm, alpha=0.8)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=pyplot.cm.coolwarm,
           s=20)
ax.text(1, 1, 'Score:'+str(score), fontsize=15,
        bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})


_times.append(time.time())
checkTime(TMAX_Q2C, "Q2C")

pyplot.show()
