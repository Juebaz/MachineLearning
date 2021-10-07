
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

# *** TODO Q2C ***
# Créez ici une grille permettant d'afficher les régions de
# décision pour chaque classifieur
# Indice : numpy.meshgrid pourrait vous être utile ici
# N'utilisez pas un pas trop petit!



# Create a grid here to display the decision regions for each
# decision regions for each classifier
# Tip: numpy.meshgrid might be useful here
# Don't use a too small step size!

# Entraînez le discriminant linéaire implémenté
# Train the implemented linear discriminant
discriminant = DiscriminantLineaire()

discriminant.fit(X, y)
ypred = discriminant.predict(X)

# Testez la performance du discriminant linéaire pour le problème
# Test the performance of the linear discriminant for the problem

score = discriminant.score(X,y)

# Création de la figure à afficher
# Create the figure to display
fig = pyplot.figure()
ax = fig.add_subplot(111)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, discriminant, xx, yy,
              cmap=pyplot.cm.coolwarm, alpha=0.8)
# Utilisez la grille que vous avez créée plus haut
# pour afficher les régions de décision, de même
# que les points colorés selon leur vraie classe
# N'oubliez pas la légende !
# Use the grid you created above
# to display the decision regions, as well as
# coloured points according to their true class
# Don't forget the legend!

ax.set_title("Title")   # À modifier / to be modified
ax.set_xlabel("X Axis")  # À modifier / to be modified
ax.set_ylabel("Y Axis")  # À modifier / to be modified
#ax.contourf() # À compléter / to be completed
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=pyplot.cm.coolwarm,
           s=20)
pyplot.show()
# ******
