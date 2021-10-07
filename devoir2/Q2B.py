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
pandas.set_option('display.max_colwidth', 0)


matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)


_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")


# Implémentation du discriminant linéaire
# Implementation of the linear discriminant
class DiscriminantLineaire:
    def __init__(self, eta=1e-2, epsilon=1e-2, max_iter=1000):

        # Cette fonction est déjà codée, utilisez les variables membres
        # qu'elle définit dans les autres fonctions
        # This function is already coded, use the member variables it
        # defines in other functions
        self.eta = eta

        # Epsilon et max_iter sont des paramètres du critère d'arrêt
        # max_iter est le nombre maximum de mises à jour des poids,
        # alors qu'epsilon est un seuil sur la différent minimale entre les
        # erreurs faites entre deux itérations consécutives pour continuer l'entraînement
        # Epsilon and max_iter are parameters for the stop criterion
        # max_iter is the maximum number of weights updates allowed, while
        # epsilon is a threshold on the minimum error difference observed between
        # two consecutive training steps.
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit(self, X, y):
 
        # Implémentez la fonction d'entraînement selon les équations développées
        # à la question précédente.
        # Implement the training function according to the equations developed at the
        # previous question

        # On initialise les poids aléatoirement
        # Weights are randomly initialized
        w = numpy.random.rand(X.shape[1]+1)
        w0 = 0
        
        for i in range(self.max_iter):
            N = X.shape[0]

            for n in range(N):
                
                x = (numpy.append([1], X[n, :]))
                y_pred = numpy.sum(w.T*x + w0)

                if (y_pred >= 0):
                    y_pred = 1
                
                else:
                    y_pred = 0

                normXSquare = numpy.linalg.norm(X[n, :])**2
            
                D_w = (-1/normXSquare) * numpy.sum(x * (y[n] - y_pred))
                D_w0 = (-1/normXSquare) * numpy.sum(y[n] - y_pred)

                w = w - self.eta * D_w
                w0 = w0 - self.eta * D_w0

        # Copie des poids entraînés dans une variable membre pour les conserver
        # Copy trained weights in a member variable for storing
        print(w)
        self.w = w
        self.w0 = w0

    def predict(self, X):
        
        Y = numpy.zeros(X.shape[0])
        
        for n in range(X.shape[0]):
            y_pred =1;
            x = (numpy.append([1], X[n, :]))
            h = numpy.sum(self.w*x+self.w0)
            if (h >= 0):
                    y_pred = 1
            else:
                    y_pred = 0
            Y[n] = y_pred
        return Y
        
        # ******

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = numpy.sum(y == predictions) / len(y)
        print(accuracy)
        return accuracy 
