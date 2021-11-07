from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_l_bfgs_b
from sklearn.exceptions import ConvergenceWarning
import warnings
from IPython import display
import collections
import time
import numpy
import itertools
import pandas
pandas.set_option('display.max_colwidth', 0)


# Nous ne voulons pas être signalé par ce type d'avertissement, non pertinent pour le devoir
# We don't want to be signaled of this warning, irrelevant for the homework
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fonction pour vérifier le temps d'exécution
# Function to verify execution time
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, ".format(maxduration) +
              "mais votre code a requis {0:.4f} secondes! ".format(duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")


# Définition des durées d'exécution maximales pour chaque classifieur
# Definition of maximum execution time for each classifier
TMAX_EVAL = 3.0
TMAX_FIT = 2.0


# Implémentation du discriminant à noyau
# Kernel discriminant implementation
class DiscriminantANoyau:

    def __init__(self, lambda_, sigma, verbose=True):
        self.lambda_ = lambda_
        self.sigma = sigma
        self.verbose = verbose

    def fit(self, X, y):
        # Implémentez la fonction d'entraînement du classifieur, selon les équations développées à Q3A
        # Implement the training function of the classifier, according to the equations developed in Q3A

        # *** TODO ***
        # Vous devez écrire une fonction nommée evaluateFunc, qui reçoit un seul argument, soit les valeurs
        # des paramètres pour lesquels on souhaite connaître l'erreur et le gradient. Cette fonction sera
        # appelée à répétition par l'optimiseur de scipy, qui l'utilisera pour minimiser l'erreur et
        # obtenir un jeu de paramètres optimal.
        # You must write a function named evaluateFunc, which receives only one argument, either the values
        # of the parameters for which you want to know the error and the gradient. This function will be
        # called repeatedly by the scipy optimizer, which will use it to minimize the error and
        # to obtain an optimal set of parameters.
        def evaluateFunc(params):
            # Ecrire ici le code de la fonction evaluateFunc calculant l'erreur
            # et le gradient de params.
            # Write here the code of the evaluateFunc function calculating the error
            # and the gradient of params.
            return err, grad
        # ******

        # *** TODO ***
        # Initialisez aléatoirement les paramètres alpha^t et w_0 (l'optimiseur requiert une valeur initiale,
        # et nous ne pouvons pas simplement n'utiliser que des zéros pour différentes raisons). Stockez ces
        # valeurs initiales aléatoires dans un array numpy nommé "params". Déterminez également les bornes à
        # utiliser sur ces paramètres et stockez les dans une variable nommée "bounds".
        # Indice : les paramètres peuvent-ils avoir une valeur maximale (au-dessus de laquelle ils ne veulent
        # plus rien dire)? Une valeur minimale? Référez-vous à la documentation de fmin_l_bfgs_b pour savoir
        # comment indiquer l'absence de bornes.
        # Randomly initialize the parameters alpha^t and w_0 (the optimizer requires an initial value,
        # and we can't just use zeros for various reasons). Store these
        # random initial values in a numpy array named "params". Also determine the bounds to use on these
        # parameters and store them in a variable named "bounds".
        # Hint: Can the parameters have a maximum value (above which they mean nothing)?
        # mean anything)? A minimum value? Refer to the documentation of fmin_l_bfgs_b to know
        # how to indicate the absence of bounds.
        # ******

        # À ce stade, trois choses devraient être définies :
        # - Une fonction d'évaluation nommée evaluateFunc, capable de retourner l'erreur et le gradient d'erreur
        #   pour chaque paramètre, pour une configuration d'alpha et w_0 donnée.
        # - Un tableau numpy nommé params de même taille que le nombre de paramètres à entraîner.
        # - Une liste nommée bounds contenant les bornes que l'optimiseur doit respecter pour chaque paramètre.
        # On appelle maintenant l'optimiseur avec ces informations et on conserve les valeurs dans params
        # At this point, three things should be defined:
        # - An evaluation function named evaluateFunc, capable of returning the error and error gradient
        # for each parameter, for a given alpha and w_0 configuration.
        # - A numpy array named params of the same size as the number of parameters to train.
        # - A list named bounds containing the bounds that the optimizer must respect for each parameter.
        # We now call the optimizer with this information and we keep the values in params
        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(
            evaluateFunc, params, bounds=bounds)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        if self.verbose:
            # On affiche quelques statistiques / display some statistics
            print("Entraînement terminé après {it} itérations et {calls} appels à evaluateFunc".format(
                it=infos['nit'], calls=infos['funcalls']))
            print("\tErreur minimale : {:.5f}".format(minval))
            print("\tL'algorithme a convergé" if infos['warnflag']
                  == 0 else "\tL'algorithme n'a PAS convergé")
            print(
                "\tGradients des paramètres à la convergence (ou à l'épuisement des ressources) :")
            print(infos['grad'])

        # *** TODO ***
        # Stockez les paramètres optimisés de la façon suivante :
        # - Le vecteur alpha dans self.alphas;
        # - Le biais w_0 dans self.w0.
        # Store the optimized parameters as follows:
        # - The alpha vector in self.alphas;
        # - The bias w_0 in self.w0.
        # ******

        # On retient également le jeu d'entraînement, qui pourra vous être utile pour les autres fonctions.
        # We also retain the training set, which can be useful for other functions.
        self.X, self.y = X, y
        return self

    def predict(self, X):

        # *** TODO ***
        # Implémentez la fonction d'inférence (prédiction). Vous pouvez supposer que fit() a préalablement été
        # exécuté et que les variables membres alphas, w0, X et y existent. N'oubliez pas que ce classifieur doit
        # retourner -1 ou 1
        # Implement the inference (prediction) function. You can assume that fit() has been previously executed and that the
        # executed and that the member variables alphas, w0, X and y exist. Remember that this classifier must
        # return -1 or 1
        # ******

        return predictions

    def score(self, X, y):

        # *** TODO ***
        # Implémentez la fonction retournant le score (accuracy) du classifieur sur les données reçues en
        # argument. Vous pouvez supposer que fit() a préalablement été exécutée
        # Indice : réutiliser votre implémentation de predict() réduit de beaucoup la taille de cette fonction!
        # Implement the function returning the accuracy of the classifier on the data received as
        # argument. You can assume that fit() has already been executed
        # Hint: reusing your implementation of predict() reduces the size of this function by a lot!
        # ******

        return score
