from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy
import time

from matplotlib import pyplot
from IPython import display

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_l_bfgs_b

import pandas
pandas.set_option('display.max_colwidth', 0)

# Nous ne voulons pas être signalé par ce type d'avertissement, non pertinent pour le devoir
# We don't want to be signaled of this warning, irrelevant for the homework
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


class DiscriminantANoyau:

    def __init__(self, lambda_, sigma, verbose=True):
        # Cette fonction est déjà codée pour vous, vous n'avez qu'à utiliser les variables membres qu'elle
        # définit dans les autres fonctions de cette classe. Lambda et sigma sont définis dans l'énoncé.
        # verbose permet d'afficher certaines statistiquesen lien avec la convergence de fmin_l_bfgs_b.
        # This function is already coded for you, you just have to use the member variables it defines in
        # in the other functions of this class. Lambda and sigma are defined in the statement.
        # verbose allows you to display some statistics related to the convergence of the fmin_l_bfgs_b.
        self.lambda_ = lambda_
        self.sigma = sigma
        self.verbose = verbose

    #Calcul de la fonction discriminante
    def discriminant(self, X, x_t, y, parametres):
        h = numpy.zeros(x_t.shape[0])
        for i in range(x_t.shape[0]):
            h[i] = numpy.sum(parametres[1:]*y*numpy.exp(-(numpy.linalg.norm(X - x_t[i], axis=1, ord=2)**2)
                                                        / (self.sigma**2))) + parametres[0]
        return h

    def fit(self, X, y):
        # Implémentez la fonction d'entraînement du classifieur, selon les équations développées à Q3A
        # Implement the training function of the classifier, according to the equations developed in Q3A

        def evaluateFunc(params):
            # Ecrire ici le code de la fonction evaluateFunc calculant l'erreur
            # et le gradient de params.

            classes = self.discriminant(X, X, y, params)
            mauvaise_classes = numpy.zeros(classes.shape[0])

            #Déterminer les observations males classées
            for i in range(classes.shape[0]):
                if classes[i] * y[i] < 0:
                    mauvaise_classes[i] = i
            x_mauvaise_classe = X[numpy.where(mauvaise_classes != 0)]
            y_mauvaise_classe = y[numpy.where(mauvaise_classes != 0)]

            #Calcul de l'erreur
            err = numpy.sum(1 - y_mauvaise_classe * self.discriminant(X, x_mauvaise_classe, y, params)) \
                + (self.lambda_ * numpy.sum(params[0:]))

            #Calculs des gradients des paramètres
            grad = params.copy()
            grad[0] = -numpy.sum(y_mauvaise_classe)
            for i in range(1, len(grad)):
                grad[i] = -numpy.sum(y[i-1]*y_mauvaise_classe*numpy.exp(-(
                    numpy.linalg.norm((x_mauvaise_classe - X[i-1]), axis=1, ord=2) ** 2) / (self.sigma**2))) + self.lambda_
            return err, grad

        #Assignation aléatoire des valeurs de départ des paramètres
        params = numpy.random.rand(X.shape[0] + 1)

        #Limites des paramètres alpha et w0
        bounds = [(None, None)]*(X.shape[0] + 1)
        for i in range(1, X.shape[0] + 1):
            bounds[i] = (0, numpy.inf)

        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(
            evaluateFunc, params, bounds=bounds)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        if self.verbose:
            # On affiche quelques statistiques / display some statistics
            print("Entraînement terminé après {it} itérations et {calls} appels à evaluateFunc".format(it=infos['nit'],
                                                                                                       calls=infos[
                                                                                                           'funcalls']))
            print("\tErreur minimale : {:.5f}".format(minval))
            print("\tL'algorithme a convergé" if infos['warnflag']
                  == 0 else "\tL'algorithme n'a PAS convergé")
            print(
                "\tGradients des paramètres à la convergence (ou à l'épuisement des ressources) :")
            print(infos['grad'])

        self.alphas, self.w0 = params[1:], params[0]

        self.X, self.y = X, y

        return self

    def predict(self, X):

        params = numpy.append(self.w0, self.alphas)
        predictions = self.discriminant(self.X, X, self.y, params)

        for i in range(predictions.shape[0]):
            if predictions[i] > 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        return predictions

    def score(self, X, y):
        classe_predi = self.predict(X)
        y_bin = numpy.copy(y)
        for i in range(y.shape[0]):
            if y[i] <= 0:
                y_bin[i] = -1
        score = sum(classe_predi == y)/y_bin.shape[0]
        return score


if __name__ == "__main__":
    results = {'Classifier': ["DiscriminantANoyau"],
               'Range_lambda': [],
               'Range_sigma': [],
               'Best_lambda': [],
               'Best_sigma': [],
               'Error_train': [],
               'Error_test': [],
               }
    #Créer le jeu de données
    data = make_moons(n_samples=800, noise=0.3)
    X = data[0]
    y = data[1]
    for i in range(y.shape[0]):
        if y[i] < 1:
            y[i] = -1
    #Séparation des données
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.5, random_state=2)
    _times.append(time.time())

    #Valeurs de lambda_
    range_lambda = [0.7, 1, 2]

    results['Range_lambda'].append(range_lambda)

    #Valeurs de sigma
    range_sigma = [0.5, 0.9, 1]

    results['Range_sigma'].append(range_sigma)

    #Recherche des meilleurs paramètres
    high_score = 0
    for lambda_ in range_lambda:
        for sigma in range_sigma:
            classifieur = DiscriminantANoyau(lambda_, sigma)
            classifieur.fit(X_train, y_train)
            scr = classifieur.score(X_val, y_val)
            if scr > high_score:
                high_score = scr
                best_lambda = lambda_
                best_sigma = sigma

    results['Best_lambda'].append(best_lambda)

    results['Best_sigma'].append(best_sigma)

    #Utilisation des meilleurs paramètres
    instance_opt = DiscriminantANoyau(best_lambda, best_sigma)
    instance_opt.fit(X_train, y_train)

    #Erreur d'entrainement
    err_train = 1 - (instance_opt.score(X_train, y_train))

    results['Error_train'].append(err_train)

    #Erreurs de validation
    err_test = 1 - instance_opt.score(X_val, y_val)

    results['Error_test'].append(err_test)

    #Création de la grille des classificateurs
    xx, yy = numpy.meshgrid(numpy.arange(min(X[:, 0]) - 0.2, max(X[:, 0]) + 0.2, 0.02),
                            numpy.arange(min(X[:, 1]) - 0.2, max(X[:, 1]) + 0.2, 0.02))

    fig = pyplot
    fig.scatter(X_val[:, 0], X_val[:, 1], c=y_val)
    zz = instance_opt.predict(numpy.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    fig.contourf(xx, yy, zz, alpha=0.7)
    fig.suptitle("Graphique des régions de décisions")

    # On affiche la figure
    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation")
    pyplot.show()

    # Affichage des résultats
    df = pandas.DataFrame(results)
    display.display(df)
