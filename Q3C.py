from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.datasets import load_iris
from matplotlib import pyplot
import matplotlib
from IPython import display
import collections
import pandas
import numpy
import time
from math import log

matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)

# Jeux de données / datasets

# Méthodes d'évaluation / evaluation methods


class ClassifieurAvecRejet:

    def __init__(self, _lambda=1):
        # _lambda est le coût de rejet
        # _lambda is the cost of reject
        self._lambda = _lambda

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = numpy.unique(y)
        nbOfClasses = len(self._classes)

        # calculate mean, var, and prior probability  for each class
        self._mean = numpy.zeros(
            (nbOfClasses, n_features), dtype=numpy.float64)
        self._var = numpy.zeros((nbOfClasses, n_features), dtype=numpy.float64)
        self._priors = numpy.zeros(nbOfClasses, dtype=numpy.float64)

        for index, classe in enumerate(self._classes):
            X_c = X[y == classe]
            self._mean[index, :] = X_c.mean(axis=0)
            self._var[index, :] = X_c.var(axis=0)
            self._priors[index] = X_c.shape[0] / float(n_samples)

    def _gaussianDensityFct(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = numpy.exp(-((x - mean) ** 2) / (2 * var))
        denominator = numpy.sqrt(2 * numpy.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        # *** TODO Q3C ***
        # Retournez la probabilité d'appartenance à chaque classe
        # pour les données passées en argument.
        # Cette fonction peut supposer que fit() a préalablement été appelé.
        # Indice: calculez les termes de l'équation de Bayes séparément.
        matrix = numpy.zeros(
            (X.shape[0], len(self._classes)), dtype=numpy.float64)

        for i, x in enumerate(X):

            probForXtoBeC_i = []

            for index, c in enumerate(self._classes):
                prior = numpy.log(self._priors[index])
                posterior = numpy.sum(
                    numpy.log(self._gaussianDensityFct(index, x)))
                posterior = prior + posterior
                matrix[i, index] = posterior
        return matrix

    def predict(self, X):
        # *** TODO Q3C ***
        # Retournez les prédictions de classes pour les données
        # passées en argument.
        # Cette fonction peut supposer que fit() a préalablement été appelé.
        # Indice: vous pouvez appeler predict_proba() pour éviter une
        # redondance du code.
        probForEachXtoBeEachC = self.predict_proba(X)
        classWithHighiestProbForEachX = []

        for index, x in enumerate(X):
            prob = probForEachXtoBeEachC[index, :]
            denominateur = numpy.sum(prob)
            probNormalise = prob/denominateur

            probMax = probNormalise[numpy.argmax(probNormalise)]

            rejectClass = len(self._classes)

            if log(probMax) < log(1-self._lambda):
                classWithHighiestProbForEachX.append(rejectClass)

            else:
                classe = self._classes[numpy.argmax(prob)]
                classWithHighiestProbForEachX.append(classe)

        return classWithHighiestProbForEachX

        # *****

    def score(self, X, y):
        # *** TODO Q3C ***
        # Retournez le score de performance, tenant compte des données rejetées
        # si lambda < 1.0, pour les données passées en argument.
        # Cette fonction peut supposer que fit() a préalablement été exécuté.
        # somme du coût des rejets et du coût des mauvais classements
        # *****
        predictions = self.predict(X)
        rejectClass = len(self._classes)
        scores = []
        accuracy = numpy.sum(y == predictions) / len(y)
        rejectScore = (self._lambda*numpy.sum(y == rejectClass))/len(y)
            
        return accuracy + rejectScore


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    # X, y = datasets.make_classification(
    #     n_samples=1000, n_features=10, n_classes=2, random_state=123
    # )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = ClassifieurAvecRejet(0.9)
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print(nb.score(X, y))
    print(predictions)
