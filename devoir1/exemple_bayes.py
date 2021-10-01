from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.datasets import load_iris
from matplotlib import pyplot
import matplotlib
from IPython import display
import collections
import pandas
import time
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from math import sqrt, exp, pi
import numpy as np


class NaiveBayes:
    def fit(self, XTrain, y):
        n_samples, n_features = XTrain.shape
        self._classes = np.unique(y)
        nbOfClasses = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((nbOfClasses, n_features), dtype=np.float64)
        self._var = np.zeros((nbOfClasses, n_features), dtype=np.float64)
        self._priors = np.zeros(nbOfClasses, dtype=np.float64)

        for index, classe in enumerate(self._classes):
            # on veut les data qui ont la classe C
            xWithClasseC = XTrain[y == classe]
            self._mean[index, :] = xWithClasseC.mean(axis=0)
            self._var[index, :] = xWithClasseC.var(axis=0)
            # nb of samples with classe C / nb total de sample
            self._priors[index] = xWithClasseC.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        print(np.array(y_pred))
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        
        print(posteriors)
        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    # X, y = datasets.make_classification(
    #     n_samples=1000, n_features=10, n_classes=2, random_state=123
    # )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print(predictions)
    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
