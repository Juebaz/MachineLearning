
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

pandas.set_option('display.max_colwidth', 0)


X, y = make_classification(n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           n_clusters_per_class=1)


discriminant = DiscriminantLineaire()

discriminant.fit(X,y)
# ypred = discriminant.predict(X)
discriminant.score(X,y)


l = LinearDiscriminantAnalysis()
l.fit(X,y)
l.score(X,y)
