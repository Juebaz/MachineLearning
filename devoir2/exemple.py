from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot
import time
import numpy
from numpy.random import random
from scipy import interpolate


def pdf(X):
    return 0.4 * norm(0, 1).pdf(X) + 0.6 * norm(5, 1).pdf(X)


# Durée maximale d'exécution pour la question
# Maximum execution time for this question
TMAX_Q1A = 1.0

# *** TODO Q1A ***
# Complétez la fonction sample(n), qui génère n
# données suivant la distribution mentionnée dans l'énoncé
minBorn = -5
maxBorn = 10


def sample(N):
    x = numpy.linspace(minBorn, maxBorn, N)
    y = pdf(x)                        # probability density function, pdf
    cdf_y = numpy.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y, x)    # this is a function
    return inverse_cdf


def return_samples(N):
    # let's generate some samples according to the chosen pdf, f(x)
    uniform_samples = random(int(N))
    required_samples = sample(N)(uniform_samples)
    return required_samples


N = 1000000
N2 = 50


x = numpy.linspace(minBorn, maxBorn, 100000)

pyplot.plot(x, pdf(x))
pyplot.hist(return_samples(N), bins=25, range=(x.min(), x.max()), density=True)


pyplot.show()
