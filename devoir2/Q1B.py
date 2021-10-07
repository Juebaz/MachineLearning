from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot
import time
import numpy
from numpy.random import random
from scipy import interpolate

import matplotlib
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)


_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")

# Définition de la fonction de densité de probabilité (PDF) de la densité-mélange
# Definition of the mixture model probability density function


def pdf(X):
    return 0.4 * norm(0, 1).pdf(X) + 0.6 * norm(5, 1).pdf(X)


TMAX_Q1B = 2.5
_times.append(time.time())

minBorn = -5
maxBorn = 10

N1 = 50
N2 = 10000


def sample(N):
    x = numpy.linspace(minBorn, maxBorn, N)
    y = pdf(x)

    cdf_y = numpy.cumsum(y)
    cdf_y_normalize = cdf_y/cdf_y.max()
    inverse_cdf = interpolate.interp1d(cdf_y_normalize, x)

    uniform_samples = random(int(N))

    samples = inverse_cdf(uniform_samples)
    return samples


# ** TODO Q1B ***
# À partir des échantillons 50 et 10 000 données de la question précédente
# faites une estimation de la distribution des données avec un noyau boxcar.
# Pour chaque taille de jeu (50 et 10 000), présentez les distributions
# estimées avec des tailles de noyau (bandwidth) de {0.3, 1, 2, 5}, dans
# la même figure, mais tracées avec des couleurs différentes.
# From the 50 and 10,000 data samples in the previous question
# make an estimate of the distribution of the data with a boxcar kernel.
# For each set size (50 and 10,000), present the distributions
# estimated with kernel sizes (bandwidth) of {0.3, 1, 2, 5}, in
# the same figure, but plotted with different colours.
# ******
x = sample(N2).reshape(-1, 1)
kde = KernelDensity(kernel='tophat', bandwidth=1).fit(x)
s = numpy.linspace(minBorn, maxBorn, N2)


e = kde.score_samples(s.reshape(-1, 1))
pyplot.plot(s, e)


x_d = numpy.linspace(minBorn, maxBorn, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

pyplot.fill_between(x_d, density, alpha=0.5)
pyplot.plot(x, numpy.full_like(x, -0.1), '|k', markeredgewidth=1)

# Affichage du graphique
_times.append(time.time())
checkTime(TMAX_Q1B, "Q1B")
pyplot.show()
