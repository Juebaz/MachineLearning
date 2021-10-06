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


sub, (ax1, ax2) = pyplot.subplots(1, 2)

x = numpy.linspace(minBorn, maxBorn, 100000)
ax1.plot(x, pdf(x))
ax2.plot(x, pdf(x))
ax1.hist(sample(N1), bins=25, range=(x.min(), x.max()),
         density=True, color="skyblue", ec="skyblue")
ax2.hist(sample(N2), bins=25, range=(x.min(), x.max()),
         density=True, color="skyblue", ec="skyblue")

ax1.set_title("N = 50")
ax1.set_xlabel("value")
ax1.set_ylabel("frequency")

ax2.set_title("N = 100000")
ax2.set_xlabel("value")
ax2.set_ylabel("frequency")


pyplot.show()
