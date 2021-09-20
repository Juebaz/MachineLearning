from sklearn import datasets
import numpy
from matplotlib import pyplot
import time
from scipy.integrate._ivp.common import norm
# Durée d'exécution maximale / maximum execution time


def createLegendHandelsFor(data, colormap, lables):
    X = data.target
    Y = data.data
    norm = pyplot.Normalize(Y.min(), X.max())

    return [pyplot.Line2D([0, 0], [0, 0], color=colormap(norm(i)), marker='o', linestyle='', label=label)
            for i, label in enumerate(lables)]


ExecutionMaxTime = 0.5

data = datasets.load_iris()
mesurementCombinaison = [(i, j) for i in range(4) for j in range(i+1, 4)]

fig, subfigs = pyplot.subplots(2, 3, tight_layout=True)

t1 = time.time()

class_labels = data.target_names
cmap = pyplot.get_cmap('viridis')


for (f1, f2), subfig in zip(mesurementCombinaison, subfigs.reshape(-1)):
    xname = data.feature_names[f1]
    yname = data.feature_names[f2]
    subfig.scatter(data.data[:, f1], data.data[:, f2],
                   c=data.target, cmap='viridis')
    subfig.set_xlabel(xname)
    subfig.set_ylabel(yname)

handles = createLegendHandelsFor(data, cmap, class_labels)
fig.legend(handles=handles, loc="upper right")
pyplot.show()

t2 = time.time()

### Ne pas modifier / do not modify ###
duration = t2 - t1
if duration > ExecutionMaxTime:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2A " +
          f"met trop de temps à s'exécuter! Le temps maximum " +
          f"permis est de {ExecutionMaxTime:.4f} secondes, mais votre " +
          f"code a requis {duration:.4f} secondes! Assurez-vous " +
          f"que vous ne faites pas d'appels bloquants (par " +
          f"exemple à show()) dans cette boucle!\x1b[0m")
