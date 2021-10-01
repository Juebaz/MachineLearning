from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import minmax_scale
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
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


# Définition de la durée d'exécution maximales pour la question
TMAX_Q3A = 15

# Dictionnaire pour enregistrer les paramètres évalués
results = collections.defaultdict(list)
results['Questions'] = ["Quel est l\'impact de la disposition des données dans l'espace?",
                        "Quel est le K optimal?",
                        ]
results['Discussion'] = []

_times.append(time.time())

# *** TODO Q3A ***
# Testez un classifeur K plus proches voisins sur le jeu de donnée
# L'évaluation doit être faite en utilisant une approche leave-one-out
# Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
# "uniform" et "distance" comme valeur de l'argument "weights".
# N'oubliez pas de normaliser le jeu de données en utilisant minmax_scale!

dataset = load_iris()

K = [1, 3, 5, 7, 11, 13, 15, 25, 35, 45]
X = dataset.data
y = dataset.target

for k in K:

    classifier = KNeighborsClassifier(n_neighbors=k)
    loo = LeaveOneOut()

    leaveOneOutResults = []
    for train_index, test_index in loo.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        classifier.fit(X_train, y_train)
        leaveOneOutResults.append(classifier.score(X_test, y_test))
    
    results[k] =  numpy.mean(numpy.array(leaveOneOutResults))

# Stockez les performances obtenues (précision moyenne pour chaque valeur de k)
# dans deux listes, scoresUniformWeights pour weights=uniform et
# scoresDistanceWeights pour weights=distance
# Le premier élément de chacune de ces listes devrait contenir la précision
# pour k=1, le second la précision pour k=3, et ainsi de suite.
# Store the obtained performances (average precision for each value of k)
# in two lists, scoresUniformWeights for weights=uniform and
# scoresDistanceWeights for weights=distance
# The first element of each of these lists should contain the precision
# for k=1, the second the precision for k=3, and so on.
scoresUniformWeights = []
scoresDistanceWeights = []
# ******

_times.append(time.time())
checkTime(TMAX_Q3A, "Q3A")

# *** TODO Q3A ***
# Produisez un graphique contenant deux courbes, l'une pour weights=uniform
# et l'autre pour weights=distance. L'axe x de la figure doit être le nombre
# de voisins et l'axe y la performance en leave-one-out
# Produce a graph containing two curves, one for weights=uniform
# and the other for weights=distance. The x axis of the figure should be the number
# of neighbours and the y-axis the leave-one-out performance

fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.set_title('Title')  # À modifier / to be modified
# ax.plot(X, y, 'r--', label="Distance weights") # À compléter / to be completed
# ax.plot(X, y, 'b--', label="Uniform weights")  # À compléter / to be completed
# ax.set_xticks(X, X)
ax.grid(axis='x')
ax.set_xlabel("Values of K")
ax.set_ylabel("Accuracy (%)")

# ******

# *** TODO Q3A ***

# Répondez aux quelques questions pour la discussion
# Answer a few questions for discussion

# Quel est l'impact de la disposition des données dans l'espace?
# What is the impact of the layout of the data in the space?
answer = "Impact de la disposition \
          des données dans l'espace: "
results['Discussion'].append(answer)

# Quel est le nombre de voisins $k$ optimal à utiliser?
# What is the optimal number of $k$ neighbors to use?
answer = "Valeur de k optimale?: \
          "
results['Discussion'].append(answer)
# ******

# Affichage des erreurs
df = pandas.DataFrame(results)
display.display(df)
