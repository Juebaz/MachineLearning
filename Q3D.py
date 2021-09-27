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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Q3C import ClassifieurAvecRejet

matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)

# Jeux de données / datasets

# Méthodes d'évaluation / evaluation methods


# Durée d'exécution maximale / maximum execution time
TMAX_Q3D = 1.0

cmap = pyplot.cm.Spectral


def createLegendHandelsFor(data, colormap, lables):
    X = data.target
    Y = data.data
    norm = pyplot.Normalize(Y.min(), X.max())

    return [pyplot.Line2D([0, 0], [0, 0], color=colormap(norm(i)), marker='o', linestyle='', label=label)
            for i, label in enumerate(lables)]



def make_meshgrid(x, y, h=0.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, classificator, xx, yy, **params):
    Z = classificator.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# Dictionnaire pour enregistrer les erreurs selon les classifieurs
# Dictionary to record classification errors
erreurs = collections.defaultdict(list)
erreurs['Classifieurs'] = []

data = load_iris()
# Créer une liste contenant toutes les paires possibles
# Create a list with all possible pairs
pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]

# Tester le classifieur avec différents lambda pour toutes les paires
# Test the classifier with different lambda over all pairs
for (f1, f2) in pairs:
    f1_name = data.feature_names[f1]
    f2_name = data.feature_names[f2]

    # *** TODO Q3D ***
    # Créez un jeu de données contenant seulement les
    # mesures désignées par f1 et f2

    X = data.data[:, [f1, f2]]
    Y = data.target

    class_labels = data.target_names

    # Créez grille permettant d'afficher régions de décision pour chaque classifieur
    # Indices : numpy.meshgrid pourrait vous être utile, mais n'utilisez pas un pas trop petit!
    # Create a grid for displaying various decision regions for each classifier
    # Tips: numpy.meshgrid can be useful, but don't use a too small step size
    # ******
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # On initialise les classifieurs avec différents paramètres lambda
    classifieurs = [ClassifieurAvecRejet(0.1),
                    ClassifieurAvecRejet(0.3),
                    ClassifieurAvecRejet(0.5),
                    ClassifieurAvecRejet(0.999)]

    # Créer une figure à plusieurs sous-graphes pour pouvoir montrer,
    # pour chaque configuration, les régions de décisions, incluant
    # la zone de rejet
    # Create a figure with several sub-plots to show, for all configuration,
    # decision regions, including the reject zone
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                                   tight_layout=True)
    t1 = time.time()
    for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):
        clf_name = clf.__class__.__name__ + " (\u03BB={})".format(clf._lambda)
        if clf_name not in erreurs['Classifieurs']:
            erreurs['Classifieurs'].append(clf_name)

        # *** TODO Q3D ***
        # Entraînez le classifieur
        # Train the classifier
        train = clf.fit(X, Y)
        predictions = clf.predict(X)

        # Stockez la valeur de l'erreur dans la variable err
        # Store error value in variable err
        err = clf.score(X, Y)

        # Utilisez la grille pour afficher les régions de décision,
        # INCLUANT LA ZONE DE REJET, de même que les points colorés selon
        # leur vraie classe
        # Use the grid to display the decision regions, INCLUDING THE
        # REJECT ZONE, as well the points coloured according to their
        # real labels.
        # ******

        plot_contours(subfig, clf, xx, yy, cmap=cmap, alpha=0.7)

        subfig.scatter(X0, X1, c=Y, cmap=cmap,
                       s=20, edgecolors='k')

        # Ajouter l'erreur pour affichage
        # Add error for displaying
        erreurs[f'{f1_name} {f2_name}'].append(err)

        # Ajouter un titre et des étiquettes d'axes
        # Add title and axis labels
        subfig.set_title("\u03BB="+str(clf._lambda))
        subfig.set_xlabel(data.feature_names[f1])
        subfig.set_ylabel(data.feature_names[f2])
        handles = createLegendHandelsFor(data, cmap, class_labels)
        fig.legend(handles=handles, loc="upper right")

    ### Ne pas modifier / do not modify ###
    t2 = time.time()
    duration = t2 - t1
    if duration > TMAX_Q3D:
        print(f"\x1b[31m[ATTENTION] Votre code pour la question Q3D " +
              f"met trop de temps à s'exécuter! Le temps maximum " +
              f"permis est de {TMAX_Q3D:.4f} secondes, mais votre " +
              f"code a requis {duration:.4f} secondes! Assurez-vous " +
              f"que vous ne faites pas d'appels bloquants (par " +
              f"exemple à show()) dans cette boucle!\x1b[0m")

# Affichage des erreurs / error display
clfs = erreurs.pop('Classifieurs')
df = pandas.DataFrame(erreurs, index=clfs)
display.display(df)
pyplot.show()
