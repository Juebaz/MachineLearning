from sklearn import datasets
from matplotlib import pyplot
import time
from scipy.integrate._ivp.common import norm
import collections
from sklearn import svm, datasets
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
import time
# import pandas
# from IPython import display


TMAX_Q2B = 1.5
ERRMAX_Q2B = 0.22


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, classificator, xx, yy, **params):
    Z = classificator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris = datasets.load_iris()

pairsOfFeatures = [(i, j) for i in range(4) for j in range(i+1, 4)]

regularizeParam = 1.0  # SVM regularization parameter

models = (svm.SVC(kernel='linear', C=regularizeParam),
          svm.LinearSVC(C=regularizeParam, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=regularizeParam),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=regularizeParam))

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Dictionnaire pour enregistrer les erreurs selon les classifieurs
# Dictionary to record errors according to classifiers
erreurs = collections.defaultdict(list)
erreurs['Classifieurs'] = []

# Traiter par paires de mesures avec réentraînement
# Process with pairs of measures with retraining
for (f1, f2) in pairsOfFeatures:
    f1_name = iris.feature_names[f1]
    f2_name = iris.feature_names[f2]

    X = iris.data[:, [f1, f2]]
    y = iris.target

    # Initialisez différents classifieurs dans la liste nommée 'classifieurs'
    # Initialize various classifiers in the list named 'classifiers'
    # classifieurs = []

    # Créez grille permettant d'afficher régions de décision pour chaque classifieur
    # Indices : numpy.meshgrid pourrait vous être utile, mais n'utilisez pas un pas trop petit!
    # Create a grid for displaying various decision regions for each classifier
    # Tips: numpy.meshgrid can be useful, but don't use a too small step size
    # *******

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # Créer une figure à plusieurs sous-graphes
    # Create a figure with several subplots
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                                   tight_layout=True)
    t1 = time.time()
    for classificator, subfig, title in zip(models, subfigs.reshape(-1), titles):
        clf_name = classificator.__class__.__name__
        if clf_name not in erreurs['Classifieurs']:
            erreurs['Classifieurs'].append(clf_name)

        # *** TODO Q2B ***
        # Entraînez le classifieur
        # Train the classifier
        trainModels = classificator.fit(X, y)

        # Obtenez et affichez son erreur (1 - accuracy)
        # Obtain and display its error (1 - accuracy)

        # Stockez la valeur de cette erreur dans la variable err
        # Store this error in variable err

        # Utilisez la grille que vous avez créée plus haut
        # pour afficher les régions de décision, de même
        # que les points colorés selon leur vraie classe
        # Use the grid created below to display decision regions
        # as well dots of the samples coloured according to their
        # real class label
        # ******
        plot_contours(subfig, classificator, xx, yy,
                      cmap=pyplot.cm.coolwarm, alpha=0.8)
        subfig.scatter(X0, X1, c=y, cmap=pyplot.cm.coolwarm,
                       s=20, edgecolors='k')
        subfig.set_xlim(xx.min(), xx.max())
        subfig.set_ylim(yy.min(), yy.max())
        subfig.set_xlabel('Sepal length')
        subfig.set_ylabel('Sepal width')
        subfig.set_xticks(())
        subfig.set_yticks(())
        subfig.set_title(title)
        # Ajouter l'erreur pour affichage
        # Add error for displaying
        # erreurs[f'{f1_name} {f2_name}'].append(err)

        ### Ne pas modifier / do not modify ###
        # if err > ERRMAX_Q2B:
        #     print(f"\x1b[31m[ATTENTION] Votre code pour la " +
        #           f"question Q2B ne produit pas les performances" +
        #           f"attendues! Le taux d'erreur maximal attendu " +
        #           f"est de {ERRMAX_Q2B:.3f}, mais l'erreur " +
        #           f"rapportée dans votre code est de {err:.3f}!\x1b[0m")

    pyplot.show()
    t2 = time.time()
    duration = t2 - t1
    if duration > TMAX_Q2B:
        print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2B " +
              f"met trop de temps à s'exécuter! Le temps maximum " +
              f"permis est de {TMAX_Q2B:.4f} secondes, mais votre " +
              f"code a requis {duration:.4f} secondes! Assurez-vous " +
              f"que vous ne faites pas d'appels bloquants (par " +
              f"exemple à show()) dans cette boucle!\x1b[0m")

# Affichage des erreurs / display errors
# clfs = erreurs.pop('Classifieurs')
# df = pandas.DataFrame(erreurs, index=clfs)
# display.display(df)
