from sklearn import datasets
from matplotlib import pyplot
import time
from scipy.integrate._ivp.common import norm
import collections
from sklearn import datasets, svm, model_selection, metrics
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
import time
import pandas
from sklearn.model_selection import RepeatedKFold
from IPython import display


iris = datasets.load_iris()
# Durées d'exécutions maximales / maximum execution duration
TMAX_Q2C_1 = 0.5
TMAX_Q2C_2 = 2.
TMAX_Q2C_3 = 2.

# Erreurs maximales attendues / maximum error expected
ERRMAX_Q2C_1 = 0.07
ERRMAX_Q2C_2 = 0.07
ERRMAX_Q2C_3 = 0.07


erreurs = collections.OrderedDict()

t1 = time.time()

classificator = QuadraticDiscriminantAnalysis()

X = iris.data
Y = iris.target

trainModels = classificator.fit(X, Y)
err = 1 - metrics.accuracy_score(classificator.predict(X), Y)

erreurs['Jeu entier'] = err

### Ne pas modifier / do not modify ###
t2 = time.time()
duration = t2 - t1
if duration > TMAX_Q2C_1:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2C.1 " +
          f"met trop de temps à s'exécuter! Le temps maximum " +
          f"permis est de {TMAX_Q2C_1:.4f} secondes, mais votre " +
          f"code a requis {duration:.4f} secondes! Assurez-vous " +
          f"que vous ne faites pas d'appels bloquants (par " +
          f"exemple à show()) dans cette boucle!\x1b[0m")
if err > ERRMAX_Q2C_1:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2C.1 ne " +
          f"produit pas les performances attendues! Le taux " +
          f"d'erreur maximal attendu est de {ERRMAX_Q2C_1:.3f}, " +
          f"mais l'erreur rapportée dans votre code est de " +
          f"{err:.3f}!\x1b[0m")

t1 = time.time()


# *** TODO Q2C.2 ***

errors = []
for x in range(0, 99):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, train_size=0.50, test_size=0.50)

    classificator.fit(X_train, Y_train)

    err_test = 1 - \
        metrics.accuracy_score(classificator.predict(X_test), Y_test)
    errors.append(err_test)

avgError = sum(errors)/100

erreurs['50/50 Train/Valid'] = avgError

### Ne pas modifier / do not modify ###
t2 = time.time()
duration = t2 - t1
if duration > TMAX_Q2C_2:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2C.2 " +
          f"met trop de temps à s'exécuter! Le temps maximum " +
          f"permis est de {TMAX_Q2C_2:.4f} secondes, mais votre " +
          f"code a requis {duration:.4f} secondes! Assurez-vous " +
          f"que vous ne faites pas d'appels bloquants (par " +
          f"exemple à show()) dans cette boucle!\x1b[0m")
if avgError > ERRMAX_Q2C_2:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2C.2 ne " +
          f"produit pas les performances attendues! Le taux " +
          f"d'erreur maximal attendu est de {ERRMAX_Q2C_2:.3f}, " +
          f"mais l'erreur rapportée dans votre code est de " +
          f"{avgError:.3f}!\x1b[0m")


t1 = time.time()


# *** TODO Q2C.3 ***

crossValGenerator = RepeatedKFold(n_splits=3, n_repeats=100, random_state=2652124)

err = 1- model_selection.cross_val_score(classificator, X, Y, cv=crossValGenerator)


avgError = np.mean(err)

erreurs['Validation 3-fold'] = avgError

### Ne pas modifier / do not modify ###
t2 = time.time()
duration = t2 - t1
if duration > TMAX_Q2C_3:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2C.3 " +
          f"met trop de temps à s'exécuter! Le temps maximum " +
          f"permis est de {TMAX_Q2C_3:.4f} secondes, mais votre " +
          f"code a requis {duration:.4f} secondes! Assurez-vous " +
          f"que vous ne faites pas d'appels bloquants (par " +
          f"exemple à show()) dans cette boucle!\x1b[0m")
if avgError > ERRMAX_Q2C_3:
    print(f"\x1b[31m[ATTENTION] Votre code pour la question Q2C.3 ne " +
          f"produit pas les performances attendues! Le taux " +
          f"d'erreur maximal attendu est de {ERRMAX_Q2C_3:.3f}, " +
          f"mais l'erreur rapportée dans votre code est de " +
          f"{avgError:.3f}!\x1b[0m")
df = pandas.DataFrame(erreurs, index=['Erreurs'])
display.display(df)
