
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


matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)


_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")  # Durée d'exécution maximale


# Maximum execution duration
TMAX_Q2D = 60

# Dictionnaire pour enregistrer les paramètres évalués
# Dictionnary for recording evaluated parameters
params = collections.defaultdict(list)
params['classifier'] = []

_times.append(time.time())

# *** TODO Q2D ***

# Chargez les données "Breast cancer Wisconsin" et normalisez les de
# manière les valeurs minimum/maximum tiennent dans le domaine [0, 1]
# Load "Breast cancer Wisconsin" dataset and normalize it in order to
# get their minimum/maximum values in the [0, 1] domain

data = load_breast_cancer()
features = data['data']
R = data['target']
features = minmax_scale(features)


# Comparez les diverses approches demandées dans l'énoncé sur Breast Cancer
# Initialisez votre discriminant linéaire avec les paramètres suivants :
# DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
# N'oubliez pas que l'évaluation doit être faite par une validation
# croisée à K=3 plis!
# Compare the various approaches requested in the statement on Breast Cancer
# Initialize your linear discriminant with the following parameters:
# DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
# Don't forget that the evaluation must be done by a cross validation
# with K=3 folds!

# Initialisation des différents classifieurs
# Initialize the various classifiers
classifiers = [DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000),
               # Ajustez les hyperparamètres! / Adjust the hyperparameters!
               LinearDiscriminantAnalysis(tol=1e-10),
               LogisticRegression(tol=0.00000001, max_iter=10000, C=100.0,),
               Perceptron(alpha=0.1, l1_ratio=0.015, max_iter=10000,
                          tol=0.00001, eta0=0.1, validation_fraction=0.01)
               ]

# Création du tableau pour accumuler les résultats
# Create the table to accumulate the results
results = {'Classifiers': [],
           'Train_err': [],
           'Valid_err': [],
           'Exec_time': [],
           'Comments': [],
           }

for clf in classifiers:
    clf_name = clf.__class__.__name__
    if clf_name not in results['Classifiers']:
        results['Classifiers'].append(clf_name)

    # Boucle d'entraînement à faire
    # Training loop to be done
    model = clf.fit(features, R)

    clf.predict(features)
    err = 1-clf.score(features, R)
    # Validation croisée (K=3) à faire
    # Cross-validation (K=3) to be don
    #cross_validate
    kf = KFold(n_splits=3, shuffle=True, random_state=2652124)
    kf.split(features)
    
    #avgError = numpy.mean(err)
   
    # Mesure du temps d'exécution à faire
    # Measuring execution time to be done

    # Ajoutez l'erreur d'entraînement dans la variable train_err
    # Add training error in variable train_err
    train_err = err  # Remplacez le 0 par la valeur / remplace the 0 with the value
    results['Train_err'].append(train_err)

    # Ajoutez l'erreur de validation dans la variable valid_err
    # Add validation error in variable valid_err
    valid_err = 0  # Remplacez le 0 par la valeur / remplace the 0 with the value
    results['Valid_err'].append(valid_err)

    # Ajoutez le temps de calcul mesuré dans la variable exec_time
    # Add measure execution time in variable exec_time
    exec_time = 0  # Remplacer le 0 par la valeur / remplace the 0 with the value
    results['Exec_time'].append(exec_time)

# ******

_times.append(time.time())
checkTime(TMAX_Q2D, "Q2D")


# *** TODO Q4C ***
# Ajoutez les commentaires et les hyperparamètres
# utilisés pour chaque classifieur demandé
# Add comments and hyperparameters used for each
# classifier requested

# Ajoutez vos commentaires
# Add your comments
comments = "Commentaires pour le \
            DiscriminantLineaire ici."
results['Comments'].append(comments)

# Ajoutez vos commentaires et HP pour le LinearDiscriminantAnalysis
# Add your comments and HP for LinearDiscriminantAnalysis
comments = "Commentaires & HP pour le \
            LinearDiscriminantAnalysis ici."
results['Comments'].append(comments)

# Ajoutez vos commentaires et HP pour le LogisticRegression
# Add your comments and HP for LogisticRegression
comments = "Commentaires & HP pour la \
            LogisticRegression ici."
results['Comments'].append(comments)

# Ajoutez vos commentaires et HP pour le Perceptron
# Add your comments and HP for Perceptron
comments = "Commentaires & HP pour le \
            Perceptron ici."
results['Comments'].append(comments)

# *****


# Affichage des erreurs
df = pandas.DataFrame(results)
display.display(df)
