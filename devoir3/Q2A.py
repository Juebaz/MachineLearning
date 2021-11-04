from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import minmax_scale
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from http.client import HTTPConnection
from io import BytesIO
from IPython import display
import itertools
import collections
import numpy
import requests
import time
import pandas

from sklearn.preprocessing import MinMaxScaler


pandas.set_option('display.max_colwidth', 0)


# Nous ne voulons pas être signalé par ce type d'avertissement, non pertinent pour le devoir
# We don't want to be signaled of this warning, irrelevant for the homework
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Fonction pour vérifier le temps d'exécution
# Function to verify execution time
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, ".format(maxduration) +
              "mais votre code a requis {0:.4f} secondes! ".format(duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")

# Fonction pour obtenir jeu de données Pendigits, ne pas modifier
# Function to obtain the Pendigits dataset, do not modify


def fetchPendigits(file_id):
    """
    Cette fonction télécharge le jeu de données pendigits et le
    retourne sous forme de deux tableaux numpy. Le premier élément
    retourné par cette fonction est un tableau de 10992x16 qui
    contient les données des caractères; le second élément est un
    vecteur de 10992 qui contient les valeurs cible (étiquettes).
    This function download dataset pendigits and return it as numpy
    arrays. The first element returned is a 10992x16 array with the
    characters data; the second element is a vector of size 10992 that
    contains the targets (labels).
    """
    URL = "https://drive.google.com/uc?"
    session = requests.Session()

    r = session.get(URL, params={'id': file_id}, stream=True)
    error_msg = f'ERREUR: impossible de télécharger UCI Pendigits (code={r.status_code})'
    assert(r.status_code == 200), error_msg

    params = {'id': file_id, 'confirm': 'download_warning'}
    r = session.get(URL, params=params, stream=True)
    stream = BytesIO(r.content)
    dataPendigits = numpy.load(stream)
    return dataPendigits[:, :-1].astype('float32'), dataPendigits[:, -1]


# Définition des durées d'exécution maximales pour chaque classifieur
# Definition of maximum execution time for each classifier
TMAX_KNN = 40
TMAX_SVM = 200
TMAX_MLP = 400
TMAX_EVAL = 80
DRIVE_ID = '1OxSkSF2RLDUcAyFd1MpG3RYBHOX8ni4V'


# Question 2A


# Initialisation du jeu de données Pendigits
# Initializing Pendigits dataset
print('Téléchargement du jeu de données Pendigits...')
data, target = fetchPendigits(DRIVE_ID)
print('Téléchargement terminé.')

# normalise chaque dimension entre 0 et 1
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Sépare les données de test et de train
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=5492, random_state=42)

# Creation of the table to accumulate results
results = {'Classifiers': [],
           'Optimal_hp1': [],
           'Optimal_hp2': [],
           'Time_train': [],
           'Time_test': [],
           'Score_train': [],
           'Score_test': [],
           }

_times.append(time.time())

# knn
clf_knn = KNeighborsClassifier()
clf_name = clf_knn.__class__.__name__
if clf_name not in results['Classifiers']:
    results['Classifiers'].append(clf_name)


n_neighbors_possible = [1, 3, 5, 10, 20, 50, 100]
weights_possible = ["uniform", "distance"]

time_train = 0

optimal_hp1 = 0
optimal_hp2 = 0
score_train = 0

for n_neighbors, weights in itertools.product(n_neighbors_possible, weights_possible):
    classifieur_kpp = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights)
    classifieur_kpp.fit(X_train, y_train)
    score = classifieur_kpp.score(X_test, y_test)

    if score > score_train:
        optimal_hp1 = n_neighbors
        optimal_hp2 = weights
        score_train = score


results['Time_train'].append(time_train)

results['Score_train'].append(score_train)
results['Optimal_hp1'].append(n_neighbors_possible)
results['Optimal_hp2'].append(weights_possible)
_times.append(time.time())
checkTime(TMAX_KNN, "k-plus proches voisins")

# svm a noyau gausin
clf_svc = SVC()
clf_name = clf_svc.__class__.__name__
if clf_name not in results['Classifiers']:
    results['Classifiers'].append(clf_name)

C_possible = [0.1, 1, 5, 10, 100, 1000]
gamma_possible = [0.1, 0.2, 0.5]

optimal_hp1 = 0
optimal_hp2 = 0
score_train = 0

for C, gamma in itertools.product(C_possible, gamma_possible):

    # On initialise le classfieur kPP avec les hyperparametres de la grille
    classifieur_svm = SVC(C=C, kernel='rbf', gamma=gamma)

    classifieur_svm.fit(X_train, y_train)

    score = classifieur_svm.score(X_test, y_test)
    print(C, gamma, score)
    if score > score_train:
        optimal_hp1 = C
        optimal_hp2 = gamma
        score_train = score

print(optimal_hp1, optimal_hp2, score_train)

results['Time_train'].append(time_train)
results['Optimal_hp1'].append(optimal_hp1)
results['Optimal_hp2'].append(optimal_hp2)
_times.append(time.time())
checkTime(TMAX_SVM, "Support vector machine")


# perceptron multicouches

clf_mlp = MLPClassifier()
clf_name = clf_mlp.__class__.__name__
if clf_name not in results['Classifiers']:
    results['Classifiers'].append(clf_name)
