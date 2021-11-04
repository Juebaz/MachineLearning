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

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV


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

def fitWithCrossValidation(clf, X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    return numpy.mean(scores)


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


param_grid = {'n_neighbors': [1, 3, 5, 10, 50, 100],
              'weights': ["uniform", "distance"]}

start_time = time.time()
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1)
time_train = time.time() - start_time

grid.fit(X_train, y_train)

score_train_knn = grid.best_score_


results['Time_train'].append(time_train)

optimal_hp1_knn = grid.best_params_['n_neighbors']
optimal_hp2_knn = grid.best_params_['weights']

results['Score_train'].append(score_train_knn)
results['Optimal_hp1'].append(grid.best_params_['n_neighbors'])
results['Optimal_hp2'].append(grid.best_params_['weights'])
_times.append(time.time())
checkTime(TMAX_KNN, "k-plus proches voisins")

# # svm a noyau gausin
clf_svc = SVC()
clf_name = clf_svc.__class__.__name__
if clf_name not in results['Classifiers']:
    results['Classifiers'].append(clf_name)


param_grid = {'C': [0.1, 1, 50, 10, 100],
              'gamma': [0.1, 0.2, 0.5]}

start_time = time.time()
grid = GridSearchCV(SVC(), param_grid,
                    refit=True, n_jobs=-1)
time_train = time.time() - start_time

grid.fit(X_train, y_train)

score_train_svm = grid.best_score_


results['Time_train'].append(time_train)

optimal_hp1_svm = grid.best_params_['C']
optimal_hp2_svm = grid.best_params_['gamma']


results['Score_train'].append(score_train_svm)
results['Optimal_hp1'].append(grid.best_params_['C'])
results['Optimal_hp2'].append(grid.best_params_['gamma'])
_times.append(time.time())
checkTime(TMAX_SVM, "Support vector machine")


# # perceptron multicouches

clf_mlp = MLPClassifier()
clf_name = clf_mlp.__class__.__name__
if clf_name not in results['Classifiers']:
    results['Classifiers'].append(clf_name)

param_grid = {'hidden_layer_sizes': [5, 10, 100],
              'activation': ['relu', 'identity', 'tanh', 'logistic']}

start_time = time.time()
grid = GridSearchCV(MLPClassifier(max_iter=1000), param_grid,
                    refit=True, n_jobs=-1)
time_train = time.time() - start_time

grid.fit(X_train, y_train)

score_train_svm = grid.best_score_


results['Time_train'].append(time_train)

optimal_hp2_pcm = grid.best_params_['activation']
optimal_hp1_pcm = grid.best_params_['hidden_layer_sizes']


results['Score_train'].append(score_train_svm)
results['Optimal_hp1'].append(grid.best_params_['activation'])
results['Optimal_hp2'].append(grid.best_params_['hidden_layer_sizes'])

_times.append(time.time())
checkTime(TMAX_MLP, "Perceptron multicouche")


# # partie 2 apres avoir optimizer

model_final_kpp = KNeighborsClassifier(
    n_neighbors=optimal_hp1_knn, weights=optimal_hp2_knn)
model_final_svm = SVC(C=optimal_hp1_svm, gamma=optimal_hp2_svm)
model_final_mlp = MLPClassifier(
    hidden_layer_sizes=optimal_hp1_pcm, activation=optimal_hp2_pcm)

# On reentraine les modeles sur le jeu de donnees entrainement complet
start_time = time.time()
model_final_kpp.fit(X_train, y_train)
score_knn = model_final_kpp.score(X_test, y_test)
results['Time_test'].append(time.time() - start_time)

start_time = time.time()
model_final_svm.fit(X_train, y_train)
score_svm = model_final_svm.score(X_test, y_test)
results['Time_test'].append(time.time() - start_time)

start_time = time.time()
model_final_mlp.fit(X_train, y_train)
score_mlp = model_final_mlp.score(X_test, y_test)
results['Time_test'].append(time.time() - start_time)


results['Score_test'].append(score_knn)
results['Score_test'].append(score_svm)
results['Score_test'].append(score_mlp)

_times.append(time.time())
checkTime(TMAX_EVAL, "Evaluation des modèles")

# Affichage des résultats
# Display results
df = pandas.DataFrame(results)
display.display(df)
print(results)
