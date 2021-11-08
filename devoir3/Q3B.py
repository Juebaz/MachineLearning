import pandas
pandas.set_option('display.max_colwidth', 0)

from sklearn.datasets import make_moons
from IPython import display
from sklearn.model_selection import train_test_split
# Nous ne voulons pas être signalé par ce type d'avertissement, non pertinent pour le devoir
# We don't want to be signaled of this warning, irrelevant for the homework
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from matplotlib import pyplot


import time
from scipy.optimize import fmin_l_bfgs_b
import numpy
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question)+
              "Le temps maximum permis est de {0:.4f} secondes, ".format(maxduration)+
              "mais votre code a requis {0:.4f} secondes! ".format(duration)+
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")

# Définition des durées d'exécution maximales pour chaque classifieur
# Definition of maximum execution time for each classifier
TMAX_EVAL = 3.0
TMAX_FIT = 2.0
class DiscriminantANoyau:

    def __init__(self, lambda_, sigma, verbose=True):
        # Cette fonction est déjà codée pour vous, vous n'avez qu'à utiliser les variables membres qu'elle
        # définit dans les autres fonctions de cette classe. Lambda et sigma sont définis dans l'énoncé.
        # verbose permet d'afficher certaines statistiquesen lien avec la convergence de fmin_l_bfgs_b.
        # This function is already coded for you, you just have to use the member variables it defines in
        # in the other functions of this class. Lambda and sigma are defined in the statement.
        # verbose allows you to display some statistics related to the convergence of the fmin_l_bfgs_b.
        self.lambda_ = lambda_
        self.sigma = sigma
        self.verbose = verbose

    def discriminant(self, X, x_t,y, parametres):
        h = numpy.empty(x_t.shape[0])
        for i in range(x_t.shape[0]):
            h[i]= numpy.sum(parametres[1:]*y*numpy.exp(-(numpy.linalg.norm(X-x_t[i],axis=1,ord=2)**2)/(self.sigma**2)))+parametres[0]
        return h
    def fit(self, X, y):
        # Implémentez la fonction d'entraînement du classifieur, selon les équations développées à Q3A
        # Implement the training function of the classifier, according to the equations developed in Q3A

        # *** TODO ***
        # Vous devez écrire une fonction nommée evaluateFunc, qui reçoit un seul argument, soit les valeurs
        # des paramètres pour lesquels on souhaite connaître l'erreur et le gradient. Cette fonction sera
        # appelée à répétition par l'optimiseur de scipy, qui l'utilisera pour minimiser l'erreur et
        # obtenir un jeu de paramètres optimal.
        # You must write a function named evaluateFunc, which receives only one argument, either the values
        # of the parameters for which you want to know the error and the gradient. This function will be
        # called repeatedly by the scipy optimizer, which will use it to minimize the error and
        # to obtain an optimal set of parameters.
        def evaluateFunc(params):
            # Ecrire ici le code de la fonction evaluateFunc calculant l'erreur
            # et le gradient de params.
            mauvais_class = numpy.where(self.discriminant(X,X,y,params) * y <=0)[0]
            x_mauvaise_classe = X[mauvais_class]
            y_mauvaise_classe = y[mauvais_class]

            err=numpy.sum(1 - y_mauvaise_classe*self.discriminant(X,x_mauvaise_classe,y,params))+(self.lambda_*numpy.sum(params[0:]))

            grad=numpy.zeros(params.shape[0])
            grad[0]=-numpy.sum(y_mauvaise_classe)
            for i in range(1,grad.shape[0]):
                grad[i]=-numpy.sum(y[i-1]*y_mauvaise_classe*numpy.exp(-(numpy.linalg.norm((x_mauvaise_classe-X[i-1]),axis=1,ord=2)**2)/(self.sigma**2)))+self.lambda_
            # Write here the code of the evaluateFunc function calculating the error
            # and the gradient of params.
            return err, grad

        # ******

        # *** TODO ***
        # Initialisez aléatoirement les paramètres alpha^t et w_0 (l'optimiseur requiert une valeur initiale,
        # et nous ne pouvons pas simplement n'utiliser que des zéros pour différentes raisons). Stockez ces
        # valeurs initiales aléatoires dans un array numpy nommé "params". Déterminez également les bornes à
        # utiliser sur ces paramètres et stockez les dans une variable nommée "bounds".
        # Indice : les paramètres peuvent-ils avoir une valeur maximale (au-dessus de laquelle ils ne veulent
        # plus rien dire)? Une valeur minimale? Référez-vous à la documentation de fmin_l_bfgs_b pour savoir
        # comment indiquer l'absence de bornes.
        # Randomly initialize the parameters alpha^t and w_0 (the optimizer requires an initial value,
        # and we can't just use zeros for various reasons). Store these
        # random initial values in a numpy array named "params". Also determine the bounds to use on these
        # parameters and store them in a variable named "bounds".
        # Hint: Can the parameters have a maximum value (above which they mean nothing)?
        # mean anything)? A minimum value? Refer to the documentation of fmin_l_bfgs_b to know
        # how to indicate the absence of bounds.
        # ******
        params=numpy.random.rand(X.shape[0]+1)

        bounds=[(None,None)]*(X.shape[0]+1)
        for i in range(1,X.shape[0]+1):
            bounds[i]=(0,numpy.inf)
        # À ce stade, trois choses devraient être définies :
        # - Une fonction d'évaluation nommée evaluateFunc, capable de retourner l'erreur et le gradient d'erreur
        #   pour chaque paramètre, pour une configuration d'alpha et w_0 donnée.
        # - Un tableau numpy nommé params de même taille que le nombre de paramètres à entraîner.
        # - Une liste nommée bounds contenant les bornes que l'optimiseur doit respecter pour chaque paramètre.
        # On appelle maintenant l'optimiseur avec ces informations et on conserve les valeurs dans params
        # At this point, three things should be defined:
        # - An evaluation function named evaluateFunc, capable of returning the error and error gradient
        # for each parameter, for a given alpha and w_0 configuration.
        # - A numpy array named params of the same size as the number of parameters to train.
        # - A list named bounds containing the bounds that the optimizer must respect for each parameter.
        # We now call the optimizer with this information and we keep the values in params
        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(evaluateFunc, params, bounds=bounds)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        if self.verbose:
            # On affiche quelques statistiques / display some statistics
            print("Entraînement terminé après {it} itérations et {calls} appels à evaluateFunc".format(it=infos['nit'],
                                                                                                       calls=infos[
                                                                                                           'funcalls']))
            print("\tErreur minimale : {:.5f}".format(minval))
            print("\tL'algorithme a convergé" if infos['warnflag'] == 0 else "\tL'algorithme n'a PAS convergé")
            print("\tGradients des paramètres à la convergence (ou à l'épuisement des ressources) :")
            print(infos['grad'])

        # *** TODO ***
        # Stockez les paramètres optimisés de la façon suivante :
        # - Le vecteur alpha dans self.alphas;
        # - Le biais w_0 dans self.w0.
        # Store the optimized parameters as follows:
        # - The alpha vector in self.alphas;
        # - The bias w_0 in self.w0.
        # ******
        self.alphas=params[1:]
        self.w0=params[0]
        # On retient également le jeu d'entraînement, qui pourra vous être utile pour les autres fonctions.
        # We also retain the training set, which can be useful for other functions.
        self.X, self.y = X, y
        return self

    def predict(self, X):
        # *** TODO ***
        # Implémentez la fonction d'inférence (prédiction). Vous pouvez supposer que fit() a préalablement été
        # exécuté et que les variables membres alphas, w0, X et y existent. N'oubliez pas que ce classifieur doit
        # retourner -1 ou 1
        # Implement the inference (prediction) function. You can assume that fit() has been previously executed and that the
        # executed and that the member variables alphas, w0, X and y exist. Remember that this classifier must
        # return -1 or 1
        # ******
        params = numpy.append(self.w0,self.alphas)
        predictions=self.discriminant(self.X,X,self.y,params)
        for i in range(predictions.shape[0]):
            if predictions[i]>0:
                predictions[i]=1
            else:
                predictions[i]=-1
        return predictions

    def score(self, X, y):
        # *** TODO ***
        # Implémentez la fonction retournant le score (accuracy) du classifieur sur les données reçues en
        # argument. Vous pouvez supposer que fit() a préalablement été exécutée
        # Indice : réutiliser votre implémentation de predict() réduit de beaucoup la taille de cette fonction!
        # Implement the function returning the accuracy of the classifier on the data received as
        # argument. You can assume that fit() has already been executed
        # Hint: reusing your implementation of predict() reduces the size of this function by a lot!
        # ******
        classe_predi=self.predict(X)
        y_bin=numpy.copy(y)
        for i in range(y.shape[0]):
            if y[i]<=0:
                y_bin[i]=-1
        score=sum(classe_predi==y)/y_bin.shape[0]
        return score

if __name__=="__main__":
    results = {'Classifier': ["DiscriminantANoyau"],
               'Range_lambda': [],
               'Range_sigma': [],
               'Best_lambda': [],
               'Best_sigma': [],
               'Error_train': [],
               'Error_test': [],
               }

    # *** TODO ***
    # Créez le jeu de données à partir de la fonction make_moons, tel que demandé dans l'énoncé
    # N'oubliez pas de vous assurer que les valeurs possibles de y sont bel et bien dans -1 et 1, et non 0 et 1!
    # Create the dataset from the make_moons function, as requested in the statement
    # Remember to make sure that the possible values of y are in -1 and 1, not 0 and 1!
    # ******
    data=make_moons(n_samples=800,noise=0.3)
    X=data[0]
    y=data[1]
    for i in range(y.shape[0]):
        if y[i] < 1:
            y[i]=-1
    # *** TODO ***
    # Séparez le jeu de données en deux parts égales, l'une pour l'entraînement et l'autre pour le test.
    # Separate the dataset into two equal parts, one for training and one for testing.
    # ******
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.5,random_state=2)
    _times.append(time.time())

    # *** TODO ***
    # Indiquez la plage de recherche pour le
    # paramètre Lambda en la mettant dans
    # la liste de la variable range_lambda.
    # Specify the search range for the
    # parameter Lambda by putting it in
    # the list of the variable range_lambda.
    range_lambda = [0.7,1,2.5,5]
    # ******

    results['Range_lambda'].append(range_lambda)

    # *** TODO ***
    # Indiquez la plage de recherche pour le
    # paramètre Sigma en la mettant dans
    # la liste de la variable range_sigma.
    # Specify the search range for the
    # parameter Sigma by putting it in
    # the list of the variable range_sigma.
    range_sigma = [0.3,0.5,0.9,1]
    # ******

    results['Range_sigma'].append(range_sigma)

    # *** TODO ***
    # Optimisez ici les paramètres lambda et sigma de votre
    # classifieur en effectuant une recherche en grille.
    # Optimize here the lambda and sigma parameters of your
    # classifier by performing a grid search.
    # ******
    high_score = 0
    for lambda_ in range_lambda:
        for sigma in range_sigma:
            classifieur=DiscriminantANoyau(lambda_,sigma)
            classifieur.fit(X_train,y_train)
            scr=classifieur.score(X_val,y_val)
            if scr>high_score:
                high_score=scr
                best_lambda=lambda_
                best_sigma=sigma
    # *** TODO ***
    # Indiquez la valeur optimale pour le
    # paramètre Lambda en la mettant dans la
    # variable best_lambda en remplaçant le 0.
    # Specify the optimal value for the
    # parameter Lambda by putting it in the
    # variable best_lambda by replacing the 0.

    # ******

    results['Best_lambda'].append(best_lambda)

    # *** TODO ***
    # Indiquez la valeur optimale pour le
    # paramètre Sigma en la mettant dans la
    # variable best_sigma en remplaçant le 0.
    # Specify the optimal value for the
    # parameter Sigma by putting it in the
    # variable best_sigma by replacing the 0.
    # ******

    results['Best_sigma'].append(best_sigma)

    # *** TODO ***
    # Une fois les paramètres lambda et
    # sigma de votre classifieur optimisés,
    # créez une instance de ce classifieur
    # en utilisant ces paramètres optimaux.
    # Once the lambda and sigma parameters of your classifier are optimized,
    # create an instance of this classifier
    # using these optimal parameters.
    # ******
    instance_opt=DiscriminantANoyau(best_lambda,best_sigma)
    instance_opt.fit(X_train,y_train)


    # *** TODO ***
    # Indicate the error rate obtained on the training set in the variable
    # err_train variable by replacing the 0.
    err_train = 1-(instance_opt.score(X_train,y_train))

    # ******

    results['Error_train'].append(err_train)

    # *** TODO ***
    # Indiquez le taux d'erreur obtenu sur le
    # jeu de test dans la variable
    # err_test en remplaçant le 0.
    # Indicate the error rate obtained on the
    # test set in the err_test variable by replacing the 0.
    err_test = 1-instance_opt.score(X_val,y_val)
    # ******

    results['Error_test'].append(err_test)

    # *** TODO ***
    # Créez ici une grille permettant d'afficher les régions de décision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous être utile ici
    # Par la suite, affichez les régions de décision dans la même figure que les données de test.
    # Note : un pas de 0.02 pour le meshgrid est recommandé
    # ******

    # On affiche la figure
    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation")
    pyplot.show()

    # Affichage des résultats
    df = pandas.DataFrame(results)
    display.display(df)