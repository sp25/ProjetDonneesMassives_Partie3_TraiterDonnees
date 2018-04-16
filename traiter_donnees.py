import numpy as np
import random
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import datetime
from math import sqrt, log, exp
from sklearn.svm import SVR


def pretraitement_donnees(fichier, columns):

    datasetTotal = pandas.read_table('dataset/' + fichier, delimiter=',', header=0, usecols=columns)

    datasetTotal['tempC'] = preprocessing.scale(datasetTotal['temp'])
    datasetTotal.drop('temp', axis=1, inplace=True)

    datasetTotal['windspeedC'] = preprocessing.scale(datasetTotal['windspeed'])
    datasetTotal['humidityC'] = preprocessing.scale(datasetTotal['humidity'])

    # À partir de l'attribut datetime, crééer des nouveaux attributs (month, day, hours)
    datasetTotal['time'] = pandas.to_datetime(datasetTotal['datetime'])
    datasetTotal['month'] = datasetTotal.time.dt.month
    datasetTotal['day'] = datasetTotal.time.dt.dayofweek

    datasetTotal['hour'] = datasetTotal.time.dt.hour

    dates = datasetTotal['datetime'].values.tolist()
    datasetTotal.drop('time', axis=1, inplace=True)
    datasetTotal.drop('datetime', axis=1, inplace=True)

    return datasetTotal, dates

# Algorithme final
# Prediction séparémment de la variable "casual" et "registered".
# Puisque
def explorer_random_forest_predire_casual_et_registered(nbr_estimators):
    randomstate = 42
    dataset, dates = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 6,7,8, 9, 10, 11])
    X = dataset[['season','holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC', 'month', 'day', 'hour']]
    Y = dataset[['casual']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)

    # appliquer la transformation logarithmique
    y_train['casual'] = y_train['casual'].apply(transfLog)

    # entraîner et prédire la variable casual
    rfCasual = RandomForestRegressor(n_estimators=nbr_estimators, oob_score=False, random_state=randomstate)
    rfCasual.fit(X_train, y_train.values.ravel())
    predictedForestRegre2 = rfCasual.predict(X_test)

    # faire la transformation inverse de la logarithmique
    predictedForestRegre22 = []
    for x in predictedForestRegre2:
        predictedForestRegre22.append(transfExp(x))

    #rmsle_score = rmsle_selon_interval_min_max(y_test, predictedForestRegre22, 250, 500)
    rmsle_score = rmsle(y_test, predictedForestRegre22, X_train)
    print("RMSLE casual : " + str(rmsle_score))

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC','month', 'day','hour']]  # 'casual', 'registered'
    Y = dataset[['registered']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    # appliquer la transformation logarithmique
    y_train['registered'] = y_train['registered'].apply(transfLog)

    # entraîner et prédire la variable registered
    rfRegistered = RandomForestRegressor(n_estimators=nbr_estimators, oob_score=False, random_state=randomstate)
    rfRegistered.fit(X_train, y_train.values.ravel())
    predictedForestRegre1 = rfRegistered.predict(X_test)

    # faire la transformation inverse de la logarithmique
    predictedForestRegre11 = []
    for x in predictedForestRegre1:
        predictedForestRegre11.append(transfExp(x))

    #rmsle_score = rmsle_selon_interval_min_max(y_test, predictedForestRegre11, 250, 500)
    rmsle_score = rmsle(y_test, predictedForestRegre11, X_train)
    print("RMSLE Registered : " + str(rmsle_score))

    # faire la somme des deux prédictions
    somme = [x + y for x, y in zip(predictedForestRegre11, predictedForestRegre22)]

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['count']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    # Évaluer le résultat de la somme des deux prédictions
    #rmsle_score = rmsle_selon_interval_min_max(y_test, somme, 250, 500)
    rmsle_score = rmsle(y_test, somme, X_train)
    print("RMSLE Forest Regressor combiné : " + str(rmsle_score))

    # print("Importance registered")
    # importances1 = list(rfRegistered.feature_importances_)
    # feature_importances1 = [(feature, round(importance1, 2)) for feature, importance1 in zip(['season', 'holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC','month', 'day', 'hour'], importances1)]
    # feature_importances1 = sorted(feature_importances1, key=lambda x: x[1], reverse=True)
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances1]
    #
    # print("Importance casual")
    # importances = list(rfCasual.feature_importances_)
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(['season', 'holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC','month', 'day', 'hour'], importances)]
    # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    return rfCasual, rfRegistered

# permet de préparer le fichier final à soumettre dans Kaggle
def soumettreKaggle_casual_et_registered():
    rfCasual, rfRegistered = explorer_random_forest_predire_casual_et_registered(30)
    dataset, dates = pretraitement_donnees('test.csv', [0, 1, 2, 3, 4, 5, 6, 7, 8])

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC', 'month', 'day', 'hour']]
    predictedCasualLog = rfCasual.predict(X)
    predictedCasual = []
    for x in predictedCasualLog:
        predictedCasual.append(transfExp(x))

    predictedRegisteredLog = rfRegistered.predict(X)
    predictedRegistered = []
    for x in predictedRegisteredLog:
        predictedRegistered.append(transfExp(x))

    somme = [x + y for x, y in zip(predictedCasual, predictedRegistered)]

    file = open("C:/temp/algo_final.csv", "w")
    for index in range(len(dates)):
        file.write(str(dates[index]) + "," + str(int(round(somme[index]))) + "\n")
    file.close()


###################################
## Autre algogrithmes testés
#####################################

# Prédiction de casual et registered sans appliquer la transformation logarithmique
def explorer_random_forest_predire_casual_et_registered_sans_log():
    randomstate = 42
    dataset, dates = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 6,7,8, 9, 10, 11])
    X = dataset[['season','holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['casual']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    rfCasual = RandomForestRegressor(n_estimators=30, oob_score=False, random_state=randomstate)
    rfCasual.fit(X_train, y_train.values.ravel())
    predictedForestRegre2 = rfCasual.predict(X_test)

    rmsle_score = rmsle(y_test, predictedForestRegre2, X_test)
    print("RMSLE casual : " + str(rmsle_score))

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC','month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['registered']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    rfRegistered = RandomForestRegressor(n_estimators=30, oob_score=False, random_state=randomstate)
    rfRegistered.fit(X_train, y_train.values.ravel())
    predictedForestRegre1 = rfRegistered.predict(X_test)

    rmsle_score = rmsle(y_test, predictedForestRegre1, X_test)
    print("RMSLE Registered : " + str(rmsle_score))

    somme = [x + y for x, y in zip(predictedForestRegre1, predictedForestRegre2)]

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['count']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    rmsle_score = rmsle(y_test, somme, X_test)
    print("RMSLE Forest Regressor combiné : " + str(rmsle_score))

    return rfCasual, rfRegistered

# analyse préliminaire des différents algorithmes.
def explorer_algorithmes():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday','tempC', 'weather', 'month', 'day', 'hour']]
    Y = dataset[['casual']]#'casual', 'registered'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)#,

    clf_tree_regr = tree.DecisionTreeRegressor(random_state=randomstate)
    clf_tree_regr.fit(X_train, y_train)
    predicted1 = clf_tree_regr.predict(X_test)
    imprimerResultatsDansConsole("DecisionTreeRegressor", y_test, predicted1)

    rfg = RandomForestRegressor(n_estimators=40, oob_score=True, random_state=randomstate,  max_depth = 25)
    rfg.fit(X_train, y_train.values.ravel())
    predicted4 = rfg.predict(X_test)
    imprimerResultatsDansConsole("RandomForestRegressor", y_test, predicted4)

    clf_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=randomstate)
    clf_gbr.fit(X_train, y_train.values.ravel())
    clf_gbr.score(X_test, y_test)
    predicted5 = clf_gbr.predict(X_test)
    imprimerResultatsDansConsole("GradientBoostingRegressor", y_test, predicted5)

    clf = SVR(C=1.0, epsilon=0.2)
    clf.fit(X_train, y_train.values.ravel())
    predicted7 = clf.predict(X_test)
    imprimerResultatsDansConsole("SVR", y_test, predicted7)

# Prédire le nombre de vélos basé uniquement sur la variable 'count'
def explorer_random_forest_predire_uniquement_count():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday','tempC', 'weather', 'month', 'day', 'hour']]
    Y = dataset[['count']]#'casual', 'registered'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)#,

    rfg = RandomForestRegressor(n_estimators=40, oob_score=True, random_state=randomstate)
    rfg.fit(X_train, y_train.values.ravel())
    predicted4 = rfg.predict(X_test)
    imprimerResultatsDansConsole("RandomForestRegressor", y_test, predicted4)

# Prédire le nombre de vélos basé uniquement sur la variable 'count' mais en appliquant la transformation logarithmique
def explorer_random_forest_predire_uniquement_count_AVEC_log():
    randomstate = 42
    dataset, dates = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC', 'month', 'day','hour']]
    Y = dataset[['count']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)#,

    y_train['count'] = y_train['count'].apply(transfLog)

    rfg = RandomForestRegressor(n_estimators=40, oob_score=True, random_state=randomstate)
    rfg.fit(X_train, y_train.values.ravel())
    predicted4 = rfg.predict(X_test)

    predicted44 = []
    for x in predicted4:
        predicted44.append(transfExp(x))

    rmsle_score = rmsle(y_test, predicted44, X_test)
    print("RMSLE casual : " + str(rmsle_score))

    return rfg

# Permet d'extraire dans un fichier CSV les résultats de la prédiction de casual dépassant un certain seuil
def extraireDansCsvResultsPiresQueMoyenne():
    randomstate = 42
    dataset, dates = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 6,7,8, 9, 10, 11])
    X = dataset[['season','holiday', 'workingday', 'tempC', 'weather', 'windspeedC', 'humidityC', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['casual']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)

    y_train['casual'] = y_train['casual'].apply(transfLog)

    rf = RandomForestRegressor(n_estimators=40, oob_score=False, random_state=randomstate)
    rf.fit(X_train, y_train.values.ravel())
    predictedForestRegre2 = rf.predict(X_test)

    predictedForestRegre22 = []
    for x in predictedForestRegre2:
        predictedForestRegre22.append(transfExp(x))

    indexMauvaisResultat = extraire_enregistrements_selon_seuil_rmsle(y_test, predictedForestRegre22, 0.5)

    mauvaisResultat= []
    #mauvaisResultat.append(X_test.columns.values)

    for index in indexMauvaisResultat:
        mauvaisResultat.append(X_test.iloc[index[0]].tolist())

    np.savetxt('C:/temp/mauvais4.csv', mauvaisResultat, delimiter=',')
    np.savetxt('C:/temp/mauvais4_result.csv', indexMauvaisResultat, delimiter=',')

def explorer_nbrArbre_randomForest():
    for nbEstimators in [10, 20, 30, 40, 50, 60]:
        print("Nombre estimateurs " + str(nbEstimators))
        explorer_random_forest_predire_casual_et_registered(nbEstimators)

def explorer_profondeurMax_randomForest():
    for profondeurMax in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print("Profondeur max " + str(profondeurMax))
        explorer_random_forest_predire_casual_et_registered(30, profondeurMax)

def explorer_maxFeature_randomForest():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]
    Y = dataset[['count']]  # 'casual', 'registered'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)

    results = []
    for maxFeature in ['auto', 'sqrt', 'log2']:
        rfg = RandomForestRegressor(n_estimators=40, oob_score=False, random_state=randomstate, max_features=maxFeature)
        rfg.fit(X_train, y_train.values.ravel())
        predicted4 = rfg.predict(X_test)
        print("RMSLE " + str(maxFeature) + " : " + str(rmsle(y_test, predicted4)))


def imprimerResultatsDansConsole(methode, y_test, predicted):
    print("RMSLE " + methode + " : " + str(rmsle(y_test, predicted)))
    print("MSE " + methode + " : " + str(mean_squared_error(y_test, predicted)))
    print("RMSE " + methode + " : " + str(sqrt(mean_squared_error(y_test, predicted))))
    print("MAE " + methode + " : " + str(mean_absolute_error(y_test, predicted)))
    print("************************************************************************")

def transfLog(x):
    return log(x + 1)

def transfExp(x):
    return exp(x) - 1

# pris du site de Kaggle
def rmsle(real, predicted, test):

    compteur = 0;
    sum=0.0

    for x in range(len(predicted)):
        if predicted[x]<0 or real.values[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real.values[x]+1)
        res = (p - r) ** 2
        sum = sum + res
        compteur += 1
    print("Nombre élément retenus : " + str(compteur))
    print("Nombre élément dataset : " + str(len(predicted)))
    print("Pourcentage : " + str((compteur / len(predicted)) * 100))
    return (sum/compteur)**0.5

def rmsle_selon_interval_min_max(real, predicted, seuil_min, seuil_max):

    compteur = 0;
    sum=0.0
    for x in range(len(predicted)):
        if real.values[x] >= seuil_min and real.values[x] <= seuil_max:
            if predicted[x]<0 or real.values[x]<0: #check for negative values
                continue
            p = np.log(predicted[x]+1)
            r = np.log(real.values[x]+1)
            res = (p - r)**2
            sum = sum + res
            compteur += 1
    print("Nombre élément retenus : " + str(compteur))
    print("Nombre élément dataset : " + str(len(predicted)))
    print("Pourcentage : " + str((compteur / len(predicted)) * 100))
    return (sum / compteur) ** 0.5

def extraire_enregistrements_selon_seuil_rmsle(real, predicted, seuil):
    mauvaisResultat = []
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real.values[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real.values[x]+1)
        res = (p - r)**2
        if res > seuil:
            mauvaisResultat.append([x,predicted[x], real.values[x].tolist()[0], res])
        sum = sum + res

    return mauvaisResultat