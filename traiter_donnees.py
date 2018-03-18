import numpy as np
import random
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import datetime
from math import sqrt
from sklearn.svm import SVR

def pretraitement_donnees(fichier, columns):
    # ************************************************************************************
    datasetTotal = pandas.read_table('dataset/' + fichier, delimiter=',', header=0, usecols=columns)

    # Conserver l'attribut temperature qui avait montré le plus d'intérêts à l'étape précédente
    datasetTotal['tempC'] = preprocessing.scale(datasetTotal['temp'])
    datasetTotal.drop('temp', axis=1, inplace=True)

    # À partir de l'attribut datetime, crééer des nouveaux attributs (month, day, hours)
    datasetTotal['time'] = pandas.to_datetime(datasetTotal['datetime'])
    datasetTotal['month'] = datasetTotal.time.dt.month
    datasetTotal['day'] = datasetTotal.time.dt.dayofweek

    datasetTotal['hour'] = datasetTotal.time.dt.hour
    #datasetTotal['hour'] = pandas.to_datetime(datasetTotal['datetime']).apply(compartimenterHeures)

    datasetTotal.drop('time', axis=1, inplace=True)
    datasetTotal.drop('datetime', axis=1, inplace=True)


    return datasetTotal

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

def explorer_random_forest_predire_uniquement_count():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday','tempC', 'weather', 'month', 'day', 'hour']]
    Y = dataset[['count']]#'casual', 'registered'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)#,

    rfg = RandomForestRegressor(n_estimators=40, oob_score=True, random_state=randomstate)
    rfg.fit(X_train, y_train.values.ravel())
    predicted4 = rfg.predict(X_test)
    imprimerResultatsDansConsole("RandomForestRegressor", y_test, predicted4)

def explorer_random_forest_predire_casual_et_registered():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 9, 10, 11])
    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['casual']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    rf = RandomForestRegressor(n_estimators=40, oob_score=False, random_state=randomstate)
    rf.fit(X_train, y_train.values.ravel())
    predictedForestRegre2 = rf.predict(X_test)
    rmsle_score = rmsle(y_test, predictedForestRegre2)
    print("RMSLE casual : " + str(rmsle_score))

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['registered']]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    rf = RandomForestRegressor(n_estimators=40, oob_score=False, random_state=randomstate)
    rf.fit(X_train, y_train.values.ravel())
    predictedForestRegre1 = rf.predict(X_test)
    rmsle_score = rmsle(y_test, predictedForestRegre1)
    print("RMSLE Registered : " + str(rmsle_score))

    somme = [x + y for x, y in zip(predictedForestRegre1, predictedForestRegre2)]

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]  # 'casual', 'registered'
    Y = dataset[['count']]

    # train, test = charger_donnees_train_test(dataset, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)  # ,

    rmsle_score = rmsle(y_test, somme)
    print("RMSLE Forest Regressor combiné : " + str(rmsle_score))

def explorer_nbrArbre_randomForest():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]
    Y = dataset[['count']]  # 'casual', 'registered'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)

    results = []
    for nbEstimators in [10, 20, 30, 40, 50, 60, 70 , 80 , 90, 100, 110, 200, 300]:
        rfg = RandomForestRegressor(n_estimators=nbEstimators, oob_score=False, random_state=randomstate)
        rfg.fit(X_train, y_train.values.ravel())
        predicted4 = rfg.predict(X_test)
        print("RMSLE " + str(nbEstimators) + " : " + str(rmsle(y_test, predicted4)))

def explorer_profondeurMax_randomForest():
    randomstate = 42
    dataset = pretraitement_donnees('train.csv', [0, 1, 2, 3, 4, 5, 9, 10, 11])

    X = dataset[['season', 'holiday', 'workingday', 'tempC', 'weather', 'month', 'day', 'hour']]
    Y = dataset[['count']]  # 'casual', 'registered'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=randomstate, shuffle=True)

    results = []
    for profondeurMax in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        rfg = RandomForestRegressor(n_estimators=40, oob_score=False, random_state=randomstate, max_depth=profondeurMax)
        rfg.fit(X_train, y_train.values.ravel())
        predicted4 = rfg.predict(X_test)
        print("RMSLE " + str(profondeurMax) + " : " + str(rmsle(y_test, predicted4)))

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

def compartimenterHeures(x):
    h = x.time()
    if h < datetime.time(5):
        return 1#"matin"
    elif h < datetime.time(10):
        return 2#"rush_am"
    elif h < datetime.time(15):
        return 3#"pm"
    elif h < datetime.time(20):
        return 4#"rush_pm"
    else:
        return 5#"soir"

# pris du site de Kaggle
def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real.values[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real.values[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5