#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn import preprocessing
import pylab as pl
import time
import os

def main(data, k, foutModel, classifyMetric):

        # loads data
        print ("Loading data...")
        X_data, y_data = load_svmlight_file(data)
        # splits data
        print ("Spliting data...")
        X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.5, random_state = 5)

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fazer a normalizacao dos dados #######
        #scaler = preprocessing.MinMaxScaler()
        #X_train = scaler.fit_transform(X_train_dense)
        #X_test = scaler.fit_transform(X_test_dense)
        
        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=k, metric=classifyMetric)

        print ('Fitting knn')
        neigh.fit(X_train, y_train)

        

        # predicao do classificador
        print ('Predicting...')
        start = time.time()

        y_pred = neigh.predict(X_test)
        
        end = time.time()
        execTime = str((end - start))

        # mostra o resultado do classificador na base de teste
        #print ('Accuracy: ',  neigh.score(X_test, y_test))
        accuracy = str(neigh.score(X_test, y_test))
        foutModel.write('Accuracy: '+accuracy+"\n")

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        foutModel.write(str(cm)+"\n")

        #print(classification_report(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9]))
        foutModel.write(str(classification_report(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9]))+"\n")

        f1s = str(f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted'))

        foutModel.write("f1score: "+str(f1s)+"\n")

        filePoints = open("results.csv","a+")
        filePoints.write(os.path.basename(foutModel.name)+", "+str(k)+", "+classifyMetric+", "+accuracy+", "+f1s+", "+execTime+"\n")