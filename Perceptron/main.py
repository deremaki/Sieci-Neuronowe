import numpy as np
import csv
import sys
import os
import copy
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from visualizations import *

#from mlp import MLP

def read_csv(filePath):
    elements = []
    with open(filePath, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        rows.__next__()
        for row in rows:
            elements.append([float(i) for i in row])
    return elements

def get_inputs(elements):
    inputs = []
    for row in elements:
            [inputs.append(row[:-1])]
    return inputs

def get_outputs(elements):
    outputs = []
    for row in elements:
            outputs.append(row[-1])
    return outputs

def main(argv):
     
    np.random.seed(123815)

  #  trainFilePath = argv[1]
  #  testFilePath = argv[2]
     
    # trainFilePath = os.getcwd() + "\data\\regression\data.activation.train.100.csv"
    # testFilePath = os.getcwd() + "\data\\regression\data.activation.test.100.csv"

    trainFilePath = os.getcwd() + "\data\\classification\data.simple.train.100.csv"
    testFilePath = os.getcwd() + "\data\\classification\data.simple.test.100.csv"

    #true - regression, false - classification
    regression = False

    perceptron_visualization = True
    regression_visualization = True
    classification_visualization = True


    train_elements = read_csv(trainFilePath)
    test_elements = read_csv(testFilePath)

    train_X = get_inputs(train_elements)
    train_Y = get_outputs(train_elements)

    test_X = get_inputs(test_elements)
    test_Y = get_outputs(test_elements)

    n = len(train_elements)

    if(regression):
        clf = MLPRegressor(hidden_layer_sizes=(5), activation='logistic', solver='lbfgs')

        if(perceptron_visualization == True):
            print("visualize")
            weights = []

            for i in range(n):
                X = np.asarray(train_X[i]).reshape(1, -1)
                Y = np.ravel(np.asarray(train_Y[i]).reshape(1, -1))
                clf.fit(X, Y)
                
                if(i%99==0):
                    old_weights = copy.deepcopy(weights)
                    weights = copy.deepcopy(clf.coefs_)
                    draw_perceptron(i, old_weights, weights)
        else:          
            clf.fit(train_X, train_Y)
            
        print("fitted")
        predicted_Y = clf.predict(test_X)

        error = 0
        for i in range(n):
            error += abs(test_Y[i]-predicted_Y[i])
        mean_error = error/n

        print('Mean error: ', mean_error)

        if(regression_visualization):
            draw_regression(test_X, test_Y, predicted_Y)

    else: #classification
        
        clf = MLPClassifier(hidden_layer_sizes=(5), activation='logistic', solver='lbfgs')

        if(perceptron_visualization == True):
            print("visualize")
            weights = []

            for i in range(n):
                X = np.asarray(train_X[i]).reshape(1, -1)
                Y = np.ravel(np.asarray(train_Y[i]).reshape(1, -1))
                clf.fit(X, Y)
                
                if(i%99==0):
                    old_weights = copy.deepcopy(weights)
                    weights = copy.deepcopy(clf.coefs_)
                    draw_perceptron(i, old_weights, weights)

        else:    
            clf.fit(train_X, train_Y)


        print("fitted")
        predicted_Y = clf.predict(test_X)

        correctly_classified = 0

        for i in range(n):
            if(test_Y[i] == predicted_Y[i]):
                correctly_classified += 1
        accuracy = correctly_classified/len(predicted_Y)*100

        print('Accuracy: ', accuracy, '%')

        if(classification_visualization):
            draw_classification(test_elements, clf)


if(__name__ == "__main__"):
    main(sys.argv[1:])