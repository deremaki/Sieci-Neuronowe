import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nn_visualization import *

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def draw_regression(test_X, test_Y, predict_Y):

    plt.plot(test_X, test_Y, 'r-', test_X, predict_Y, 'b--')

    plt.title("MLP Regression")
    plt.show()


def draw_classification(test_elements_array, classifier):
    colors=['r', 'b', 'g']
    markers = [u'x', u'+', u'*']
    h = .02
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # elements = []
    # for point in mesh_points:
    #     elements.append(point[0], point[1], 1)

    z = np.array(classifier.predict(mesh_points))
    z = z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, z, cmap=cmap_light)

    for e in test_elements_array:
        plt.scatter(e[0], e[1], marker=markers[int(e[2])%3-1], cmap=cmap_bold, s=15, c=colors[int(e[2])%3-1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("MLP Classification")
    plt.show()


def draw_perceptron(iteration, old_weights, weights):
    
    network = []
    for i in range(len(weights)):
        network.append(len(weights[i]))

    network.append(len(weights[len(weights)-1][0])) #output layer

    perceptron = DrawNN(network, iteration, old_weights, weights)
    perceptron.draw()