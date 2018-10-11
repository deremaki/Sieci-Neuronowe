import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def draw_classification(train_elements_array, classifier):
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

    for e in train_elements_array:
        plt.scatter(e[0], e[1], marker=markers[int(e[2])%3-1], cmap=cmap_bold, s=15, c=colors[int(e[2])%3-1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("MLP Classification")
    plt.show()
