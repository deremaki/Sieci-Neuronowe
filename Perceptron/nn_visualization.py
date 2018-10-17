from matplotlib import pyplot as plt
from math import cos, sin, atan
import numpy as np
from scipy import ndimage

def get_normalized_width(min_x, max_x, x):
    return (x-min_x)/(max_x-min_x)

def min_max_weights(weight):
    mx = 0
    mn = 0
    for i in range(len(weight)):
        if(i==0):
            mx = weight[i].max()
            mn = weight[i].min()
        else:
            if(weight[i].max() > mx):
                mx = weight[i].max()
            if(weight[i].min() < mn):
                mn = weight[i].min()
    return mn, mx




class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        plt.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weights, old_weights, layer_no, min_weight, max_weight):
        self.vertical_distance_between_layers = 10
        self.horizontal_distance_between_neurons = 8
        self.neuron_radius = 1
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        self.old_weights = old_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.layer_no = layer_no

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, n1i, n2i, n1l, n2l):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = plt.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        xy = line.get_xydata()
        current_weight = self.previous_layer.weights[n2i, n1i]
        line.set_color("black")
        line.set_label("test")
        width = get_normalized_width(self.min_weight, self.max_weight, current_weight)
        width = width * 2 + 0.25
        line.set_linewidth(width)
        plt.gca().add_line(line)
        label = ""
        if (len(self.old_weights)!=0 or (len(self.previous_layer.old_weights)!=0)):
            difference = self.previous_layer.weights[n2i, n1i] - self.previous_layer.old_weights[n2i, n1i]
            if(difference > 0):
                label = ("%.2f" % current_weight) + "\n(+%.4f)" % difference
            else:
                label = ("%.2f" % current_weight) + "\n(%.4f)" % difference
            if(difference>0):
                line.set_color("green")
            if(difference<0):
                line.set_color("red")
        else:
            label = ("%.2f" % current_weight)
        vertical_shift = 0
        if(n2i%2==1):
            vertical_shift = 1
        plt.text((xy[0][0] + xy[1][0])/2 + 0.5, (xy[0][1] + xy[1][1])/2 + vertical_shift, label, fontsize=7, color="blue")

    def draw(self, layerType=0):
        for i in range(len(self.neurons)):
            self.neurons[i].draw( self.neuron_radius )
            if self.previous_layer:
                for j in range(len(self.previous_layer.neurons)):
                    self.__line_between_two_neurons(self.neurons[i], self.previous_layer.neurons[j], i, j, self.layer_no, self.previous_layer.layer_no)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            plt.text(x_text, self.y, 'Input Layer', fontsize = 8)
        elif layerType == -1:
            plt.text(x_text, self.y, 'Output Layer', fontsize = 8)
        else:
            plt.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 8)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, iter_no):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0
        self.iter_no = iter_no

    def add_layer(self, number_of_neurons, weights, old_weights, layer_no, min_weight, max_weight ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, weights, old_weights, layer_no, min_weight, max_weight)
        self.layers.append(layer)

    def draw(self):
        plt.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        plt.axis('scaled')
        plt.axis('off')
        plt.title( 'Neural Network architecture, iter: ' + str(self.iter_no), fontsize=10 )
        plt.show()

class DrawNN():
    def __init__( self, neural_network, iter_no, old_weights, weights):
        self.neural_network = neural_network
        self.iter_no = iter_no
        self.old_weights = old_weights
        self.weights = weights
        min_w, max_w = min_max_weights(weights)
        self.max_weight = max_w
        self.min_weight = min_w

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer, self.iter_no )
        for i in range(len(self.neural_network)):
            if(i==len(self.neural_network)-1):
                    network.add_layer(self.neural_network[i], [],[], i, self.min_weight, self.max_weight)
            else:
                if(len(self.old_weights)==0):
                    network.add_layer(self.neural_network[i], self.weights[i], [], i, self.min_weight, self.max_weight)
                else:
                    network.add_layer(self.neural_network[i], self.weights[i], self.old_weights[i], i, self.min_weight, self.max_weight)
        network.draw()