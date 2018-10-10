import numpy as np

np.random.seed(123815)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return x*(1-x)

class Neuron:
    def __init__(self, inputs_no):
        self.inputs_no = inputs_no
        self.set_weights( [np.random.uniform(-1,1) for x in range(0, self.inputs_no) ] )
    
    def set_weights(self, weights):
        self.weights = weights

    def sum(self, inputs):
        return sum(val*self.weights[i] for i,val in enumerate(inputs))

class NeuronLayer:
    def __init__(self, neurons_no, inputs_no):
        self.neurons_no = neurons_no
        self.neurons = [Neuron(inputs_no) for _ in range(0, self.neurons_no)]

class MLP:
    def __init__(self, inputs_no, outputs_no, neurons_in_hl, hidden_layers_no):
            self.inputs_no = inputs_no
            self.outputs_no = outputs_no
            self.neurons_in_hl = neurons_in_hl
            self.hidden_layers_no = hidden_layers_no

            #input layer
            self.layers = [NeuronLayer(self.neurons_in_hl, self.inputs_no)]

            #hidden layers
            self.layers += [NeuronLayer(self.neurons_in_hl, self.neurons_in_hl) for _ in range(0, self.hidden_layers_no)]

            #output layer
            self.layers += [NeuronLayer(self.outputs_no, self.neurons_in_hl)]