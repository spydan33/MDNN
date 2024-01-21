import numpy as numpy

class layer:
    def __init__(self, N_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputes, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputes):
        self.output = np.dot(inputes, self.weights) + self.biases
    weights: False

class activation_relu:
    def forword(self, inputes):
        self.output = np.maximum(0,inputs)

class activation_softmax:
    def forword(self,inputs):
        exp_values = np.exp.(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
