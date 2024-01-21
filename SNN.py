from keras.models import Sequential
from keras.layers import Dense
import numpy as np
class SNN:
    def relu(x):
        return np.maximum(0, x)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def forward_propagation(inputs, weights, biases):
        hidden_layer_input = np.dot(inputs, weights) + biases
        
        print(hidden_layer_input)
        hidden_layer_output = SNN.relu(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights) + biases
        output = SNN.sigmoid(output_layer_input)
        return output

    def build_neural_network(n_inputs, n_outputs, layers, neurons):
        model = Sequential()
        
        # Input layer
        model.add(Dense(neurons, input_dim=n_inputs, activation='relu'))
        
        # L-1 hidden layers, each with N neurons
        for _ in range(layers-1):
            model.add(Dense(neurons, activation='relu'))
        
        # Output layer
        model.add(Dense(n_outputs, activation='softmax')) # Or another activation function depending on your use case
        
        return model
        