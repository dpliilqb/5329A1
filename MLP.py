import os
import pickle

import numpy as np
from Modules import Activation, HiddenLayer, Layer, SoftmaxLayer, DropoutLayer, BatchNormalizationLayer

class MLP:
    # for initiallization, the code will create all layers automatically based on the provided parameters.
    def __init__(self):
        # initialize layers
        self.layers = []
        self.params = []

    def add(self, layer):
        self.layers.append(layer)

    # Forward process. Pass parameters within layers and save output.
    def forward(self, input, training=True):
        for layer in self.layers:
            # For Droupout, there's a training sign to tell it the current process is forward.
            if isinstance(layer, DropoutLayer):
                output = layer.forward(input, training=training)
            else:
                output = layer.forward(input)
            input = output

        return output

    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv
        # MSE
        error = y - y_hat
        loss = error ** 2
        # calculate the MSE's delta of the output layer
        delta = -error * activation_deriv(y_hat)
        # return loss and delta
        return loss, delta

    # backward progress
    def backward(self, delta, y_pred=None, y_true=None):
        if isinstance(self.layers[-1], SoftmaxLayer):
            delta = y_pred - y_true
            delta = self.layers[-1].backward(delta, True)
        else:
            delta = self.layers[-1].backward(delta)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!
    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'W') and getattr(layer, 'W', None) is not None:
                layer.W -= lr * layer.grad_W
                grad_b_sum = np.sum(layer.grad_b, axis=0)
                layer.b -= lr * grad_b_sum

    # define the training function
    # it will return all losses within the whole training process.

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output

    def save_model(self, path = "Saved Model", filename = "model.h5"):
        parameters = [layer.get_wnb() for layer in self.layers if
                      hasattr(layer, 'get_wng')]
        joint_path = os.path.join(path, filename)
        with open(joint_path, 'wb') as file:
            pickle.dump(parameters, file)

    def load_model(self, path = "Saved Model", filename = "model.h5"):
        joint_path = os.path.join(path, filename)
        with open(joint_path, 'rb') as file:
            parameters = pickle.load(file)
        for layer, param in zip(self.layers, parameters):
            if hasattr(layer, 'set_wnb'):
                layer.set_weights(param)