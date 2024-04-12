import os
import pickle

import numpy as np
from Modules import Layer, SoftmaxLayer, DropoutLayer

class MLP:
    """
    A simple implementation of a Multilayer Perceptron (MLP) neural network.

    Attributes:
        layers (list): A list to store the layers of the neural network.
    """
    def __init__(self):
        """
        Initializes an empty MLP model with no layers.
        """
        self.layers = []

    def add(self, layer):
        """
        Add a layer to the neural network.

        Args:
            layer (Layer): The layer to be added to the network.
        """
        self.layers.append(layer)

    def forward(self, input, training=True):
        """
        Perform the forward propagation through the network.

        Args:
            input (array-like): The input data.
            training (bool): Flag to indicate whether the forward pass is for training.

        Returns:
            output (array-like): The output of the last layer after forward propagation.
        """
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                # For Dropout, there's a training sign to tell it the current process is forward.
                output = layer.forward(input, training=training)
            else:
                output = layer.forward(input)
            input = output
        return output

    def backward(self, delta, y_pred=None, y_true=None):
        """
        Perform backward propagation through the network.

        Args:
            delta (array-like): The initial gradient delta.
            y_pred (array-like): Predicted labels (used if the last layer is Softmax).
            y_true (array-like): True labels (used if the last layer is Softmax).
        """
        if isinstance(self.layers[-1], SoftmaxLayer):
            # The gradient of cross entropy and softmax is (y_pred - y_true)
            delta = y_pred - y_true
            delta = self.layers[-1].backward(delta, True)
        else:
            delta = self.layers[-1].backward(delta)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def update(self, lr):
        """
        Update the network weights using gradients computed during backpropagation.

        Args:
            lr (float): The learning rate.
        """
        for layer in self.layers:
            if hasattr(layer, 'W') and getattr(layer, 'W') is not None:
                layer.W -= lr * layer.grad_W
                layer.b -= lr * layer.grad_b

    def predict(self, x):
        """
        Predict the output for given input using the trained network.

        Args:
            x (array-like): The input data.

        Returns:
            output (array-like): The predicted output.
        """
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output

    def save_model(self, path="Saved Model", filename="model.h5"):
        """
        Save the model parameters to a file.

        Args:
            path (str): The directory path to save the model.
            filename (str): The filename to save the model.
        """
        # model_dict: store layers and params.
        model_dict = {"layers": self.layers, "params": []}

        for layer in self.layers:
            if hasattr(layer, 'get_wnb'):
                model_dict["params"].append(layer.get_wnb)
            else:
                model_dict["params"].append({})
        joint_path = os.path.join(path, filename)
        with open(joint_path, 'wb') as file:
            pickle.dump(model_dict, file)

    def load_model(self, path="Saved Models", filename="model.h5"):
        """
        Load the model parameters from a file.

        Args:
            path (str): The directory path to load the model from.
            filename (str): The filename to load the model from.
        """
        joint_path = os.path.join(path, filename)
        with open(joint_path, 'rb') as file:
            model_dict = pickle.load(file)
            self.layers = model_dict["layers"]
        for layer, param in zip(model_dict["layers"], model_dict["params"]):
            if hasattr(layer, 'set_wnb'):
                layer.set_wnb(param)
