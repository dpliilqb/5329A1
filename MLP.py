import numpy as np
from Modules import Activation, HiddenLayer, Layer, SoftmaxLayer, DropoutLayer, BatchNormalizationLayer, GELULayer

class MLP:
    # for initiallization, the code will create all layers automatically based on the provided parameters.
    def __init__(self, layers, activation):
        # initialize layers
        self.layers = []
        self.params = []
        self.activation = activation
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

    # backward progress
    def backward(self, delta):
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!
    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self, X, y, learning_rate=0.1, epochs=100):
        X = np.array(X)
        y = np.array(y)
        to_return = np.zeros(epochs)

        for k in range(epochs):
            loss = np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])

                # forward pass
                y_hat = self.forward(X[i])

                # backward pass
                loss[it], delta = self.criterion_MSE(y[i], y_hat)
                self.backward(delta)
                y
                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        return output