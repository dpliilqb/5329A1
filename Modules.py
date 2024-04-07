import numpy as np

class Activation(object):
    def __init__(self, activation='tanh'):
        self.f = self.__tanh
        self.f_deriv = self.__tanh_deriv
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_deriv(self, a):
        return np.where(a > 0, 1, 0)


class Layer:
    def __init__(self):
        pass
    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError

class SoftmaxLayer(Layer):
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        N = output_gradient.shape[0]
        dZ = self.output * (output_gradient - np.sum(output_gradient * self.output, axis=1, keepdims=True))
        return dZ

class DropoutLayer(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape) / (1 - self.dropout_rate)
            return input * self.mask
        else:
            return input

    def backward(self, output_gradient):
        return output_gradient * self.mask

class BatchNormalizationLayer(Layer):
    def __init__(self, num_features, epsilon=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = epsilon
        self.num_features = num_features
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        self.momentum = 0.9

    def forward(self, input, training=True):
        if training:
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.x_normalized = (input - batch_mean) / np.sqrt(batch_var + self.epsilon)
            output = self.gamma * self.x_normalized + self.beta
            self.input = input
            return output
        else:
            input_normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * input_normalized + self.beta

    def backward(self, output_gradient):
        N = self.input.shape[0]

        # Gradient of gamma and beta
        self.grad_gamma = np.sum(output_gradient * self.x_normalized, axis=0)
        self.grad_beta = np.sum(output_gradient, axis=0)

        # Compute gradients of input
        x_mu = self.input - np.mean(self.input, axis=0)
        std_inv = 1. / np.sqrt(np.var(self.input, axis=0) + self.epsilon)

        dx_normalized = output_gradient * self.gamma
        dvar = np.sum(dx_normalized * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dx_normalized * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        grad_input = dx_normalized * std_inv + dvar * 2 * x_mu / N + dmu / N
        return grad_input

class GELULayer(Layer):
    def forward(self, input):
        self.input = input
        return 0.5 * input * (1 + np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * np.power(input, 3))))

    def backward(self, output_gradient):
        x = self.input
        tanh_out = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        sec_h_square = 1 / np.cosh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) ** 2
        first_term = 0.5 * (1 + tanh_out)
        second_term = 0.5 * x * (1 + 0.044715 * 3 * np.power(x, 2)) * np.sqrt(2 / np.pi) * sec_h_square
        grad_input = output_gradient * (first_term + second_term)
        return grad_input

class HiddenLayer(Layer):
    def __init__(self, n_in, n_out,
                 activation_last_layer='tanh', activation='tanh', W=None, b=None):

        self.input = None
        self.output = None
        self.activation = Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        # if activation == 'logistic':
        #     self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out, )

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    # the forward and backward progress (in the hidden layer level) for each training epoch
    # please learn the week2 lec contents carefully to understand these codes.
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta