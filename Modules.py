import numpy as np
from scipy.special import erf


class Activation(object):
    """
    This class handles different activation functions for a neural network. Activation functions are crucial
    as they introduce non-linear properties to the network, which allows the network to learn more complex functions.

    Attributes:
        f (function): The activation function to be used in the network.
        f_deriv (function): The derivative of the activation function, used during the backpropagation to update weights.
    """

    def __tanh(self, x):
        """
        Computes the hyperbolic tangent of x element-wise.
        """
        return np.tanh(x)

    def __tanh_deriv(self, a):
        """
        Derivative of the hyperbolic tangent function. Assumes input 'a' is already the output of a tanh function.
        """
        return 1.0 - a ** 2

    def __logistic(self, x):
        """
        Logistic (sigmoid) function.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        """
        Derivative of the logistic function. Assumes input 'a' is the output of a logistic function.
        """
        return a * (1 - a)

    def __relu(self, x):
        """
        Rectified Linear Unit (ReLU) function, which outputs the max of 0 and x.
        """
        return np.maximum(0, x)

    def __relu_deriv(self, a):
        """
        Derivative of the ReLU function. Returns 1 for all inputs greater than 0.
        """
        return np.where(a > 0, 1, 0)

    def __gelu(self, x):
        """
        Gaussian Error Linear Unit (GELU) function, a smoother version of ReLU.
        """
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def __gelu_deriv(self, x):
        """
        Derivative of the GELU function.
        """
        return 0.5 * (1 + erf(x / np.sqrt(2))) + (x * np.exp(-0.5 * x ** 2)) / (np.sqrt(2 * np.pi))

    def __init__(self, activation='tanh'):
        """
        Initializes the Activation class with the specified activation function.
        """
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'gelu':
            self.f = self.__gelu
            self.f_deriv = self.__gelu_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv


class Layer:
    """
    A base class for layers in a neural network. This class is intended to be subclassed by specific types of layers
    that implement their own logic for forward and backward passes.

    Attributes:
        W (numpy.ndarray): The weight matrix for this layer. It should be initialized in derived classes.
        b (numpy.ndarray): The bias vector for this layer. It should be initialized in derived classes.
        grad_W (numpy.ndarray): Gradient of the loss with respect to the weight matrix, to be computed during backpropagation.
        grad_b (numpy.ndarray): Gradient of the loss with respect to the bias vector, to be computed during backpropagation.
    """
    def __init__(self):
        """
        Initializes the Layer with default values. Actual layers will override these with specific initial values
        appropriate to their type of computation.
        """
        self.W = None      # Weights of the layer, to be defined in subclass
        self.b = None      # Biases of the layer, to be defined in subclass
        self.grad_W = None # Gradient of weights, to be calculated during backpropagation
        self.grad_b = None # Gradient of biases, to be calculated during backpropagation

    def forward(self, input):
        raise NotImplementedError("Must be implemented in subclass.")

    def backward(self, input):
        raise NotImplementedError("Must be implemented in subclass.")

class SoftmaxLayer(Layer):
    """
    A softmax layer class that extends the base Layer class, specifically for use in classification tasks
    where softmax function is used to convert logits to probabilities which are easier to interpret.

    Attributes:
        output (numpy.ndarray): Stores the output of the softmax function to be used in backward pass.
    """
    def __init__(self):
        """
        Initializes the Softmax layer by calling the initializer of the base Layer class.
        """
        super().__init__()
        self.output = None

    def forward(self, input):
        """
        Performs the forward pass using the softmax activation function.

        Args:
            input (numpy.ndarray): Input data matrix of shape (N, C) where N is the batch size and
                                   C is the number of classes.

        Returns:
            numpy.ndarray: The result of applying the softmax function to the input, same shape as input.
        """
        # Numerical stability improvement by subtracting max from each row before exponentiating
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, is_cross_entropy=False):
        """
        Performs the backward pass of the softmax layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss function with respect to the output of this layer.
            is_cross_entropy (bool): Flag indicating whether the next layer's loss function is cross-entropy,
                                     which simplifies the gradient during backpropagation.

        Returns:
            numpy.ndarray: Gradient of the loss function with respect to the input of this layer.
        """
        if is_cross_entropy:
            # Directly return the gradient (y_pred - y_true) if the loss function is cross-entropy loss,
            # which simplifies the derivative
            dZ = output_gradient
        else:
            # Compute the general form gradient of softmax
            dZ = self.output * (output_gradient - np.sum(output_gradient * self.output, axis=1, keepdims=True))

        return dZ

class DropoutLayer(Layer):
    """
    A dropout layer that randomly disables a fraction of its neurons during training, preventing them from co-adapting
    too much. This helps to prevent overfitting by making the network's predictions less sensitive to the specific
    weights of individual neurons.

    Attributes:
        dropout_rate (float): The probability of dropping a neuron's activation during training.
        mask (numpy.ndarray): A binary mask indicating which neurons are kept during a forward pass. The same mask
                              is used during the backward pass to ensure consistency in gradients propagation.
    """
    def __init__(self, dropout_rate=0.5):
        """
        Initializes the Dropout layer with a specified dropout rate.

        Args:
            dropout_rate (float): Fraction of the neurons to drop out during the training phase.
        """
        super().__init__()
        self.dropout_rate = dropout_rate  # Probability of setting a neuron to zero
        self.mask = None

    def forward(self, input, training=True):
        """
        Computes the forward pass through the dropout layer.

        Args:
            input (numpy.ndarray): Input data from the previous layer.
            training (bool): If True, the dropout layer will randomly nullify some of the inputs. If False,
                             the layer will perform no dropout and just pass the input through.

        Returns:
            numpy.ndarray: The output of the dropout layer which is either element-wise multiplied by the dropout
                           mask (during training) or the unchanged input (during testing/prediction).
        """
        if training:
            # Generate a binary mask based on the dropout rate
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape) / (1 - self.dropout_rate)
            return input * self.mask  # Apply dropout to input by element-wise multiplication
        else:
            # During testing, no dropout is applied
            return input

    def backward(self, output_gradient):
        """
        Backward pass through the dropout layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss function with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient of the loss function with respect to the input of this layer, adjusted for the
                           dropout by multiplying with the same mask used during the forward pass.
        """
        # Only pass gradient where the neuron was not dropped out
        return output_gradient * self.mask

class BatchNormalizationLayer(Layer):
    """
    A batch normalization layer normalizes the input layer by adjusting and scaling the activations.
    This helps to mitigate the problem known as internal covariate shift, where the distribution of
    network activations changes during training.

    Attributes:
        gamma (numpy.ndarray): Scale factors which are learned during training.
        beta (numpy.ndarray): Shift factors which are learned during training.
        epsilon (float): A small constant added to the variance to avoid division by zero.
        num_features (int): The number of features in the input.
        running_mean (numpy.ndarray): The running mean of the features, used during inference.
        running_var (numpy.ndarray): The running variance of the features, used during inference.
        momentum (float): The momentum factor for updating running_mean and running_var.
    """
    def __init__(self, num_features, epsilon=1e-5):
        """
        Initializes the BatchNormalization layer.

        Args:
            num_features (int): Number of features in the input data.
            epsilon (float): Small float added to variance to avoid dividing by zero.
        """
        super().__init__()
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = epsilon
        self.num_features = num_features
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        self.momentum = 0.9

    def forward(self, input, training=True):
        """
        Forward pass for the batch normalization layer.

        Args:
            input (numpy.ndarray): Input data for normalization.
            training (bool): If True, the layer is in training mode; otherwise, it's in inference mode.

        Returns:
            numpy.ndarray: The normalized and scaled output.
        """
        if training:
            # Calculate mean and variance from the current batch
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)
            # Update running mean and variance for inference mode
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            # Normalize the batch and scale and shift
            self.x_normalized = (input - batch_mean) / np.sqrt(batch_var + self.epsilon)
            output = self.gamma * self.x_normalized + self.beta
            self.input = input  # Store input for use in backward pass
            return output
        else:
            # Normalize using running mean and variance for inference
            input_normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * input_normalized + self.beta

    def backward(self, output_gradient):
        """
        Backward pass for the batch normalization layer.

        Args:
            output_gradient (numpy.ndarray): Gradient of the loss function with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient of the loss function with respect to the input of this layer.
        """
        N = self.input.shape[0]  # Batch size

        # Gradient of gamma and beta
        self.grad_gamma = np.sum(output_gradient * self.x_normalized, axis=0)
        self.grad_beta = np.sum(output_gradient, axis=0)

        # Gradient of the input
        x_mu = self.input - np.mean(self.input, axis=0)
        std_inv = 1. / np.sqrt(np.var(self.input, axis=0) + self.epsilon)

        dx_normalized = output_gradient * self.gamma
        dvar = np.sum(dx_normalized * x_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dx_normalized * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

        grad_input = dx_normalized * std_inv + dvar * 2 * x_mu / N + dmu / N
        return grad_input

class HiddenLayer(Layer):
    """
    Represents a hidden layer in a neural network, which is a fully connected layer applying an activation function.

    Attributes:
        input (numpy.ndarray): The input to the layer.
        output (numpy.ndarray): The output from the layer.
        activation (callable): The activation function applied to the layer outputs.
        activation_deriv (callable): The derivative of the activation function, used during backpropagation.
        W (numpy.ndarray): Weight matrix for the layer.
        b (numpy.ndarray): Bias vector for the layer.
        grad_W (numpy.ndarray): Gradient of the layer's weights.
        grad_b (numpy.ndarray): Gradient of the layer's biases.
    """
    def __init__(self, n_in, n_out, activation=''):
        """
        Initializes the hidden layer with specified input size, output size, and activation function.

        Args:
            n_in (int): Number of input units.
            n_out (int): Number of output units.
            activation (str): Type of activation function to use ('logistic', 'tanh', 'relu', etc.).
        """
        super().__init__()
        self.input = None
        self.output = None
        self.activation = Activation(activation).f if activation is not None else None
        self.activation_deriv = Activation(activation).f_deriv if activation is not None else None

        # Initialize weights using a uniform distribution with a scale factor derived from the layer's size
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        # Optional modification for logistic activation for better initialization
        if activation == 'logistic':
            self.W *= 4

        # Initialize biases to zero
        self.b = np.zeros(n_out)

        # Initialize gradients as zeros
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):
        """
        Compute the forward pass through the hidden layer.

        Args:
            input (numpy.ndarray): Input data or the output from the previous layer.

        Returns:
            numpy.ndarray: Activated output from the layer.
        """
        # Linear transformation
        lin_output = np.dot(input, self.W) + self.b
        # Apply activation function (if any)
        self.output = self.activation(lin_output) if self.activation else lin_output
        self.input = input
        return self.output

    def backward(self, delta):
        """
        Perform backpropagation for this layer.

        Args:
            delta (numpy.ndarray): The gradient of the loss function with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient of the loss function with respect to the input of this layer.
        """
        # Compute gradients of weights and biases
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0)

        # Propagate gradient back to the previous layer (if not output layer)
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta

    def get_wnb(self):
        """
        Get current weights and biases of the layer.

        Returns:
            dict: Dictionary containing weights ('W') and biases ('b').
        """
        return {"W": self.W, "b": self.b}

    def set_wnb(self, param):
        """
        Set weights and biases with provided parameters.

        Args:
            param (dict): Dictionary containing weights ('W') and biases ('b') to be set.
        """
        self.W = param.get("W", "W not found")
        self.b = param.get("b", "b not found")