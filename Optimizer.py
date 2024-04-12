import numpy as np

class Momentum_Optimizer:
    '''
    A class that implements the Momentum optimization algorithm, commonly used to accelerate the convergence
    in the training of neural networks by stabilizing the updates.

    Attributes:
        layers (list): A list of layers of the neural network that have weights and biases.
        lr (float): Learning rate, determines the step size at each iteration while moving toward a minimum of a loss function.
        momentum (float): Momentum coefficient, used to improve convergence and reduce oscillations.
        w_layers (list): Filtered list of layers that actually contain weights.
        velocities_W (list): List of velocity arrays for the weights of each layer.
        velocities_b (list): List of velocity arrays for the biases of each layer.
    '''
    def __init__(self, layers, lr=0.01, momentum=0.9):
        """
        Initializes the Momentum optimizer.

        Args:
            layers (list): The list of layers from the neural network model.
            lr (float): The learning rate for the optimizer.
            momentum (float): The momentum coefficient to apply to the weight updates.
        """
        self.layers = layers
        self.lr = lr
        self.momentum = momentum

        # Prepare lists to store velocities for weights and biases for layers with weights
        self.w_layers = [layer for layer in self.layers if hasattr(layer, 'W') and getattr(layer, 'W') is not None]
        self.velocities_W = [np.zeros_like(l.W) for l in self.w_layers]
        self.velocities_b = [np.zeros_like(l.b) for l in self.w_layers]

    def update(self):
        """
        Updates the weights and biases of the network's layers using the Momentum optimization algorithm.

        Description:
        This method computes the update for each layer based on its gradient and the previous velocity.
        It adjusts the weights and biases in the direction that minimizes the loss, incorporating momentum
        to smooth out the updates.
        """
        for i, layer in enumerate(self.w_layers):
            # Update the velocity for the weights and apply it to the weights
            self.velocities_W[i] = self.momentum * self.velocities_W[i] + self.lr * layer.grad_W
            layer.W -= self.velocities_W[i]

            # Update the velocity for the biases and apply it to the biases
            self.velocities_b[i] = self.momentum * self.velocities_b[i] + self.lr * layer.grad_b
            layer.b -= self.velocities_b[i]


class Weight_Decay_Optimizer:
    """
    A class that implements weight decay optimization, often used as a regularization technique
    to prevent overfitting by adding a penalty on the size of the weights.

    Attributes:
        layers (list): A list of layers of the neural network that have weights and biases.
        w_layers (list): Filtered list of layers that actually contain weights.
        lr (float): Learning rate, determines the step size at each iteration while moving toward a minimum of a loss function.
        alpha (float): Weight decay coefficient, used as the regularization parameter.
    """
    def __init__(self, layers, lr=0.01, alpha=0.01):
        """
        Initializes the Weight Decay optimizer.

        Args:
            layers (list): The list of layers from the neural network model.
            lr (float): The learning rate for the optimizer.
            alpha (float): The weight decay coefficient, acting as the regularization term.
        """
        self.layers = layers
        self.w_layers = [layer for layer in self.layers if hasattr(layer, 'W') and getattr(layer, 'W') is not None]
        self.lr = lr
        self.alpha = alpha

    def update(self):
        """
        Updates the weights and biases of the network's layers using weight decay for regularization.

        Description:
        This method adjusts the weights by applying weight decay, which discourages large weights
        through a regularization term that is added to the loss. This helps prevent overfitting.
        """
        for i, layer in enumerate(self.w_layers):
            # Apply weight decay by modifying gradients with the weight decay term
            if self.alpha > 0:
                layer.grad_W += self.alpha * layer.W
                layer.grad_b += self.alpha * layer.b

            # Update weights and biases according to the modified gradients
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b



class Adam_Optimizer:
    """
    A class implementing the Adam (Adaptive Moment Estimation) optimization algorithm,
    which is a popular optimizer used for training deep learning models. Adam combines
    the advantages of two other extensions of stochastic gradient descent, namely
    Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).

    Attributes:
        layers (list): A list of layers in the neural network that have trainable weights and biases.
        w_layers (list): A filtered list of layers that have weights.
        lr (float): Learning rate, determines the step size at each iteration.
        beta1 (float): Exponential decay rate for the first moment estimates (similar to momentum).
        beta2 (float): Exponential decay rate for the second-moment estimates (similar to RMSprop).
        epsilon (float): A small constant to prevent any division by zero in the implementation.
        m_W (list): First moment vector (mean) for weights, initialized as zero.
        v_W (list): Second moment vector (uncentered variance) for weights, initialized as zero.
        m_b (list): First moment vector for biases, initialized as zero.
        v_b (list): Second moment vector for biases, initialized as zero.
        t (int): Timestep, used for bias-correction in moment estimates.
    """
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the Adam optimizer with the specified learning rate, decay rates, and epsilon value.
        """
        self.layers = layers
        self.w_layers = [layer for layer in self.layers if hasattr(layer, 'W') and getattr(layer, 'W') is not None]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = [np.zeros_like(layer.W) for layer in self.w_layers]
        self.v_W = [np.zeros_like(layer.W) for layer in self.w_layers]
        self.m_b = [np.zeros_like(layer.b) for layer in self.w_layers]
        self.v_b = [np.zeros_like(layer.b) for layer in self.w_layers]
        self.t = 0  # Initialization of the timestep counter

    def update(self):
        """
        Updates parameters (weights and biases) of the network layers using the Adam optimization algorithm.
        """
        self.t += 1  # Increment the timestep
        for i, layer in enumerate(self.w_layers):
            # Update the first and second moment estimates for weights
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)

            # Bias-correct the first and second moment estimates
            m_W_corrected = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_W_corrected = self.v_W[i] / (1 - self.beta2 ** self.t)

            # Compute the update for weight parameters
            W_param_update = self.lr * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
            layer.W -= W_param_update

            # Update the first and second moment estimates for biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            # Bias-correct the first and second moment estimates for biases
            m_b_corrected = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_corrected = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Compute the update for bias parameters
            b_param_update = self.lr * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
            layer.b -= b_param_update

