import numpy as np

class Momentum_Optimizer:
    '''
    This class implements the Momentum Optimization algorithm.
    '''
    def __init__(self, layers, lr=0.01, momentum=0.9):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.w_layers = [layer for layer in self.layers if hasattr(layer, 'W') and getattr(layer, 'W', None) is not None]
        self.velocities_W = [np.zeros_like(l.W) for l in self.w_layers]
        self.velocities_b = [np.zeros_like(l.b) for l in self.w_layers]

    def update(self):
        '''
        This method updates the weights of the layers.
        :return:
        '''
        for i, layer in enumerate(self.w_layers):
            self.velocities_W[i] = self.momentum * self.velocities_W[i] + self.lr * layer.grad_W
            layer.W -= self.velocities_W[i]

            # grad_b_sum = np.sum(layer.grad_b, axis=0)
            self.velocities_b[i] = self.momentum * self.velocities_b[i] + self.lr * layer.grad_b
            layer.b -= self.velocities_b[i]

class Weight_Decay_Optimizer:
    def __init__(self, layers, lr=0.01, alpha=0.01):
        self.layers = layers
        self.w_layers = [layer for layer in self.layers if hasattr(layer, 'W') and getattr(layer, 'W', None) is not None]
        self.lr = lr
        self.alpha = alpha

    def update(self):
        for i, layer in enumerate(self.w_layers):
            # 应用Weight decay
            if self.alpha > 0:
                layer.grad_W += self.alpha * layer.W
                layer.grad_b += self.alpha * layer.b

            # 更新参数
            layer.W -= self.lr * layer.grad_W
            # grad_b_sum = np.sum(layer.grad_b, axis=0)
            layer.b -= self.lr * layer.grad_b


class Adam_Optimizer:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.w_layers = [layer for layer in self.layers if hasattr(layer, 'W') and getattr(layer, 'W', None) is not None]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = [np.zeros_like(layer.W) for layer in self.w_layers]
        self.v_W = [np.zeros_like(layer.W) for layer in self.w_layers]
        self.m_b = [np.zeros_like(layer.b) for layer in self.w_layers]
        self.v_b = [np.zeros_like(layer.b) for layer in self.w_layers]
        self.t = 0

    def update(self):
        self.t += 1
        for i, layer in enumerate(self.w_layers):
            # 计算梯度的一阶矩和二阶矩估计
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)

            # 对一阶矩和二阶矩估计进行偏差校正
            m_W_corrected = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_W_corrected = self.v_W[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            W_param_update = self.lr * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            # 对一阶矩和二阶矩估计进行偏差校正
            m_b_corrected = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_corrected = self.v_b[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            b_param_update = self.lr * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)

            layer.W -= W_param_update
            # grad_b_sum = np.sum(layer.grad_b, axis=0)
            layer.b -= layer.grad_b
