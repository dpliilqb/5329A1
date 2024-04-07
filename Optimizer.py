import numpy as np

class Momentum_Optimizer:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in parameters]

    def update(self):
        for i, param in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param['grad']
            param['value'] -= self.velocities[i]

class GradientDescentOptimizer:
    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self):
        for param in self.parameters:
            # 应用Weight decay
            if self.weight_decay > 0:
                param['grad'] += self.weight_decay * param['value']

            # 更新参数
            param['value'] -= self.lr * param['grad']


class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param['value']) for param in parameters]
        self.v = [np.zeros_like(param['value']) for param in parameters]
        self.t = 0

    def update(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            # 计算梯度的一阶矩和二阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param['grad']
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param['grad'] ** 2)

            # 对一阶矩和二阶矩估计进行偏差校正
            m_corrected = self.m[i] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            param_update = self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            param['value'] -= param_update
