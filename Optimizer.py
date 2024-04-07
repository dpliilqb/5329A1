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