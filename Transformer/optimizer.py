import numpy as np
from typing import List
from modules import Parameter

class SGD:
    """SGD optimizer with momentum"""

    def __init__(self, parameters: List[Parameter], lr: float = 0.001,
                 momentum: float = 0.9, weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # velocity for momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        """update parameters"""
        for param, velocity in zip(self.parameters, self.velocities):
            # add weight decay (L2 regularization)
            if self.weight_decay > 0:
                param.grad = param.grad + self.weight_decay * param.data

            # update velocity
            velocity[:] = self.momentum * velocity - self.lr * param.grad

            # update parameter
            param.data += velocity

    def zero_grad(self):
        """zero all gradients"""
        for param in self.parameters:
            param.zero_grad()

    def set_lr(self, lr: float):
        """set learning rate (for use with schedulers)"""
        self.lr = lr


class AdamW:
    """AdamW optimizer"""

    def __init__(self, parameters: List[Parameter], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # first moment (momentum)
        self.m = [np.zeros_like(p.data) for p in parameters]
        # second moment (RMSprop)
        self.v = [np.zeros_like(p.data) for p in parameters]
        # time step
        self.t = 0

    def step(self):
        """update parameters"""
        self.t += 1

        for param, m, v in zip(self.parameters, self.m, self.v):
            # update biased first moment estimate
            m[:] = self.beta1 * m + (1 - self.beta1) * param.grad

            # update biased second moment estimate
            v[:] = self.beta2 * v + (1 - self.beta2) * (param.grad ** 2)

            # bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # update parameters with AdamW weight decay
            # AdamW: apply weight decay directly to parameters (decoupled)
            param.data = param.data * (1 - self.lr * self.weight_decay)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """zero all gradients"""
        for param in self.parameters:
            param.zero_grad()

    def set_lr(self, lr: float):
        """set learning rate (for use with schedulers)"""
        self.lr = lr
